import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import argparse
import json
import logging
import wandb
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from src.model import CubeDiff
from src.dataset import Sun360Dataset
from src.positional_encoding import generate_cubemap_positional_encoding
from src.cubemap_utils import cubemap_to_equirectangular

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    Create learning rate scheduler with warm-up as described in the paper
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up phase
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, name=None):
    """
    Save training checkpoint
    """
    if name:
        checkpoint_path = os.path.join(args.output_dir, f"{name}.pt")
    else:
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}-step-{global_step}.pt")

    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": vars(args),
    }, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model weights separately for easier loading
    if name == "final_model":
        weights_path = os.path.join(args.output_dir, "model_weights.pt")
        torch.save(model.state_dict(), weights_path)
        logger.info(f"Saved model weights to {weights_path}")


def generate_validation_images(model, validation_data, device, output_dir, step):
    """
    Generate validation images using the current model
    """
    model.eval()
    validation_output_dir = os.path.join(output_dir, f"validation_step_{step}")
    os.makedirs(validation_output_dir, exist_ok=True)

    from src.inference import generate_panorama

    for i, data in enumerate(validation_data):
        file_name = data["file_name"]

        # Get front face as conditioning
        condition_image = data["images"][0].unsqueeze(0).to(device)

        # Create empty text prompt
        empty_prompts = [""] * 6

        # Generate panorama
        generated_faces = generate_panorama(
            model=model,
            prompts=empty_prompts,  # Using empty prompts for image-only
            condition_image=condition_image,
            num_inference_steps=30,  # Faster for validation
            device=device,
            output_type="pil"
        )

        # Save individual faces
        for j, face in enumerate(generated_faces):
            face_path = os.path.join(validation_output_dir, f"img_{i}_face_{j}.png")
            face.save(face_path)

        # Save equirectangular
        try:
            equirect = cubemap_to_equirectangular(generated_faces)
            equirect_path = os.path.join(validation_output_dir, f"img_{i}_panorama.png")
            equirect.save(equirect_path)

            # Log to wandb if available
            if args.use_wandb:
                wandb.log({
                    f"validation_img_{i}": wandb.Image(equirect_path),
                    "step": step
                })
        except Exception as e:
            logger.error(f"Error generating equirectangular: {e}")

    model.train()


def load_validation_data(data_root, num_samples=4, image_size=512):
    """
    Load a subset of the validation data for generating samples during training
    """
    validation_dataset = Sun360Dataset(
        data_root=data_root,
        image_size=image_size,
        split="test"
    )

    indices = np.random.choice(
        len(validation_dataset),
        min(num_samples, len(validation_dataset)),
        replace=False
    )

    validation_data = []
    for idx in indices:
        validation_data.append(validation_dataset[idx])

    return validation_data


def calculate_v_prediction_loss(latents, noise, noise_pred, timesteps, noise_scheduler):
    """
    Calculate v-prediction loss as described in the paper
    """
    # Get alpha_t and sigma_t from timesteps
    alpha_t = noise_scheduler.alphas_cumprod[timesteps]
    sigma_t = torch.sqrt(1 - alpha_t)

    # Reshape alpha and sigma for broadcasting
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    sigma_t = sigma_t.view(-1, 1, 1, 1)

    # Calculate v-target from the paper
    v_target = alpha_t.sqrt() * noise - sigma_t * latents

    # Calculate MSE loss between predicted v and target v
    loss = F.mse_loss(v_target, noise_pred, reduction="mean")

    return loss


def train_model(args):
    """
    Train CubeDiff model with Sun360 dataset using image-only conditioning
    """
    # Set seed for reproducibility
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    logger.info(f"Configuration saved to {config_path}")

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="cubediff", config=vars(args))

    # Initialize model
    logger.info(f"Initializing model from {args.pretrained_model_path}")
    model = CubeDiff(
        pretrained_model_path=args.pretrained_model_path,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees,
        vae_scale_factor=args.vae_scale_factor
    )
    model.to(args.device)

    # Set up gradient accumulation if needed
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size}")

    # Set up optimizer with weight decay
    param_groups = [
        {
            "params": [p for n, p in model.unet.named_parameters()
                      if not any(nd in n for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.unet.named_parameters()
                      if any(nd in n for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    # Set up dataset and dataloader
    logger.info(f"Loading dataset from {args.data_root}")
    train_dataset = Sun360Dataset(
        data_root=args.data_root,
        image_size=args.image_size,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees,
        split="train"
    )

    # Set up distributed training if requested
    if args.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(train_distributed, args=(world_size, args), nprocs=world_size, join=True)
        return

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device != "cpu" else False,
        drop_last=True
    )

    # Load validation data
    validation_data = load_validation_data(
        args.data_root,
        num_samples=args.validation_samples,
        image_size=args.image_size
    )

    # Calculate total training steps
    total_steps = args.num_epochs * len(train_dataloader) // args.gradient_accumulation_steps

    # Set up learning rate scheduler with warm-up
    warmup_steps = int(args.warmup_ratio * total_steps)
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    logger.info(f"Training for {args.num_epochs} epochs, {total_steps} total steps, {warmup_steps} warmup steps")

    # Initialize noise scheduler
    noise_scheduler = model.get_noise_scheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule
    )

    # Initialize mixed precision scaler if requested
    scaler = GradScaler() if args.mixed_precision and args.device != "cpu" else None

    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=args.device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1

        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]

        logger.info(f"Resumed from epoch {start_epoch}, global step {global_step}")

    # Pre-generate empty text embeddings once
    empty_prompt = ""
    empty_text_embeddings = model.encode_text([empty_prompt] * 6, args.device)

    # Training loop
    model.train()
    logger.info("Starting training...")

    # Define nullcontext for use when mixed_precision is False
    class nullcontext:
        def __enter__(self): return None
        def __exit__(self, *args): pass

    # For gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Get data from batch
            cubemap_images = batch["images"].to(args.device)  # [B, 6, 3, H, W]

            # Reshape for processing
            batch_size, num_faces, channels, height, width = cubemap_images.shape
            images = cubemap_images.view(batch_size * num_faces, channels, height, width)

            # Training step with optional mixed precision
            with autocast() if args.mixed_precision and args.device != "cpu" else nullcontext():
                # Encode images to latents
                with torch.no_grad():
                    latents = model.vae.encode(images).latent_dist.sample() * args.vae_scale_factor

                # Add noise to latents
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size * num_faces,),
                    device=args.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Generate position encodings
                pos_enc = generate_cubemap_positional_encoding(
                    batch_size, latents.shape[2], latents.shape[3], device=args.device)

                # Apply classifier-free guidance dropout (randomly drop text condition)
                if np.random.random() < args.text_dropout_prob:
                    # Unconditional (no text)
                    text_embeddings = empty_text_embeddings.repeat(batch_size * num_faces // 6, 1, 1)
                else:
                    # Conditional (empty text, but could be replaced with real text)
                    text_embeddings = empty_text_embeddings.repeat(batch_size * num_faces // 6, 1, 1)

                # Forward pass for noise prediction
                noise_pred = model.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False
                )[0]

                # Calculate loss
                if args.v_prediction:
                    # Calculate v-prediction loss as per the paper
                    loss = calculate_v_prediction_loss(
                        latents, noise, noise_pred, timesteps, noise_scheduler
                    )
                else:
                    # Standard noise prediction loss
                    loss = F.mse_loss(noise_pred, noise, reduction="mean")

                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                accumulated_loss += loss.item()

            # Update model with optional mixed precision
            if args.mixed_precision and scaler and args.device != "cpu":
                scaler.scale(loss).backward()

                # Step optimizer after accumulation steps
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            else:
                loss.backward()

                # Step optimizer after accumulation steps
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            # Update progress for all batches
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            # Only log when optimizer is stepped
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                avg_loss = accumulated_loss
                accumulated_loss = 0.0

                progress_bar.set_postfix({
                    "loss": avg_loss,
                    "lr": lr_scheduler.get_last_lr()[0]
                })

                # Log to wandb if available
                if args.use_wandb:
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step": global_step
                    })

                # Save checkpoint periodically
                if global_step > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, lr_scheduler,
                        epoch, global_step, args,
                        name=f"checkpoint-step-{global_step}"
                    )

                    # Generate validation samples
                    generate_validation_images(
                        model, validation_data, args.device,
                        args.output_dir, global_step
                    )

        # End of epoch
        epoch_loss /= len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} completed. Average loss: {epoch_loss:.6f}")

        # Save checkpoint after each epoch
        save_checkpoint(
            model, optimizer, lr_scheduler,
            epoch, global_step, args,
            name=f"checkpoint-epoch-{epoch+1}"
        )

        # Generate validation samples after each epoch
        generate_validation_images(
            model, validation_data, args.device,
            args.output_dir, f"epoch_{epoch+1}"
        )

    # Save final model
    save_checkpoint(
        model, optimizer, lr_scheduler,
        args.num_epochs-1, global_step, args,
        name="final_model"
    )

    logger.info(f"Training complete after {global_step} steps.")

    # Close wandb if used
    if args.use_wandb:
        wandb.finish()


def train_distributed(rank, world_size, args):
    """
    Distributed training entry point
    """
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)
    args.device = f"cuda:{rank}"

    # Create model
    model = CubeDiff(
        pretrained_model_path=args.pretrained_model_path,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees,
        vae_scale_factor=args.vae_scale_factor
    )
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(args.device),
        device_ids=[rank],
        output_device=rank
    )

    # Rest of training code would go here, modified for DDP
    # ...

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CubeDiff model on Sun360 dataset")

    # Model arguments
    parser.add_argument("--pretrained_model_path", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Path to pretrained diffusion model")
    parser.add_argument("--enable_overlap", action="store_true",
                        help="Enable overlapping predictions")
    parser.add_argument("--face_overlap_degrees", type=float, default=2.5,
                        help="Overlap in degrees for each face")
    parser.add_argument("--v_prediction", action="store_true",
                        help="Use v-prediction instead of standard noise prediction")
    parser.add_argument("--vae_scale_factor", type=float, default=0.18215,
                        help="Scale factor for VAE")

    # Training arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/sun360",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of steps to accumulate gradients across")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size for training")
    parser.add_argument("--learning_rate", type=float, default=8e-5,
                        help="Learning rate (as per paper)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Portion of steps for learning rate warmup")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--text_dropout_prob", type=float, default=0.1,
                        help="Probability of dropping text condition for classifier-free guidance")

    # Diffusion parameters
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                        help="Number of timesteps for diffusion")
    parser.add_argument("--beta_start", type=float, default=0.00085,
                        help="Starting beta value")
    parser.add_argument("--beta_end", type=float, default=0.012,
                        help="Ending beta value")
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear",
                        help="Schedule for beta values")

    # Logging and saving
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--validation_samples", type=int, default=4,
                        help="Number of validation samples to generate")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")

    # Hardware and performance
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--distributed", action="store_true",
                        help="Use distributed training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Train the model
    train_model(args)
