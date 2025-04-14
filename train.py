import os
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import argparse
from positional_encoding import generate_cubemap_positional_encoding
from cubemap_dataset import CubemapDataset

from model import CubeDiff


def train(args):
    """
    Train CubeDiff model

    Args:
        args: Training arguments
    """
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project="cubediff", name=args.run_name)

    # Initialize model
    model = CubeDiff(
        pretrained_model_path=args.pretrained_model_path,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees
    )
    model.to(args.device)

    # Set up optimizer
    optimizer = AdamW(
        model.unet.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )

    # Set up dataset and dataloader
    train_dataset = CubemapDataset(
        data_root=args.data_root,
        text_file=args.text_file,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}/{args.num_epochs}")

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            # Get data from batch
            cubemap_images = batch["images"].to(args.device)  # [B, 6, 3, H, W]
            text_prompts = batch["prompts"]  # List of 6 prompts per batch

            # Reshape for processing
            batch_size, num_faces, channels, height, width = cubemap_images.shape
            images = cubemap_images.view(batch_size * num_faces, channels, height, width)

            # Encode images to latents
            with torch.no_grad():
                latents = model.vae.encode(images).latent_dist.sample() * 0.18215

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size * num_faces,),
                device=args.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings (each face has its own prompt)
            text_embeddings = []
            for i in range(batch_size):
                batch_prompts = [text_prompts[i][f] for f in range(num_faces)]
                embeddings = model.encode_text(batch_prompts, args.device)
                text_embeddings.append(embeddings)

            text_embeddings = torch.cat(text_embeddings, dim=0)

            # Add classifier-free guidance
            if args.guidance_scale > 1.0:
                uncond_embeddings = model.encode_text([""] * (batch_size * num_faces), args.device)
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                # Duplicate latents for classifier-free guidance
                noisy_latents = torch.cat([noisy_latents] * 2)
                timesteps = torch.cat([timesteps] * 2)

            # Generate position encodings
            pos_enc = generate_cubemap_positional_encoding(
                batch_size, latents.shape[2], latents.shape[3], device=args.device)

            # Forward pass
            latents_with_pos = torch.cat([noisy_latents, pos_enc], dim=1)
            noise_pred = model.unet(
                latents_with_pos,
                timesteps,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]

            # Apply classifier-free guidance if enabled
            if args.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

            # Log to wandb if enabled
            if args.use_wandb and global_step % args.log_every == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": global_step,
                    "train/epoch": epoch,
                })

            global_step += 1

            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Saved final model to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CubeDiff model")

    # Model arguments
    parser.add_argument("--pretrained_model_path", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Path to pretrained diffusion model")
    parser.add_argument("--enable_overlap", type=bool, default=True,
                        help="Enable overlapping predictions")
    parser.add_argument("--face_overlap_degrees", type=float, default=2.5,
                        help="Overlap in degrees for each face")

    # Training arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--text_file", type=str, required=True,
                        help="Path to text prompts file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log metrics every N steps")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default="cubediff",
                        help="Name for W&B run")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)
