import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import argparse
import json
import logging
import gc
import torch.distributed as dist
import torch.multiprocessing as mp

from src.model import CubeDiff
from src.dataset import Sun360Dataset
from src.positional_encoding import generate_cubemap_positional_encoding
from src.cubemap_utils import cubemap_to_equirectangular

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, name=None):
    os.makedirs(args.output_dir, exist_ok=True)
    base = f"checkpoint-epoch-{epoch}-step-{global_step}" if name is None else name
    path = os.path.join(args.output_dir, f"{base}.pt")
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": vars(args),
    }
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path}")
    if name == "final_model":
        wpath = os.path.join(args.output_dir, "model_weights.pt")
        torch.save(state['model_state_dict'], wpath)
        logger.info(f"Saved model weights to {wpath}")


def generate_validation_images(model, validation_data, device, output_dir, step):
    model.eval()
    out_dir = os.path.join(output_dir, f"validation_step_{step}")
    os.makedirs(out_dir, exist_ok=True)
    from src.inference import generate_panorama
    for i, data in enumerate(validation_data):
        fn = data["file_name"]
        cond = data["images"][0].unsqueeze(0).to(device)
        prompts = [""] * 6
        faces = generate_panorama(
            model=model,
            prompts=prompts,
            condition_image=cond,
            num_inference_steps=30,
            device=device,
            output_type="pil"
        )
        for j, face in enumerate(faces):
            face.save(os.path.join(out_dir, f"{fn}_face_{j}.png"))
        try:
            eq = cubemap_to_equirectangular(faces)
            eq.save(os.path.join(out_dir, f"{fn}_panorama.png"))
        except Exception as e:
            logger.error(f"Equirect error: {e}")
    model.train()


def load_validation_data(data_root, num_samples=4, image_size=512):
    ds = Sun360Dataset(data_root=data_root, image_size=image_size, split="test")
    idxs = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    return [ds[i] for i in idxs]


def calculate_v_prediction_loss(latents, noise, noise_pred, timesteps, noise_scheduler):
    alpha = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sigma = torch.sqrt(1 - alpha).view(-1, 1, 1, 1)
    v_target = alpha.sqrt() * noise - sigma * latents
    return F.mse_loss(v_target, noise_pred, reduction="mean")


def train_model(args):
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Config saved to {args.output_dir}/config.json")

    if args.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(train_distributed, args=(world_size, args), nprocs=world_size, join=True)
        return

    device = torch.device(args.device)
    model = CubeDiff(
        pretrained_model_path=args.pretrained_model_path,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees,
        vae_scale_factor=args.vae_scale_factor
    ).to(device)
    model.enable_gradient_checkpointing()
    torch.cuda.empty_cache()

    optimizer = AdamW(
        [
            {
                "params": [p for n, p in model.unet.named_parameters()
                           if p.requires_grad and not any(x in n for x in ["bias", "LayerNorm.weight"])] ,
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.unet.named_parameters()
                           if p.requires_grad and any(x in n for x in ["bias", "LayerNorm.weight"])] ,
                "weight_decay": 0.0
            },
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )

    train_ds = Sun360Dataset(
        data_root=args.data_root,
        image_size=args.image_size,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees,
        split="train"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"),
        drop_last=True
    )
    val_data = load_validation_data(args.data_root, args.validation_samples, args.image_size)

    total_steps = args.num_epochs * len(train_loader) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    noise_scheduler = model.get_noise_scheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule
    )
    scaler = GradScaler() if args.mixed_precision and args.device != "cpu" else None

    start_epoch, global_step = 0, 0
    if args.resume_from_checkpoint:
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)

    empty_emb = model.encode_text(["" for _ in range(6)], device)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        if args.distributed:
            raise RuntimeError("Distributed training should use train_distributed entrypoint.")

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            imgs = batch["images"].to(device)
            B, F, C, H, W = imgs.shape
            imgs = imgs.view(B * F, C, H, W)

            with autocast(enabled=args.mixed_precision and args.device != "cpu"):
                with torch.no_grad():
                    latents = model.vae.encode(imgs).latent_dist.sample() * args.vae_scale_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (B * F,),
                    device=device
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                pos_enc = generate_cubemap_positional_encoding(B, latents.shape[2], latents.shape[3], device)
                text_emb = empty_emb.repeat(B, 1, 1)
                noise_pred = model(
                    noisy,
                    timesteps,
                    encoder_hidden_states=text_emb,
                    pos_enc=pos_enc
                )
                if args.v_prediction:
                    loss = calculate_v_prediction_loss(latents, noise, noise_pred, timesteps, noise_scheduler)
                else:
                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, args,
                                    name=f"checkpoint-step-{global_step}")
                    generate_validation_images(model, val_data, args.device, args.output_dir, global_step)

            epoch_loss += loss.item() * args.gradient_accumulation_steps

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} completed. Avg loss: {avg_loss:.6f}")
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, args,
                        name=f"checkpoint-epoch-{epoch+1}")
        generate_validation_images(model, val_data, args.device, args.output_dir, f"epoch_{epoch+1}")
        gc.collect()
        torch.cuda.empty_cache()

    save_checkpoint(model, optimizer, scheduler, args.num_epochs-1, global_step, args, name="final_model")
    logger.info(f"Training complete after {global_step} steps.")


def train_distributed(rank, world_size, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    args.device = f"cuda:{rank}"
    torch.manual_seed(args.seed + rank)

    train_ds = Sun360Dataset(
        data_root=args.data_root,
        image_size=args.image_size,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees,
        split="train"
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    model = CubeDiff(
        pretrained_model_path=args.pretrained_model_path,
        enable_overlap=args.enable_overlap,
        face_overlap_degrees=args.face_overlap_degrees,
        vae_scale_factor=args.vae_scale_factor
    ).to(args.device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    total_steps = args.num_epochs * len(loader) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    noise_scheduler = model.module.get_noise_scheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule
    )
    scaler = GradScaler() if args.mixed_precision else None

    val_data = load_validation_data(args.data_root, args.validation_samples, args.image_size) if rank == 0 else None
    if rank == 0:
        empty_emb = model.module.encode_text(["" for _ in range(6)], args.device)

    global_step = 0
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Rank {rank} Epoch {epoch+1}/{args.num_epochs}")):
            imgs = batch["images"].to(args.device)
            B, F, C, H, W = imgs.shape
            imgs = imgs.view(B * F, C, H, W)

            with autocast(enabled=args.mixed_precision):
                with torch.no_grad():
                    latents = model.module.vae.encode(imgs).latent_dist.sample() * args.vae_scale_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B * F,), device=args.device
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                pos_enc = generate_cubemap_positional_encoding(B, latents.shape[2], latents.shape[3], args.device)
                text_emb = empty_emb.repeat(B, 1, 1)
                noise_pred = model(
                    noisy,
                    timesteps,
                    encoder_hidden_states=text_emb,
                    pos_enc=pos_enc
                )
                loss = calculate_v_prediction_loss(latents, noise, noise_pred, timesteps, noise_scheduler) \
                       if args.v_prediction else F.mse_loss(noise_pred, noise, reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % args.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, args,
                                    name=f"ddp-checkpoint-{global_step}")
                    generate_validation_images(model.module, val_data, args.device, args.output_dir, global_step)

            epoch_loss += loss.item() * args.gradient_accumulation_steps

        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            logger.info(f"DDP Epoch {epoch+1}/{args.num_epochs} avg loss: {avg_loss:.6f}")
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, args,
                            name=f"ddp-epoch-{epoch+1}")
            generate_validation_images(model.module, val_data, args.device, args.output_dir, f"ddp_epoch_{epoch+1}")
            gc.collect()
            torch.cuda.empty_cache()

    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, args.num_epochs-1, global_step, args, name="final_model")
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
