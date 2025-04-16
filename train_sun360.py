import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging

from src.model import CubeDiff
from src.dataset import Sun360Dataset
from src.positional_encoding import generate_cubemap_positional_encoding

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CubeDiffTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize model
        logger.info(f"Initializing model from {args.pretrained_model_path}")
        self.model = CubeDiff(
            pretrained_model_path=args.pretrained_model_path,
            enable_overlap=args.enable_overlap,
            face_overlap_degrees=args.face_overlap_degrees,
            vae_scale_factor=0.18215
        )
        self.model.to(self.device)

        # Set up dataset
        logger.info(f"Loading dataset from {args.data_root}")
        train_dataset = Sun360Dataset(
            data_root=args.data_root,
            image_size=args.image_size,
            enable_overlap=args.enable_overlap,
            face_overlap_degrees=args.face_overlap_degrees,
            split="train"
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True
        )

        # Set up optimizer
        param_groups = [
            {
                "params": [p for n, p in self.model.unet.named_parameters()
                          if not any(nd in n for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad],
                "weight_decay": 1e-2,
            },
            {
                "params": [p for n, p in self.model.unet.named_parameters()
                          if any(nd in n for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            param_groups,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Initialize noise scheduler
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        self.noise_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False
        )

        # Create dummy text embeddings for all samples (reused during training)
        logger.info("Creating dummy text embeddings")
        self.dummy_prompt = " "  # A single space as dummy prompt
        self.dummy_embeddings = None  # Will create these on first batch

    def train(self):
        logger.info(f"Starting training for {self.args.num_epochs} epochs")

        for epoch in range(self.args.num_epochs):
            self.train_epoch(epoch)

            # Save checkpoint
            checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Get data from batch
            cubemap_images = batch["images"].to(self.device)  # [B, 6, 3, H, W]

            # Reshape to process all faces together
            batch_size, num_faces, channels, height, width = cubemap_images.shape
            images = cubemap_images.view(batch_size * num_faces, channels, height, width)

            # Encode images to latents
            with torch.no_grad():
                latents = self.model.vae.encode(images).latent_dist.sample() * 0.18215

            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (batch_size * num_faces,),
                device=self.device
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Generate position encodings
            latent_height, latent_width = latents.shape[2], latents.shape[3]
            pos_enc = generate_cubemap_positional_encoding(
                batch_size, latent_height, latent_width, device=self.device)

            # Apply positional encoding
            modified_latents = noisy_latents.clone()
            modified_latents[:, :2, :, :] = modified_latents[:, :2, :, :] + pos_enc * 0.1

            # Create dummy text embeddings for all faces
            if self.dummy_embeddings is None or self.dummy_embeddings.shape[0] != batch_size * num_faces:
                dummy_prompts = [self.dummy_prompt] * (batch_size * num_faces)
                self.dummy_embeddings = self.model.encode_text(dummy_prompts, self.device)

            # Forward pass through UNet with dummy text embeddings
            noise_pred = self.model.unet(
                modified_latents,
                timesteps,
                encoder_hidden_states=self.dummy_embeddings,
                return_dict=False
            )[0]

            # Calculate simple MSE loss
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # End of epoch
        epoch_loss /= len(self.train_dataloader)
        logger.info(f"Epoch {epoch+1}/{self.args.num_epochs} completed. Average loss: {epoch_loss:.6f}")
        return epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CubeDiff with dummy text conditioning")

    # Model arguments
    parser.add_argument("--pretrained_model_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Path to pretrained model")
    parser.add_argument("--enable_overlap", action="store_true",
                        help="Enable overlapping predictions")
    parser.add_argument("--face_overlap_degrees", type=float, default=2.5,
                        help="Overlap in degrees for each face")

    # Training arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="./outputs/sun360_dummy",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size")

    # Hardware
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader workers")

    args = parser.parse_args()

    # Train the model
    trainer = CubeDiffTrainer(args)
    trainer.train()
