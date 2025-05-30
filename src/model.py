import torch
import torch.nn as nn

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from .synced_norm import replace_groupnorm_with_synced
from .attention_inflation import inflate_attention_layers
from .positional_encoding import generate_cubemap_positional_encoding

class CubeDiff(nn.Module):
    def __init__(
        self,
        pretrained_model_path="runwayml/stable-diffusion-v1-5",
        enable_overlap=True,
        face_overlap_degrees=2.5,
        vae_scale_factor=0.18215
    ):
        print(f"Loading model from {pretrained_model_path}")
        super().__init__()

        print(f"Initializing CubeDiff with model: {pretrained_model_path}")

        self.vae_scale_factor = vae_scale_factor

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_path, subfolder="unet")

        # Apply synchronized GroupNorm to VAE
        print("Replacing GroupNorm layers with synchronized versions...")
        self.vae = replace_groupnorm_with_synced(self.vae)

        # Inflate attention layers in UNet
        print("Inflating attention layers...")
        self.unet = inflate_attention_layers(self.unet)

        # Settings for overlap
        self.enable_overlap = enable_overlap
        self.face_overlap_degrees = face_overlap_degrees

        self._freeze_modules()

        print("CubeDiff model initialized.")

    def _freeze_modules(self):
        print("Freezing text encoder and VAE parameters...")
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.vae.parameters():
            param.requires_grad = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        print("Enabling gradient checkpointing for UNet...")
        self.unet.enable_gradient_checkpointing()

    def encode_text(self, prompts, device="cuda"):
        """
        Encode text prompts to embeddings

        Args:
            prompts (list): List of text prompts
            device (str): Device to use

        Returns:
            torch.Tensor: Text embeddings
        """
        if not isinstance(prompts, list):
            prompts = [prompts]

        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

        return text_embeddings

    def prepare_latents(self, batch_size=1, height=512, width=512, device="cuda"):
        """
        Prepare initial latents for all 6 cubemap faces

        Args:
            batch_size (int): Batch size
            height (int): Image height
            width (int): Image width
            device (str): Device to use

        Returns:
            torch.Tensor: Latents tensor of shape [batch_size*6, 4, h/8, w/8]
        """
        # Get latent dimensions
        latent_height = height // 8
        latent_width = width // 8

        # Generate random latents for all faces
        latents = torch.randn(
            (batch_size * 6, 4, latent_height, latent_width),
            device=device
        )

        return latents

    def prepare_conditioned_latents(self, condition_image, device="cuda"):
        """
        Prepare latents with a condition image for the first face

        Args:
            condition_image (torch.Tensor): Conditioning image tensor [1, 3, H, W]
            device (str): Device to use

        Returns:
            tuple: (latents, mask) where latents has shape [batch_size*6, 4, h/8, w/8]
                  and mask has shape [batch_size*6, 1, h/8, w/8]
        """
        # Move to device
        condition_image = condition_image.to(device)

        # Get batch size and dimensions
        batch_size = condition_image.shape[0]

        # Encode condition image to latent
        with torch.no_grad():
            condition_latent = self.vae.encode(condition_image).latent_dist.sample() * self.vae_scale_factor

        # Generate noise for other 5 faces
        latent_shape = condition_latent.shape
        other_latents = torch.randn((batch_size * 5, *latent_shape[1:]), device=device)

        # Concatenate condition latent with noise for other faces
        all_latents = torch.cat([condition_latent, other_latents], dim=0)

        # Create mask to indicate conditioning face
        mask = torch.zeros((batch_size * 6, 1, latent_shape[2], latent_shape[3]), device=device)
        mask[:batch_size] = 1.0  # First face is conditioned

        return all_latents, mask

    def get_noise_scheduler(self, num_train_timesteps=1000, beta_start=0.00085,
                           beta_end=0.012, beta_schedule="scaled_linear"):
        """Get noise scheduler for training"""
        scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            num_train_timesteps=num_train_timesteps,
            clip_sample=False
        )

        return scheduler

    def forward(self, latents, timestep, encoder_hidden_states=None, mask=None, pos_enc=None):
        """
        Forward pass through the model

        Args:
            latents (torch.Tensor): Latent tensor
            timestep (torch.Tensor): Current timestep
            encoder_hidden_states (torch.Tensor): Text embeddings
            mask (torch.Tensor, optional): Mask for conditioned faces
            pos_enc (torch.Tensor, optional): Positional encoding

        Returns:
            torch.Tensor: Predicted noise
        """
        # Get latent shape
        batch_size, channels, height, width = latents.shape

        # Generate positional encoding if not provided
        if pos_enc is None:
            pos_enc = generate_cubemap_positional_encoding(
                batch_size // 6, height, width, device=latents.device)

        # Apply positional encoding to latents
        # Clone to avoid modifying the original latents
        modified_latents = latents.clone()

        # Add positional encoding to all channels with appropriate scaling
        # The paper suggests this helps with cross-face consistency
        for i in range(channels):
            scale = 0.1 / (i+1)  # Apply diminishing scale to deeper channels
            modified_latents[:, i, :, :] = modified_latents[:, i, :, :] + pos_enc[:, 0, :, :] * scale

        # Forward pass through UNet
        noise_pred = self.unet(
            modified_latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={"scale": 1.0},
            return_dict=False
        )[0]

        return noise_pred
