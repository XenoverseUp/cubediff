from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from .positional_encoding import generate_cubemap_positional_encoding
from .cubemap_utils import crop_image_for_overlap


def generate_panorama(
    model,
    prompts,
    condition_image=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    device="cuda",
    output_type="pil",
):
    """
    Generate a panorama using CubeDiff model

    Args:
        model (CubeDiff): CubeDiff model
        prompts (str or list): Text prompt or list of 6 text prompts
        condition_image (torch.Tensor, optional): Optional conditioning image [1, 3, H, W]
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Guidance scale for classifier-free guidance
        height (int): Image height
        width (int): Image width
        device (str): Device to use
        output_type (str): Output type, either 'pil' or 'tensor'

    Returns:
        list or torch.Tensor: List of 6 PIL images or tensor of shape [6, 3, H, W]
    """
    print(f"Generating panorama with {num_inference_steps} steps...")

    model.to(device)
    model.eval()

    if isinstance(prompts, str):
        prompts = [prompts] * 6

    assert len(prompts) == 6, f"Expected 6 prompts, got {len(prompts)}"

    text_embeddings = model.encode_text(prompts, device)

    uncond_embeddings = model.encode_text([""] * 6, device)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    if condition_image is not None:
        latents, mask = model.prepare_conditioned_latents(condition_image, device)
    else:
        latents = model.prepare_latents(batch_size=1, height=height, width=width, device=device)
        mask = None

    pos_enc = generate_cubemap_positional_encoding(1, latents.shape[2], latents.shape[3], device=device)

    # Denoising loop
    with torch.no_grad():
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)

            # Get current timestep
            timestep = t.expand(latent_model_input.shape[0]).to(device)

            # Add positional encoding to each face's latents
            modified_latents = latent_model_input.clone()

            for j in range(modified_latents.shape[1]):
                channel_idx = j % 2  # Alternate between U and V coordinates
                scale = 0.1 / (j//2 + 1)  # Diminishing scale per coordinate pair
                modified_latents[:, j, :, :] = modified_latents[:, j, :, :] + pos_enc[:, channel_idx, :, :] * scale

            # Predict noise
            noise_pred = model.unet(
                modified_latents,
                timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Update latents
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if mask is not None:
                condition_latent = latents[:1]  # First face is condition

                # Apply mask to update only the conditioned face
                latents = latents * (1 - mask) + condition_latent * mask

    # Decode latents to images
    with torch.no_grad():
        images = model.vae.decode(latents / 0.18215).sample

    # Convert images to output format
    images = (images / 2 + 0.5).clamp(0, 1)

    # Reshape to separate faces
    images = images.reshape(6, 3, height, width)

    if output_type == "pil":
        # Convert to PIL images
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images_np]

        # Apply overlap cropping if enabled
        if model.enable_overlap:
            pil_images = [crop_image_for_overlap(img, model.face_overlap_degrees)
                          for img in pil_images]

        return pil_images
    else:
        return images


def visualize_attention_maps(model, latents, timestep, text_embeddings, device="cuda"):
    """
    Visualize attention maps between faces to debug cross-face consistency
    """
    # This is a debugging function that can be used to visualize how attention
    # flows between different faces of the cubemap

    attention_maps = {}

    def get_attention_hook(layer_name):
        def hook(module, input, output):
            attention_maps[layer_name] = output[1]  # Usually attention weights are second output
        return hook

    # Register hooks on attention layers
    hooks = []
    for name, module in model.unet.named_modules():
        if "attn2" in name:  # Cross-attention layers
            hooks.append(module.register_forward_hook(get_attention_hook(name)))

    # Forward pass to get attention maps
    with torch.no_grad():
        model.unet(
            latents,
            timestep,
            encoder_hidden_states=text_embeddings,
            return_dict=False
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Process and return attention maps
    return attention_maps
