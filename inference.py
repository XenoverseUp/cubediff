from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


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

    # Move model to device
    model.to(device)
    model.eval()

    # Process text prompts
    if isinstance(prompts, str):
        prompts = [prompts] * 6  # Use same prompt for all faces

    # Check number of prompts
    assert len(prompts) == 6, f"Expected 6 prompts, got {len(prompts)}"

    # Encode text prompts
    text_embeddings = model.encode_text(prompts, device)

    # Prepare classifier-free guidance
    uncond_embeddings = model.encode_text([""] * 6, device)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Initialize scheduler
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Prepare latents
    if condition_image is not None:
        latents, mask = model.prepare_conditioned_latents(condition_image, device)
    else:
        latents = model.prepare_latents(batch_size=1, height=height, width=width, device=device)
        mask = None

    # Denoising loop
    with torch.no_grad():
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)

            # Get current timestep
            timestep = t.expand(latent_model_input.shape[0]).to(device)

            # Predict noise
            noise_pred = model(latent_model_input, timestep, text_embeddings, mask)

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Update latents
            latents = scheduler.step(noise_pred, t, latents).prev_sample

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


def crop_image_for_overlap(image, overlap_degrees=2.5):
    """
    Crop image to remove overlapping regions

    Args:
        image (PIL.Image): Image to crop
        overlap_degrees (float): Amount of overlap in degrees

    Returns:
        PIL.Image: Cropped image
    """
    width, height = image.size

    # Calculate crop margins
    standard_fov = 90.0
    actual_fov = standard_fov + (2 * overlap_degrees)

    margin_ratio = overlap_degrees / actual_fov
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    # Crop image
    return image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
