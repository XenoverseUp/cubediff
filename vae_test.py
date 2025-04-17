from diffusers import AutoencoderKL
from PIL import Image

import torch
import matplotlib.pyplot as plt
from transformers.models.clip import CLIPFeatureExtractor

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
processor = CLIPFeatureExtractor.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="feature_extractor")
vae.eval()


image_path = "./rgb_1.jpeg"
image = Image.open(image_path).convert("RGB")

input_image = processor.preprocess(image, return_tensors="pt").data["pixel_values"]

print(type(input_image))

with torch.no_grad():
    latent_dist = vae.encode(input_image).latent_dist
    latents = latent_dist.sample() * 0.18215

print(latents.shape)  # Expected: [1, 4, 64, 64]

with torch.no_grad():
    reconstructed = vae.decode(latents / 0.18215).sample  # [1, 3, 512, 512]

reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)

original_display = input_image.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original_display)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(reconstructed.squeeze(0).permute(1, 2, 0))
plt.axis("off")
plt.show()
