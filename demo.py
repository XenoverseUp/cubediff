import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms

from src.model import CubeDiff
from inference import generate_panorama
from src.cubemap_utils import cubemap_to_equirectangular


def main(args):
    """
    Demo script for CubeDiff

    Args:
        args: Command-line arguments
    """
    # Load model

    if args.model_path:
        print(f"Loading model from {args.model_path}")
    model = CubeDiff(
        pretrained_model_path=args.pretrained_model_path,
        enable_overlap=True,
        face_overlap_degrees=2.5
    )

    # Load checkpoint if provided
    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")

    # Prepare device
    device = args.device if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Prepare prompts
    if args.prompt_file:
        # Load prompts from file
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines()]
            # Make sure we have exactly 6 prompts
            if len(prompts) < 6:
                prompts = prompts + [prompts[0]] * (6 - len(prompts))
            elif len(prompts) > 6:
                prompts = prompts[:6]
    else:
        # Use single prompt for all faces
        prompts = [args.prompt] * 6

    print(f"Using prompts: {prompts}")

    # Prepare conditioning image if provided
    condition_image = None
    if args.condition_image and os.path.exists(args.condition_image):
        print(f"Using condition image: {args.condition_image}")
        # Load and transform condition image
        img = Image.open(args.condition_image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        condition_image = transform(img).unsqueeze(0).to(device)

    # Generate panorama
    print("Generating panorama...")
    cubemap_faces = generate_panorama(
        model=model,
        prompts=prompts,
        condition_image=condition_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        device=device,
        output_type="pil"
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save individual faces
    for i, face in enumerate(cubemap_faces):
        face_path = os.path.join(args.output_dir, f"face_{i}.png")
        face.save(face_path)
        print(f"Saved {face_path}")

    # Convert to equirectangular and save
    print("Converting to equirectangular...")
    equirect = cubemap_to_equirectangular(
        cubemap_faces,
        output_width=args.equirect_width,
        output_height=args.equirect_height
    )
    equirect_path = os.path.join(args.output_dir, "panorama.png")
    equirect.save(equirect_path)
    print(f"Saved equirectangular panorama to {equirect_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate panorama with CubeDiff")

    # Model arguments
    parser.add_argument("--pretrained_model_path", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Path to pretrained diffusion model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained CubeDiff model checkpoint")

    # Generation arguments
    parser.add_argument("--prompt", type=str,
                        default="A beautiful landscape with mountains and trees",
                        help="Text prompt for generation")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to file with prompts (one per line)")
    parser.add_argument("--condition_image", type=str, default=None,
                        help="Path to conditioning image")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width")
    parser.add_argument("--equirect_width", type=int, default=2048,
                        help="Equirectangular output width")
    parser.add_argument("--equirect_height", type=int, default=1024,
                        help="Equirectangular output height")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    main(args)
