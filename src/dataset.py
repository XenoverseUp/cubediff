import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from .cubemap_utils import equirectangular_to_cubemap


class CubemapDataset(Dataset):
    """
    Dataset for training CubeDiff with cubemap faces
    """
    def __init__(
        self,
        data_root,
        text_file,
        image_size=512,
        enable_overlap=True,
        face_overlap_degrees=2.5,
    ):
        """
        Initialize dataset

        Args:
            data_root (str): Root directory containing panoramic images
            text_file (str): Path to JSON file with text prompts
            image_size (int): Size of images
            enable_overlap (bool): Enable overlapping predictions
            face_overlap_degrees (float): Overlap in degrees for each face
        """
        self.data_root = data_root
        self.enable_overlap = enable_overlap
        self.face_overlap_degrees = face_overlap_degrees

        # Get list of panorama files
        self.panorama_files = []
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.panorama_files.append(os.path.join(root, file))

        # Load text prompts
        with open(text_file, 'r') as f:
            self.text_data = json.load(f)

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        print(f"CubemapDataset initialized with {len(self.panorama_files)} panoramas")

    def __len__(self):
        return len(self.panorama_files)

    def __getitem__(self, idx):
        """Get item from dataset"""
        # Get panorama file path
        panorama_path = self.panorama_files[idx]
        file_name = os.path.basename(panorama_path)

        # Load panorama image
        panorama = Image.open(panorama_path).convert("RGB")

        # Convert to cubemap faces
        faces = equirectangular_to_cubemap(panorama)

        # Apply transforms to each face
        face_tensors = []
        for face in faces:
            face_tensor = self.transform(face)
            face_tensors.append(face_tensor)

        # Stack face tensors
        faces_tensor = torch.stack(face_tensors)

        # Get text prompts
        if file_name in self.text_data:
            # If file has specific prompts
            prompts = self.text_data[file_name]
        else:
            # Use default prompts
            prompts = self.text_data.get("default", [""] * 6)

        # Make sure we have 6 prompts
        if len(prompts) < 6:
            prompts = prompts + [""] * (6 - len(prompts))

        return {
            "images": faces_tensor,  # [6, 3, H, W]
            "prompts": prompts,      # List of 6 prompts
            "file_name": file_name
        }

class Sun360Dataset(Dataset):
    """
    Dataset for training CubeDiff with Sun360 dataset using only image conditioning
    """
    def __init__(
        self,
        data_root,
        image_size=512,
        enable_overlap=True,
        face_overlap_degrees=2.5,
        split="train"
    ):
        self.data_root = os.path.join(data_root, split)
        self.enable_overlap = enable_overlap
        self.face_overlap_degrees = face_overlap_degrees
        self.split = split

        # Get panoramic images from RGB folder
        self.rgb_dir = os.path.join(self.data_root, "RGB")
        self.panorama_files = []
        for file in os.listdir(self.rgb_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.panorama_files.append(os.path.join(self.rgb_dir, file))

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        print(f"Sun360Dataset initialized with {len(self.panorama_files)} panoramas")

    def __len__(self):
        return len(self.panorama_files)

    def __getitem__(self, idx):
        # Get panorama file path
        panorama_path = self.panorama_files[idx]
        file_name = os.path.basename(panorama_path)

        # Load panorama image
        panorama = Image.open(panorama_path).convert("RGB")

        # Convert equirectangular to cubemap faces
        faces = equirectangular_to_cubemap(
            panorama,
            enable_overlap=self.enable_overlap,
            overlap_degrees=self.face_overlap_degrees
        )

        # Apply transforms to each face
        face_tensors = []
        for face in faces:
            face_tensor = self.transform(face)
            face_tensors.append(face_tensor)

        # Stack face tensors
        faces_tensor = torch.stack(face_tensors)  # [6, 3, H, W]

        return {
            "images": faces_tensor,  # [6, 3, H, W]
            "file_name": file_name
        }
