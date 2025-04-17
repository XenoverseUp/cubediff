import torch
import math


def generate_cubemap_positional_encoding(batch_size, height, width, device="cuda"):
    """
    Generate positional encoding for cubemap faces.
    Uses arctan2 to map 3D coordinates to UV space.

    Args:
        batch_size: Number of panoramas in batch
        height: Height of latent feature map
        width: Width of latent feature map
        device: Device to create tensors on

    Returns:
        torch.Tensor: Position encoding of shape [batch_size*6, 2, height, width]
    """
    # Initialize position encoding tensor
    positions = torch.zeros(batch_size, 6, 2, height, width, device=device)

    # Generate normalized grid coordinates
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Create tensors of the same shape as x_grid and y_grid
    ones = torch.ones_like(x_grid)

    # Define direction vectors for each face according to the paper's convention
    # Each function maps a (x,y) position on the face to a (x,y,z) direction vector
    cube_directions = [
        lambda x, y: (ones, y, -x),           # front face (+X)
        lambda x, y: (-ones, y, x),           # back face (-X)
        lambda x, y: (x, ones, -y),           # up face (+Y)
        lambda x, y: (x, -ones, y),           # down face (-Y)
        lambda x, y: (x, y, ones),            # left face (+Z)
        lambda x, y: (-x, y, -ones)           # right face (-Z)
    ]

    # Generate position encodings for each face
    for face_idx, direction_fn in enumerate(cube_directions):
        # Convert to 3D coordinates on cube
        x, y, z = direction_fn(x_grid, y_grid)

        # Normalize to ensure we're on the unit cube
        norm = torch.sqrt(x*x + y*y + z*z)
        x = x / norm
        y = y / norm
        z = z / norm

        # Convert to UV coordinates using arctan2 as per paper equation (1)
        u = torch.atan2(x, z)  # Note: x/z is correct for UV cubemap coord system
        v = torch.atan2(y, torch.sqrt(x*x + z*z))

        # Normalize to [0, 1]
        u = (u / math.pi + 1) / 2
        v = (v / (math.pi/2) + 1) / 2

        # Store in positions tensor
        positions[:, face_idx, 0, :, :] = u
        positions[:, face_idx, 1, :, :] = v

    # Reshape to match expected format: [batch_size*6, 2, height, width]
    return positions.reshape(batch_size * 6, 2, height, width)
