import torch
import math


def generate_cubemap_positional_encoding(batch_size, height, width, device="cuda"):
    """
    Generate positional encoding for cubemap faces.

    Returns:
        torch.Tensor: Position encoding of shape [batch_size*6, 2, height, width]
    """
    # Initialize position encoding tensor
    positions = torch.zeros(batch_size, 6, 2, height, width, device=device)

    # Generate normalized grid coordinates
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Direction functions for each face (front, back, up, down, left, right)
    # Create tensors of the same shape as x_grid and y_grid
    ones = torch.ones_like(x_grid)

    # Define direction functions that return tensors
    cube_directions = [
        lambda x, y: (ones, y, -x),           # front
        lambda x, y: (-ones, y, x),           # back
        lambda x, y: (x, ones, y),            # up
        lambda x, y: (x, -ones, -y),          # down
        lambda x, y: (-y, x, ones),           # left
        lambda x, y: (y, x, -ones)            # right
    ]

    # Generate position encodings for each face
    for face_idx, direction_fn in enumerate(cube_directions):
        # Convert to 3D coordinates on cube
        x, y, z = direction_fn(x_grid, y_grid)

        # Convert to UV coordinates using arctan2
        u = torch.atan2(x, z)
        v = torch.atan2(y, torch.sqrt(x*x + z*z))

        # Normalize to [0, 1]
        u = (u / math.pi + 1) / 2
        v = (v / (math.pi/2) + 1) / 2

        # Store in positions tensor
        positions[:, face_idx, 0, :, :] = u
        positions[:, face_idx, 1, :, :] = v

    # Reshape to match expected format: [batch_size*6, 2, height, width]
    return positions.reshape(batch_size * 6, 2, height, width)
