import math
import numpy as np
from PIL import Image


def equirectangular_to_cubemap(equirect_img, face_size=512):
    """
    Convert equirectangular image to 6 cubemap faces

    Args:
        equirect_img (PIL.Image): Equirectangular image
        face_size (int): Size of each face

    Returns:
        list: List of 6 PIL images [front, back, up, down, left, right]
    """
    # Get equirectangular dimensions
    equirect_width, equirect_height = equirect_img.size
    equirect = np.array(equirect_img)

    # Initialize cubemap faces
    faces = []

    # For each face
    for face_idx in range(6):
        # Create face image
        face = np.zeros((face_size, face_size, 3), dtype=np.uint8)

        # Map each pixel
        for y in range(face_size):
            for x in range(face_size):
                # Normalize to [-1, 1]
                nx = 2 * (x + 0.5) / face_size - 1
                ny = 2 * (y + 0.5) / face_size - 1

                # Get 3D vector based on face
                if face_idx == 0:  # Front
                    vec = [1.0, ny, -nx]
                elif face_idx == 1:  # Back
                    vec = [-1.0, ny, nx]
                elif face_idx == 2:  # Up
                    vec = [nx, 1.0, -ny]
                elif face_idx == 3:  # Down
                    vec = [nx, -1.0, ny]
                elif face_idx == 4:  # Left
                    vec = [ny, nx, 1.0]
                else:  # Right
                    vec = [-ny, nx, -1.0]

                # Normalize vector
                norm = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
                vec = [v / norm for v in vec]

                # Convert to equirectangular coordinates
                phi = math.atan2(vec[2], vec[0])
                theta = math.asin(vec[1])

                # Map to equirectangular image coordinates
                u = (phi / (2 * math.pi) + 0.5) % 1.0
                v = (theta / math.pi + 0.5)

                # Sample from equirectangular image
                equirect_x = int(u * equirect_width) % equirect_width
                equirect_y = int(v * equirect_height) % equirect_height

                # Set pixel in face
                face[y, x] = equirect[equirect_y, equirect_x]

        faces.append(Image.fromarray(face))

    return faces


def cubemap_to_equirectangular(cubemap_faces, output_width=2048, output_height=1024):
    """
    Convert 6 cubemap faces to equirectangular projection

    Args:
        cubemap_faces (list): List of 6 PIL images [front, back, up, down, left, right]
        output_width (int): Width of output equirectangular image
        output_height (int): Height of output equirectangular image

    Returns:
        PIL.Image: Equirectangular image
    """
    # Convert PIL images to numpy arrays
    cube_arrays = [np.array(face) for face in cubemap_faces]
    face_size = cube_arrays[0].shape[0]

    # Initialize equirectangular image
    equirect = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # For each pixel in equirectangular image
    for y in range(output_height):
        for x in range(output_width):
            # Convert to spherical coordinates
            lat = (0.5 - y / output_height) * math.pi
            lon = (x / output_width - 0.5) * 2 * math.pi

            # Convert to 3D vector
            x_vec = math.cos(lat) * math.cos(lon)
            y_vec = math.sin(lat)
            z_vec = math.cos(lat) * math.sin(lon)

            # Determine which face to sample from
            abs_x, abs_y, abs_z = abs(x_vec), abs(y_vec), abs(z_vec)
            max_val = max(abs_x, abs_y, abs_z)

            # Sample from appropriate face
            if max_val == abs_x:
                if x_vec > 0:
                    # Front face
                    face_idx = 0
                    u, v = -z_vec / x_vec, y_vec / x_vec
                else:
                    # Back face
                    face_idx = 1
                    u, v = z_vec / abs_x, y_vec / abs_x
            elif max_val == abs_y:
                if y_vec > 0:
                    # Up face
                    face_idx = 2
                    u, v = x_vec / y_vec, -z_vec / y_vec
                else:
                    # Down face
                    face_idx = 3
                    u, v = x_vec / abs_y, z_vec / abs_y
            else:
                if z_vec > 0:
                    # Left face
                    face_idx = 4
                    u, v = x_vec / z_vec, y_vec / z_vec
                else:
                    # Right face
                    face_idx = 5
                    u, v = -x_vec / abs_z, y_vec / abs_z

            # Map to face coordinates
            u = (u + 1) / 2
            v = (v + 1) / 2

            # Clamp to face bounds
            u = max(0, min(1, u))
            v = max(0, min(1, v))

            # Convert to pixel coordinates
            px = int(u * face_size)
            py = int(v * face_size)

            # Clamp to prevent out-of-bounds access
            px = min(face_size - 1, max(0, px))
            py = min(face_size - 1, max(0, py))

            # Sample from face
            equirect[y, x] = cube_arrays[face_idx][py, px]

    return Image.fromarray(equirect)
