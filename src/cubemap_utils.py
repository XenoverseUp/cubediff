import math
import numpy as np
from PIL import Image


def equirectangular_to_cubemap(equirect_img, face_size=512, enable_overlap=True, overlap_degrees=2.5):
    """
    Convert equirectangular image to 6 cubemap faces

    Args:
        equirect_img (PIL.Image): Equirectangular image
        face_size (int): Size of each face
        enable_overlap (bool): Enable overlapping predictions
        overlap_degrees (float): Overlap in degrees for each face

    Returns:
        list: List of 6 PIL images [front, back, up, down, left, right]
    """
    # Get equirectangular dimensions
    equirect_width, equirect_height = equirect_img.size
    equirect = np.array(equirect_img)

    # Calculate actual FOV based on overlap
    fov = 90.0
    if enable_overlap:
        fov += 2 * overlap_degrees

    # Initialize cubemap faces
    faces = []

    # For each face - following the order: front, back, up, down, left, right
    for face_idx in range(6):
        # Create face image
        face = np.zeros((face_size, face_size, 3), dtype=np.uint8)

        # Map each pixel
        for y in range(face_size):
            for x in range(face_size):
                # Normalize to [-1, 1]
                nx = 2 * (x + 0.5) / face_size - 1
                ny = 2 * (y + 0.5) / face_size - 1

                # Calculate FOV multiplier
                fov_mul = math.tan(math.radians(fov / 2)) / math.tan(math.radians(90 / 2))
                nx *= fov_mul
                ny *= fov_mul

                # Get 3D vector based on face
                # Carefully follow the correct convention:
                # +X = front, -X = back, +Y = up, -Y = down, +Z = left, -Z = right
                if face_idx == 0:  # Front (+X)
                    vec = [1.0, ny, -nx]
                elif face_idx == 1:  # Back (-X)
                    vec = [-1.0, ny, nx]
                elif face_idx == 2:  # Up (+Y)
                    vec = [nx, 1.0, -ny]
                elif face_idx == 3:  # Down (-Y)
                    vec = [nx, -1.0, ny]
                elif face_idx == 4:  # Left (+Z)
                    vec = [nx, ny, 1.0]  # Fixed to match conventions
                elif face_idx == 5:  # Right (-Z)
                    vec = [-nx, ny, -1.0]  # Fixed to match conventions

                # Normalize vector
                norm = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
                vec = [v / norm for v in vec]

                # Convert to equirectangular coordinates
                phi = math.atan2(vec[2], vec[0])
                theta = math.asin(vec[1])

                # Map to equirectangular image coordinates
                u = (phi / (2 * math.pi) + 0.5) % 1.0
                v = (theta / math.pi + 0.5)

                # Sample from equirectangular image with bilinear interpolation
                # This gives better quality than nearest neighbor
                equirect_x = u * equirect_width
                equirect_y = v * equirect_height

                # Get the four surrounding pixels for interpolation
                x0, y0 = int(equirect_x), int(equirect_y)
                x1, y1 = min(x0 + 1, equirect_width - 1), min(y0 + 1, equirect_height - 1)

                # Get the four surrounding pixel values
                p00 = equirect[y0, x0]
                p01 = equirect[y0, x1]
                p10 = equirect[y1, x0]
                p11 = equirect[y1, x1]

                # Calculate weights
                wx = equirect_x - x0
                wy = equirect_y - y0

                # Bilinear interpolation
                value = (1 - wx) * (1 - wy) * p00 + wx * (1 - wy) * p01 + \
                        (1 - wx) * wy * p10 + wx * wy * p11

                # Set pixel in face
                face[y, x] = value.astype(np.uint8)

        faces.append(Image.fromarray(face))

    return faces


def cubemap_to_equirectangular(cubemap_faces, output_width=2048, output_height=1024, enable_overlap=True, overlap_degrees=2.5):
    """
    Convert 6 cubemap faces to equirectangular projection

    Args:
        cubemap_faces (list): List of 6 PIL images [front, back, up, down, left, right]
        output_width (int): Width of output equirectangular image
        output_height (int): Height of output equirectangular image
        enable_overlap (bool): Whether faces have overlapping regions
        overlap_degrees (float): Overlap in degrees for each face

    Returns:
        PIL.Image: Equirectangular image
    """
    # Crop faces if they have overlap
    if enable_overlap:
        cropped_faces = []
        for face in cubemap_faces:
            cropped_faces.append(crop_image_for_overlap(face, overlap_degrees))
        cubemap_faces = cropped_faces

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
            # Follow the same convention as in equirectangular_to_cubemap
            if max_val == abs_x:
                if x_vec > 0:
                    # Front face (+X)
                    face_idx = 0
                    u, v = -z_vec / x_vec, y_vec / x_vec
                else:
                    # Back face (-X)
                    face_idx = 1
                    u, v = z_vec / abs_x, y_vec / abs_x
            elif max_val == abs_y:
                if y_vec > 0:
                    # Up face (+Y)
                    face_idx = 2
                    u, v = x_vec / y_vec, -z_vec / y_vec
                else:
                    # Down face (-Y)
                    face_idx = 3
                    u, v = x_vec / abs_y, z_vec / abs_y
            else:
                if z_vec > 0:
                    # Left face (+Z)
                    face_idx = 4
                    u, v = x_vec / z_vec, y_vec / z_vec
                else:
                    # Right face (-Z)
                    face_idx = 5
                    u, v = -x_vec / abs_z, y_vec / abs_z

            # Map to face coordinates [0, 1]
            u = (u + 1) / 2
            v = (v + 1) / 2

            # Clamp to face bounds
            u = max(0, min(0.999, u))
            v = max(0, min(0.999, v))

            # Convert to pixel coordinates with bilinear interpolation
            px = u * face_size
            py = v * face_size

            # Get the four surrounding pixels for interpolation
            x0, y0 = int(px), int(py)
            x1, y1 = min(x0 + 1, face_size - 1), min(y0 + 1, face_size - 1)

            # Calculate weights
            wx = px - x0
            wy = py - y0

            # Get the four surrounding pixel values
            face_array = cube_arrays[face_idx]
            p00 = face_array[y0, x0]
            p01 = face_array[y0, x1]
            p10 = face_array[y1, x0]
            p11 = face_array[y1, x1]

            # Bilinear interpolation
            equirect[y, x] = ((1 - wx) * (1 - wy) * p00 + wx * (1 - wy) * p01 +
                              (1 - wx) * wy * p10 + wx * wy * p11).astype(np.uint8)

    return Image.fromarray(equirect)


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

    # Ensure margins are at least 1 pixel
    margin_x = max(1, margin_x)
    margin_y = max(1, margin_y)

    # Make sure we don't crop too much (avoid negative dimensions)
    if 2 * margin_x >= width or 2 * margin_y >= height:
        # If margins are too big, scale them down
        scale = min((width - 2) / (2 * margin_x), (height - 2) / (2 * margin_y))
        margin_x = int(margin_x * scale)
        margin_y = int(margin_y * scale)

    # Crop image
    return image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
