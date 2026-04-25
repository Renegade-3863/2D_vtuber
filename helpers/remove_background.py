from pathlib import Path
from PIL import Image
import os
import argparse
from collections import deque

import numpy as np

# Define the u2net model path relative to the project root path
project_root_path = Path(__file__).resolve().parents[1]
os.environ["U2NET_HOME"] = str(project_root_path / "models" / "u2net") 

from rembg import remove


def cleanup_white_background_residue(
    image,
    white_threshold=240,
    edge_alpha_threshold=220,
    edge_alpha_scale=0.45,
    hard_cut_alpha=36,
    fringe_luma_threshold=175,
    fringe_chroma_delta=48,
    fringe_strength=0.85,
):
    """
    Clean residual white background and edge halos after background removal.

    Strategy:
    1) Flood-fill near-white pixels connected to image borders and set alpha=0.
    2) Reduce alpha on semi-transparent near-white edge pixels to suppress white fringes.
    3) Decontaminate RGB on likely fringe pixels where white background is mixed in.
    """
    rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
    h, w, _ = rgba.shape
    rgb = rgba[:, :, :3]

    white_mask = (
        (rgb[:, :, 0] > white_threshold)
        & (rgb[:, :, 1] > white_threshold)
        & (rgb[:, :, 2] > white_threshold)
    )

    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    for x in range(w):
        if white_mask[0, x]:
            visited[0, x] = True
            queue.append((0, x))
        if white_mask[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            queue.append((h - 1, x))

    for y in range(h):
        if white_mask[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            queue.append((y, 0))
        if white_mask[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            queue.append((y, w - 1))

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while queue:
        y, x = queue.popleft()
        for dy, dx in directions:
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w and (not visited[ny, nx]) and white_mask[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    # Remove border-connected white background only.
    rgba[visited, 3] = 0

    # Suppress white edge halos while preserving non-white details.
    edge_white_mask = white_mask & (rgba[:, :, 3] < edge_alpha_threshold)
    scaled_alpha = (rgba[:, :, 3].astype(np.float32) * edge_alpha_scale).astype(np.uint8)
    rgba[edge_white_mask, 3] = scaled_alpha[edge_white_mask]

    # Remove tiny almost-invisible edge remnants.
    rgba[rgba[:, :, 3] <= hard_cut_alpha, 3] = 0

    # RGB decontamination for semi-transparent bright/gray-ish fringe.
    alpha = rgba[:, :, 3].astype(np.float32)
    rgb_f = rgba[:, :, :3].astype(np.float32)
    luma = rgb_f.mean(axis=2)
    cmax = rgb_f.max(axis=2)
    cmin = rgb_f.min(axis=2)
    chroma = cmax - cmin

    fringe_mask = (
        (alpha > 0)
        & (alpha < edge_alpha_threshold)
        & (luma > fringe_luma_threshold)
        & (chroma < fringe_chroma_delta)
    )

    if np.any(fringe_mask):
        # Estimate "whiteness" as shared gray component and suppress it.
        common = cmin
        alpha_factor = 1.0 - (alpha / 255.0)
        strength_map = np.clip(alpha_factor * fringe_strength, 0.0, 1.0)
        suppress = (common * strength_map)[:, :, None]
        rgb_f = np.clip(rgb_f - suppress, 0.0, 255.0)
        rgba[:, :, :3] = rgb_f.astype(np.uint8)

    return Image.fromarray(rgba, "RGBA")

def resize_rgba_image(image, output_size=512, resize_mode="pad"):
    """
    Resize image to output_size x output_size.
    - pad means to keep aspect ratio and pad transparent background
    """
    image = image.convert("RGBA")

    if resize_mode == "stretch":
        return image.resize((output_size, output_size), Image.Resampling.LANCZOS)

    # Default to "pad" mode
    canvas = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 0))
    src_w, src_h = image.size
    scale = min(output_size / src_w, output_size / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    offset_x = (output_size - new_w) // 2
    offset_y = (output_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y), resized)
    return canvas

def process_image_background(input_path, output_path, output_size=512, resize_mode="pad"):
    """
    Using rembg to remove the background of the input image and save the result to the designated output path.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return
    
    try:
        # Open the input image
        with Image.open(input_path) as input_image:
            # Remove the background using rembg
            output_image = remove(input_image)
            # Normalize output resolution for THA3
            output_image = resize_rgba_image(
                output_image,
                output_size=output_size,
                resize_mode=resize_mode
            )
            # Remove residual white background/halo artifacts.
            output_image = cleanup_white_background_residue(output_image)
            # Save the output image
            output_image.save(output_path)
            print(
                f"Successfully processed '{input_path}' and saved to '{output_path}' "
                f"with size {output_size}x{output_size} (mode={resize_mode})."
            )

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background of the image and add alpha channel.")
    parser.add_argument(
        "--input_image",
        type=str,
        help="Path to the input image file (Should be in data/images directory)."
    )
    args = parser.parse_args()
    input_filename = args.input_image
    
    # Generate the output filename automatically
    # ignore the extension of the input file
    base_name, ext = os.path.splitext(input_filename)
    output_filename = f"{base_name}_transparent.png"
    
    # Get the directory of the script, and deduce the input and output paths
    helpers_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(helpers_dir)
    
    # Construct the full paths to be used in the processing function
    input_image_path = os.path.join(project_root, "data", "images", input_filename)
    output_image_path = os.path.join(project_root, "data", "images", output_filename)
    
    # Perform the processing
    process_image_background(input_image_path, output_image_path)