import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from PIL import Image

from tha3.poser.modes.load_poser import load_poser
from tha3.util import extract_pytorch_image_from_PIL_image, rgba_to_numpy_image

def lerp(a, b, x):
    return a + (b - a) * x

def sample_happy(t):
    # t in [0, 0.8]
    if t <= 0.4:
        alpha = t / 0.4
        mouth = lerp(0.3, 0.8, alpha)
        open_mouth = lerp(0.1, 0.4, alpha)
        eye = lerp(0.1, 0.3, alpha)
        brow = lerp(0.15, 0.35, alpha)
    else:
        alpha = (t - 0.4) / 0.4
        mouth = lerp(0.8, 0.3, alpha)
        open_mouth = lerp(0.4, 0.1, alpha)
        eye = lerp(0.3, 0.1, alpha)
        brow = lerp(0.35, 0.15, alpha)

    return {
        "mouth_raised_corner_left": mouth,
        "mouth_raised_corner_right": mouth,
        "mouth_aaa": open_mouth,
        "eye_happy_wink_left": eye,
        "eye_happy_wink_right": eye,
        "eyebrow_happy_left": brow,
        "eyebrow_happy_right": brow,
    }

def build_name_to_index(poser):
    name_to_index = {}
    idx = 0
    for group in poser.get_pose_parameter_groups():
        for name in group.get_parameter_names():
            name_to_index[name] = idx
            idx += 1
    return name_to_index


def main():
    device = torch.device("cuda")
    poser = load_poser("separable_half", device)
    dtype = poser.get_dtype()

    img_path = Path("data/images/Talking-Head-Anime-①.png")
    pil = Image.open(img_path).convert("RGBA")
    torch_image = extract_pytorch_image_from_PIL_image(pil).to(device).to(dtype)

    name_to_index = build_name_to_index(poser)
    num_params = poser.get_num_parameters()

    fps = 20
    period = 0.8
    start = time.time()

    while True:
        t = (time.time() - start) % period
        pose = [0.0] * num_params

        for k, v in sample_happy(t).items():
            if k in name_to_index:
                pose[name_to_index[k]] = v
        
        pose_tensor = torch.tensor(pose, device=device, dtype=dtype)

        with torch.no_grad():
            out = poser.pose(torch_image, pose_tensor)[0].detach().cpu()

        img = rgba_to_numpy_image(out)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

        cv2.imshow("happy_demo", cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()