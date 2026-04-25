# pip install torch diffusers transformers accelerate pillow rembg
# 如果是 NVIDIA，建议先按你 CUDA 版本安装对应 torch

import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

POSITIVE = (
    "single anime humanoid character, upper-body shot, front-facing, standing upright, "
    "centered composition, head located in the middle area of the top half, "
    "hands below and away from the head, clear visible eyes and eyebrows, "
    "clean silhouette, high quality lineart, plain background"
)
NEGATIVE = (
    "multiple characters, side view, tilted head, closed eyes, hair covering eyes, "
    "hand near face, busy background, cropped head, low contrast face, blurry"
)

def generate_tha3_candidate(out_path="data/images/tha3_candidate.png", seed=42):
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    g = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=POSITIVE,
        negative_prompt=NEGATIVE,
        width=512,
        height=512,
        guidance_scale=7.0,
        num_inference_steps=30,
        generator=g,
    ).images[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_tha3_candidate()
