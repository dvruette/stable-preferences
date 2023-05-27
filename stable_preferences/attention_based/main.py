import math
import os
import re
from datetime import date

import hydra
import torch
from PIL import Image
from omegaconf import DictConfig

from stable_preferences.attention_based.generator import StableDiffuserWithAttentionFeedback
from stable_preferences.utils import get_free_gpu


def tile_images(images):
    size = images[0].size
    assert all(img.size == size for img in images), "All images must have the same size"

    grid_size_x = math.ceil(len(images) ** 0.5)
    grid_size_y = math.ceil(len(images) / grid_size_x)

    # Create a new blank image with the size of the tiled grid
    tiled_image = Image.new('RGB', (grid_size_x * size[0], grid_size_y * size[1]))

    # Paste the four images into the tiled image
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            idx = x + grid_size_x * y
            if idx >= len(images):
                break
            img = images[idx]
            tiled_image.paste(img, (x * size[0], y * size[1]))
    return tiled_image


@hydra.main(config_path="../configs", config_name="attention_based", version_base=None)
def main(ctx: DictConfig):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    print(f"Using device: {device}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using dtype: {dtype}")

    generator = StableDiffuserWithAttentionFeedback(
        model_ckpt=ctx.model_ckpt if hasattr(ctx, "model_ckpt") else None,
        model_name=ctx.model_name if hasattr(ctx, "model_name") else None,
        stable_diffusion_version=ctx.model_version,
        torch_dtype=dtype,
    ).to(device)

    trajectory = generator.generate(
        prompt=ctx.prompt,
        negative_prompt=ctx.negative_prompt,
        liked=list(ctx.liked_images) if ctx.liked_images else [],
        disliked=list(ctx.disliked_images) if ctx.disliked_images else [],
        seed=ctx.seed,
        n_images=ctx.n_images,
        denoising_steps=ctx.denoising_steps,
        feedback_start=ctx.feedback.start,
        feedback_end=ctx.feedback.end,
        min_weight=ctx.feedback.min_weight,
        max_weight=ctx.feedback.max_weight,
        neg_scale=ctx.feedback.neg_scale,
    )

    imgs = trajectory[:][-1]
    
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join("outputs", "images", date_str)
    os.makedirs(out_folder, exist_ok=True)
    
    n_files = max([int(f.split(".")[0].split("_")[1]) for f in os.listdir(out_folder) if re.match(r"example_[0-9_]+\.png", f)], default=0) + 1
    for i, img in enumerate(imgs):
        # each image is of the form example_ID.png. Extract the max id
        out_path = os.path.join(out_folder, f"example_{n_files}_{i}.png")
        img.save(out_path)
        print(f"Saved image to {out_path}")
    
    if len(imgs) > 1:
        tiled = tile_images(imgs)
        tiled_path = os.path.join(out_folder, f"tiled_{n_files}.png")
        tiled.save(tiled_path)
        print(f"Saved tile to {tiled_path}")


if __name__ == "__main__":
    main()
