import os
from datetime import date

import hydra
from omegaconf import DictConfig
import torch

from stable_preferences.attention_based.generator import StableDiffuserWithAttentionFeedback
from stable_preferences.utils import get_free_gpu


@hydra.main(config_path="../configs", config_name="attention_based", version_base=None)
def main(ctx: DictConfig):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    device = "cpu"
    print(f"Using device: {device}")

    dtype = torch.float16 if str(device) == "cuda" else torch.float32
    print(f"Using dtype: {dtype}")

    generator = StableDiffuserWithAttentionFeedback(
        stable_diffusion_version=ctx.model_version,
        unet_max_chunk_size=ctx.unet_max_chunk_size,
        torch_dtype=dtype,
    ).to(device)

    trajectory = generator.generate(
        prompt=ctx.prompt,
        liked=list(ctx.liked_images),
        disliked=list(ctx.disliked_images),
        field=ctx.field.field_type,
        binary_feedback_type=ctx.binary_feedback_type,
        seed=ctx.seed,
        n_images=ctx.n_images,
        guidance_scale=ctx.field.guidance_scale,
        walk_distance=ctx.field.walk_distance,
        walk_steps=ctx.field.walk_steps,
        flatten_channels=ctx.field.flatten_channels,
        denoising_steps=ctx.denoising_steps,
        **ctx.additional_args,
    )

    imgs = trajectory[:][-1]
    
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join("outputs", "images", date_str)
    os.makedirs(out_folder, exist_ok=True)
    
    for img in imgs:
        n_files = len([name for name in os.listdir(out_folder)])
        out_path = os.path.join(out_folder, f"example_{n_files}.png")
        img.save(out_path)
        print(f"Saved image to {out_path}")


if __name__ == "__main__":
    main()
