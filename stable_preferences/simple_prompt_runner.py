import os
from datetime import date

import hydra
from omegaconf import DictConfig
import torch

from stable_preferences.generation_trajectory import StableDiffuserWithBinaryFeedback
from stable_preferences.utils import get_free_gpu


@hydra.main(config_path="configs", config_name="simple_prompt_runner", version_base=None)
def main(ctx: DictConfig):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    print(f"Using device: {device}")

    dtype = torch.float16 if str(device) != "cpu" else torch.float32
    print(f"Using dtype: {dtype}")

    generator = StableDiffuserWithBinaryFeedback(
        stable_diffusion_version=ctx.model_version,
        unet_max_chunk_size=ctx.unet_max_chunk_size,
        torch_dtype=dtype,
    ).to(device)

    trajectory = generator.generate(
        prompt=ctx.prompt,
        liked=list(ctx.liked_prompts),
        disliked=list(ctx.disliked_prompts),
        field=ctx.field,
        binary_feedback_type=ctx.binary_feedback_type,
        seed=ctx.seed,
        n_images=ctx.n_images,
        walk_distance=ctx.walk_distance,
        walk_steps=ctx.walk_steps,
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
