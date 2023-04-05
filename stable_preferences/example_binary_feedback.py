import os
from datetime import date

import torch
import hydra
from omegaconf import DictConfig
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler

from stable_preferences.utils import get_free_gpu, generate_trajectory, generate_trajectory_with_binary_feedback

MODE = "binary_feedback"

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.set_default_dtype(dtype)


@hydra.main(config_path="configs", config_name="example_binary_feedback", version_base=None)
def main(ctx: DictConfig):
    if ctx.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = get_free_gpu() #"cuda" if torch.cuda.is_available() else device
    else:
        device = ctx.device
    print(f"Using device: {device}")

    scheduler = DPMSolverSinglestepScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(ctx.model_id, scheduler=scheduler, torch_dtype=dtype).to(device)

    print(f"Unet: {sum(p.numel() for p in pipe.unet.parameters()) / 1e6:.0f}M")
    print(f"VAE: {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.0f}M")
    print(f"TextEncoder: {sum(p.numel() for p in pipe.text_encoder.parameters()) / 1e6:.0f}M")

    if MODE!= "binary_feedback":
        traj = generate_trajectory(
            pipe,
            ctx.prompt,
            ctx.neg_prompt,
            ctx.cfg_scale,
            steps=ctx.steps,
            seed=ctx.seed,
            only_decode_last=True,
            device=device,
        )
    else:
        traj = generate_trajectory_with_binary_feedback(
            pipe,
            ctx.prompt,
            ctx.neg_prompt,
            list(ctx.liked_prompts),
            list(ctx.disliked_prompts),
            ctx.cfg_scale,
            steps=ctx.steps,
            seed=ctx.seed,
            only_decode_last=True,
            device=device,
        )
    img = traj[-1][-1]
    
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join("outputs", "images", date_str)
    os.makedirs(out_folder, exist_ok=True)
    n_files = len([name for name in os.listdir(out_folder)])

    out_path = os.path.join(out_folder, f"example_{n_files}.png")
    img.save(out_path)
    print(f"Saved image to {out_path}")

    try:
        img.show()
    except:
        pass


if __name__ == "__main__":
    main()
