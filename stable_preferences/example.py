import torch
import hydra
from omegaconf import DictConfig
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler

from stable_preferences.utils import generate_trajectory


dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.set_default_dtype(dtype)


@hydra.main(config_path="configs", config_name="example", version_base=None)
def main(ctx: DictConfig):
    if ctx.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = "cuda" if torch.cuda.is_available() else device
    else:
        device = ctx.device
    print(f"Using device: {device}")

    scheduler = DPMSolverSinglestepScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(ctx.model_id, scheduler=scheduler, torch_dtype=dtype).to(device)

    print(f"Unet: {sum(p.numel() for p in pipe.unet.parameters()) / 1e6:.0f}M")
    print(f"VAE: {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.0f}M")
    print(f"TextEncoder: {sum(p.numel() for p in pipe.text_encoder.parameters()) / 1e6:.0f}M")

    traj = generate_trajectory(
        pipe,
        ctx.prompt,
        ctx.neg_prompt,
        ctx.cfg_scale,
        steps=ctx.steps,
        seed=ctx.seed,
        only_last=True,
        device=device,
    )
    img = traj[-1][-1]
    img.save("example.png")
    img.show()


if __name__ == "__main__":
    main()
