import torch
import hydra
from omegaconf import DictConfig
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler

from stable_preferences.utils import get_free_gpu


dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.set_default_dtype(dtype)


@hydra.main(config_path="configs", config_name="example", version_base=None)
def main(ctx: DictConfig):
    if ctx.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = get_free_gpu() 
    else:
        device = ctx.device
    print(f"Using device: {device}")

    scheduler = DPMSolverSinglestepScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(ctx.model_id, scheduler=scheduler, torch_dtype=dtype).to(device)

    print(f"Unet: {sum(p.numel() for p in pipe.unet.parameters()) / 1e6:.0f}M")
    print(f"VAE: {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.0f}M")
    print(f"TextEncoder: {sum(p.numel() for p in pipe.text_encoder.parameters()) / 1e6:.0f}M")

    torch.manual_seed(ctx.seed)
    init_noise = torch.randn(1, 4, 64, 64)

    output = pipe(
        prompt=ctx.prompt,
        height=ctx.height,
        width=ctx.width,
        num_inference_steps=ctx.steps,
        guidance_scale=ctx.cfg_scale,
        negative_prompt=ctx.neg_prompt,
        output_type="pil",
        latents=init_noise,
    )

    img = output["images"][0]

    img.save("example.png")
    img.show()


if __name__ == "__main__":
    main()
