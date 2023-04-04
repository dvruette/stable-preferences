import os
import warnings
from tempfile import NamedTemporaryFile
from datetime import date

import torch
import hydra
import tqdm
from omegaconf import DictConfig
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler
from transformers import CLIPModel
import numpy as np


dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.set_default_dtype(dtype)

def get_free_gpu():
    try:
        with NamedTemporaryFile() as f:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{}'.format(f.name))
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        return np.argmax(memory_available)
    except:
        warnings.warn("Could not get free GPU, using CPU")
        return "cpu"

@torch.no_grad()
def generate_trajectory_with_clip_guidance(
    pipe,
    clip,
    pos_prompt="a photo of an astronaut riding a horse on mars",
    neg_prompt="",
    cfg_scale=5,
    steps=20,
    seed=42,
    show_progress=True,
    batch_size=1,
    device=None,
):
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder.to(device)
    vae.cpu()
    clip.cpu()
    unet.cpu()

    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(batch_size, 4, 64, 64, device=device)
    z = z * scheduler.init_noise_sigma

    scheduler.set_timesteps(steps, device=device)
    prompt_tokens = tokenizer([pos_prompt] + [neg_prompt], return_tensors="pt", padding=True, truncation=True).to(z.device)
    text_encoder.to(device, non_blocking=True)
    prompt_embd = text_encoder(**prompt_tokens).last_hidden_state
    text_encoder.cpu()
    pos_prompt_embd = prompt_embd[:1] # first element
    neg_prompt_embd = prompt_embd[1:2] # second element
    prompt_embd = torch.cat([pos_prompt_embd] * batch_size + [neg_prompt_embd] * batch_size)

    iterator = scheduler.timesteps
    if show_progress:
        iterator = tqdm.tqdm(scheduler.timesteps)

    traj = []
    norms = []
    for i, t in enumerate(iterator):
        if hasattr(scheduler, "sigma_t"):
            sigma = scheduler.sigma_t[t]
        elif hasattr(scheduler, "sigma"):
            sigma = scheduler.sigma[t]
        else:
            raise ValueError("Unknown scheduler, doesn't have sigma_t or sigma attribute.")

        z = scheduler.scale_model_input(z, t)
        zs = torch.cat(2*[z], dim=0)
        assert zs.shape == (batch_size * 2, 4, 64, 64)

        unet.to(device, non_blocking=True)
        unet_out = unet(zs, t, prompt_embd).sample
        unet.cpu()
        noise_cond, noise_uncond = unet_out.chunk(2)

        # lats = z.detach().requires_grad_()
        lats_x0 = z - sigma * noise_cond

        image = vae.decode((1 / vae.config.scaling_factor) * lats_x0).sample
        image = (image / 2 + .5).clamp(0, 1)
        traj.append(pipe.numpy_to_pil(image.cpu().permute(0, 2, 3, 1).float().numpy()))

        # clip = clip.to(device, non_blocking=True)

        # clip.cpu()

        guidance = noise_cond - noise_uncond
        noise_pred = noise_cond + cfg_scale*guidance

        noise_pred = noise_pred.to(z.dtype)
        z = scheduler.step(noise_pred, t, z).prev_sample
        # if not only_decode_last or t == scheduler.timesteps[-1]:
        #     vae.to(device, non_blocking=True)
        #     y = pipe.decode_latents(z)
        #     vae.cpu()
        #     traj.append(pipe.numpy_to_pil(y))
    print("norms", norms)
    return traj


@hydra.main(config_path="../configs", config_name="clip_guided", version_base=None)
def main(ctx: DictConfig):
    if ctx.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = get_free_gpu() #"cuda" if torch.cuda.is_available() else device
    else:
        device = ctx.device
    print(f"Using device: {device}")

    scheduler = DPMSolverSinglestepScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(ctx.model_id, scheduler=scheduler, torch_dtype=dtype)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    print(f"Unet: {sum(p.numel() for p in pipe.unet.parameters()) / 1e6:.0f}M")
    print(f"VAE: {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.0f}M")
    print(f"TextEncoder: {sum(p.numel() for p in pipe.text_encoder.parameters()) / 1e6:.0f}M")

    traj = generate_trajectory_with_clip_guidance(
        pipe=pipe,
        clip=clip,
        pos_prompt=ctx.prompt,
        neg_prompt=ctx.neg_prompt,
        cfg_scale=ctx.cfg_scale,
        steps=ctx.steps,
        seed=ctx.seed,
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
