import os
from datetime import date

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import tqdm
import torchvision.transforms as T
from PIL import Image
from omegaconf import DictConfig
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler
from transformers import CLIPModel, CLIPImageProcessor, AutoTokenizer

from stable_preferences.utils import get_free_gpu


dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.set_default_dtype(dtype)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

    
def spherical_distance(x, y):
    x = F.normalize(x, dim=-1).unsqueeze(1)
    y = F.normalize(y, dim=-1).unsqueeze(0)
    l = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    return l

@torch.no_grad()
def generate_trajectory_with_clip_guidance(
    pipe,
    clip,
    pos_prompt="a photo of an astronaut riding a horse on mars",
    neg_prompt="",
    pos_images=None,
    neg_images=None,
    cfg_scale=7,
    n_cuts=4,
    alpha=4000.0,
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
    pipe.to(device)
    clip.to(device)

    clip_img_size = clip.config.vision_config.image_size
    # crop = T.RandomCrop(clip_img_size)
    make_cutouts = MakeCutouts(clip_img_size, n_cuts)

    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(batch_size, 4, 64, 64, device=device)
    z = z * scheduler.init_noise_sigma

    scheduler.set_timesteps(steps, device=device)
    prompt_tokens = tokenizer([pos_prompt] + [neg_prompt], return_tensors="pt", padding=True, truncation=True).to(z.device)
    prompt_embd = text_encoder(**prompt_tokens).last_hidden_state
    pos_prompt_embd = prompt_embd[:1] # first element
    neg_prompt_embd = prompt_embd[1:2] # second element
    prompt_embd = torch.cat([pos_prompt_embd] * batch_size + [neg_prompt_embd] * batch_size)

    if pos_images is not None:
        pos_img_embd = clip.get_image_features(pos_images.to(device, dtype=z.dtype))
    if neg_images is not None:
        neg_img_embd = clip.get_image_features(neg_images.to(device, dtype=z.dtype))

    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    tokens = clip_tokenizer(["picture of a mountain lake"], return_tensors="pt").to(device)
    clip_prompt_embd = clip.get_text_features(**tokens)

    iterator = scheduler.timesteps
    if show_progress:
        iterator = tqdm.tqdm(scheduler.timesteps)

    traj = []
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

        unet_out = unet(zs, t, prompt_embd).sample
        noise_cond, noise_uncond = unet_out.chunk(2)

        # guidance = noise_cond
        # noise_pred = noise_cond
        noise_pred = noise_cond + cfg_scale*(noise_cond - noise_uncond)

        with torch.enable_grad():
            # noise_cond = noise_cond.detach().requires_grad_()
            # noise_pred = noise_cond + (noise_cond - noise_uncond)
            noise_pred = noise_pred.detach().requires_grad_()
            
            lats_x0 = z - sigma * noise_pred

            image = vae.decode((1 / vae.config.scaling_factor) * lats_x0).sample
            image = (image / 2 + .5).clamp(0, 1)
            traj.append(pipe.numpy_to_pil(image.detach().cpu().permute(0, 2, 3, 1).float().numpy()))

            # cropped_imgs = []
            # for _ in range(n_cuts):
            #     # cropped = crop(image.to(torch.float32)).to(image.dtype)
            #     cropped = crop(image)
            #     cropped_imgs.append(F.adaptive_avg_pool2d(cropped, (clip_img_size, clip_img_size)))
            # cropped_img = torch.stack(cropped_imgs).view(-1, 3, clip_img_size, clip_img_size)
            cropped_img = make_cutouts(image)
            img_embd = clip.get_image_features(cropped_img)

            losses = []
            prompt_dist = spherical_distance(img_embd, clip_prompt_embd)
            losses.append(prompt_dist.view(n_cuts, -1).mean(dim=0))

            # if pos_images is not None:
            #     pos_dist = spherical_distance(img_embd, pos_img_embd)  # shape: (batch_size * n_cuts, num_pos)
            #     losses.append(pos_dist.view(n_cuts, -1).mean(dim=0))

            # if neg_images is not None:
            #     neg_dist = spherical_distance(img_embd, neg_img_embd)
            #     losses.append(neg_dist.view(n_cuts, -1).mean(dim=0))

            # compute gradient
            gradient = torch.autograd.grad(torch.stack(losses).sum(), noise_pred)[0]
            mag = gradient.square().sum().sqrt()
            gradient = gradient * mag.clamp(max=0.01) / mag
            # apply gradient to noise
            noise_pred = noise_pred.detach() - (gradient.detach() * alpha * sigma**2)


        noise_pred = noise_pred.to(z.dtype)
        z = scheduler.step(noise_pred, t, z).prev_sample
    return traj


@hydra.main(config_path="../configs", config_name="clip_guided", version_base=None)
def main(ctx: DictConfig):
    # load models
    scheduler = DPMSolverSinglestepScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(ctx.model_id, scheduler=scheduler, torch_dtype=dtype)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    print(f"Unet: {sum(p.numel() for p in pipe.unet.parameters()) / 1e6:.0f}M")
    print(f"VAE: {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.0f}M")
    print(f"TextEncoder: {sum(p.numel() for p in pipe.text_encoder.parameters()) / 1e6:.0f}M")

    # load feedback images
    if ctx.liked_images:
        imgs = [Image.open(f) for f in ctx.liked_images]
        liked_imgs = processor.preprocess(imgs, return_tensors="pt")["pixel_values"]
    else:
        liked_imgs = None
    if ctx.disliked_images:
        imgs = [Image.open(f) for f in ctx.liked_images]
        disliked_imgs = processor.preprocess(imgs, return_tensors="pt")["pixel_values"]
    else:
        disliked_imgs = None

    # select device
    if ctx.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = get_free_gpu() #"cuda" if torch.cuda.is_available() else device
    else:
        device = ctx.device
    print(f"Using device: {device}")

    # generate trajectory
    traj = generate_trajectory_with_clip_guidance(
        pipe=pipe,
        clip=clip,
        pos_prompt=ctx.prompt,
        neg_prompt=ctx.neg_prompt,
        pos_images=liked_imgs,
        neg_images=disliked_imgs,
        cfg_scale=ctx.cfg_scale,
        steps=ctx.steps,
        seed=ctx.seed,
        device=device,
    )
    
    # write generated image(s) to disk
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join("outputs", "images", date_str)
    os.makedirs(out_folder, exist_ok=True)
    n_files = len([name for name in os.listdir(out_folder)])

    if ctx.save == "all":
        traj_path = os.path.join(out_folder, f"trajectory_{n_files}")
        os.makedirs(traj_path, exist_ok=True)

        for i, imgs in enumerate(traj):
            for j, img in enumerate(imgs):
                out_path = os.path.join(traj_path, f"{i}_{j}.png")
                img.save(out_path)
        print(f"Saved trajectory to {traj_path}")
    elif ctx.save == "last":
        imgs = traj[-1]
        for i, img in enumerate(imgs):
            out_path = os.path.join(out_folder, f"image_{n_files}_{i}.png")
            img.save(out_path)
            print(f"Saved image to {out_path}")



if __name__ == "__main__":
    main()
