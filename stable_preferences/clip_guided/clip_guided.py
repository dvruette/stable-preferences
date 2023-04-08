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
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


@torch.no_grad()
def generate_trajectory_with_clip_guidance(
    pipe,
    clip,
    clip_tokenizer,
    feature_extractor,
    pos_prompt="a photo of an astronaut riding a horse on mars",
    neg_prompt="",
    clip_prompt="",
    pos_images=None,
    neg_images=None,
    cfg_scale=7,
    n_cuts=4,
    use_cutouts=False,
    clip_scale=100.0,
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

    slice_size = unet.config.attention_head_dim // 2
    unet.set_attention_slice(slice_size)

    clip_img_size = feature_extractor.size["shortest_edge"]
    make_cutouts = MakeCutouts(clip_img_size, n_cuts)
    resize = T.Resize(clip_img_size)
    normalize = T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    scheduler.set_timesteps(steps, device=device)

    prompt_tokens = tokenizer(
        [pos_prompt] + [neg_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    prompt_embd = text_encoder(prompt_tokens.input_ids).last_hidden_state
    pos_prompt_embd = prompt_embd[:1] # first element
    neg_prompt_embd = prompt_embd[1:2] # second element
    prompt_embd = torch.cat([pos_prompt_embd] * batch_size + [neg_prompt_embd] * batch_size)

    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(batch_size, 4, 64, 64, device=device, dtype=prompt_embd.dtype)
    z = z * scheduler.init_noise_sigma

    if pos_images is not None:
        pos_img_embd = clip.get_image_features(pos_images.to(device, dtype=z.dtype))
    if neg_images is not None:
        neg_img_embd = clip.get_image_features(neg_images.to(device, dtype=z.dtype))

    if clip_prompt:
        tokens = clip_tokenizer(
            [clip_prompt],
            truncation=True,
            return_tensors="pt",
        ).to(device)
        clip_prompt_embd = clip.get_text_features(tokens.input_ids)
        clip_prompt_embd = clip_prompt_embd / clip_prompt_embd.norm(p=2, dim=-1, keepdim=True)


    iterator = scheduler.timesteps
    if show_progress:
        iterator = tqdm.tqdm(scheduler.timesteps)

    traj = []
    for i, t in enumerate(iterator):
        with torch.enable_grad():
            z_cond = z.detach().requires_grad_()
            zs = torch.cat([z_cond, z], dim=0)
            zs = scheduler.scale_model_input(zs, t)

            unet_out = unet(zs, t, prompt_embd).sample
            noise_cond, noise_uncond = unet_out.chunk(2)

            alpha_prod_t = scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            
            lats_x0 = (z_cond - beta_prod_t ** (0.5) * noise_cond) / alpha_prod_t ** (0.5)
            fac = torch.sqrt(beta_prod_t)
            sample_x0 = lats_x0 * (fac) + z_cond * (1 - fac)

            sample_x0 = 1 / vae.config.scaling_factor * sample_x0
            image = vae.decode(sample_x0).sample
            image = (image / 2 + 0.5).clamp(0, 1)

            if use_cutouts:
                cropped_img = torch.cat([make_cutouts(image), resize(image)], dim=0)
            else:
                cropped_img = resize(image)
            cropped_img = normalize(cropped_img).to(z.dtype)
            img_embd = clip.get_image_features(cropped_img)
            img_embd = img_embd / img_embd.norm(p=2, dim=-1, keepdim=True)

            losses = []
            # pos_dist = spherical_distance(img_embd, pos_img_embd)
            # losses.append(pos_dist.mean())

            # if clip_prompt:
            #     prompt_dist = spherical_distance(img_embd, clip_prompt_embd)
            #     if use_cutouts:
            #         prompt_dist = prompt_dist.view(n_cuts + 1, batch_size, -1)
            #         losses.append(prompt_dist[:-1].sum(2).mean(0).mean())
            #     else:
            #         prompt_dist = prompt_dist.view(1, batch_size, -1)
            #     losses.append(prompt_dist[-1].mean()) # resized

            if pos_images is not None:
                pos_dist = spherical_distance(img_embd, pos_img_embd)
                if use_cutouts:
                    pos_dist = pos_dist.view(n_cuts + 1, batch_size, -1)
                    losses.append(pos_dist[:-1].sum(2).mean(0).mean())
                    losses.append(pos_dist[-1].mean())
                else:
                    losses.append(pos_dist.mean())

            # if neg_images is not None:
            #     neg_dist = -spherical_distance(img_embd, neg_img_embd)
            #     if use_cutouts:
            #         neg_dist = neg_dist.view(n_cuts + 1, batch_size, -1)
            #         losses.append(neg_dist[:-1].sum(2).mean(0).mean())
            #         losses.append(neg_dist[-1].mean())
            #     else:
            #         losses.append(neg_dist.mean())

            # compute gradient
            loss = torch.stack(losses).sum()
            grads = -torch.autograd.grad(loss, z_cond)[0]
            mag = grads.square().sum().sqrt()
            grads = grads * mag.clamp(max=0.025) / mag

            # apply gradient to noise
            # noise_pred = (noise_cond + cfg_scale*(noise_cond - noise_uncond)).detach()
            # noise_pred = noise_pred.detach() - (clip_scale * torch.sqrt(beta_prod_t) * grads.detach())
            noise_cond = noise_cond.detach() - (clip_scale * torch.sqrt(beta_prod_t) * grads.detach())
            noise_pred = noise_cond + cfg_scale*(noise_cond - noise_uncond)

        z = scheduler.step(noise_pred, t, z).prev_sample
        traj.append(pipe.numpy_to_pil(image.detach().cpu().permute(0, 2, 3, 1).float().numpy()))
    
    # scale and decode the image latents with vae
    x0 = 1 / vae.config.scaling_factor * z
    image = vae.decode(x0).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    traj.append(pipe.numpy_to_pil(image.detach().cpu().permute(0, 2, 3, 1).float().numpy()))

    return traj


@hydra.main(config_path="../configs", config_name="clip_guided", version_base=None)
def main(ctx: DictConfig):
    # load models
    scheduler = DPMSolverSinglestepScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(ctx.model_id, scheduler=scheduler, torch_dtype=torch.float16)
    clip = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16)
    processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

    clip.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)

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
        clip_tokenizer=clip_tokenizer,
        feature_extractor=processor,
        pos_prompt=ctx.prompt,
        neg_prompt=ctx.neg_prompt,
        clip_prompt=ctx.clip_prompt,
        pos_images=liked_imgs,
        neg_images=disliked_imgs,
        cfg_scale=ctx.cfg_scale,
        clip_scale=ctx.clip_scale,
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
