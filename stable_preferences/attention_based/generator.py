import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention import BasicTransformerBlock


def attn_with_weights(
    attn: nn.Module,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    weights=None,  # shape: (batch_size, sequence_length)
):
    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)

    if weights is not None:
        if weights.shape[0] != 1:
            weights = weights.repeat_interleave(attn.heads, dim=0)
        attention_probs = attention_probs * weights[:, None]
        attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


class StableDiffuserWithAttentionFeedback(nn.Module):

    def __init__(self,
            model_ckpt: Optional[str] = None,
            model_name: Optional[str] = None,
            stable_diffusion_version: str = "1.5",
            unet_max_chunk_size=8,
            torch_dtype=torch.float32,
        ):
        super().__init__()
        if model_name is None:
            if stable_diffusion_version == "1.5":
                model_name = "runwayml/stable-diffusion-v1-5"
            elif stable_diffusion_version == "2.1":
                model_name = "stabilityai/stable-diffusion-2-1"
            else:
                raise ValueError(f"Unknown stable diffusion version: {stable_diffusion_version}. Version must be either '1.5' or '2.1'")

        # scheduler = DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
        # scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        # scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

        if model_ckpt is not None:
            pipe = StableDiffusionPipeline.from_ckpt(model_ckpt, scheduler=scheduler, torch_dtype=torch_dtype, safety_checker=None)
            pipe.scheduler = scheduler
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler, torch_dtype=torch_dtype, safety_checker=None)

        self.pipeline = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = scheduler

        print(f"Unet: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.0f}M")
        print(f"VAE: {sum(p.numel() for p in self.vae.parameters()) / 1e6:.0f}M")
        print(f"TextEncoder: {sum(p.numel() for p in self.text_encoder.parameters()) / 1e6:.0f}M")

        self.unet_max_chunk_size = unet_max_chunk_size
        self.dtype = torch_dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device):
        self.pipeline.to(device)
        return super().to(device)

    def initialize_prompts(self, prompts: List[str]):

        prompt_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
        )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = prompt_tokens.attention_mask.to(self.device)
        else:
            attention_mask = None

        prompt_embd = self.text_encoder(
            input_ids=prompt_tokens.input_ids.to(self.device),
            attention_mask=attention_mask,
        ).last_hidden_state

        return prompt_embd
    
    def get_unet_hidden_states(self, z_all, t, prompt_embd):
        cached_hidden_states = []
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                def new_forward(self, hidden_states, *args, **kwargs):
                    cached_hidden_states.append(hidden_states.clone().detach().cpu())
                    return self.old_forward(hidden_states, *args, **kwargs)
                
                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)
        
        # run forward pass to cache hidden states, output can be discarded
        _ = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return cached_hidden_states
    
    def unet_forward_with_cached_hidden_states(
        self,
        z_all,
        t,
        prompt_embd,
        cached_pos_hiddens: Optional[List[torch.Tensor]] = None,
        cached_neg_hiddens: Optional[List[torch.Tensor]] = None,
        pos_weights=(0.8, 0.8),
        neg_weights=(0.5, 0.5),
    ):
        if cached_pos_hiddens is None and cached_neg_hiddens is None:
            return self.unet(z_all, t, encoder_hidden_states=prompt_embd)
        
        local_pos_weights = torch.linspace(*pos_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()
        local_neg_weights = torch.linspace(*neg_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()

        for block, pos_weight, neg_weight in zip(
            self.unet.down_blocks + [self.unet.mid_block] + self.unet.up_blocks,
            local_pos_weights + [pos_weights[1]] + local_pos_weights[::-1],
            local_neg_weights + [neg_weights[1]] + local_neg_weights[::-1],
        ):
            for module in block.modules():
                if isinstance(module, BasicTransformerBlock):
                    def new_forward(self, hidden_states, pos_weight=pos_weight, neg_weight=neg_weight, **kwargs):
                        cond_hiddens, uncond_hiddens = hidden_states.chunk(2, dim=0)
                        batch_size, d_model = cond_hiddens.shape[:2]
                        device, dtype = hidden_states.device, hidden_states.dtype

                        weights = torch.ones(batch_size, d_model, device=device, dtype=dtype)

                        if cached_pos_hiddens is not None:
                            cached_pos_hs = cached_pos_hiddens.pop(0).to(hidden_states.device)
                            cond_pos_hs = torch.cat([cond_hiddens, cached_pos_hs], dim=1)
                            pos_weights = weights.clone().repeat(1, 1 + cached_pos_hs.shape[1] // d_model)
                            pos_weights[:, d_model:] = pos_weight
                            out_pos = attn_with_weights(self, cond_hiddens, encoder_hidden_states=cond_pos_hs, weights=pos_weights)
                        else:
                            out_pos = self.old_forward(cond_hiddens)

                        if cached_neg_hiddens is not None:
                            cached_neg_hs = cached_neg_hiddens.pop(0).to(hidden_states.device)
                            uncond_neg_hs = torch.cat([uncond_hiddens, cached_neg_hs], dim=1)
                            neg_weights = weights.clone().repeat(1, 1 + cached_neg_hs.shape[1] // d_model)
                            neg_weights[:, d_model:] = neg_weight
                            out_neg = attn_with_weights(self, uncond_hiddens, encoder_hidden_states=uncond_neg_hs, weights=neg_weights)
                        else:
                            out_neg = self.old_forward(uncond_hiddens)

                        out = torch.cat([out_pos, out_neg], dim=0)
                        return out
                    
                    module.attn1.old_forward = module.attn1.forward
                    module.attn1.forward = new_forward.__get__(module.attn1)

        out = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return out
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "a photo of an astronaut riding a horse on mars",
        negative_prompt: str = "",
        liked: List[str] = [],
        disliked: List[str] = [],
        seed: int = 42,
        n_images: int = 1,
        guidance_scale: float = 8.0,
        denoising_steps: int = 20,
        only_decode_last: bool = False,
        feedback_time: Tuple[float, float] = (0.25, 0.75),
        min_weight: float = 0.05,
        max_weight: float = 0.8,
        neg_scale: float = 0.5,
        pos_bottleneck_scale: float = 1.0,
        neg_bottleneck_scale: float = 1.0,
    ):
        """
        Generate a trajectory of images with binary feedback.
        The feedback can be given as a list of liked and disliked prompts, or as images, which then get inverted.
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(n_images, 4, 64, 64, device=self.device, dtype=self.dtype)

        # out = self.pipeline(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     latents=z,
        #     guidance_scale=guidance_scale,
        #     num_inference_steps=denoising_steps,
        #     num_images_per_prompt=n_images,
        # )
        # return [out.images]

        if liked and len(liked) > 0:
            pos_images = [self.image_to_tensor(img) for img in liked]
            pos_images = torch.stack(pos_images).to(self.device, dtype=self.dtype)
            pos_latents = self.vae.config.scaling_factor * self.vae.encode(pos_images).latent_dist.sample()
        else:
            pos_latents = torch.tensor([], device=self.device, dtype=self.dtype)

        if liked and len(disliked) > 0:
            neg_images = [self.image_to_tensor(img) for img in disliked]
            neg_images = torch.stack(neg_images).to(self.device, dtype=self.dtype)
            neg_latents = self.vae.config.scaling_factor * self.vae.encode(neg_images).latent_dist.sample()
        else:
            neg_latents = torch.tensor([], device=self.device, dtype=self.dtype)
        
        cond_prompt_embd, uncond_prompt_embd, null_prompt_embd = self.initialize_prompts([prompt, negative_prompt, ""]).split(1)
        batched_prompt_embd = torch.cat([cond_prompt_embd, uncond_prompt_embd], dim=0)
        batched_prompt_embd = batched_prompt_embd.repeat_interleave(n_images, dim=0)

        self.scheduler.set_timesteps(denoising_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        z = z * self.scheduler.init_noise_sigma

        num_warmup_steps = len(timesteps) - denoising_steps * self.scheduler.order

        ref_start_idx = round(len(timesteps) * (1 - feedback_time[1]))
        ref_end_idx = round(len(timesteps) * (1 - feedback_time[0]))

        traj = []
        with tqdm(total=denoising_steps) as pbar:
            for i, t in enumerate(timesteps):
                if hasattr(self.scheduler, "sigma_t"):
                    sigma = self.scheduler.sigma_t[t]
                elif hasattr(self.scheduler, "sigmas"):
                    sigma = self.scheduler.sigmas[i]
                else:
                    sigma = 0
                alpha_hat = 1 / (sigma**2 + 1)

                z_single = self.scheduler.scale_model_input(z, t)
                z_all = torch.cat([z_single] * 2, dim=0)
                z_ref = torch.cat([pos_latents, neg_latents], dim=0)

                
                if i >= ref_start_idx and i <= ref_end_idx:
                    weight = max_weight
                else:
                    weight = min_weight
                pos_ws = (weight, weight * pos_bottleneck_scale)
                neg_ws = (weight * neg_scale, weight * neg_scale * neg_bottleneck_scale)

                if z_ref.size(0) > 0:
                    noise = torch.randn_like(z_ref)
                    if isinstance(self.scheduler, EulerAncestralDiscreteScheduler):
                        z_ref_noised = alpha_hat**0.5 * z_ref + (1 - alpha_hat)**0.5 * noise
                    else:
                        z_ref_noised = self.scheduler.add_noise(z_ref, noise, t)
                    # ref_prompt_embd = torch.cat([cond_prompt_embd] * pos_latents.size(0) + [null_prompt_embd] * neg_latents.size(0), dim=0)
                    ref_prompt_embd = torch.cat([cond_prompt_embd] * (pos_latents.size(0) + neg_latents.size(0)), dim=0)

                    cached_hidden_states = self.get_unet_hidden_states(z_ref_noised, t, ref_prompt_embd)

                    n_pos, n_neg = pos_latents.shape[0], neg_latents.shape[0]
                    cached_pos_hs, cached_neg_hs = [], []
                    for hs in cached_hidden_states:
                        cached_pos, cached_neg = hs.split([n_pos, n_neg], dim=0)
                        cached_pos = cached_pos.view(1, -1, *cached_pos.shape[2:]).expand(n_images, -1, -1)
                        cached_neg = cached_neg.view(1, -1, *cached_neg.shape[2:]).expand(n_images, -1, -1)
                        cached_pos_hs.append(cached_pos)
                        cached_neg_hs.append(cached_neg)

                    if n_pos == 0:
                        cached_pos_hs = None
                    if n_neg == 0:
                        cached_neg_hs = None
                else:
                    cached_pos_hs, cached_neg_hs = None, None
                
                unet_out = self.unet_forward_with_cached_hidden_states(
                    z_all,
                    t,
                    prompt_embd=batched_prompt_embd,
                    cached_pos_hiddens=cached_pos_hs,
                    cached_neg_hiddens=cached_neg_hs,
                    pos_weights=pos_ws,
                    neg_weights=neg_ws,
                ).sample

                noise_cond, noise_uncond = unet_out.chunk(2)
                guidance = noise_cond - noise_uncond
                noise_pred = noise_uncond + guidance_scale * guidance
                z = self.scheduler.step(noise_pred, t, z).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    pbar.update()
                    if not only_decode_last or i == len(self.scheduler.timesteps) - 1:
                        if i < len(timesteps) - 1:
                            lats_x0 = (z_single - (1 - alpha_hat)**0.5 * noise_cond) / alpha_hat**0.5
                            sqrt_beta_hat = (1 - alpha_hat)**0.5
                            pred_x0 = sqrt_beta_hat*lats_x0 + (1 - sqrt_beta_hat)*z_single
                        else:
                            pred_x0 = z

                        y = self.pipeline.decode_latents(pred_x0)
                        piled = self.pipeline.numpy_to_pil(y)
                        os.makedirs("outputs/trajectory", exist_ok=True)
                        for j, img in enumerate(piled):
                            img.save(f"outputs/trajectory/{i}_{j}.png")
                        traj.append(piled)

        return traj
    
    @staticmethod
    def image_to_tensor(image: str):
        """
        Convert a PIL image to a torch tensor.
        """
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512, 512))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)
