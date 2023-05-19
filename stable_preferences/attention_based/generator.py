import math
import os
from typing import Literal, List

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from einops import repeat, rearrange, pack, unpack
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler
from diffusers.models.attention import BasicTransformerBlock

from stable_preferences.fields import walk_in_field


class StableDiffuserWithAttentionFeedback(nn.Module):

    def __init__(self, 
            stable_diffusion_version: str = "1.5",
            unet_max_chunk_size=8,
            torch_dtype=torch.float32,
        ):
        super().__init__()

        if stable_diffusion_version == "1.5":
            model_name = "runwayml/stable-diffusion-v1-5"
        elif stable_diffusion_version == "2.1":
            model_name = "stabilityai/stable-diffusion-2-1"
        else:
            raise ValueError(f"Unknown stable diffusion version: {stable_diffusion_version}. Version must be either '1.5' or '2.1'")

        scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler, torch_dtype=torch_dtype, safety_checker=None)
        # pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

        self.pipeline = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler

        print(f"Unet: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.0f}M")
        print(f"VAE: {sum(p.numel() for p in self.vae.parameters()) / 1e6:.0f}M")
        print(f"TextEncoder: {sum(p.numel() for p in self.text_encoder.parameters()) / 1e6:.0f}M")

        self.unet_max_chunk_size = unet_max_chunk_size
        self.dtype = torch_dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_prompts(self, cond_prompt, uncond_prompt=""):

        prompt_tokens = self.tokenizer(
            [uncond_prompt, cond_prompt],
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
        )
        prompt_embd = prompt_embd[0]

        return prompt_embd[0], prompt_embd[1]
    
    def get_unet_hidden_states(self, z_all, t, batched_prompt_embd):
        cached_hidden_states = []
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                def new_forward(self, hidden_states, *args, **kwargs):
                    cached_hidden_states.append(hidden_states.clone().detach().cpu())
                    return self.old_forward(hidden_states, *args, **kwargs)
                
                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)
        
        # run forward pass to cache hidden states, output can be discarded
        _ = self.unet(z_all, t, encoder_hidden_states=batched_prompt_embd)

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
        batched_prompt_embd,
        cached_hidden_states,
    ):
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                def new_forward(self, hidden_states, *args, encoder_hidden_states=None, **kwargs):
                    cached_hs = cached_hidden_states.pop(0).to(hidden_states.device)
                    if encoder_hidden_states is not None:
                        print(encoder_hidden_states.shape)
                    hs = torch.cat([hidden_states, cached_hs], dim=1)
                    # out = self.old_forward(hs, *args, **kwargs)
                    out = self.old_forward(hidden_states, *args, encoder_hidden_states=hs, **kwargs)
                    return out
                
                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)

        out = self.unet(z_all, t, encoder_hidden_states=batched_prompt_embd)

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
        field: Literal["constant_direction"] = "constant_direction",
        binary_feedback_type: Literal["prompt", "image"] = "image_direct",
        seed: int = 42,
        n_images: int = 1,
        guidance_scale: float = 8.0,
        walk_distance: float = 1.0,
        walk_steps: int = 50,
        flatten_channels: bool = True,
        denoising_steps: int = 20,
        show_progress: bool = True,
        only_decode_last: bool = False,
        **kwargs
    ):
        """
        Generate a trajectory of images with binary feedback.
        The feedback can be given as a list of liked and disliked prompts, or as images, which then get inverted.
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(n_images, 4, 64, 64, device=self.device, dtype=self.dtype)

        # out = self.pipeline(prompt, guidance_scale=guidance_scale, num_inference_steps=denoising_steps, latents=z)
        # return [out.images]

        pos_images = [self.image_to_tensor(img) for img in liked]
        neg_images = [self.image_to_tensor(img) for img in disliked]
        pos_images = torch.stack(pos_images).to(self.device, dtype=self.dtype)
        neg_images = torch.stack(neg_images).to(self.device, dtype=self.dtype)
        pos_latents = self.vae.config.scaling_factor * self.vae.encode(pos_images).latent_dist.sample()
        neg_latents = self.vae.config.scaling_factor * self.vae.encode(neg_images).latent_dist.sample()
        
        uncond_prompt_embd, cond_prompt_embd = self.initialize_prompts(prompt, negative_prompt)

        self.scheduler.set_timesteps(denoising_steps, device=self.device)

        iterator = self.scheduler.timesteps
        if show_progress:
            iterator = tqdm(iterator)

        # z = torch.randn(n_images, 4, 64, 64, device=self.device, dtype=self.dtype)
        z = z * self.scheduler.init_noise_sigma

        traj = []
        for i, t in enumerate(iterator):
            z_all = torch.cat([z] * 2, dim=0)
            z_all = self.scheduler.scale_model_input(z_all, t)
            batched_prompt_embd = torch.stack([uncond_prompt_embd, cond_prompt_embd], dim=0)
            
            # z_ref = torch.cat([z_all[:1], pos_latents], dim=0)
            z_ref = torch.cat([pos_latents, pos_latents], dim=0)
            # z_ref = torch.cat([neg_latents, pos_latents], dim=0)
            z_ref = self.scheduler.scale_model_input(z_ref, t)
            noise = torch.randn_like(z_ref)
            z_ref_noised = self.scheduler.add_noise(z_ref, noise, t)

            cached_hidden_states = self.get_unet_hidden_states(z_ref_noised, t, batched_prompt_embd)
            
            unet_out = self.unet_forward_with_cached_hidden_states(
                z_all,
                t,
                batched_prompt_embd,
                cached_hidden_states
            ).sample
            # unet_out = self.unet(z_all, t, encoder_hidden_states=batched_prompt_embd).sample

            noise_uncond, noise_cond = unet_out.chunk(2)

            guidance = noise_cond - noise_uncond
            noise_pred = noise_uncond + guidance_scale * guidance

            z = self.scheduler.step(noise_pred, t, z).prev_sample

            if not only_decode_last or i == len(self.scheduler.timesteps) - 1:
                y = self.pipeline.decode_latents(z)
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
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)
