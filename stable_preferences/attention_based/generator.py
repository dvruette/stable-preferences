import math
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
        pipe = StableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler, torch_dtype=torch_dtype)

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

    def initialize_prompts(
        self,
        liked_prompts: List[str],
        disliked_prompts: List[str],
        num_steps: int,
    ):
        """
        Initialize prompt feedback for the trajectory generation.
        This function just embeds the prompts and returns the embeddings, per step. 
        This provides the flexibility, to have a framework working with a potentially different prompt for each step, i.e. compatible with null text inversion.
        """

        if len(liked_prompts) == 0 and len(disliked_prompts) == 0:
            return torch.zeros((num_steps, 0, 0, 0)), torch.zeros((num_steps, 0, 0, 0))

        prompt_tokens = self.tokenizer(
            liked_prompts + disliked_prompts,
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
            input_ids=prompt_tokens["input_ids"].to(self.device),
            attention_mask=attention_mask,
        ).last_hidden_state
        liked_prompts_embd = prompt_embd[: len(liked_prompts)]
        disliked_prompts_embd = prompt_embd[len(liked_prompts) :]
        liked_prompts_embds = repeat(liked_prompts_embd, 'prompts a b -> steps prompts a b', steps=num_steps)
        disliked_prompts_embds = repeat(disliked_prompts_embd, 'prompts a b -> steps prompts a b', steps=num_steps)

        return liked_prompts_embds, disliked_prompts_embds
    
    def get_unet_hidden_states(self, z_all, t, batched_prompt_embd):
        cached_hidden_states = []
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                def new_forward(self, hidden_states, **kwargs):
                    cached_hidden_states.append(hidden_states.clone().detach().cpu())
                    return self.old_forward(hidden_states, **kwargs)
                
                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)
        
        # run forward pass to cache hidden states, output can be discarded
        _ = self.unet(z_all, t, batched_prompt_embd)

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
                def new_forward(self, hidden_states, **kwargs):
                    cached_hs = cached_hidden_states.pop(0).to(hidden_states.device)
                    hs = torch.cat([hidden_states, cached_hs], dim=1)
                    out = self.old_forward(hs, **kwargs)
                    return out[:, :hidden_states.shape[1]]
                
                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)

        out = self.unet(z_all, t, batched_prompt_embd)

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
        only_decode_last: bool = True,
        **kwargs
    ):
        """
        Generate a trajectory of images with binary feedback.
        The feedback can be given as a list of liked and disliked prompts, or as images, which then get inverted.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if binary_feedback_type == "prompt":
            liked_prompts_embds, disliked_prompts_embds = self.initialize_prompts(
                liked, disliked, denoising_steps
            )
        elif binary_feedback_type == "image_inversion":
            raise NotImplementedError("Image feedback is not implemented yet")
        elif binary_feedback_type == "image_direct":
            pos_images = [self.image_to_tensor(img) for img in liked]
            neg_images = [self.image_to_tensor(img) for img in disliked]
            pos_images = torch.stack(pos_images).to(self.device)
            neg_images = torch.stack(neg_images).to(self.device)
            pos_latents = self.vae.encode(pos_images).latent_dist.sample()
            neg_latents = self.vae.encode(neg_images).latent_dist.sample()
        else:
            raise ValueError(f"Binary feedback type {binary_feedback_type} is not supported.")
        
        cond_prompt_embds, uncond_prompt_embds = self.initialize_prompts([prompt], [""], denoising_steps) # shape: "steps 1 a b"
        cond_prompt_embds = rearrange(cond_prompt_embds, "steps 1 a b -> steps a b")
        uncond_prompt_embds = rearrange(uncond_prompt_embds, "steps 1 a b -> steps a b")

        self.scheduler.set_timesteps(denoising_steps, device=self.device)

        iterator = self.scheduler.timesteps
        if show_progress:
            iterator = tqdm(iterator)

        traj = []
        z = torch.randn(n_images, 4, 64, 64, device=self.device, dtype=self.dtype)
        z = z * self.scheduler.init_noise_sigma
        for i, t in enumerate(iterator):
            z_single = self.scheduler.scale_model_input(z, t)
            z_all = repeat(z_single, "batch a b c -> (batch prompts) a b c", prompts=2) # we generate the next z for all prompts and then combine
            
            z_ref = torch.cat([pos_latents, neg_latents], dim=0)
            z_ref = self.scheduler.scale_model_input(z_ref, t)
            noise = torch.randn_like(z_ref)
            z_ref_noised = self.scheduler.add_noise(z_ref, noise, t)

            cond_prompt_embd = cond_prompt_embds[i]
            uncond_prompt_embd = uncond_prompt_embds[i]
            prompt_embd, ps = pack([cond_prompt_embd, uncond_prompt_embd], '* a b')
            batched_prompt_embd = repeat(prompt_embd, 'prompts a b -> (batch prompts) a b', batch=n_images)

            cached_hidden_states = self.get_unet_hidden_states(z_ref_noised, t, batched_prompt_embd)
            
            unet_out = self.unet_forward_with_cached_hidden_states(
                z_all,
                t,
                batched_prompt_embd,
                cached_hidden_states
            ).sample

            unet_out = rearrange(unet_out, "(batch prompts) a b c -> batch prompts a b c", batch=n_images)
            noise_liked, noise_disliked = unpack(
                unet_out,
                [(len(liked),), (len(disliked),)],
                'batch * a b c'
            )

            # noise_destinations = walk_in_field(
            #     latent_uncond_batch=noise_uncond,
            #     latent_cond_batch=noise_cond,
            #     field_points_pos=noise_liked,
            #     field_points_neg=noise_disliked,
            #     field_type=field,
            #     guidance_scale=guidance_scale,
            #     walk_distance=walk_distance,
            #     n_steps=walk_steps,
            #     flatten_channels=flatten_channels,
            #     **kwargs,
            # )

            noise_liked, noise_disliked = noise_liked.squeeze(1), noise_disliked.squeeze(1)
            guidance = noise_liked - noise_disliked
            noise_destinations = noise_liked + guidance_scale * guidance

            # print("guidance norm: ",(noise_cond-noise_destinations).view(n_images, -1).norm(dim=1))
            noise_destinations = noise_destinations.to(z.dtype)
            z = self.scheduler.step(noise_destinations, t, z).prev_sample

            if not only_decode_last or i == len(self.scheduler.timesteps) - 1:
                y = self.decode_latents(z)
                piled = self.numpy_to_pil(y)
                traj.append(piled)

        return traj
    
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
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
    
    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images
