import math
from typing import Literal, List

import torch
import torch.nn as nn
from PIL import Image
from einops import repeat, rearrange, pack, unpack
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler

from stable_preferences.spaces import img_batch_to_space
from stable_preferences.fields import walk_in_field


class StableDiffuserWithBinaryFeedback(nn.Module):

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

    def chunked_unet_forward(
        self,
        z_all,
        t,
        batched_prompt_embd,
    ):
        """
        Forward pass for the diffusion model, chunked to avoid memory issues.
        """
        n_chunks = math.ceil(z_all.shape[0] / self.unet_max_chunk_size)
        z_all_chunks = torch.chunk(z_all, n_chunks, dim=0)
        batched_prompt_embd_chunks = torch.chunk(batched_prompt_embd, n_chunks, dim=0)
        unet_out_all = []
        for z_all_chunk, batched_prompt_embd_chunk in zip(z_all_chunks, batched_prompt_embd_chunks):
            unet_out = self.unet(z_all_chunk, t, batched_prompt_embd_chunk).sample     
            unet_out = unet_out.to(torch.float32)
            unet_out_all.append(unet_out)
        unet_out_all,_ = pack(unet_out_all, '* a b c')
        return unet_out_all

    def optimize_noise(
            self,
            noise_cond,
            in_space_destinations,
            space_converter,
        ):
        """
        Optimize the noise to get to the destination in the space.
        """

        current_points = noise_cond.clone().detach().requires_grad_(True)

        loss_t = torch.nn.functional.mse_loss(space_converter(current_points), in_space_destinations, reduction='sum')
        loss_t.backward()
        grad_norm = current_points.grad.norm()
        lr = 0.3 # torch.tensor(1/3) * norm_to_walk / max(grad_norm,1.0)

        convergence_threshold = 1e-5
        max_iterations = 100

        iterations_used=0
        print(f"Initial loss from optimization: {torch.nn.functional.mse_loss(space_converter(current_points), in_space_destinations, reduction='sum')}")
        for iteration in range(max_iterations):
            current_points.grad.zero_()

            current_points_in_space = space_converter(current_points)
            loss = torch.nn.functional.mse_loss(current_points_in_space, in_space_destinations, reduction='sum')
            if loss.item() < convergence_threshold:
                print(f"Converged after {iteration} iterations")
                break
            loss.backward()

            # Full gradient descent update
            with torch.no_grad():
                current_points -= lr * current_points.grad

            iterations_used += 1

        print(f"Final loss from optimization: {torch.nn.functional.mse_loss(space_converter(current_points), in_space_destinations, reduction='sum')} in {iterations_used} iterations")
        return current_points.clone().detach().requires_grad_(False)

        
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "a photo of an astronaut riding a horse on mars",
        liked: List[str] = [],
        disliked: List[str] = [],
        field: Literal["constant_direction"] = "constant_direction",
        space: Literal["latent_noise"] = "latent_noise",
        binary_feedback_type: Literal["prompt", "image"] = "prompt",
        seed: int = 42,
        n_images: int = 1,
        walk_distance: float = 8.0,
        walk_steps: int = 1,
        denoising_steps: int = 20,
        show_progress: bool = True,
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
            raise NotImplementedError("Image feedback is not implemented yet")
            # add the images into the kwargs and then use them in the field
            # this is hacky, but i guess it's ok.
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
        norms = []
        z = torch.randn(n_images, 4, 64, 64, device=self.device, dtype=self.dtype)
        z = z * self.scheduler.init_noise_sigma
        for i, t in enumerate(iterator):
            z_single = self.scheduler.scale_model_input(z, t)
            z_all = repeat(z_single, "batch a b c -> (batch prompts) a b c", prompts=2 + len(liked) + len(disliked)) # we generate the next z for all prompts and then combine

            liked_prompts_embd = liked_prompts_embds[i]
            disliked_prompts_embd = disliked_prompts_embds[i]
            cond_prompt_embd = cond_prompt_embds[i]
            uncond_prompt_embd = uncond_prompt_embds[i]
            prompt_embd, ps = pack(
                [a for a in [cond_prompt_embd, uncond_prompt_embd, liked_prompts_embd, disliked_prompts_embd] if 0 not in a.shape], # avoid empty concatenation throwing errors 
                '* a b')
            batched_prompt_embd = repeat(prompt_embd, 'prompts a b -> (batch prompts) a b', batch=n_images)
            
            unet_out = self.chunked_unet_forward(
                z_all,
                t,
                batched_prompt_embd,
            )
            unet_out = rearrange(unet_out, "(batch prompts) a b c -> batch prompts a b c", batch=n_images)
            noise_cond, noise_uncond, noise_liked, noise_disliked = unpack(
                unet_out,
                [(),(),(len(liked),), (len(disliked),)],
                'batch * a b c'
            )

            space_converter = img_batch_to_space(space)
            in_space_destinations = walk_in_field(
                space_converter(noise_uncond),
                space_converter(noise_cond),
                space_converter(noise_liked),
                space_converter(noise_disliked),
                field_type=field,
                walk_distance=walk_distance,
                n_steps=walk_steps,
                **kwargs
            )
            with torch.enable_grad():
                noise_destinations = self.optimize_noise(
                    noise_cond.detach(),
                    in_space_destinations.detach(),
                    space_converter,
                )

            # cfg_vector = noise_cond - noise_uncond
            # noise_destinations = noise_cond + walk_distance*cfg_vector

            print("guidance norm: ",(noise_cond-noise_destinations).view(n_images, -1).norm(dim=1))
            noise_destinations = noise_destinations.to(z.dtype)
            z = self.scheduler.step(noise_destinations, t, z).prev_sample

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
