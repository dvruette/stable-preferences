from typing import List, Optional, Union

import torch
import tqdm
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision
from transformers import CLIPImageProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    # DiffusionPipeline,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    DPMSolverSinglestepScheduler,
)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class CLIPGuidedStableDiffusion(nn.Module):
    """CLIP guided stable diffusion based on the amazing repo by @crowsonkb and @Jack000
    - https://github.com/Jack000/glid-3-xl
    - https://github.dev/crowsonkb/k-diffusion
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        clip_model: CLIPModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler],
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor

        self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        self.cut_out_size = (
            feature_extractor.size
            if isinstance(feature_extractor.size, int)
            else feature_extractor.size["shortest_edge"]
        )
        self.make_cutouts = MakeCutouts(self.cut_out_size)

        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.clip_model, False)

    @property
    def device(self):
        return next(self.parameters()).device

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        self.enable_attention_slicing(None)

    def freeze_vae(self):
        set_requires_grad(self.vae, False)

    def unfreeze_vae(self):
        set_requires_grad(self.vae, True)

    def freeze_unet(self):
        set_requires_grad(self.unet, False)

    def unfreeze_unet(self):
        set_requires_grad(self.unet, True)

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

    @torch.enable_grad()
    def cond_fn(
        self,
        latents,
        timestep,
        index,
        text_embeddings,
        noise_pred_original,
        text_embeddings_clip,
        clip_guidance_scale,
        num_cutouts,
        use_cutouts=True,
    ):  
        latents = latents.detach().requires_grad_()

        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        print("z_cond norm", latents.norm().item())

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        print("noise_pred norm", noise_pred.norm().item())

        if isinstance(self.scheduler, (DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            print("lats_x0 norm", pred_original_sample.norm().item())

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)

            print(alpha_prod_t.item(), clip_guidance_scale)
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        print("sample norm", sample.norm().item())

        sample = 1 / self.vae.config.scaling_factor * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        print("img_norm", image.norm().item())

        if use_cutouts:
            img = self.make_cutouts(image, num_cutouts)
        else:
            img = transforms.Resize(self.cut_out_size)(image)
        img = self.normalize(img).to(latents.dtype)
        print("img_norm", img.norm().item())

        image_embeddings_clip = self.clip_model.get_image_features(img)
        image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

        if use_cutouts:
            dists = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip)
            dists = dists.view([num_cutouts, sample.shape[0], -1])
            loss = dists.sum(2).mean(0).sum() * clip_guidance_scale
        else:
            loss = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip).mean() * clip_guidance_scale


        grads = -torch.autograd.grad(loss, latents)[0]

        noise_pred = noise_pred - torch.sqrt(beta_prod_t) * grads
        print(f"t={timestep.item()} | z.norm={latents.norm().item():.2f} | loss={loss.item():.2f} | grads.mean={grads.mean().item():.2f} | grads.std={grads.std().item():.2f} | grads.norm={grads.norm().item():.2f} | noise_pred.norm={noise_pred.norm().item():.2f}")
        return noise_pred, latents, image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        clip_guidance_scale: Optional[float] = 100,
        clip_prompt: Optional[Union[str, List[str]]] = None,
        num_cutouts: Optional[int] = 4,
        use_cutouts: Optional[bool] = True,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        batch_size: Optional[int] = 1,
    ):
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        # duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        if clip_guidance_scale > 0:
            if clip_prompt is not None:
                clip_text_input = self.tokenizer(
                    clip_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(self.device)
            else:
                clip_text_input = text_input.input_ids.to(self.device)
            text_embeddings_clip = self.clip_model.get_text_features(clip_text_input)
            text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
            # duplicate text embeddings clip for each generation per prompt
            text_embeddings_clip = text_embeddings_clip.repeat_interleave(num_images_per_prompt, dim=0)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype

        torch.manual_seed(42)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype)

        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma


        images = []
        for i, t in enumerate(tqdm.tqdm(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # # predict the noise residual
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # # perform classifier free guidance
            # if do_classifier_free_guidance:
            #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # perform clip guidance
            if clip_guidance_scale > 0:
                text_embeddings_for_guidance = (
                    text_embeddings.chunk(2)[1] if do_classifier_free_guidance else text_embeddings
                )
                noise_pred, latents, image = self.cond_fn(
                    latents,
                    t,
                    i,
                    text_embeddings_for_guidance,
                    None,
                    text_embeddings_clip,
                    clip_guidance_scale,
                    num_cutouts,
                    use_cutouts,
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred.detach(), t, latents.detach()).prev_sample

            image = image.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                image = self.numpy_to_pil(image)
            images.append(image)

        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        images.append(image)
        return images