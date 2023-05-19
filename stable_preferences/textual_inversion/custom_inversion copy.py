import os
import functools

import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import random
import hydra
import torchvision.transforms as T
import PIL
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset
from PIL import Image
from omegaconf import DictConfig
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.utils import is_wandb_available
# from accelerate.utils import set_seed
# from accelerate import Accelerator

from stable_preferences.human_preference_dataset.utils import HumanPreferenceDatasetReader
from stable_preferences.utils import get_free_gpu

def torch_batched_ema(x: torch.FloatTensor, beta=0.99):
    """Computes the exponential moving average of a tensor along the second dimension."""
    assert x.dim() <= 2
    out = []
    for i in range(x.shape[0]):
        if i == 0:
            out.append((1 - beta) * x[i])
        else:
            out.append(beta * out[-1] + (1 - beta) * x[i])
    return torch.stack(out, dim=0)


@torch.no_grad()
def generate_samples_from_latents(
    latents: torch.FloatTensor,
    prompt_embeds: torch.FloatTensor,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    num_inference_steps: int = 20,
    guidance_scale: float = 8.0,
):  
    def decode_latents(latents):
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    # Prepare unconditional embedding
    uncond_tokens = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(latents.device)
    if hasattr(text_encoder, "use_attention_mask") and text_encoder.use_attention_mask:
        attention_mask = uncond_tokens["attention_mask"]
    else:
        attention_mask = None
    uncond_embeds = text_encoder(input_ids=uncond_tokens["input_ids"], attention_mask=attention_mask).last_hidden_state
    uncond_embeds = uncond_embeds.expand(latents.shape[0], -1, -1)
    
    # Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=latents.device)
    timesteps = scheduler.timesteps

    # Denoising loop
    for _, t in enumerate(tqdm.tqdm(timesteps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        cond_embeds = prompt_embeds[None, t].expand(latents.shape[0], -1, -1)
        embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=embeds,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Post-processing
    image = decode_latents(latents)
    image = StableDiffusionPipeline.numpy_to_pil(image)
    return image


class TextualInversionDataset(Dataset):
    """Class to create a dataset for textual inversion."""
    def __init__(
        self,
        images,
        size=512,
        repeats=100,
        flip_p=0.05,
        rotate_degrees=5,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.rotate_degrees = rotate_degrees

        self.images = images
        self.num_images = len(images)
        self._length = self.num_images * repeats

        self.interpolation = PIL.Image.BICUBIC # TODO: implement other kind of interpolations if we want to (e.g. bicubic, lanczos, nearest neighbor)
        
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=self.flip_p),
            T.RandomRotation(self.rotate_degrees, interpolation=self.interpolation),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = self.images[i % len(self.images)]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        return {
            "pixel_values": torch.from_numpy(image).permute(2, 0, 1),
        }


@hydra.main(config_path="../configs", config_name="custom_inversion", version_base=None)
def main(ctx: DictConfig):
    if ctx.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    if ctx.device == "auto":
        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = "cpu"
        device = get_free_gpu() if torch.cuda.is_available() else device
    else:
        device = ctx.device
    print(f"Using device: {device}")
        
    dtype = "fp16" if torch.cuda.is_available() else "no"
    weight_dtype = torch.float16 if dtype == "fp16" else torch.float32
    # weight_dtype = torch.float32
    
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=ctx.training.gradient_accumulation_steps,
    #     mixed_precision=dtype,
    #     log_with=ctx.report_to
    # )
    # device = accelerator.device
    print(f"Using device: {device}")


    # if ctx.training.seed is not None:
    #     set_seed(ctx.training.seed)
    
    # Load stable diffusion model
    noise_scheduler = DDPMScheduler.from_pretrained(ctx.model_id, subfolder="scheduler", torch_dtype=weight_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(ctx.model_id, scheduler=noise_scheduler, torch_dtype=weight_dtype)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    # slice_size = unet.config.attention_head_dim // 2
    # unet.set_attention_slice(slice_size)
    
    # try:
    #     import xformers
    #     unet.enable_xformers_memory_efficient_attention()
    #     print("Using xformers for memory efficient attention")
    # except:
    #     print("Not using xformers for memory efficient attention. Make sure to have downloaded the xformers library and have a GPU available.")
    
    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    dataset_path = to_absolute_path(ctx.data.human_preference_dataset_path)
    reader = HumanPreferenceDatasetReader(dataset_path)
    example = reader.get_example_by_id(ctx.data.training_example_id)
    image = example["images"][example["human_preference"]]
    prompt = example["prompt"]

    image.save("target_image.png")


    # Dataset and DataLoaders creation
    train_dataset = TextualInversionDataset(
        images=[image],
        size=ctx.data.resolution,
        repeats=ctx.data.repeats,
        center_crop=ctx.data.center_crop,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=ctx.training.batch_size, shuffle=True, num_workers=ctx.training.dataloader_num_workers
    )
    
    text_encoder.to(device)
    prompt_tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=ctx.training.max_length, return_tensors="pt").to(device)
    # IMPORTANT: stable diffusion 1.5 was trained without masking the padding tokens
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = prompt_tokens["attention_mask"]
    else:
        attention_mask = None
    prompt_embeds = text_encoder(input_ids=prompt_tokens["input_ids"], attention_mask=attention_mask).last_hidden_state

    num_timesteps = noise_scheduler.config.num_train_timesteps
    if ctx.training.embedding_type in ["naive", "cumulative", "cumulative_decay"]:
        prompt_embeds = prompt_embeds.expand(num_timesteps, -1, -1)
        embedding = torch.zeros_like(prompt_embeds)
    elif ctx.training.embedding_type == "constant":
        embedding = prompt_embeds
    embedding = torch.nn.Parameter(embedding, requires_grad=True)

    # Initilize the optimizer
    optimizer = torch.optim.AdamW(
        [embedding],
        lr=ctx.training.lr,
        betas=(ctx.training.adam_beta1, ctx.training.adam_beta2),
        weight_decay=ctx.training.adam_weight_decay,
        eps=ctx.training.adam_epsilon,
    )
    
    
    # embedding, optimizer, train_dataloader = accelerator.prepare(
    #     embedding, optimizer, train_dataloader
    # )
    
    # Move vae and unet to device
    prompt_embeds = prompt_embeds.to(device, dtype=weight_dtype)
    embedding = embedding.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    
    # print("Accelerator device: ", accelerator.device)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.eval()

    def update_embedding(embedding: torch.FloatTensor):
        if ctx.training.embedding_type == "naive":
            return prompt_embeds + embedding
        elif ctx.training.embedding_type == "cumulative":
            return prompt_embeds + embedding.cumsum(dim=0)
        elif ctx.training.embedding_type == "cumulative_decay":
            emb = embedding.view(num_timesteps, -1)
            cum_emb = torch_batched_ema(emb.t(), beta=ctx.training.embedding_decay).t()
            return prompt_embeds + cum_emb.reshape(embedding.shape)
        elif ctx.training.embedding_type == "constant":
            return embedding.expand(num_timesteps, -1, -1)
        else:
            raise ValueError(f"Unknown embedding type {ctx.training.embedding_type}")

    learned_embeds = update_embedding(embedding)

    generate_fn = functools.partial(
        generate_samples_from_latents,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=noise_scheduler,
        num_inference_steps=20,
        guidance_scale=8,
    )

    latent_res = unet.config.sample_size
    eval_latents = noise_scheduler.init_noise_sigma * torch.randn(ctx.training.eval_batch_size, unet.in_channels, latent_res, latent_res, dtype=weight_dtype)
    eval_imgs = generate_fn(eval_latents.to(device), learned_embeds.to(device))
    text_encoder.cpu()
    
    print(f"Saving eval images to {os.path.join(os.getcwd(), ctx.output_dir, 'eval_images')}")
    out_folder = os.path.join(ctx.output_dir, "eval_images")
    os.makedirs(out_folder, exist_ok=True)
    for i, img in enumerate(eval_imgs):
        img.save(os.path.join(out_folder, f"img_0_{i}.png"))

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    prog_bar = tqdm.trange(ctx.training.steps)
    prog_bar.set_description("Steps")

    dl = iter(train_dataloader)

    for i in range(ctx.training.steps):
        for j in range(ctx.training.gradient_accumulation_steps):
            try:
                batch = next(dl)
            except StopIteration:
                dl = iter(train_dataloader)
                batch = next(dl)
            
            # with accelerator.accumulate(embedding):
            if True:
                latents = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor
                
                # sample the noise to add to latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                # Add noise to the latents according to noise magnitude at each timestep (forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings for conditioning
                encoder_hidden_states = learned_embeds.gather(0, timesteps.view(bsz, 1, 1).expand(-1, *learned_embeds.shape[1:]))
                
                # Predict noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # accelerator.backward(loss)
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

        # update learned embeddings based on updated parameters
        learned_embeds = update_embedding(embedding)
        
        #accelerator.wait_for_everyone()
        prog_bar.set_postfix({"loss": loss.item()})
        prog_bar.update()

        if (i + 1) % ctx.training.eval_steps == 0:
            with torch.no_grad():
                text_encoder.to(device)
                eval_imgs = generate_fn(eval_latents.to(device), learned_embeds.to(device))
                text_encoder.cpu()
                out_folder = os.path.join(ctx.output_dir, "eval_images")
                os.makedirs(out_folder, exist_ok=True)
                for j, img in enumerate(eval_imgs):
                    img.save(os.path.join(out_folder, f"img_{i}_{j}.png"))
            
        if (i + 1) % ctx.training.save_steps == 0:
            os.makedirs(os.path.join(ctx.output_dir, "embeddings"), exist_ok=True)
            save_path = os.path.join(ctx.output_dir, "embeddings", f"step_{i+1}.pt")
            torch.save(learned_embeds.detach().cpu(), save_path)
    
    os.makedirs(os.path.join(ctx.output_dir, "embeddings"), exist_ok=True)
    save_path = os.path.join(ctx.output_dir, "embeddings", f"final.pt")
    torch.save(learned_embeds.detach().cpu(), save_path)
    # accelerator.end_training()


if __name__ == "__main__":
    main()
