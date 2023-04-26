import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import random
import hydra
import torchvision.transforms as T
from torch.utils.data import Dataset
import PIL
import os
from PIL import Image
from omegaconf import DictConfig
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from accelerate import Accelerator

from stable_preferences.human_preference_dataset.utils import HumanPreferenceDatasetReader
from stable_preferences.utils import get_free_gpu


class TextualInversionDataset(Dataset):
    """Class to create a dataset for textual inversion.
    
    """
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
            T.RandomRotation(self.rotate_degrees, resample=self.interpolation),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % len(self.images)])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


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
    
    accelerator = Accelerator(
        gradient_accumulation_steps=ctx.training.gradient_accumulation_steps,
        mixed_precision=dtype,
        log_with=ctx.report_to
    )
    
    if ctx.training.seed is not None:
        set_seed(ctx.training.seed)
    
    tokenizer = CLIPTokenizer.from_pretrained(ctx.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(ctx.model_id, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(ctx.model_id, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(ctx.model_id, subfolder="unet", torch_dtype=weight_dtype)
    
    noise_scheduler = DDPMScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    
    # Load stable diffusion model
    
    if (torch.cuda.is_available() and is_xformers_available()):
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        print("Using xformers for memory efficient attention")
    else:
        print("Not using xformers for memory efficient attention. Make sure to have downloaded the xformers library and have a GPU available.")
    
    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)


    reader = HumanPreferenceDatasetReader(ctx.data.human_preference_dataset_path)
    example = reader.get_example_by_id(ctx.data.training_example_id)
    image = example["images"][example["human_preference"]]
    prompt = example["prompt"]

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
    
    
    prompt_tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=ctx.max_length, return_tensors="pt")
    prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
    prompt_embd = text_encoder.get_text_features(**prompt_tokens)

    num_timesteps = noise_scheduler.config.num_train_timesteps
    if ctx.training.embedding_type == "naive":
        embedding = prompt_embd.expand(num_timesteps, -1, -1).detach().clone().requires_grad_(True)
    elif ctx.training.embedding_type == "cumulative":
        embedding = torch.zeros(num_timesteps, *prompt_embd.shape[1:], device=device, requires_grad=True)

    # Initilize the optimizer
    optimizer = torch.optim.AdamW(
        embedding,
        lr=ctx.training.lr,
        betas=(ctx.training.adam_beta1, ctx.training.adam_beta2),
        weight_decay=ctx.training.adam_weight_decay,
        eps=ctx.training.adam_epsilon,
    )
    
    
    embedding, optimizer, train_dataloader = accelerator.prepare(
        embedding, optimizer, train_dataloader
    )
    
    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    print(accelerator.device)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.eval()
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    prog_bar = tqdm.trange(ctx.training.steps)
    prog_bar.set_description("Steps")
        
    for i in range(ctx.training.steps):
        for step, batch in enumerate(train_dataloader):
            latents = vae.encode(batch["pixel_values"].to(device, dtype = weight_dtype)).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor
            
            # sample the noise to add to latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            
            # Add noise to the latents according to noise magnitude at each timestep (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings for conditioning
            if ctx.training.embedding_type == "naive":
                embd = embedding
            elif ctx.training.embedding_type == "cumulative":
                embd = prompt_embd + embedding.cumsum(dim=0)
            else:
                raise ValueError(f"Unknown embedding type {ctx.training.embedding_type}")
            encoder_hidden_states = embd.gather(0, timesteps.view(bsz, 1, 1).expand(-1, *embedding.shape[1:]))
            
            # Predict noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()
        
        #accelerator.wait_for_everyone()
        prog_bar.set_postfix({"loss": loss.item()})
        prog_bar.update()
            
        if (i + 1) % ctx.training.save_steps == 0:
            if ctx.training.embedding_type == "naive":
                learned_embeds = embedding
            elif ctx.training.embedding_type == "cumulative":
                learned_embeds = prompt_embd + embedding.cumsum(dim=0)
            save_path = os.path.join(ctx.output_dir, f"learned_embeds_{ctx.placeholder_token}_{i+1}.pt")
            torch.save(learned_embeds.detach().cpu(), save_path)
    

    if ctx.training.embedding_type == "naive":
        learned_embeds = embedding
    elif ctx.training.embedding_type == "cumulative":
        learned_embeds = prompt_embd + embedding.cumsum(dim=0)
    save_path = os.path.join(ctx.output_dir, f"learned_embeds_{ctx.placeholder_token}_final.pt")
    torch.save(learned_embeds.detach().cpu(), save_path)
    accelerator.end_training()


if __name__ == "__main__":
    main()
