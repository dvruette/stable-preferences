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
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator

if is_wandb_available():
    import wandb
    
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.15.0.dev0")

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

class TextualInversionDataset(Dataset):
    """Class to create a dataset for textual inversion.
    
    """
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = PIL.Image.BICUBIC # TODO: implement other kind of interpolations if we want to (e.g. bicubic, lanczos, nearest neighbor)
        
        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = T.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

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

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
    

def print_gpu_memory(indices=[0]):
    total_memory = 0
    total_allocated_memory = 0
    total_free_memory = 0
    for idx in indices:
        props = torch.cuda.get_device_properties(idx)
        total_memory += props.total_memory
        reserved_memory = torch.cuda.memory_reserved(idx)
        allocated_memory = torch.cuda.memory_allocated(idx)
        free_memory = reserved_memory - allocated_memory
        total_allocated_memory += allocated_memory
        total_free_memory += free_memory
        print(f"GPU {idx}:")
        print(f"\tTotal memory: {props.total_memory/1024**2:.2f} MB")
        print(f"\tReserved memory: {reserved_memory/1024**2:.2f} MB")
        print(f"\tAllocated memory: {allocated_memory/1024**2:.2f} MB")
        print(f"\tFree memory: {free_memory/1024**2:.2f} MB")
    print(f"Total GPU memory: {total_memory/1024**2:.2f} MB")
    print(f"Total allocated GPU memory: {total_allocated_memory/1024**2:.2f} MB")
    print(f"Total free GPU memory: {total_free_memory/1024**2:.2f} MB")


@hydra.main(config_path="../configs", config_name="textual_inversion", version_base=None)
def main(ctx: DictConfig):
    if ctx.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    if ctx.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = "cpu"
        device = "cuda" if torch.cuda.is_available() else device
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
    
    if ctx.seed is not None:
        set_seed(ctx.seed)
    
    tokenizer = CLIPTokenizer.from_pretrained(ctx.model_id, subfolder="tokenizer")
    num_added_tokens = tokenizer.add_tokens(ctx.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {ctx.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
        
    # Get the token ids of both the init token and the placeholder token
    initializer_token_id = tokenizer.encode(ctx.initializer_token, add_special_tokens=False)[0]
    print(f"Initializer token id: {initializer_token_id}")
    place_holder_token_id = tokenizer.convert_tokens_to_ids(ctx.placeholder_token)
    print(f"Placeholder token id: {place_holder_token_id}")
    
    # Load stable diffusion model
    text_encoder = CLIPTextModel.from_pretrained(ctx.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(ctx.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(ctx.model_id, subfolder="unet")
    
    if (torch.cuda.is_available() and is_xformers_available()):
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        print("Using xformers for memory efficient attention")
    else:
        print("Not using xformers for memory efficient attention. Make sure to have downloaded the xformers library and have a GPU available.")
    
    # Resize token embeddings since we have added a token
    text_encoder.resize_token_embeddings(len(tokenizer))
    print(f"Text encoder has {len(tokenizer)} tokens")
    
    # Initialize the newly added placeholder token with the embeddings of the initializer token which will be optimized later
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[place_holder_token_id] = token_embeds[initializer_token_id] 
    
    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all the parameters except for the token embeddings in the text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    #text_encoder.requires_grad_(False) Dimitri's work 

    # Dataset and DataLoaders creation
    train_dataset = TextualInversionDataset(
        data_root=ctx.train_data_dir,
        tokenizer=tokenizer,
        size=ctx.resolution,
        placeholder_token=ctx.placeholder_token,
        repeats=ctx.repeats,
        learnable_property=ctx.learnable_property,
        center_crop=ctx.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=ctx.training.batch_size, shuffle=True, num_workers=ctx.training.dataloader_num_workers
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    
    # embedding = torch.randn(1, 1, text_encoder.config.hidden_size, device=device, requires_grad=True)

    # Initilize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=ctx.training.lr,
        betas=(ctx.training.adam_beta1, ctx.training.adam_beta2),
        weight_decay=ctx.training.adam_weight_decay,
        eps=ctx.training.adam_epsilon,
    )
    
    
    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
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
        text_encoder.train()
        #os.system("nvcc --version")
        
        for step, batch in enumerate(train_dataloader):
            
            #print_gpu_memory()
            latents = vae.encode(batch["pixel_values"].to(device, dtype = weight_dtype)).latent_dist.sample().detach()

            latents = latents * vae.config.scaling_factor
            
            # sample the noise to add to latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            timesteps = timesteps.long()
            
            # Add noise to the latents according to noise magnitude at each timestep (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings for conditioning
           
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0].to(dtype=weight_dtype)
            
            #print(encoder_hidden_states)

            
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
            
            # Zero out the gradients for all the token embedding except the newly added 
            # embedding for the concept we are learning
            grads = text_encoder.get_input_embeddings().weight.grad
            index_grads_to_zero = torch.arange(len(tokenizer)) != place_holder_token_id
            grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        #accelerator.wait_for_everyone()
        prog_bar.set_postfix({"loss": loss.item()})
        prog_bar.update()
            
        if (i + 1) % ctx.training.save_steps == 0:
            learned_embeds = text_encoder.get_input_embeddings().weight[place_holder_token_id]
            learned_embeds_dict = {ctx.placeholder_token: learned_embeds.detach().cpu()}
            save_path = os.path.join(ctx.output_dir, f"learned_embeds_{ctx.placeholder_token}_{i+1}.pt")
            torch.save(learned_embeds_dict, save_path)
    
    learned_embeds = text_encoder.get_input_embeddings().weight[place_holder_token_id]
    learned_embeds_dict = {ctx.placeholder_token: learned_embeds.detach().cpu()}
    save_path = os.path.join(ctx.output_dir, f"learned_embeds_{ctx.placeholder_token}_final.pt")
    torch.save(learned_embeds_dict, save_path)
    accelerator.end_training()


if __name__ == "__main__":
    main()
