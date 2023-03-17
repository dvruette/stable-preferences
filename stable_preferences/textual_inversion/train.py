from typing import List

import torch
import torch.nn.functional as F
import tqdm
import hydra
import torchvision.transforms as T
from hydra.utils import to_absolute_path
from PIL import Image
from omegaconf import DictConfig
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel


dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch.set_default_dtype(dtype)


def load_images(input_paths: List[str]):
    # load images to tensor with torchvision
    transform = T.Compose([
        T.Resize(512),
        T.ToTensor(),
    ])

    imgs = []
    for path in input_paths:
        img = Image.open(to_absolute_path(path)).convert("RGB")
        img = 2*transform(img) - 1
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


@hydra.main(config_path="../configs", config_name="textual_inversion", version_base=None)
def main(ctx: DictConfig):
    if ctx.device == "auto":
        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = "cpu"
        device = "cuda" if torch.cuda.is_available() else device
    else:
        device = ctx.device
    print(f"Using device: {device}")

    # tokenizer = CLIPTokenizer.from_pretrained(ctx.model_id, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(ctx.model_id, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(ctx.model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(ctx.model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(ctx.model_id, subfolder="unet").to(device)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    embedding = torch.randn(1, 1, text_encoder.config.hidden_size, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([embedding], lr=ctx.training.lr)

    batch_size = ctx.training.batch_size
    x = load_images([ctx.image]).to(device)
    x = x.expand(batch_size, -1, -1, -1)
    x_enc = vae.encode(x)

    prog_bar = tqdm.trange(ctx.training.steps)

    for i in range(ctx.training.steps):
        z = x_enc.latent_dist.sample().detach()
        z = z * vae.config.scaling_factor

        noise = torch.randn_like(z)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=z.device)
        timesteps = timesteps.long()
        noisy_z = noise_scheduler.add_noise(z, noise, timesteps)
        model_pred = unet(noisy_z, timesteps, embedding.expand(batch_size, -1, -1)).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(z, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prog_bar.set_postfix({"loss": loss.item()})
        prog_bar.update()

        if (i + 1) % ctx.training.save_steps == 0:
            with torch.no_grad():
                torch.save(embedding, f"embedding_{i+1}.pt")


if __name__ == "__main__":
    main()
