import hydra
import torch
from omegaconf import DictConfig
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel

from stable_preferences.data import load_images
from stable_preferences.few_shot.model import IA3Config, IA3Model


@hydra.main(config_path="../configs", config_name="few_shot", version_base=None)
def main(ctx: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_scheduler = DDPMScheduler.from_pretrained(ctx.model.pretrained_name, subfolder="scheduler").to(device)
    vae = AutoencoderKL.from_pretrained(ctx.model.pretrained_name, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(ctx.model.pretrained_name, subfolder="unet").to(device)

    config = IA3Config(target_modules=ctx.model.target_modules)

