import os
import torch
import argparse
from lora_model import LoRA_HP_SD

parser = argparse.ArgumentParser("generate images from a unet lora checkpoint")
parser.add_argument("--unet_weight", default="", type=str)
parser.add_argument("--prompt", default="a man on mars", type=str)
parser.add_argument("--output_folder", default="./outputs", type=str)
# parser.add_argument("--negative_prompt", default="Weird image. ", type=str)
# parser.add_argument("--world_size", default=-1, type=int)
# parser.add_argument("--rank", default=-1, type=int)

args = parser.parse_args()

model = LoRA_HP_SD(unet_weight=args.unet_weight, results_folder=args.output_folder)
print(f"Initialized LoRA model, generating image...")
# Generate the image

image = model.generate_images_from_prompts(args.prompt)
print(f"Generated image, saving to {args.output_folder}")
print(f"Finished.")
