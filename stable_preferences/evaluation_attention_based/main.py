import math
import os
import re
from datetime import date

import hydra
import torch
from PIL import Image
from omegaconf import DictConfig
from os.path import join, exists
from glob import glob
from stable_preferences.attention_based.generator import StableDiffuserWithAttentionFeedback
from stable_preferences.utils import get_free_gpu
from stable_preferences.human_preference_dataset.prompts import sample_prompts
import pdb
from stable_preferences.evaluation.eval import *
import numpy as np
import json

def tile_images(images):
    size = images[0].size
    assert all(img.size == size for img in images), "All images must have the same size"

    grid_size_x = math.ceil(len(images) ** 0.5)
    grid_size_y = math.ceil(len(images) / grid_size_x)

    # Create a new blank image with the size of the tiled grid
    tiled_image = Image.new('RGB', (grid_size_x * size[0], grid_size_y * size[1]))

    # Paste the four images into the tiled image
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            idx = x + grid_size_x * y
            if idx >= len(images):
                break
            img = images[idx]
            tiled_image.paste(img, (x * size[0], y * size[1]))
    return tiled_image


@hydra.main(config_path="../configs", config_name="evaluation_attention_based", version_base=None)
def main(ctx: DictConfig):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    #device ="cuda:2"
    print(f"Using device: {device}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using dtype: {dtype}")

    generator = StableDiffuserWithAttentionFeedback(
        model_ckpt=ctx.model_ckpt if hasattr(ctx, "model_ckpt") else None,
        model_name=ctx.model_name if hasattr(ctx, "model_name") else None,
        stable_diffusion_version=ctx.model_version,
        unet_max_chunk_size=ctx.unet_max_chunk_size,
        torch_dtype=dtype,
    ).to(device)
    
    
        
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join("outputs", "rounds", date_str)
    experiment_paths = sorted(glob(join(out_folder, 'experiment_*')))
    n_experiment = len(experiment_paths)
        
    out_folder = os.path.join(out_folder, "experiment_" + str(n_experiment))
    os.makedirs(out_folder, exist_ok=True)
 
    prompt=sample_prompts(max_num_prompts=128, seed=42)[n_experiment] if ctx.sample_prompt else ctx.prompt
    
    liked=list(ctx.liked_images) if ctx.liked_images else []
    disliked=list(ctx.disliked_images) if ctx.disliked_images else []
    
    score_dic = dict(prompt = prompt, rounds = [])
    
    for round in range(ctx.n_rounds):
        trajectory = generator.generate(
            prompt=prompt,
            negative_prompt=ctx.negative_prompt,
            liked=liked,
            disliked=disliked,
            seed=ctx.seed,
            n_images=ctx.n_images,
            denoising_steps=ctx.denoising_steps,
        )
        scores = np.zeros(ctx.n_images)
        out_paths = []
        imgs = trajectory[:][-1]
        for i, img in enumerate(imgs):
            # each image is of the form example_ID.png. Extract the max id
            out_path = os.path.join(out_folder, f"round_{round}_image_{i}.png")
            out_paths.append(out_path)
            img.save(out_path)
            print(f"Saved image to {out_path}")
            scores[i] = clip_score(out_path, prompt)
            score_dic['rounds'].append([dict(image_path=out_path,score=scores[i])])
        if len(imgs) > 1:
            tiled = tile_images(imgs)
            tiled_path = os.path.join(out_folder, f"tiled_round_{round}.png")
            tiled.save(tiled_path)
            print(f"Saved tile to {tiled_path}")
        #pdb.set_trace()
        liked.append(out_paths[np.argmax(scores)])
        disliked = disliked + np.delete(out_paths, np.argmax(scores)).tolist()
        
    with open(os.path.join(out_folder, "scores.json"), 'w', encoding ='utf8') as json_file:
        json.dump(score_dic, json_file, ensure_ascii = False)


if __name__ == "__main__":
    main()
