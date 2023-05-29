import math
import os
import glob
from datetime import date

import hydra
import torch
import pandas as pd
import tqdm
import numpy as np
from PIL import Image
from omegaconf import DictConfig
from stable_preferences.attention_based.generator import StableDiffuserWithAttentionFeedback
from stable_preferences.utils import get_free_gpu
from stable_preferences.human_preference_dataset.prompts import sample_prompts

from stable_preferences.evaluation.automatic_eval.image_similarity import ImageSimilarity
from stable_preferences.evaluation.automatic_eval.image_diversity import ImageDiversity
from stable_preferences.evaluation.automatic_eval.hps import HumanPreferenceScore

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


def word_dropout(x, p=0.5):
    words = x.split()
    words = [word for word in words if np.random.rand() > p]
    return " ".join(words)

def get_prompts(path_to_dir):
    num_prompts = len(glob.glob(os.path.join(path_to_dir, "*.txt")))
    prompts = []
    for i in range(num_prompts):
        with open(os.path.join(path_to_dir, f"{i+1}.txt"), "r") as f:
            prompt = f.read()
        prompts.append(prompt)
    return prompts

def get_images(path_to_dir):
    num_images = len(glob.glob(os.path.join(path_to_dir, "*.jpg")))
    images = []
    for i in range(num_images):
        image = Image.open(os.path.join(path_to_dir, f"{i+1}.jpg"))
        images.append(image)
    return images

@hydra.main(config_path="../configs", config_name="evaluation_attention_based_visual", version_base=None)
def main(ctx: DictConfig):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    device ="cuda:0"
    print(f"Using device: {device}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using dtype: {dtype}")

    generator = StableDiffuserWithAttentionFeedback(
        model_ckpt=ctx.model_ckpt if hasattr(ctx, "model_ckpt") else None,
        model_name=ctx.model_name if hasattr(ctx, "model_name") else None,
        stable_diffusion_version=ctx.model_version,
        torch_dtype=dtype,
    ).to(device)
    # generator.pipeline.enable_xformers_memory_efficient_attention()
    
    date_str = date.today().strftime("%Y-%m-%d")
    
    out_folder = os.path.join(ctx.output_path, "outputs", "rounds", date_str) if hasattr(ctx, "output_path") else os.path.join("outputs", "rounds", date_str)
    experiment_paths = sorted(glob.glob(os.path.join(out_folder, 'experiment_*')))
    n_experiment = len(experiment_paths)
        
    out_folder = os.path.join(out_folder, "experiment_" + str(n_experiment))
    os.makedirs(out_folder, exist_ok=True)
    
    prompts = get_prompts(ctx.images_prompts_dir)
    images = get_images(ctx.images_prompts_dir)

    print(f"Found {len(prompts)} prompts and {len(images)} images")
    
    # scoring_model = ClipScore(device=device)
    hps_model = HumanPreferenceScore(weight_path="stable_preferences/evaluation/resources/hpc.pt", device=device)
    img_similarity_model = ImageSimilarity(device=device)
    img_diversity_model = ImageDiversity(device=device)
    
    init_liked_paths = list(ctx.liked_images) if ctx.liked_images else []
    init_disliked_paths = list(ctx.disliked_images) if ctx.disliked_images else []
    init_liked = [Image.open(img_path) for img_path in init_liked_paths]
    init_disliked = [Image.open(img_path) for img_path in init_disliked_paths]

    metrics = []
    with torch.inference_mode():
        for idx, prompt in enumerate(tqdm.tqdm(prompts, smoothing=0.01)):
            
            reference_image = images[idx]
            
            print(f"Prompt {idx + 1}/{len(prompts)}: {prompt}")

            liked_paths = init_liked_paths.copy()
            disliked_paths = init_disliked_paths.copy()
            liked = init_liked.copy()
            disliked = init_disliked.copy()

            if ctx.seed is None:
                seed = torch.randint(0, 2 ** 32, (1,)).item()
            else:
                seed = ctx.seed

            for i in range(ctx.n_rounds):
                ps = [prompt] * ctx.n_images
                if ctx.prompt_dropout > 0:
                    ps = [word_dropout(p, ctx.prompt_dropout) for p in ps]
                trajectory = generator.generate(
                    prompt=ps,
                    negative_prompt=ctx.negative_prompt,
                    liked=liked,
                    disliked=disliked,
                    seed=seed,
                    n_images=ctx.n_images,
                    denoising_steps=ctx.denoising_steps,
                    guidance_scale=ctx.guidance_scale,
                    feedback_start=ctx.feedback.start,
                    feedback_end=ctx.feedback.end,
                    min_weight=ctx.feedback.min_weight,
                    max_weight=ctx.feedback.max_weight,
                    neg_scale=ctx.feedback.neg_scale,
                )
                imgs = trajectory[-1]
                
                img_sim_scores = img_similarity_model.compute([reference_image], imgs)[0]

                if len(liked) > 0:
                    pos_sims = img_similarity_model.compute(imgs, liked)
                    pos_sims = np.mean(pos_sims, axis=1)
                else:
                    pos_sims = [None] * len(imgs)
                
                if len(disliked) > 0:
                    neg_sims = img_similarity_model.compute(imgs, disliked)
                    neg_sims = np.mean(neg_sims, axis=1)
                else:
                    neg_sims = [None] * len(imgs)
                    
                hp_scores = hps_model.compute(prompt, imgs)                
                
                round_diversity = img_diversity_model.compute(imgs)

                out_paths = []
                for j, (p, img, hps, target_img_sim, pos_sim, neg_sim) in enumerate(zip(ps, imgs, hp_scores, img_sim_scores, pos_sims, neg_sims)):
                    # each image is of the form example_ID.xpng. Extract the max id
                    out_path = os.path.join(out_folder, f"prompt_{idx}_round_{i}_image_{j}.png")
                    out_paths.append(out_path)
                    img.save(out_path)
                    print(f"Saved image to {out_path}")

                    metrics.append({
                        "round": i,
                        "prompt": p,
                        "prompt_idx": idx,
                        "image_idx": j,
                        "image": out_path,
                        "hps": hps,
                        "target_img_sim": target_img_sim,
                        "round_diversity": round_diversity,
                        "pos_sim": pos_sim,
                        "neg_sim": neg_sim,
                        "seed": seed,
                        "liked": liked_paths,
                        "disliked": disliked_paths,
                    })
                if len(imgs) > 1:
                    tiled = tile_images(imgs)
                    tiled_path = os.path.join(out_folder, f"prompt_{idx}_tiled_round_{i}.png")
                    tiled.save(tiled_path)
                    print(f"Saved tile to {tiled_path}")

                liked_idx = np.argmax(img_sim_scores)
                disliked_idx = np.argmin(img_sim_scores)
                liked.append(imgs[liked_idx])
                # disliked = disliked + np.delete(out_paths, np.argmax(scores)).tolist()
                disliked.append(imgs[disliked_idx])
                liked_paths.append(out_paths[liked_idx])
                disliked_paths.append(out_paths[disliked_idx])
                print(f"Similarities to target image: {img_sim_scores}")
                print(f"HPS scores: {hp_scores}")
                print(f"In-batch similarity: {round_diversity}")
                print(f"Pos. similarities: {pos_sims}")
                print(f"Neg. similarities: {neg_sims}")

            pd.DataFrame(metrics).to_csv(os.path.join(out_folder, "metrics.csv"), index=False)
            print(f"Saved metrics to {out_folder}/metrics.csv")


if __name__ == "__main__":
    main()
