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
from stable_preferences.attention_based.generator import (
    StableDiffuserWithAttentionFeedback,
    apply_unet_lora_weights,
)
from stable_preferences.utils import get_free_gpu
from stable_preferences.human_preference_dataset.prompts import sample_prompts

from stable_preferences.evaluation.automatic_eval.image_similarity import (
    ImageSimilarity,
)
from stable_preferences.evaluation.automatic_eval.hps import HumanPreferenceScore


def tile_images(images):
    size = images[0].size
    assert all(img.size == size for img in images), "All images must have the same size"

    grid_size_x = math.ceil(len(images) ** 0.5)
    grid_size_y = math.ceil(len(images) / grid_size_x)

    # Create a new blank image with the size of the tiled grid
    tiled_image = Image.new("RGB", (grid_size_x * size[0], grid_size_y * size[1]))

    # Paste the four images into the tiled image
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            idx = x + grid_size_x * y
            if idx >= len(images):
                break
            img = images[idx]
            tiled_image.paste(img, (x * size[0], y * size[1]))
    return tiled_image


@hydra.main(
    config_path="../configs",
    config_name="evaluation_attention_based",
    version_base=None,
)
def main(ctx: DictConfig):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    device = get_free_gpu() if torch.cuda.is_available() else device
    # device ="cuda:2"
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
    # generator.pipeline.enable_xformers_memory_efficient_attention()

    # After the pipeline was generated inside the generator, apply LoRA if specified
    if ctx.lora_weights:
        print(f"Applying LoRA weights from {ctx.lora_weights}")
        apply_unet_lora_weights(
            pipeline=generator.pipeline, unet_path=ctx.lora_weights, device=device
        )

    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join("outputs", "rounds", date_str)
    experiment_paths = sorted(glob.glob(os.path.join(out_folder, "experiment_*")))
    n_experiment = len(experiment_paths)

    out_folder = os.path.join(out_folder, "experiment_" + str(n_experiment))
    os.makedirs(out_folder, exist_ok=True)

    if ctx.sample_prompt:
        prompts = sample_prompts(max_num_prompts=ctx.num_prompts, seed=0)
    else:
        prompts = [ctx.prompt]

    # scoring_model = ClipScore(device=device)
    hps_model = HumanPreferenceScore(
        weight_path="stable_preferences/evaluation/resources/hpc.pt", device=device
    )
    img_similarity_model = ImageSimilarity(device=device)

    init_liked = list(ctx.liked_images) if ctx.liked_images else []
    init_disliked = list(ctx.disliked_images) if ctx.disliked_images else []
    init_liked = [Image.open(img_path) for img_path in init_liked]
    init_disliked = [Image.open(img_path) for img_path in init_disliked]

    metrics = []
    with torch.inference_mode():
        for prompt_idx, prompt in enumerate(tqdm.tqdm(prompts, smoothing=0.01)):
            print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")

            liked = init_liked.copy()
            disliked = init_disliked.copy()

            if ctx.seed is None:
                seed = torch.randint(0, 2**32, (1,)).item()
            else:
                seed = ctx.seed

            for i in range(ctx.n_rounds):
                trajectory = generator.generate(
                    prompt=prompt,
                    negative_prompt=ctx.negative_prompt,
                    liked=liked,
                    disliked=disliked,
                    seed=seed,
                    n_images=ctx.n_images,
                    denoising_steps=ctx.denoising_steps,
                )
                imgs = trajectory[-1]

                hp_scores = hps_model.compute(prompt, imgs)

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

                out_paths = []
                for j, (img, hps, pos_sim, neg_sim) in enumerate(
                    zip(imgs, hp_scores, pos_sims, neg_sims)
                ):
                    # each image is of the form example_ID.xpng. Extract the max id
                    out_path = os.path.join(
                        out_folder, f"prompt_{prompt_idx}_round_{i}_image_{j}.png"
                    )
                    out_paths.append(out_path)
                    img.save(out_path)
                    print(f"Saved image to {out_path}")

                    metrics.append(
                        {
                            "round": i,
                            "prompt": prompt,
                            "image_idx": j,
                            "image": out_path,
                            "hps": hps,
                            "pos_sim": pos_sim,
                            "neg_sim": neg_sim,
                            "seed": ctx.seed,
                            "liked": liked,
                            "disliked": disliked,
                        }
                    )
                if len(imgs) > 1:
                    tiled = tile_images(imgs)
                    tiled_path = os.path.join(
                        out_folder, f"prompt_{prompt_idx}_tiled_round_{i}.png"
                    )
                    tiled.save(tiled_path)
                    print(f"Saved tile to {tiled_path}")

                liked.append(imgs[np.argmax(hp_scores)])
                # disliked = disliked + np.delete(out_paths, np.argmax(scores)).tolist()
                disliked.append(imgs[np.argmin(hp_scores)])
                print(f"HPS scores: {hp_scores}")
                print(f"Pos. similarities: {pos_sims}")
                print(f"Neg. similarities: {neg_sims}")

            pd.DataFrame(metrics).to_csv(
                os.path.join(out_folder, "metrics.csv"), index=False
            )
            print(f"Saved metrics to {out_folder}/metrics.csv")


if __name__ == "__main__":
    main()
