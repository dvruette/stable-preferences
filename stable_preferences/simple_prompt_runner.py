import hydra
from omegaconf import DictConfig
import os
from datetime import date
from generation_trajectory import StableDiffuserWithBinaryFeedback


@hydra.main(config_path="configs", config_name="simple_prompt_runner", version_base=None)
def main(ctx: DictConfig):

    generator = StableDiffuserWithBinaryFeedback(
        ctx.model_version,
        ctx.unet_max_chunk_size,
        ctx.n_images,
        ctx.walk_distance,
        ctx.walk_steps,
        ctx.denoising_steps,
    )

    trajectory = generator.generate(
        ctx.prompt,
        ctx.liked_prompts,
        ctx.disliked_prompts,
        field=ctx.field,
        space=ctx.space,
        binary_feedback_type='prompt',
        seed=ctx.seed,
        **ctx.additional_args,
    )

    imgs = trajectory[:][-1]
    
    date_str = date.today().strftime("%Y-%m-%d")
    out_folder = os.path.join("outputs", "images", date_str)
    os.makedirs(out_folder, exist_ok=True)
    
    for img in imgs:
        n_files = len([name for name in os.listdir(out_folder)])
        out_path = os.path.join(out_folder, f"example_{n_files}.png")
        img.save(out_path)
        print(f"Saved image to {out_path}")


if __name__ == "__main__":
    main()
