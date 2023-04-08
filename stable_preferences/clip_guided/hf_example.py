import os

from diffusers import DPMSolverSinglestepScheduler, StableDiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel
import torch

from stable_preferences.utils import get_free_gpu
from stable_preferences.clip_guided.clip_guided_stable_diffusion import CLIPGuidedStableDiffusion


def main():
    # feature_extractor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    # clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16)
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", torch_dtype=torch.float16)

    model_id = "runwayml/stable-diffusion-v1-5"
    # model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_id, subfolder="scheduler")
    guided_pipeline = CLIPGuidedStableDiffusion(
        vae=pipe.vae,
        unet=pipe.unet,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        scheduler=scheduler,
        clip_model=clip_model,
        feature_extractor=feature_extractor,
    )
    guided_pipeline.enable_attention_slicing()
    guided_pipeline.freeze_unet()
    guided_pipeline.freeze_vae()
    device = get_free_gpu()
    clip_model.to(device)
    guided_pipeline = guided_pipeline.to(device)

    # prompt = "fantasy book cover, full moon, fantasy forest landscape, golden vector elements, fantasy magic, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece"
    # prompt = "a beautiful sunset"
    prompt = "picture of a mountain lake"

    torch.manual_seed(42)

    out_folder = "./clip_guided_sd"
    os.makedirs(out_folder, exist_ok=True)

    for i in range(1):
        images = guided_pipeline(
            prompt,
            num_inference_steps=20,
            guidance_scale=0.0,
            clip_guidance_scale=100,
            num_cutouts=4,
            use_cutouts=False,
        )

        n_files = len([name for name in os.listdir(out_folder)])
        traj_dir = os.path.join(out_folder, f"traj_{n_files}")
        os.makedirs(traj_dir, exist_ok=True)
        for i, imgs in enumerate(images):
            for j, image in enumerate(imgs):
                image.save(os.path.join(traj_dir, f"image_{i}_{j}.png"))


if __name__ == "__main__":
    main()
