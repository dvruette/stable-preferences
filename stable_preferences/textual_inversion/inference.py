import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import hydra
from omegaconf import DictConfig


    
@hydra.main(config_path="../configs", config_name="textual_inversion_inference", version_base=None)
def main(ctx: DictConfig):
    if ctx.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = "cpu"
        device = "cuda" if torch.cuda.is_available() else device
    else:
        device = ctx.device
    print(f"Using device: {device}")
    
    stored_embeddings = torch.load(ctx.embeddings_path)
    trained_token = list(stored_embeddings.keys())[0]
    embeds = stored_embeddings[trained_token] 
    
    # Load text encoder and tokenizer 
    tokenizer = CLIPTokenizer.from_pretrained(ctx.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(ctx.model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    
    # Get the type of the embeddings in the tokenizer and convert the learnt one to that type
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)
    
    token = trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

    # resize the token embeddings as we have add one token
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    
    pipe = StableDiffusionPipeline.from_pretrained(
        ctx.model_id,
        torch_dtype=torch.float16,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    ).to(device)
    
    prompt = ctx.prompt

    num_samples = ctx.num_generations
    for i in range(num_samples):
        images = pipe(prompt, num_images_per_prompt=ctx.num_samples_per_generation, num_inference_steps=50, guidance_scale=7.5).images
        for j in range(len(images)):
            images[j].save(f"{ctx.output_dir}/{prompt}_{i+1}_{j+1}.png") # i=generation, j=sample
    
    
if __name__ == "__main__":
    main()