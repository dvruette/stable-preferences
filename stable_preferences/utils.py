import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm


def display_images(images, n_cols=4, size=4):
    n_rows = int(np.ceil(len(images) / n_cols))
    fig = plt.figure(figsize=(size * n_cols, size * n_rows))
    for i, img in enumerate(images):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    return fig


@torch.no_grad()
def generate_trajectory(
    pipe,
    pos_prompt="a photo of an astronaut riding a horse on mars",
    neg_prompt="",
    cfg_scale=5,
    steps=20,
    seed=42,
    show_progress=True,
    batch_size=1,
    only_last=False,
    device=None,
):  
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    # vae = pipe.vae
    tokenizer = pipe.tokenizer

    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(batch_size, 4, 64, 64, device=device)
    z = z * scheduler.init_noise_sigma

    scheduler.set_timesteps(steps, device=device)
    prompt_tokens = tokenizer([pos_prompt] + [neg_prompt], return_tensors="pt", padding=True, truncation=True).to(z.device)
    prompt_embd = text_encoder(**prompt_tokens).last_hidden_state
    pos_prompt_embd = prompt_embd[:1]
    neg_prompt_embd = prompt_embd[-1:]
    prompt_embd = torch.cat([pos_prompt_embd] * batch_size + [neg_prompt_embd] * batch_size)

    iterator = scheduler.timesteps
    if show_progress:
        iterator = tqdm.tqdm(scheduler.timesteps)

    traj = []
    for _, t in enumerate(iterator):
        z = scheduler.scale_model_input(z, t)
        unet_out = unet(torch.cat([z, z], dim=0), t, prompt_embd)
        noise_cond, noise_uncond = unet_out.sample.chunk(2)
        
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        z = scheduler.step(noise_pred, t, z).prev_sample
        if not only_last or t == scheduler.timesteps[-1]:
            y = pipe.decode_latents(z)
            traj.append(pipe.numpy_to_pil(y))
    return traj
