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
def generate_trajectory_with_binary_feedback(
    pipe,
    pos_prompt="a photo of an astronaut riding a horse on mars",
    neg_prompt="",
    liked_prompts=[],
    disliked_prompts=[],
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
    if liked_prompts==[]: liked_prompts = [""]
    if disliked_prompts==[]: disliked_prompts = [""]
    prompt_tokens = tokenizer([pos_prompt] + [neg_prompt] + liked_prompts + disliked_prompts, return_tensors="pt", padding=True, truncation=True).to(z.device)
    prompt_embd = text_encoder(**prompt_tokens).last_hidden_state
    pos_prompt_embd = prompt_embd[:1] # first element
    neg_prompt_embd = prompt_embd[1:2] # second element
    liked_prompts_embd = prompt_embd[2:2+len(liked_prompts)]
    disliked_prompts_embd = prompt_embd[2+len(liked_prompts):]
    # print("pos_prompt_embd.shape", pos_prompt_embd.shape)
    # print("liked_prompt_embd.shape", liked_prompts_embd.shape)
    prompt_embd = torch.cat([pos_prompt_embd] * batch_size + [neg_prompt_embd] * batch_size + [p.unsqueeze(0) for p in liked_prompts_embd for _ in range(batch_size)] + [p.unsqueeze(0) for p in disliked_prompts_embd for _ in range(batch_size)])

    iterator = scheduler.timesteps
    if show_progress:
        iterator = tqdm.tqdm(scheduler.timesteps)

    traj = []
    norms = []
    for iteration, t in enumerate(iterator):
        z = scheduler.scale_model_input(z, t)
        zs = torch.cat((2+len(liked_prompts)+len(disliked_prompts))*[z], dim=0)

        assert(zs.shape==(batch_size*(2+len(liked_prompts)+len(disliked_prompts)),4,64,64))
        unet_out = unet(zs, t, prompt_embd) # layout: pos promt in all samples, neg prompt in all samples, first liked prompt in all samples, second liked prompt in all samples, etc.
        noise_cond, noise_uncond, noise_liked, noise_disliked = torch.tensor_split(unet_out.sample,[batch_size,2*batch_size,(2+len(liked_prompts))*batch_size])

        # aggregate the noise from the liked and disliked prompts per sample and obtain one mean vector per sample
        mean_noise_liked = torch.stack(noise_liked.split(batch_size)).mean(axis=0)
        mean_noise_disliked = torch.stack(noise_disliked.split(batch_size)).mean(axis=0)
        # print("mean noise liked: ",mean_noise_liked.shape)
        
        # cfg_vector = (noise_cond-noise_uncond)
        # cfg_noramlizer = 1
        # if mean_noise_liked.shape[0] != 0: # this is the case if there are liked prompts, otherwise the mean vector is empty
        #     cfg_vector += (mean_noise_liked-noise_cond)
        #     cfg_noramlizer += 1
        # if mean_noise_disliked.shape[0] != 0:
        #     cfg_vector += (noise_cond-mean_noise_disliked)   
        #     cfg_noramlizer += 1
        # noise_pred = noise_uncond + cfg_scale * cfg_vector/cfg_noramlizer
        alpha = 0.6
        if t<=5000: # i think stepps range from 1000 (first) to 0 (last)
            # I observed that the norm of the preference vector becomes very large, so I normalize it to the norm of the cfg vector
            # making the norm equal for both results in very high norms, because the noise_cond is "not happy", therefore it helps allowing the 
            # preference vector, if happy not to scale up to make the cond unhappy. 
            preference_vector = (mean_noise_liked-mean_noise_disliked)
            preference_vector = preference_vector / preference_vector.norm() * min((noise_cond-noise_uncond).norm(),preference_vector.norm())
            # scale the cond vector up such that overall we have the same norm than if we would only used the standard cfg 
            noise_pred = noise_cond + (alpha*preference_vector + (1-alpha)*(noise_cond-noise_uncond))*cfg_scale
            norms.append((preference_vector.norm().item(), (noise_cond-noise_uncond).norm().item()))
        else:
            # standard cfg
            noise_pred = noise_cond + (noise_cond-noise_uncond)*cfg_scale

        z = scheduler.step(noise_pred, t, z).prev_sample
        if not only_last or t == scheduler.timesteps[-1]:
            y = pipe.decode_latents(z)
            traj.append(pipe.numpy_to_pil(y))
    print("norms", norms)
    return traj
    

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
