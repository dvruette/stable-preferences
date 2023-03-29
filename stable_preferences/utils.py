from typing import Literal, List

import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from stable_preferences.models.unet_utils import unet_encode, unet_decode


def display_images(images, n_cols=4, size=4):
    n_rows = int(np.ceil(len(images) / n_cols))
    fig = plt.figure(figsize=(size * n_cols, size * n_rows))
    for i, img in enumerate(images):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    return fig


def aggregate_latents(
    x: torch.Tensor,
    pos_z: List[torch.Tensor],
    neg_z: List[torch.Tensor],
    aggregation: Literal["mean", "lda", "flow", "softmax"] = "mean",
    softmax_temp: float = 0.1,
):
    # input: list of length n_samples, elements with shape [batch_size, 4, 64, 64]
    # sanity checks:
    assert len(pos_z) > 0
    assert len(neg_z) > 0
    assert pos_z[0].shape == neg_z[0].shape
    batch_size = pos_z[0].shape[0]

    if aggregation == "mean":
        mean_pos = torch.stack(pos_z).mean(dim=0)
        mean_neg = torch.stack(neg_z).mean(dim=0)
        return mean_pos - mean_neg
    elif aggregation == "lda":
        discriminants = []
        for i in range(batch_size):
            pos = torch.stack([x[i] for x in pos_z]).detach().cpu().numpy()
            neg = torch.stack([x[i] for x in neg_z]).detach().cpu().numpy()
            coefs = []
            for channel in range(pos.shape[1]):
                y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
                X = np.concatenate([pos[:, channel], neg[:, channel]]).reshape(len(pos) + len(neg), -1)
                lda = LinearDiscriminantAnalysis(n_components=1)
                lda.fit(X, y)
                coefs.append(torch.from_numpy(lda.coef_.reshape(1, 64, 64)))
            discriminants.append(torch.cat(coefs, dim=0))
        direction = torch.stack(discriminants).to(pos_z[0].device, dtype=pos_z[0].dtype, non_blocking=True)
        return direction / 64
    elif aggregation == "flow":
        pos = torch.stack(pos_z, dim=1)
        neg = torch.stack(neg_z, dim=1)

        pos_dists = (pos - x).view(pos.size(0), pos.size(1), -1)
        neg_dists = (neg - x).view(neg.size(0), neg.size(1), -1)
        pos_dists = pos_dists.norm(dim=-1)
        neg_dists = neg_dists.norm(dim=-1)

        # pos_score = 1 / (pos_dists**2 + 1)
        # neg_score = 1 / (neg_dists**2 + 1)
        pos_score = torch.exp(-pos_dists**2)
        neg_score = torch.exp(-neg_dists**2)
        pos_score = pos_score / pos_score.sum(dim=-1, keepdim=True)
        neg_score = neg_score / neg_score.sum(dim=-1, keepdim=True)

        pos_avg = (pos * pos_score[:, :, None, None, None]).sum(dim=1)
        neg_avg = (neg * neg_score[:, :, None, None, None]).sum(dim=1)
        diff = pos_avg - neg_avg
        # diff = diff / diff.norm(dim=-1, keepdim=True)
        return diff
    elif aggregation == "softmax":
        pos = torch.stack(pos_z, dim=1)
        neg = torch.stack(neg_z, dim=1)

        pos_sims = (pos * x).view(pos.size(0), pos.size(1), -1)
        neg_sims = (neg * x).view(neg.size(0), neg.size(1), -1)
        pos_sims = pos_sims.sum(dim=-1) / softmax_temp / torch.sqrt(torch.tensor(4 * 64 * 64, device=pos.device, dtype=pos.dtype))
        neg_sims = neg_sims.sum(dim=-1) / softmax_temp / torch.sqrt(torch.tensor(4 * 64 * 64, device=neg.device, dtype=neg.dtype))

        pos_score = torch.softmax(pos_sims, dim=-1)
        neg_score = torch.softmax(neg_sims, dim=-1)

        pos_avg = (pos * pos_score[:, :, None, None, None]).sum(dim=1)
        neg_avg = (neg * neg_score[:, :, None, None, None]).sum(dim=1)
        diff = pos_avg - neg_avg
        # diff = diff / diff.norm(dim=-1, keepdim=True)
        return diff
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


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
    alpha=0.7,
    beta=1.0,
    aggregation: Literal["mean", "lda", "flow", "softmax"] = "flow",
    latent_space: Literal["noise", "unet"] = "noise",
    show_progress=True,
    batch_size=1,
    only_decode_last=False,
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
        assert zs.shape == (batch_size * (2+len(liked_prompts)+len(disliked_prompts)), 4, 64, 64)

        if latent_space == "noise":
            unet_out = unet(zs, t, prompt_embd).sample # layout: pos promt in all samples, neg prompt in all samples, first liked prompt in all samples, second liked prompt in all samples, etc.
            noise_cond, noise_uncond, noise_liked, noise_disliked = torch.tensor_split(unet_out, [batch_size, 2*batch_size, (2+len(liked_prompts))*batch_size])
            noise_cond = noise_cond.to(torch.float32)
            noise_uncond = noise_uncond.to(torch.float32)
            noise_liked = noise_liked.to(torch.float32)
            noise_disliked = noise_disliked.to(torch.float32)

            if aggregation in ["flow", "softmax"]:
                flow_steps = 100
                flow_scale = 1.0

                cond_scale = noise_cond.view(batch_size, -1).norm().view(batch_size, 1, 1, 1)
                uncond_scale = noise_uncond.view(batch_size, -1).norm().view(batch_size, 1, 1, 1)

                for _ in range(flow_steps):
                    pref_cond = aggregate_latents(
                        noise_cond,
                        noise_liked.split(batch_size),
                        noise_disliked.split(batch_size),
                        aggregation=aggregation,
                    )
                    pref_uncond = aggregate_latents(
                        noise_uncond,
                        noise_liked.split(batch_size),
                        noise_disliked.split(batch_size),
                        aggregation=aggregation,
                    )
                    noise_cond += flow_scale/flow_steps * pref_cond
                    noise_uncond -= flow_scale/flow_steps * pref_uncond

                noise_cond *= cond_scale / noise_cond.view(batch_size, -1).norm().view(batch_size, 1, 1, 1)
                noise_uncond *= uncond_scale / noise_uncond.view(batch_size, -1).norm().view(batch_size, 1, 1, 1)

                guidance = noise_cond - noise_uncond
                noise_pred = noise_cond + cfg_scale*guidance

            else:
                preference = aggregate_latents(
                    noise_cond,
                    noise_liked.split(batch_size),
                    noise_disliked.split(batch_size),
                    aggregation=aggregation,
                )

                if t <= 5000: # i think stepps range from 1000 (first) to 0 (last)
                    # I observed that the norm of the preference vector becomes very large, so I normalize it to the norm of the cfg vector
                    # making the norm equal for both results in very high norms, because the noise_cond is "not happy", therefore it helps allowing the 
                    # preference vector, if happy not to scale up to make the cond unhappy.
                    norms.append((preference.norm().item(), (noise_cond - noise_uncond).norm().item()))
                    cfg_vector = noise_cond - noise_uncond
                    pref_norm = preference.view(batch_size, -1).norm().view(batch_size, 1, 1, 1)
                    cfg_norm = cfg_vector.view(batch_size, -1).norm().view(batch_size, 1, 1, 1)

                    # scale the cond vector up such that overall we have the same norm than if we would only used the standard cfg
                    preference = preference * (min(pref_norm, cfg_norm) / pref_norm)
                    guidance = alpha*preference + (1 - alpha)*cfg_vector

                    # guidance_mag = alpha*pref_norm + (1 - alpha)*cfg_norm
                    # guidance_norm = guidance.view(batch_size, -1).norm().view(batch_size, 1, 1, 1)
                    # guidance = guidance * (guidance_mag / guidance_norm)

                    # cfg_vec = noise_cond - noise_uncond
                    # guidance = cfg_vec + beta*preference

                    noise_pred = noise_cond + cfg_scale*guidance
                else:
                    # standard cfg
                    noise_pred = noise_cond + (noise_cond - noise_uncond)*cfg_scale
        elif latent_space == "unet":
            sample, emb, resids, fwd_upsample = unet_encode(unet, zs, t, prompt_embd)
            latent_cond, latent_uncond, latent_pos, latent_neg = torch.tensor_split(sample, [batch_size, 2*batch_size, (2+len(liked_prompts))*batch_size])
            preference = aggregate_latents(
                latent_pos.split(batch_size),
                latent_neg.split(batch_size),
                aggregation=aggregation,
            )
            norms.append((preference.norm().item(), latent_cond.norm().item()))
            pref_latent_cond = 10*preference + latent_cond
            # pref_latent_cond = torch.stack(latent_pos.split(batch_size)).mean(dim=0)
            pref_latent_cond = pref_latent_cond / pref_latent_cond.norm() * latent_cond.norm()
            agg_sample = torch.cat([pref_latent_cond, latent_uncond], dim=0)

            # truncate residuals and emb
            resids = tuple(r[:agg_sample.size(0)] for r in resids)
            emb = emb[:agg_sample.size(0)]

            unet_out = unet_decode(unet, agg_sample, emb, prompt_embd, resids, fwd_upsample).sample
            noise_cond, noise_uncond = unet_out[:batch_size], unet_out[batch_size:2*batch_size]
            noise_pred = noise_uncond + (cfg_scale + 1)*(noise_cond - noise_uncond)
        else:
            raise ValueError(f"Unknown latent space: {latent_space}")

        noise_pred = noise_pred.to(z.dtype)
        z = scheduler.step(noise_pred, t, z).prev_sample
        if not only_decode_last or t == scheduler.timesteps[-1]:
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
    only_decode_last=False,
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
        if not only_decode_last or t == scheduler.timesteps[-1]:
            y = pipe.decode_latents(z)
            traj.append(pipe.numpy_to_pil(y))
    return traj
