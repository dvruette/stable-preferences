import functools
from typing import Literal

import torch
import torch.nn.functional as F
from einops import repeat, rearrange, pack, unpack
from torch.nn.functional import normalize

def walk_in_field(
    latent_uncond_batch: torch.Tensor,
    latent_cond_batch: torch.Tensor,
    field_points_pos: torch.Tensor,
    field_points_neg: torch.Tensor,
    field_type: Literal["constant_direction", "kernel"] = "throw error",
    walk_distance: float = "throw error",
    guidance_scale: float = "throw error",
    walk_type: Literal["pre_guidance", "joint", "post_guidance"] = "throw error",
    n_steps: int = "throw error",
    flatten_channels = "error",
    preference_portion: float = "throw error",
    **kwargs,  # any further arguments that are needed for the field type
):
    """
    Walk in a field of a specific type having binary feedback vectors. It takes n_steps which sum up to the walk_distance.
    E.g. when walking in a field with a constant direction, the n_steps don't matter.
    Gives back a new latent_batch with the points where the walk ended.

    The calling function then can optimize the noise space with gradient descent to reach those points.

    format of field_points_pos (written in einops format): [batch liked_points a b c]
    """
    assert latent_uncond_batch.shape == latent_cond_batch.shape, "latent_uncond_batch and latent_cond_batch must have the same shape"
    ### flatten all dimensions except the batch dimension
    original_shape = latent_cond_batch.shape
    batch_size, num_channels = latent_cond_batch.shape[:2]

    if flatten_channels:
        latent_uncond_batch = latent_uncond_batch.reshape(batch_size, -1)
        latent_cond_batch = latent_cond_batch.reshape(batch_size, -1) 
        field_points_pos = rearrange(field_points_pos, "batch liked_points a b c -> batch liked_points (a b c)")
        field_points_neg = rearrange(field_points_neg, "batch disliked_points a b c -> batch disliked_points (a b c)")
    else:
        latent_uncond_batch = latent_uncond_batch.reshape(batch_size, num_channels, -1)
        latent_cond_batch = latent_cond_batch.reshape(batch_size, num_channels, -1)
        field_points_pos = rearrange(field_points_pos, "batch liked_points a b c -> batch liked_points a (b c)")
        field_points_neg = rearrange(field_points_neg, "batch disliked_points a b c -> batch disliked_points a (b c)")


    if field_type == "constant_direction":
        step_fn = functools.partial(constant_direction_step, **kwargs)
    elif field_type == "kernel":
        step_fn = functools.partial(kernel_step, field_type=field_type, **kwargs)
    elif field_type == "smoothed_inverse_polynomial":
        step_fn = functools.partial(normalized_field_step, field_type=field_type, **kwargs)
    else:
        raise NotImplementedError(f"Field type {field_type} not implemented.")
    
    single_step_fn = functools.partial(
        get_single_step,
        step_fn=step_fn,
        latent_uncond_batch=latent_uncond_batch.clone().detach(),
        latent_cond_batch=latent_cond_batch.clone().detach(),
        field_points_pos=field_points_pos,
        field_points_neg=field_points_neg,
        num_channels=num_channels,
        flatten_channels=flatten_channels,
    )
    
    if len(field_points_pos) == 0 or len(field_points_neg) == 0:
        print("Warning: No positive points or no negative points. Using conditional directions only. Ignoring all liked and disliked images.")
        output = latent_cond_batch + guidance_scale * (latent_cond_batch - latent_uncond_batch)
        return output.reshape(original_shape)

    if walk_type == "joint":
        step_size = guidance_scale / n_steps
        walking_points = latent_cond_batch.clone().detach()
        for _ in range(n_steps):
            preference_direction = single_step_fn(walking_points)
            
            conditional_direction = latent_cond_batch - latent_uncond_batch
            walk_direction = preference_portion * preference_direction + (1 - preference_portion) * conditional_direction
            walking_points =  walking_points + step_size * walk_direction
        output = walking_points
    elif walk_type == "pre_guidance":
        step_size = walk_distance / n_steps
        cond_scale = latent_cond_batch.norm(dim=-1, keepdim=True)
        uncond_scale = latent_uncond_batch.norm(dim=-1, keepdim=True)

        for _ in range(n_steps):
            preference_cond = single_step_fn(latent_cond_batch)
            preference_uncond = single_step_fn(latent_uncond_batch)

            latent_cond_batch += step_size * preference_cond
            latent_uncond_batch += step_size * preference_uncond

        # latent_cond_batch = latent_cond_batch / latent_cond_batch.norm(dim=-1, keepdim=True) * cond_scale
        # latent_uncond_batch = latent_uncond_batch / latent_uncond_batch.norm(dim=-1, keepdim=True) * uncond_scale
        
        guidance = latent_cond_batch - latent_uncond_batch
        output = latent_cond_batch + guidance_scale * guidance
    elif walk_type == "post_guidance":
        step_size = guidance_scale * walk_distance / n_steps
        guidance = latent_cond_batch - latent_uncond_batch
        walk_points = latent_cond_batch + guidance_scale * guidance
        for _ in range(n_steps):
            preference_direction = single_step_fn(walk_points)
            walk_points = walk_points + step_size * preference_direction
        output = walk_points

    return output.reshape(original_shape)


def get_single_step(
    xs: torch.Tensor,
    step_fn,
    latent_uncond_batch: torch.Tensor,
    latent_cond_batch: torch.Tensor,
    field_points_pos: torch.Tensor,
    field_points_neg: torch.Tensor,
    num_channels: int,
    flatten_channels: bool = True,
):
    if flatten_channels:
        return step_fn(
            walking_points=xs,
            latent_uncond_points=latent_uncond_batch,
            latent_cond_points=latent_cond_batch,
            field_points_pos=field_points_pos,
            field_points_neg=field_points_neg,
        )
    else:
        preference_per_channel = []
        for channel_idx in range(num_channels):
            preference_per_channel.append(
                step_fn(
                    walking_points=xs[:, channel_idx],
                    latent_uncond_points=latent_uncond_batch[:, channel_idx],
                    latent_cond_points=latent_cond_batch[:, channel_idx],
                    field_points_pos=field_points_pos[:, :, channel_idx],
                    field_points_neg=field_points_neg[:, :, channel_idx],
                )
            )
        return torch.stack(preference_per_channel, dim=1)


def kernel_step(
    walking_points: torch.Tensor,
    latent_uncond_points: torch.Tensor,
    latent_cond_points: torch.Tensor,
    field_points_pos: torch.Tensor,
    field_points_neg: torch.Tensor,
    kernel_function: Literal["rbf", "inv_square"] = "inv_square",
    **kwargs,
):
    if kernel_function == "rbf":
        def log_kernel_fn(x, y):
            return -torch.norm(x - y, dim=-1) ** 2
    elif kernel_function == "inv_square":
        def log_kernel_fn(x, y):
            return -2 * torch.log(torch.norm(x - y, dim=-1))

    pos_logscore = log_kernel_fn(walking_points.unsqueeze(1), field_points_pos)
    neg_logscore = log_kernel_fn(walking_points.unsqueeze(1), field_points_neg)
    pos_logsumexp = torch.logsumexp(pos_logscore, dim=1, keepdim=True)
    neg_logsumexp = torch.logsumexp(neg_logscore, dim=1, keepdim=True)
    weights_pos = torch.exp(pos_logscore - pos_logsumexp)
    weights_neg = torch.exp(neg_logscore - neg_logsumexp)

    weighted_field_points_pos = (field_points_pos * weights_pos.unsqueeze(-1)).sum(dim=1)
    weighted_field_points_neg = (field_points_neg * weights_neg.unsqueeze(-1)).sum(dim=1)
    preference_directions = weighted_field_points_pos - weighted_field_points_neg
    return preference_directions


def constant_direction_step(
    walking_points: torch.Tensor,
    latent_uncond_points: torch.Tensor,
    latent_cond_points: torch.Tensor,
    field_points_pos: torch.Tensor,
    field_points_neg: torch.Tensor,
    clip_preference_vec: bool = False,
    **kwargs,
):
    """
    Take one step in a constant direction field.

    Format of field_points_pos (written in einops format): [batch liked_points (a b c)]
    """
    conditional_directions = latent_cond_points - latent_uncond_points
    preference_directions = field_points_pos.mean(axis=1) - field_points_neg.mean(axis=1)
    if clip_preference_vec:
        preference_directions = F.normalize(preference_directions, dim=1)
        preference_directions *= torch.minimum(conditional_directions.norm(dim=1), preference_directions.norm(dim=1)).view(-1,1)

    return preference_directions




### COPIED FROM DEBUGGED MAIN

def normalized_field_step(
        walking_points,
        latent_uncond_points,
        latent_cond_points,
        field_points_pos,
        field_points_neg,
        **kwargs
    ):
    """
    Take one step in a polynomial field of the form SUM_i |x-p_i|^coefficient
    """
    field_type = kwargs['field_type']
    smoothing_strength = kwargs['smoothing_strength']
    poly_coefficient = kwargs['poly_coefficient']
    if field_type == "polynomial":
        pot_grad = polynomial_distance_potential(poly_coefficient)
    elif field_type == "smoothed_inverse_polynomial":
        pot_grad = smoothed_inverse_potential(smoothing_strength, poly_coefficient)
    else:
        raise ValueError(f"Unknown field type {field_type}")
    field_points_pos = rearrange(field_points_pos, 'batch liked_points a -> liked_points batch a')
    field_points_neg = rearrange(field_points_neg, 'batch disliked_points a -> disliked_points batch a')
    # preference_portion = kwargs['preference_portion']
    conditional_directions = latent_cond_points - latent_uncond_points
    if 0 not in field_points_pos.shape and 0 not in field_points_neg.shape:
        # pot_grad = polynomial_distance_potential(coefficient)
        summed_pot_grad = torch.zeros_like(conditional_directions)
        for p_i in field_points_pos:
            summed_pot_grad += pot_grad(walking_points, p_i)
        for p_i in field_points_neg:
            summed_pot_grad -= pot_grad(walking_points, p_i)
        walk_direction_preference = normalize(-summed_pot_grad)
        # walk_direction = preference_portion * walk_direction_preference + (1-preference_portion) * conditional_directions
    else:
        print('Warning: No conditional points or no unconditional points. Using conditional directions only. Ignoring all liked and disliked images.')
        walk_direction_preference = conditional_directions
    return walk_direction_preference




def polynomial_distance_potential(coefficient):
    # defines the potential field of SUM_i (x-p_i)^coefficient
    # the field potential is to be minimized
    def potential_grad(x, p_i):
        assert len(p_i.shape) == 2, f'p_i must be a batched vector, but has shape {p_i.shape}'
        # p_i = repeat(p_i, 'd -> b d', b=x.shape[0])
        d = torch.norm(x-p_i, dim=1)
        inv_walk_direction = coefficient * (x-p_i) * d.reshape(-1,1)**(coefficient)
        return inv_walk_direction
    return potential_grad
        
def smoothed_inverse_potential(smoothing_radius, distance_strength):
    def potential_grad(x,p_i):
        assert len(p_i.shape) == 2, f'p_i must be a batched vector, but has shape {p_i.shape}'
        d = torch.norm(x-p_i, dim=1)
        # print(d)
        inv_walk_direction = (x-p_i) * (1/(d**distance_strength+smoothing_radius)).reshape(-1,1)
        return inv_walk_direction
    return potential_grad