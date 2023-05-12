import functools
from typing import Literal

import torch
import torch.nn.functional as F
from einops import repeat, rearrange, pack, unpack


def walk_in_field(
    latent_uncond_batch: torch.Tensor,
    latent_cond_batch: torch.Tensor,
    field_points_pos: torch.Tensor,
    field_points_neg: torch.Tensor,
    field_type: Literal["constant_direction", "kernel"] = "constant_direction",
    walk_distance: float = 1.0,
    n_steps: int = 1,
    flatten_channels: bool = True,
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

    step_size = walk_distance / n_steps
    if field_type == "constant_direction":
        step_fn = functools.partial(constant_direction_step, step_size=step_size, **kwargs)
    elif field_type == "kernel":
        step_fn = functools.partial(kernel_step, step_size=step_size, **kwargs)
    else:
        raise NotImplementedError(f"Field type {field_type} not implemented.")
    
    if len(field_points_pos) == 0 or len(field_points_neg) == 0:
        print("Warning: No positive points or no negative points. Using conditional directions only. Ignoring all liked and disliked images.")
        output = latent_cond_batch + walk_distance * (latent_cond_batch - latent_uncond_batch)
        return output.reshape(original_shape)

    walking_points = latent_cond_batch.clone().detach()
    for _ in range(n_steps):
        if flatten_channels:
            walking_points = step_fn(
                walking_points=walking_points,
                latent_uncond_points=latent_uncond_batch,
                latent_cond_points=latent_cond_batch,
                field_points_pos=field_points_pos,
                field_points_neg=field_points_neg,
            )
        else:
            walking_points_per_channel = []
            for channel_idx in range(num_channels):
                walking_points_per_channel.append(
                    step_fn(
                        walking_points=walking_points[:, channel_idx],
                        latent_uncond_points=latent_uncond_batch[:, channel_idx],
                        latent_cond_points=latent_cond_batch[:, channel_idx],
                        field_points_pos=field_points_pos[:, :, channel_idx],
                        field_points_neg=field_points_neg[:, :, channel_idx],
                    )
                )
            walking_points = torch.stack(walking_points_per_channel, dim=1)
    return walking_points.reshape(original_shape)


def kernel_step(
    step_size: float,
    walking_points: torch.Tensor,
    latent_uncond_points: torch.Tensor,
    latent_cond_points: torch.Tensor,
    field_points_pos: torch.Tensor,
    field_points_neg: torch.Tensor,
    preference_portion: float = 0.5,
    kernel_function: Literal["rbf", "inv_square"] = "inv_square",
    **kwargs,
):
    if kernel_function == "rbf":
        def kernel_fn(x, y):
            return torch.exp(-torch.norm(x - y, dim=-1) ** 2)
    elif kernel_function == "inv_square":
        def kernel_fn(x, y):
            return 1 / torch.norm(x - y, dim=-1) ** 2
        
    conditional_directions = latent_cond_points - latent_uncond_points

    scores_pos = kernel_fn(walking_points.unsqueeze(1), field_points_pos)
    scores_neg = kernel_fn(walking_points.unsqueeze(1), field_points_neg)
    weights_pos = scores_pos / scores_pos.sum(dim=-1, keepdim=True)
    weights_neg = scores_neg / scores_neg.sum(dim=-1, keepdim=True)

    weighted_field_points_pos = (field_points_pos * weights_pos.unsqueeze(-1)).sum(dim=1)
    weighted_field_points_neg = (field_points_neg * weights_neg.unsqueeze(-1)).sum(dim=1)
    preference_directions = weighted_field_points_pos - weighted_field_points_neg

    walk_direction = preference_portion * preference_directions + (1 - preference_portion) * conditional_directions
    return walking_points + step_size * walk_direction


def constant_direction_step(
    step_size: float,
    walking_points: torch.Tensor,
    latent_uncond_points: torch.Tensor,
    latent_cond_points: torch.Tensor,
    field_points_pos: torch.Tensor,
    field_points_neg: torch.Tensor,
    preference_portion: float = 0.5,
    clip_preference_vec: bool = False,
    **kwargs,
):
    """
    Take one step in a constant direction field.

    Format of field_points_pos (written in einops format): [batch liked_points (a b c)]
    """
    conditional_directions = latent_cond_points - latent_uncond_points
    # print("normal step", latent_cond_points.shape, latent_uncond_points.shape, field_points_pos.shape, field_points_neg.shape)
    preference_directions = field_points_pos.mean(axis=1) - field_points_neg.mean(axis=1)
    if clip_preference_vec:
        preference_directions = F.normalize(preference_directions, dim=1)
        preference_directions *= torch.minimum(conditional_directions.norm(dim=1), preference_directions.norm(dim=1)).view(-1,1)
    walk_direction = preference_portion * preference_directions + (1 - preference_portion) * conditional_directions

    return walking_points + walk_direction * step_size
