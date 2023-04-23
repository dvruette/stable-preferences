from torch.nn.functional import normalize
from einops import repeat, rearrange, pack, unpack
import torch


def walk_in_field(
        latent_uncond_batch,
        latent_cond_batch,
        field_points_pos,
        field_points_neg,
        field_type,
        walk_distance, 
        n_steps=1,
        # any further arguments that are needed for the field type
        **kwargs
        ):
    """
    Walk in a field of a specific type having binary feedback vectors. It takes n_steps which sum up to the walk_distance.
    E.g. when walking in a field with a constant direction, the n_steps don't matter.
    Gives back a new latent_batch with the points where the walk ended.

    The calling function then can optimize the noise space with gradient descent to reach those points.

    format of field_points_pos (written in einops format): [batch liked_points a b c]
    """
    assert latent_uncond_batch.shape == latent_cond_batch.shape, 'latent_uncond_batch and latent_cond_batch must have the same shape'
    ### flatten all dimensions except the batch dimension
    original_shape = latent_cond_batch.shape
    latent_uncond_batch = latent_uncond_batch.reshape(latent_uncond_batch.shape[0], -1)
    latent_cond_batch = latent_cond_batch.reshape(latent_cond_batch.shape[0], -1) 
    field_points_pos = rearrange(field_points_pos, 'batch liked_points a b c -> batch liked_points (a b c)')
    field_points_neg = rearrange(field_points_neg, 'batch disliked_points a b c -> batch disliked_points (a b c)')

    one_step = walk_distance / n_steps
    walking_points = latent_cond_batch.clone().detach()
    for i in range(n_steps):
        if field_type=='constant_direction':
            walking_points = constant_direction_step(
                    one_step,
                    walking_points,
                    latent_uncond_batch,
                    latent_cond_batch,
                    field_points_pos,
                    field_points_neg,
                    **kwargs
                    )
        else:
            raise NotImplementedError(f'Field type {field_type} not implemented.')
        
    return walking_points.reshape(original_shape)
            

            




def constant_direction_step(
        one_step,
        walking_points,
        latent_uncond_points,
        latent_cond_points,
        field_points_pos,
        field_points_neg,
        **kwargs
        ):
    """
    Take one step in a constant direction field.

    Format of field_points_pos (written in einops format): [batch liked_points (a b c)]
    """
    preference_portion = kwargs['preference_portion']
    conditional_directions = latent_cond_points - latent_uncond_points
    if 0 not in field_points_pos.shape and 0 not in field_points_neg.shape:
        print("normal step", latent_cond_points.shape, latent_uncond_points.shape, field_points_pos.shape, field_points_neg.shape)
        preference_directions = field_points_pos.mean(axis=1) - field_points_neg.mean(axis=1)
        if 'clip_preference_vec' in kwargs and kwargs['clip_preference_vec']:
            preference_directions = normalize(preference_directions, dim=1)
            preference_directions *= torch.minimum(conditional_directions.norm(dim=1), preference_directions.norm(dim=1)).view(-1,1)
        walk_direction = preference_portion * preference_directions + (1-preference_portion) * conditional_directions
    else:
        print('Warning: No conditional points or no unconditional points. Using conditional directions only. Ignoring all liked and disliked images.')
        walk_direction = conditional_directions
    return walking_points + walk_direction*one_step
