




def img_batch_to_space(space_type):
    """
    Convert a batch of images (noise space) to a specific space. 
    Gives back a differentiable function to map into the space which can then be used to optimize the input images
    """
    if space_type=='latent_noise':
        return lambda x: x