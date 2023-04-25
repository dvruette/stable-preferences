from typing import Literal, List
import torch
from einops import repeat, rearrange, pack, unpack
from tqdm import tqdm
from spaces import img_batch_to_space
from fields import walk_in_field
from scipy.optimize import minimize
import torch.optim as optim
from stable_preferences.utils import get_free_gpu
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler





class StableDiffuserWithBinaryFeedback:

    def __init__(self, 
            stable_diffusion_version: str = "1.5",
            unet_max_chunk_size=8,
            n_images=1,
            walk_distance=8,
            walk_steps=1,
            denoising_steps=20,
            show_progress=True,
        ):
        if stable_diffusion_version == "1.5":
            model_name = "runwayml/stable-diffusion-v1-5"
        elif stable_diffusion_version == "2.1":
            model_name = "stabilityai/stable-diffusion-2-1"
        else:
            raise ValueError(f"Unknown stable diffusion version: {stable_diffusion_version}. Version bust be either '1.5' or '2.1'")
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = get_free_gpu() if torch.cuda.is_available() else self.device

        print(f"Using device: {self.device}")
        dtype = torch.float16 if str(self.device)!='cpu' else torch.float32
        print(f"Using dtype: {dtype}")
        torch.set_default_dtype(dtype) # bad style to change global default, but this is how it is done in the original code

        self.scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.diffusion_pipe = StableDiffusionPipeline.from_pretrained(model_name, scheduler=self.scheduler, torch_dtype=dtype).to(self.device)

        print(f"Unet: {sum(p.numel() for p in self.diffusion_pipe.unet.parameters()) / 1e6:.0f}M")
        print(f"VAE: {sum(p.numel() for p in self.diffusion_pipe.vae.parameters()) / 1e6:.0f}M")
        print(f"TextEncoder: {sum(p.numel() for p in self.diffusion_pipe.text_encoder.parameters()) / 1e6:.0f}M")

        self.unet_max_chunk_size = unet_max_chunk_size
        self.n_images = n_images
        self.walk_distance = walk_distance
        self.walk_steps = walk_steps
        self.denoising_steps = denoising_steps
        self.show_progress = show_progress
        

    def initialize_prompts(
        self,
        liked_prompts: List[str],
        disliked_prompts: List[str],
    ):
        """
        Initialize prompt feedback for the trajectory generation.
        This function just embeds the prompts and returns the embeddings, per step. 
        This provides the flexibility, to have a framework working with a potentially different prompt for each step, i.e. compatible with null text inversion.
        """

        if len(liked_prompts) == 0 and len(disliked_prompts) == 0:
            return torch.zeros((self.denoising_steps, 0, 0, 0)), torch.zeros((self.denoising_steps, 0, 0, 0))

        text_encoder = self.diffusion_pipe.text_encoder
        tokenizer = self.diffusion_pipe.tokenizer

        prompt_tokens = tokenizer(
            liked_prompts + disliked_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        prompt_embd = text_encoder(**prompt_tokens).last_hidden_state
        liked_prompts_embd = prompt_embd[: len(liked_prompts)]
        disliked_prompts_embd = prompt_embd[len(liked_prompts) :]
        liked_prompts_embds = repeat(liked_prompts_embd, 'prompts a b -> steps prompts a b', steps=self.denoising_steps)
        disliked_prompts_embds = repeat(disliked_prompts_embd, 'prompts a b -> steps prompts a b', steps=self.denoising_steps)

        return liked_prompts_embds, disliked_prompts_embds

    def chunked_unet_forward(
        self,
        unet,
        z_all,
        t,
        batched_prompt_embd,
    ):
        """
        Forward pass for the diffusion model, chunked to avoid memory issues.
        """
        n_chunks = z_all.shape[0] // self.unet_max_chunk_size
        z_all_chunks = torch.chunk(z_all, n_chunks, dim=0)
        batched_prompt_embd_chunks = torch.chunk(batched_prompt_embd, n_chunks, dim=0)
        unet_out_all = []
        for z_all_chunk, batched_prompt_embd_chunk in zip(z_all_chunks, batched_prompt_embd_chunks):
            unet_out = unet(z_all_chunk, t, batched_prompt_embd_chunk).sample     
            unet_out = unet_out.to(torch.float32)
            unet_out_all.append(unet_out)
        unet_out_all,_ = pack(unet_out_all, '* a b c')
        return unet_out_all

    def optimize_noise(
            self,
            noise_cond,
            in_space_destinations,
            space_converter,
        ):
        """
        Optimize the noise to get to the destination in the space.
        """

        current_points = noise_cond.clone().detach().requires_grad_(True)

        loss_t = torch.nn.functional.mse_loss(space_converter(current_points), in_space_destinations, reduction='sum')
        loss_t.backward()
        grad_norm = current_points.grad.norm()
        lr = 0.3 # torch.tensor(1/3) * norm_to_walk / max(grad_norm,1.0)

        convergence_threshold = 1e-5
        max_iterations = 100

        iterations_used=0
        print(f"Initial loss from optimization: {torch.nn.functional.mse_loss(space_converter(current_points), in_space_destinations, reduction='sum')} in {iterations_used} many iterations")
        for iteration in range(max_iterations):
            current_points.grad.zero_()

            current_points_in_space = space_converter(current_points)
            loss = torch.nn.functional.mse_loss(current_points_in_space, in_space_destinations, reduction='sum')
            if loss.item() < convergence_threshold:
                print(f"Converged after {iteration} iterations")
                break
            loss.backward()

            # Full gradient descent update
            with torch.no_grad():
                current_points -= lr * current_points.grad

            iterations_used += 1

        print(f"Final loss from optimization: {torch.nn.functional.mse_loss(space_converter(current_points), in_space_destinations, reduction='sum')} in {iterations_used} many iterations")
        return current_points.clone().detach().requires_grad_(False)

        
    @torch.no_grad()
    def generate(
        self,
        prompt="a photo of an astronaut riding a horse on mars",
        liked=[],
        disliked=[],
        field: Literal["constant_direction"] = "constant_direction",
        space: Literal["latent_noise"] = "latent_noise",
        binary_feedback_type: Literal["prompt", "image"] = "prompt",
        seed=42,
        **kwargs
    ):
        """
        Generate a trajectory of images with binary feedback.
        The feedback can be given as a list of liked and disliked prompts, or as images, which then get inverted.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if binary_feedback_type == "prompt":
            liked_prompts_embds, disliked_prompts_embds = self.initialize_prompts(
                liked, disliked
            )
        elif binary_feedback_type == "image":
            raise NotImplementedError("Image feedback is not implemented yet")
        cond_prompt_embds, uncond_prompt_embds = self.initialize_prompts(
            [prompt], [""]
        ) # shape: "steps 1 a b"
        cond_prompt_embds = rearrange(cond_prompt_embds, "steps 1 a b -> steps a b")
        uncond_prompt_embds = rearrange(uncond_prompt_embds, "steps 1 a b -> steps a b")

        scheduler = self.diffusion_pipe.scheduler
        unet = self.diffusion_pipe.unet

        scheduler.set_timesteps(self.denoising_steps, device=self.device)

        iterator = scheduler.timesteps
        if self.show_progress:
            iterator = tqdm(scheduler.timesteps)
        traj = []
        norms = []
        z = torch.randn(self.n_images, 4, 64, 64, device=self.device)
        z = z * scheduler.init_noise_sigma
        for iteration, t in enumerate(iterator):
            z_single = scheduler.scale_model_input(z, t)
            z_all = repeat(z_single, "batch a b c -> (batch prompts) a b c", prompts=2 + len(liked) + len(disliked)) # we generate the next z for all prompts and then combine

            liked_prompts_embd = liked_prompts_embds[iteration]
            disliked_prompts_embd = disliked_prompts_embds[iteration]
            cond_prompt_embd = cond_prompt_embds[iteration]
            uncond_prompt_embd = uncond_prompt_embds[iteration]
            prompt_embd, ps = pack(
                [a for a in [cond_prompt_embd, uncond_prompt_embd, liked_prompts_embd, disliked_prompts_embd] if 0 not in a.shape], # avoid empty concatenation throwing errors 
                '* a b')
            batched_prompt_embd = repeat(prompt_embd, 'prompts a b -> (batch prompts) a b', batch=self.n_images)
            
            unet_out = self.chunked_unet_forward(
                unet,
                z_all,
                t,
                batched_prompt_embd,
            )
            unet_out = rearrange(unet_out, "(batch prompts) a b c -> batch prompts a b c", batch=self.n_images)
            noise_cond, noise_uncond, noise_liked, noise_disliked = unpack(
                unet_out,
                [(),(),(len(liked),), (len(disliked),)],
                'batch * a b c'
            )

            space_converter = img_batch_to_space(space)
            in_space_destinations = walk_in_field(
                space_converter(noise_uncond),
                space_converter(noise_cond),
                space_converter(noise_liked),
                space_converter(noise_disliked),
                field,
                self.walk_distance,
                n_steps=self.walk_steps,
                **kwargs
            )
            with torch.enable_grad():
                noise_destinations = self.optimize_noise(
                    noise_cond.detach(),
                    in_space_destinations.detach(),
                    space_converter,
                )

            # cfg_vector = noise_cond - noise_uncond
            # noise_destinations = noise_cond + self.walk_distance*cfg_vector

            print("guidance norm: ",(noise_cond-noise_destinations).view(self.n_images, -1).norm(dim=1))
            noise_destinations = noise_destinations.to(z.dtype)
            z = scheduler.step(noise_destinations, t, z).prev_sample

            y = self.diffusion_pipe.decode_latents(z)
            piled = self.diffusion_pipe.numpy_to_pil(y)
            traj.append(piled)

        return traj