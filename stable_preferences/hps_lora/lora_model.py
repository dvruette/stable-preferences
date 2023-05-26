import os
import torch
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers import StableDiffusionPipeline
import json
import time

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else DEVICE


class LoRA_HP_SD:
    def __init__(
        self,
        unet_weight: str = None,
        prompts_filepath: str = None,
        results_folder="./outputs",
        negative_prompt="Weird image.",
        world_size=-1,
        rank=-1,
        device=None,
    ):
        self.unet_weight = unet_weight
        self.negative_prompt = negative_prompt
        self.prompts = prompts_filepath
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.folder = os.path.join(results_folder, self.run_id)
        os.makedirs(self.folder, exist_ok=True)

        self.use_command_line_rank = world_size > 0 and rank >= 0
        self.local_rank = 0
        self.rank = rank
        self.world_size = world_size
        self.device = device if device else DEVICE
        print(f"Using device: {self.device}")

        self._get_rank_world_size()

        if prompts_filepath:
            with open(self.prompts) as f:
                self.pairs = json.load(f)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        ).to(self.device)

        if self.unet_weight:
            print(f"Loading LoRA unet weights from {self.unet_weight}")
            self._load_unet_weights()
        else:
            print("No unet weights provided, using default unet weights")

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def _get_rank_world_size(self):
        for v in (
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
        ):
            if v in os.environ:
                self.local_rank = int(os.environ[v])
                if self.use_command_line_rank:
                    assert self.local_rank == 0
                break

        if self.use_command_line_rank:
            pass
        else:
            self.rank = 0
            for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
                if v in os.environ:
                    self.rank = int(os.environ[v])
                    break

        if self.use_command_line_rank:
            pass
        else:
            self.world_size = 1
            for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
                if v in os.environ:
                    self.world_size = int(os.environ[v])
                    break

    def _load_unet_weights(self):
        model_weight = torch.load(self.unet_weight, map_location="cpu")
        unet = self.pipeline.unet
        lora_attn_procs = {}
        lora_rank = list(
            set(
                [
                    v.size(0)
                    for k, v in model_weight.items()
                    if k.endswith("down.weight")
                ]
            )
        )
        assert len(lora_rank) == 1
        lora_rank = lora_rank[0]
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_rank,
            ).to(self.device)
        unet.set_attn_processor(lora_attn_procs)
        unet.load_state_dict(model_weight, strict=False)

    def generate_images(self):
        generator = torch.Generator(device=self.device).manual_seed(self.rank + 1)

        for i, pair in enumerate(self.pairs):
            if i % self.world_size != self.rank:
                continue
            if os.path.exists(f"{self.folder}/{i}.jpg"):
                continue
            with torch.no_grad():
                raw_images = self.pipeline(
                    [pair],
                    num_inference_steps=50,
                    generator=generator,
                    negative_prompt=[self.negative_prompt],
                ).images
            for j, image in enumerate(raw_images):
                image.save(f"{self.folder}/{i}.jpg", quality=90)

    def generate_images_from_prompts(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]

        generated_images = []
        generator = torch.Generator(device=self.device).manual_seed(self.rank + 1)

        for i, prompt in enumerate(prompts):
            image_filename = f"{self.folder}/{prompt.replace(' ', '_')}.jpg"

            if os.path.exists(image_filename):
                continue

            with torch.no_grad():
                raw_images = self.pipeline(
                    prompt,
                    num_inference_steps=50,
                    generator=generator,
                    negative_prompt=self.negative_prompt,
                ).images

            for j, image in enumerate(raw_images):
                print(f"Storing image to {image_filename}")
                image.save(image_filename, quality=90)
                generated_images.append(image)

        return generated_images
