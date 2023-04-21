import torch
from PIL import Image
import clip

from torchmetrics.functional.multimodal import clip_score
from functools import partial


class ClipScore:
    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cpu") -> None:
        """
        Initialize the ClipScore class.

        Args:
            clip_model (str): The name or path of the CLIP model to use.
            device (str): The device to run the model on, defaults to "cpu".
        """
        if not device:
            # Use MPS if available, otherwise use CPU
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            # Use CUDA if available, otherwise use CPU or MPS
            self.device = "cuda" if torch.cuda.is_available() else self.device
        else:
            self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.clip_score_fn = partial(
            clip_score, model_name_or_path="openai/clip-vit-base-patch16"
        )

    def compute_clip_score(self, image: Image, text_prompt: str) -> float:
        """
        Compute the CLIP score for a given image and text prompt.

        Args:
            image (PIL.Image): The image to compute the CLIP score for.
            text_prompt (str): The text prompt to compute the CLIP score for.

        Returns:
            float: The CLIP score.
        """
        # Preprocess image
        image_tensor = (self.preprocess(image).unsqueeze(0).to(self.device) * 255).type(
            torch.uint8
        )
        print(image_tensor.shape)

        clip_score = self.clip_score_fn(image_tensor, text_prompt).detach()
        return float(clip_score)

    def compute_clip_score_from_path(self, image_path: str, text_prompt: str) -> float:
        """
        Compute the CLIP score for a given image path and text prompt.

        Args:
            image_path (str): The path to the image to compute the CLIP score for.
            text_prompt (str): The text prompt to compute the CLIP score for.

        Returns:
            float: The CLIP score.
        """
        # Load image from the path and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Call the compute_clip_score method with the loaded image
        return self.compute_clip_score(image, text_prompt)


# Example usage
# clip_score_calculator = ClipScore()
# image_path = "path/to/your/image.jpg"
# text_prompt = "A description of the image"
# score_from_path = clip_score_calculator.compute_clip_score_from_path(
#     image_path, text_prompt
# )
# print("CLIP Score from path:", score_from_path)

# image = Image.open(image_path).convert("RGB")  # Pass a PIL Image object
# score = clip_score_calculator.compute_clip_score(image, text_prompt)
# print("CLIP Score from image:", score)
