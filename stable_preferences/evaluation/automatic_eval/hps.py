import torch
import clip
import os
from PIL import Image


class HumanPreferenceScore:
    """
    Compute human preference score from text and images.
    """

    def __init__(self, weight_path: str = "./weights"):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = "cuda" if torch.cuda.is_available() else device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        params = torch.load(
            weight_path,
            map_location=device,
        )["state_dict"]
        self.model.load_state_dict(params)
        self.device = device

    def compute_from_paths(self, text, image_paths):
        """
        ::param text: str
        ::param image_paths: list of str
        ::return: torch.Tensor
        """
        images = [Image.open(image_path) for image_path in image_paths]
        return self.compute(text, images)

    def compute(self, text, images):
        """
        ::param text: str
        ::param images: list of PIL.Image
        ::return: torch.Tensor
        """
        processed_images = []
        for image in images:
            processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
            processed_images.append(processed_image)
        processed_images = torch.cat(processed_images, dim=0)
        text = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(processed_images)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            hps = image_features @ text_features.T

        return hps
