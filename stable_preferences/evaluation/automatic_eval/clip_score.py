import torch
from PIL import Image
import clip


class ClipScore:
    def __init__(self, 
        clip_model: str = "ViT-L/14@336px", #"ViT-B/32",
        # open_clip_dataset: str = "laion2b_s39b_b160k",
        device: str = None) -> None:
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
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=open_clip_dataset, device=self.device)

    def compute(self, text_prompt: str, image: Image) -> float:
        """
        Compute the CLIP score for a given image and text prompt.

        Args:
            image (PIL.Image): The image to compute the CLIP score for.
            text_prompt (str): The text prompt to compute the CLIP score for.

        Returns:
            float: The CLIP score.
        """
        # Preprocess image
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([text_prompt]).to(self.device)
        # print(image_tensor.shape)

        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the CLIP score
        similarity = (100.0 * image_features @ text_features.T)
        return similarity.item()

    def compute_from_path(self, text_prompt: str, image_path: str) -> float:
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
        return self.compute(image, text_prompt)


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
