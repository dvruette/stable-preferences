from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from typing import Union
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else DEVICE
print(f"Directional Similarity: using {DEVICE}")


class DirectionalSimilarity(nn.Module):
    def __init__(
        self,
        tokenizer: CLIPTokenizer = None,
        text_encoder: CLIPTextModelWithProjection = None,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        """
        Initialize the DirectionalSimilarity class with the specified tokenizer, text_encoder,
        image_processor, and image_encoder.

        :param tokenizer: The tokenizer to tokenize the text.
        :param text_encoder: The text encoder to encode the text.
        :param image_processor: The image processor to preprocess the images.
        :param image_encoder: The image encoder to encode the images.
        """
        super().__init__()

        if not tokenizer:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        else:
            self.tokenizer = tokenizer
        if not text_encoder:
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(DEVICE)
        else:
            self.text_encoder = text_encoder
        if not image_processor:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        else:
            self.image_processor = image_processor
        if not image_encoder:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(DEVICE)
        else:
            self.image_encoder = image_encoder

    def preprocess_image(self, image: Image.Image) -> dict:
        """
        Preprocess the input image using the image processor.

        :param image: The input PIL Image object.
        :return: A dictionary containing the preprocessed image tensor.
        """
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(DEVICE)}

    def tokenize_text(self, text: str) -> dict:
        """
        Tokenize the input text using the tokenizer.

        :param text: The input text.
        :return: A dictionary containing the tokenized input_ids tensor.
        """
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(DEVICE)}

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode the input image using the image encoder.

        :param image: The input PIL Image object.
        :return: A tensor containing the encoded image features.
        """
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode the input text using the text encoder.

        :param text: The input text.
        :return: A tensor containing the encoded text features.
        """
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(
        self,
        img_feat_one: torch.Tensor,
        img_feat_two: torch.Tensor,
        text_feat_one: torch.Tensor,
        text_feat_two: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the directional similarity between two pairs of image and text features.

        :param img_feat_one: The first image feature tensor.
        :param img_feat_two: The second image feature tensor.
        :param text_feat_one: The first text feature tensor.
        :param text_feat_two: The second text feature tensor.
        :return: A tensor containing the directional similarity.
        """
        sim_direction = F.cosine_similarity(
            img_feat_two - img_feat_one, text_feat_two - text_feat_one
        )
        return sim_direction

    def forward(
        self,
        image_one: Image.Image,
        image_two: Image.Image,
        caption_one: str,
        caption_two: str,
    ) -> torch.Tensor:
        """
        Calculate the directional similarity between two image-caption pairs.

        :param image_one: The first input PIL Image object.
        :param image_two: The second input PIL Image object.
        :param caption_one: The first input caption text.
        :param caption_two: The second input caption text.
        :return: A tensor containing the directional similarity.
        """
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity


# Example usage
# clip_id = "openai/clip-vit-large-patch14"
# tokenizer = CLIPTokenizer.from_pretrained(clip_id)
# text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
# image_processor = CLIPImageProcessor.from_pretrained(clip_id)
# image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

directional_similarity_calculator = DirectionalSimilarity(
    # tokenizer, text_encoder, image_processor, image_encoder
)

image_one = Image.open(
    "/nese/mit/group/evlab/u/luwo/projects/projects/stable-preferences/example_1.png"
).convert("RGB")
image_two = Image.open(
    "/nese/mit/group/evlab/u/luwo/projects/projects/stable-preferences/example_2.png"
).convert("RGB")
caption_one = "An astronaut on mars"
caption_two = "An astronaut on a horse on mars"

directional_similarity = directional_similarity_calculator(
    image_one, image_two, caption_one, caption_two
)
print("Directional Similarity:", directional_similarity.item())
