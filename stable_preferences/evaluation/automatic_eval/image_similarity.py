import torch
from PIL import Image
import numpy as np
import clip

from clip_score import ClipScore


class ImageSimilarity:
    def __init__(
        self,
        positive_images,
        negative_images,
        generated_image,
        clip_model: str = "ViT-B/32",
        device: str = None,
    ):
        self.positive_images = positive_images
        self.negative_images = negative_images
        self.generated_image = generated_image
        self.clip_score_calculator = ClipScore()

    def _get_similarity(self, reference_images, method):
        similarities = []
        for image in reference_images:
            if method == "path":
                similarity = self.clip_score_calculator.compute_clip_score_from_path(
                    image, self.generated_image
                )
            elif method == "tensor":
                similarity = self.clip_score_calculator.compute_clip_score(
                    image, self.generated_image
                )
            similarities.append(similarity)
        return min(similarities), max(similarities), np.mean(similarities)

    def compute_similarity_from_path(self):
        pos_min, pos_max, pos_mean = self._get_similarity(self.positive_images, "path")
        neg_min, neg_max, neg_mean = self._get_similarity(self.negative_images, "path")
        return {
            "positive": {"min": pos_min, "max": pos_max, "mean": pos_mean},
            "negative": {"min": neg_min, "max": neg_max, "mean": neg_mean},
        }

    def compute_similarity_from_tensor(self):
        pos_min, pos_max, pos_mean = self._get_similarity(
            self.positive_images, "tensor"
        )
        neg_min, neg_max, neg_mean = self._get_similarity(
            self.negative_images, "tensor"
        )
        return {
            "positive": {"min": pos_min, "max": pos_max, "mean": pos_mean},
            "negative": {"min": neg_min, "max": neg_max, "mean": neg_mean},
        }
