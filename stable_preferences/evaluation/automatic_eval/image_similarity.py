import torch
import numpy as np
# import clip
import clip


class ImageSimilarity:
    def __init__(
        self,
        clip_model: str = "ViT-L/14@336px", #"ViT-B/32",
        # open_clip_dataset: str = "laion2b_s39b_b160k",
        device: str = None,
    ):
        if not device:
            # Use MPS if available, otherwise use CPU
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            # Use CUDA if available, otherwise use CPU or MPS
            self.device = "cuda" if torch.cuda.is_available() else self.device
        else:
            self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=open_clip_dataset, device=self.device)

    @torch.no_grad()
    def compute(self, imgs1, imgs2):
        # Preprocess image
        processed1, processed2 = [], []
        for img in imgs1:
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            processed1.append(img)
        processed1 = torch.cat(processed1, dim=0)
        for img in imgs2:
            img = self.preprocess(img).unsqueeze(0).to(self.device)
            processed2.append(img)
        processed2 = torch.cat(processed2, dim=0)

        imgs1_features = self.model.encode_image(processed1)
        imgs2_features = self.model.encode_image(processed2)
        imgs1_features /= imgs1_features.norm(dim=-1, keepdim=True)
        imgs2_features /= imgs2_features.norm(dim=-1, keepdim=True)

        # Compute the CLIP score
        similarity = (100.0 * imgs1_features @ imgs2_features.T)
        return similarity.cpu().numpy()

    def _get_similarity(self, reference_images, method):
        raise NotImplementedError("Old image similarity methods are broken, sorry :( Use `ImageSimilarity.compute` instead.")
        similarities = []
        for image in reference_images:
            if method == "path":
                similarity = self.clip_score_calculator.compute_from_path(
                    image, self.generated_image
                )
            elif method == "tensor":
                similarity = self.clip_score_calculator.compute(
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
