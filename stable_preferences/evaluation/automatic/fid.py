import torch
import torchvision.transforms as transforms
from PIL import Image
from openai import CLIP
from scipy.linalg import sqrtm


class FIDScore:
    """
    Frechet Inception Distance (FID) is a metric to measure the distance between two distributions of images.
    """

    def __init__(self, clip_model="ViT-B/32", device="cpu"):
        if not device:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.device = "cuda" if torch.cuda.is_available() else self.device
        else:
            self.device = device
        self.model, self.preprocess = CLIP.load(clip_model, device=self.device)

    def get_image_features(self, image_paths):
        features = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features, _ = self.model.encode_image(image_tensor)
            features.append(image_features.cpu().numpy())
        return torch.tensor(features).squeeze()

    def compute_fid_score(self, image_paths1, image_paths2):
        features1 = self.get_image_features(image_paths1)
        features2 = self.get_image_features(image_paths2)

        mu1, mu2 = features1.mean(0), features2.mean(0)
        sigma1, sigma2 = torch.cov(features1, rowvar=False), torch.cov(
            features2, rowvar=False
        )

        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.numpy().dot(sigma2.numpy()), disp=False)
        covmean = torch.from_numpy(covmean).float()

        fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()


# Example usage
fid_calculator = FIDScore()
image_paths1 = ["path/to/your/image1A.jpg", "path/to/your/image2A.jpg"]
image_paths2 = ["path/to/your/image1B.jpg", "path/to/your/image2B.jpg"]
fid_score = fid_calculator.compute_fid_score(image_paths1, image_paths2)
print("FID Score:", fid_score)
