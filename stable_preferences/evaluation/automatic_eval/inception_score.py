import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Tuple


class InceptionScore:
    def __init__(self, device: str = "cpu"):
        """
        Initialize the InceptionScore class with the specified device.

        :param device: The device to run the calculations on. Defaults to "cpu".
        """
        if not device:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.device = "cuda" if torch.cuda.is_available() else self.device
        else:
            self.device = device

        self.model = models.inception_v3(pretrained=True).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_pred(self, img_path: str) -> torch.Tensor:
        """
        Get the predicted probabilities for an image using the Inception-V3 model.

        :param img_path: The path to the image file.
        :return: A tensor containing the predicted probabilities.
        """
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img_tensor)
        return torch.nn.functional.softmax(pred, dim=1).cpu().numpy()

    def compute_inception_score(
        self, image_paths: List[str], n_splits: int = 10
    ) -> Tuple[float, float]:
        """
        Compute the Inception Score for a list of images.

        :param image_paths: A list of paths to the images.
        :param n_splits: The number of splits to use for calculating the score. Defaults to 10.
        :return: A tuple containing the mean and standard deviation of the Inception Scores.
        """
        preds = []
        for img_path in image_paths:
            preds.append(self.get_pred(img_path))
        preds = torch.tensor(preds).squeeze()

        split_scores = []
        for k in range(n_splits):
            part = preds[
                k * (len(preds) // n_splits) : (k + 1) * (len(preds) // n_splits), :
            ]
            py = torch.mean(part, 0)
            scores = torch.exp(torch.mean(torch.log(torch.clamp(part / py, min=1e-16))))
            split_scores.append(scores)

        return (
            torch.mean(torch.stack(split_scores)).item(),
            torch.std(torch.stack(split_scores)).item(),
        )


# Example usage
inception_score_calculator = InceptionScore()
image_paths = [
    "path/to/your/image1.jpg",
    "path/to/your/image2.jpg",
    "path/to/your/image3.jpg",
]
mean, std = inception_score_calculator.compute_inception_score(image_paths)
print("Inception Score: Mean =", mean, "Std =", std)
