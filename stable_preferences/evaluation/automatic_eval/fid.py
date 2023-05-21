import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.image.fid import FrechetInceptionDistance


class ImageFID:
    """
    Compute Frechet Inception Distance (FID) from images.
    """

    def __init__(self, feature=64):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fid = FrechetInceptionDistance(feature=feature).to(device)
        self.device = device

    def compute_from_directory(self, directory, real=True):
        """
        ::param directory: str
        ::param real: bool
        ::return: float
        """
        image_paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(("jpg", "png", "jpeg"))
        ]
        images = [Image.open(image_path) for image_path in image_paths]
        tensor_images = torch.stack([ToTensor()(img) for img in images]).to(self.device)

        self.fid.update(tensor_images, real=real)
        return self.fid.compute()

    def compute_from_tensor(self, images, real=True):
        """
        ::param images: torch.Tensor
        ::param real: bool
        ::return: float
        """
        self.fid.update(images.to(self.device), real=real)
        return self.fid.compute()


# Computing FID from a directory of images
fid_computer = ImageFID()
fid_score = fid_computer.compute_from_directory("/path/to/your/directory", real=True)
print(fid_score)

# Computing FID from a tensor of images
imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
fid_computer = ImageFID()
fid_computer.compute_from_tensor(imgs_dist1, real=True)
fid_score = fid_computer.compute_from_tensor(imgs_dist2, real=False)
print(fid_score)
