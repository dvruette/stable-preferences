from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from hydra.utils import to_absolute_path


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.paths = image_paths
        if transform is None:
            self.transform = T.Compose([
                T.Resize(512),
                T.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(to_absolute_path(path)).convert("RGB")
        img = 2*self.transform(img) - 1
        return img


def load_images(image_paths: List[str]) -> List[torch.Tensor]:
    if not image_paths:
        return []
    ds = ImageDataset(image_paths)
    imgs = [img for img in ds]
    return imgs
