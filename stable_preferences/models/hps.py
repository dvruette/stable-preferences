"""
Human preference score (finetuned clip model)

The pretrained human preference classifier can be downloaded from 
[OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EWDmzdoqa1tEgFIGgR5E7gYBTaQktJcxoOYRoTHWzwzNcw?e=b7rgYW).
Before running the human preference classifier, please make sure you have set up the CLIP environment 
as specified in the [official repo](https://github.com/openai/CLIP).
"""

import torch
import clip
from PIL import Image


class HumanPreferenceScore:
    def __init__(self, model_path, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            device = "cuda" if torch.cuda.is_available() else device

        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        params = torch.load(model_path, map_location=device)["state_dict"]
        self.model.load_state_dict(params)

    def infer_score(self, image_paths, prompt):
        images = [
            self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            for image_path in image_paths
        ]
        images = torch.cat(images, dim=0)
        text = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            hps = image_features @ text_features.T

        return hps


# Example usage
hps_obj = HumanPreferenceScore(
    "/Users/lukas/Desktop/projects/diffusion/align_sd/weights/hpc.pt"
)
img_path_1 = "/Users/lukas/Desktop/projects/diffusion/align_sd/assets/vis1.png"
img_path_2 = "/Users/lukas/Desktop/projects/diffusion/align_sd/assets/vis2.png"
image_paths = [img_path_1, img_path_2]
prompt = "your prompt here"
hps = hps_obj.infer(image_paths, prompt)
print(hps)
