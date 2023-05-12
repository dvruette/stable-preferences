import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else DEVICE
print("DEVICE:", DEVICE)

#
# CLIP
#

from stable_preferences.evaluation.automatic.clip_score import ClipScore

example_paths = [
    "/nese/mit/group/evlab/u/luwo/projects/projects/stable-preferences/example_1.png",
    "/nese/mit/group/evlab/u/luwo/projects/projects/stable-preferences/example_2.png",
    "/nese/mit/group/evlab/u/luwo/projects/projects/stable-preferences/example_3.png",
    "/nese/mit/group/evlab/u/luwo/projects/projects/stable-preferences/static/example_img/dot matrix beautiful swooping kingfisher unsplash monstrous tesselated surreal eldritch chiaroscuro HDR enchanted mystical whimsical bold artwork by leonid afremov and lois van baarle.png",
]

PROMPT = "a photo of an astronaut riding a horse on mars"

# compute the CLIP score for each image

clip_score_calculator = ClipScore()
scores = [
    clip_score_calculator.compute_clip_score_from_path(path, PROMPT)
    for path in example_paths
]

# print the scores
for i, (path, score) in enumerate(zip(example_paths, scores)):
    print(f"CLIP Score for image {i}:", score)
