## Evaluation Usage

Make sure you unzip the resources directory within the evaluation directory.

We can currently evaluate w
1. Single image with prompt {clip, hpc, pap}-scores
2. Directional similarity of img_1 + prompt_1 to img_2 + prompt_2
3. FID from set of generated to set of real images
4. InceptionScore

(1) In order to run the evaluation scores on a single image with its corresponding generating description, run:
```
python eval.py --image_path /path/to/image --prompt "Image description" --clip --hps --pap
```

The currently available evaluation metrics are 
```
--clip : standard clip score 
--hps : human preference score trained on the Discord interactions
--pap : pick-a-pic score trained on the pick-a-pic dataset
```

(2) In order to compute the FID score, use the ImageFID class and call compute_from_directory for a path to the real image directory and a path to the generated image directory. It will update and return the metric, i.e. we can iteratively call the method and it will update the metric during the process

(3) For the directional similarity, you can call the class with tensors (2 images, 2 prompts). Example usage can be found in the file.

(4) Example is in inception_score file.
