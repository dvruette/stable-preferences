### Download data
Download data from [Better Aligning Text-to-Image Models with Human Preference](https://github.com/tgxs002/align_sd). The data is available on their OneDrive and has to be unzipped in:
```
stable-preferences/stable_preferences/human_preference_dataset
```
which will create the "dataset" directory automatically.

Also download the model weights and store them in 
```
stable-preferences/stable_preferences/human_preference_dataset/weights
```


### Compute the human preference score
In order to compute the human preference score (HPC), set up the clip environment as specified in the [clip repository](https://github.com/openai/CLIP).
A model inference class is located in hpc.py. You can feed the images as paths or PIL.image and compute the HPC.


### Estimate the preference field 
Estimation methods: 


