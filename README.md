# Stable Preferences

Group project for Generative Visual Models

## Overview

- [🧨 Diffusers](https://huggingface.co/docs/diffusers/index) for diffusion model stuff
- [Hydra](https://hydra.cc/docs/intro/) for handling configs

## Setup

Create virtual env (alternatively, you can use `conda` if preferred):
```bash
python3 -m venv .venv  # create new virtual environment
source .venv/bin/activate  # activate it
pip install -r requirements.txt  # install requirements
pip install -e .  # install current repository in editable mode
```

If everything worked well, you should be able to execute the example script (might take a while to download the model).
This should generate "an astronaut riding a horse on mars" and save it in `example.png`.
```bash
python stable_preferences/example.py
```

The script is configured using Hydra, which means you can pass command line arguments as follows (commas need to be escape as `\,`):
```
python stable_preferences/example.py prompt="a cute corgi playing the trumpet\, best quality\, trending on artstation\, exceptional detail" neg_prompt="bad art\, worst quality" cfg_scale=10 seed=42
```
