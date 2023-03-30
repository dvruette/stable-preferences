#!/bin/bash

#SBATCH -n 10 #full node
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --gres=gpumem:10g
#SBATCH --gpus=1
#SBATCH --output=analysis2.out
#SBATCH --error=analysis2.err
#SBATCH --open-mode=truncate

module load gcc/8.2.0 python/3.9.9
module load eth_proxy
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate stable-preferences
python3 inference.py

# interactive shell: srun --gres=gpumem:20g --gpus=1 --mem-per-cpu=16G --pty inference.sh
# run job: sbatch --gres=gpumem:20g --gpus=1 --mem-per-cpu=16G inference.sh