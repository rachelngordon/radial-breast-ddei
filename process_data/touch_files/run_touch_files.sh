#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=touch_files.err
#SBATCH --output=touch_files.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=touch_files
#SBATCH --mem-per-gpu=120000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=gpuq
#SBATCH --time=400

# Run the training script with srun
bash touch_files.sh