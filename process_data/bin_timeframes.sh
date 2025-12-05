#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=bin_timeframes.err
#SBATCH --output=bin_timeframes.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=bin_timeframes
#SBATCH --mem-per-gpu=120000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=gpuq
#SBATCH --time=1440

# Load Micromamba
source /gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/etc/profile.d/micromamba.sh

# Activate your Micromamba environment
micromamba activate ddei

# Run the training script with srun
python3 bin_timeframes.py