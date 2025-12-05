#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=undersample.err
#SBATCH --output=undersample.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=undersample
#SBATCH --mem-per-gpu=120000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=gpuq
#SBATCH --time=14400

# Load Micromamba
source /gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/etc/profile.d/micromamba.sh

# Activate your Micromamba environment
micromamba activate recon_mri

# Run the training script with srun
bash undersample_all_data.sh