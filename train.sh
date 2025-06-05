#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=train_ei_loss.err
#SBATCH --output=train_ei_loss.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=train_ei_loss
#SBATCH --mem-per-gpu=490000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=gpuq
#SBATCH --time=1440

# Load Micromamba
source /gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/etc/profile.d/micromamba.sh

# Activate your Micromamba environment
micromamba activate recon_mri

# Run the training script with srun
python3 train.py