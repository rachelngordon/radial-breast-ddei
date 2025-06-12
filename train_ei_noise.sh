#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/ei_loss_temp_noise.err
#SBATCH --output=logs/ei_loss_temp_noise.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ei_loss_temp_noise
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
python3 train.py --config configs/config_ei_noise.yaml --exp_name ei_loss_temp_noise