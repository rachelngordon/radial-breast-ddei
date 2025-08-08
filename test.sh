#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/test_no_aug.err
#SBATCH --output=logs/test_no_aug.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test_no_aug
#SBATCH --mem-per-gpu=80000
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
python3 train_aug.py --config configs/config_mc_debug.yaml --exp_name test_no_aug