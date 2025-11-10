#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/train_dro_ei_spatial_encode_both.err
#SBATCH --output=logs/train_dro_ei_spatial_encode_both.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=train_dro_ei_spatial_encode_both
#SBATCH --mem-per-gpu=50000
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
python3 train_dro.py --config configs/config_ei_spatial_single_gpu.yaml --exp_name train_dro_ei_spatial_encode_both