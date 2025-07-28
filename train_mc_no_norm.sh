#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/lsfpnet_mc_svd_mag_noise_3layers_no_norm.err
#SBATCH --output=logs/lsfpnet_mc_svd_mag_noise_3layers_no_norm.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=lsfpnet_mc_svd_mag_noise_3layers_no_norm
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
python3 train.py --config configs/config_mc_no_norm.yaml --exp_name lsfpnet_mc_svd_mag_noise_3layers_no_norm