#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/lsfp_mc_8spf_svd_detach_uv_lr0001_no_time_encoding.err
#SBATCH --output=logs/lsfp_mc_8spf_svd_detach_uv_lr0001_no_time_encoding.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=lsfp_mc_8spf_svd_detach_uv_lr0001_no_time_encoding
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
python3 train.py --config configs/config_mc_8spf_lr0001.yaml --exp_name lsfp_mc_8spf_svd_detach_uv_lr0001_no_time_encoding