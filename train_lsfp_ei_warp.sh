#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/lsfpnet_ei_eval_warp.err
#SBATCH --output=logs/lsfpnet_ei_eval_warp.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=lsfpnet_ei_eval_warp
#SBATCH --mem-per-gpu=100000
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
python3 train.py --config configs/config_ei_warp.yaml --exp_name lsfpnet_ei_eval_warp