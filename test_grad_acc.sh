#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/test_grad_acc.err
#SBATCH --output=logs/test_grad_acc.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test_grad_acc
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
python3 train_acc.py --config configs/config_acc4_ei_spatial_zf.yaml --exp_name test_grad_acc