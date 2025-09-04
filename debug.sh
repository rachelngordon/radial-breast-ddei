#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/debug_new_block.err
#SBATCH --output=logs/debug_new_block.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=debug_new_block
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

python3 train.py --config configs/config_mc_8spf.yaml --exp_name debug_new_block