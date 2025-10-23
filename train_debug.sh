#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/test_debug.err
#SBATCH --output=logs/test_debug.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test_debug
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
python3 train_zf.py --config configs/config_mc_zf_debug.yaml --exp_name debug_plots