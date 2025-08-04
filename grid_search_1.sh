#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/losses_grid_search_1.err
#SBATCH --output=logs/losses_grid_search_1.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=losses_grid_search_1
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
python3 grid_search_batch.py --total-batches 3 --current-batch 1