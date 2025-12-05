#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=process_missing_data.err
#SBATCH --output=process_missing_data.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=process_missing_data
#SBATCH --mem-per-gpu=120000
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
bash process_missing_data.sh