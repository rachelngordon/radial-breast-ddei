#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/ei_rotate_encode_time.err
#SBATCH --output=logs/ei_rotate_encode_time.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ei_rotate_encode_time
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
python3 train.py --config configs/config_ei_rotate_encode_time.yaml --exp_name ei_rotate_encode_time --from_checkpoint True