#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/lsfp_ei_warp_aug_encoding_no_reg.err
#SBATCH --output=logs/lsfp_ei_warp_aug_encoding_no_reg.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=lsfp_ei_warp_aug_encoding_no_reg
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
python3 train_fpg.py --config configs/config_ei_warp_aug_enc.yaml --exp_name lsfp_ei_warp_aug_encoding_no_reg