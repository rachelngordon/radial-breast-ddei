#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/lsfp_ei_warp_40slices_aug_encoding_no_reg.err
#SBATCH --output=logs/lsfp_ei_warp_40slices_aug_encoding_no_reg.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=4
#SBATCH --job-name=lsfp_ei_warp_40slices_aug_encoding_no_reg
#SBATCH --mem-per-gpu=60000
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
NCCL_SOCKET_IFNAME=ib0 torchrun --nproc_per_node=4 train_distributed.py --config configs/config_ei_warp_slices_aug_enc.yaml --exp_name lsfp_ei_warp_40slices_aug_encoding_no_reg