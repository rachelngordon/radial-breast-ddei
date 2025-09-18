#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/mc_baseline_multigpu.err
#SBATCH --output=logs/mc_baseline_multigpu.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=2
#SBATCH --job-name=mc_baseline_multigpu
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
# python3 new_train.py --config configs/config_mc_dist.yaml --exp_name test_dist
torchrun --nproc_per_node=2 new_train.py --config configs/config_no_encoding_detach_uv.yaml --exp_name mc_baseline_multigpu