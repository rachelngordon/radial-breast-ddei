#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/grasp_recon_no_zf_for_eval.err
#SBATCH --output=logs/grasp_recon_no_zf_for_eval.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=grasp_recon_no_zf_for_eval
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
python3 grasp_recon_for_val.py --slices 83 --data_dir /ess/scratch/scratch1/rachelgordon/non_zf_data_83_slices/zf_kspace