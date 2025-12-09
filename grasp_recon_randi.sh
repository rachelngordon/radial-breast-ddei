#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/grasp_slice_largest_region.err
#SBATCH --output=logs/grasp_slice_largest_region.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=grasp_slice_largest_region
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
python grasp_recon_largest_slices.py   --csv_path data/largest_tumor_slices.csv   --data_dir /ess/scratch/scratch1/rachelgordon/zf_data_192_slices/zf_kspace   --spokes_per_frame 2 4 8 16 24 36   --total_spokes 288