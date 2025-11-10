#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/rebuild_missing_grasp.err
#SBATCH --output=logs/rebuild_missing_grasp.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=rebuild_missing_grasp
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

python rebuild_missing_grasp.py --slices 192 --data_dir /ess/scratch/scratch1/rachelgordon/zf_data_192_slices/zf_kspace --split_json /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json --spokes 2,4,16,24,36 --missing_csv /ess/scratch/scratch1/rachelgordon/zf_data_192_slices/missing_grasp_val_raw.csv --id_suffix _2