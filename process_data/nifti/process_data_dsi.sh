#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=1
#SBATCH --error=logs/zero_pad_data_200.err
#SBATCH --output=logs/zero_pad_data_200.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=zero_pad_data_200
#SBATCH --mem-per-gpu=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=general
#SBATCH --time=700


# Load Micromamba
source /home/rachelgordon/micromamba/etc/profile.d/mamba.sh

# Activate your Micromamba environment
micromamba activate recon_mri

# Run the training script with srun
bash process_all_data.sh /net/scratch2/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_ /net/scratch2/rachelgordon/zf_data_192_slices 192
