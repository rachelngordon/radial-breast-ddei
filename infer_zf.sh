#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/test_infer.err
#SBATCH --output=logs/test_infer.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test_infer
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
python infer_zf.py \
  --config output/zero_pad_ei_spatial/config.yaml \
  --checkpoint output/zero_pad_ei_spatial/zero_pad_ei_spatial_model.pth \
  --exp_name zero_pad_ei_spatial \
  --patient_id fastMRI_breast_015 \
  --slice_idx 0 \
  --spokes_per_frame 8 \
  --num_frames 36 \
  --chunk_size 24 \
  --chunk_overlap 12 \
  --norm frame \
  --device cuda:0
