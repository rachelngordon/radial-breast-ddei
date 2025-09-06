#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/encode_af_detach_uv_no_lowk_dc.err
#SBATCH --output=logs/encode_af_detach_uv_no_lowk_dc.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=encode_af_detach_uv_no_lowk_dc
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
python3 train.py --config configs/config_encode_af_detach_uv.yaml --exp_name encode_af_detach_uv_no_lowk_dc --from_checkpoint True