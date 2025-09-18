#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/submitit_logs/mc_baseline_submitit/%j_0_log.err
#SBATCH --gpus-per-node=2
#SBATCH --job-name=mc_baseline_submitit
#SBATCH --mem-per-gpu=50000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/submitit_logs/mc_baseline_submitit/%j_0_log.out
#SBATCH --partition=gpuq
#SBATCH --signal=USR2@90
#SBATCH --time=1
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/submitit_logs/mc_baseline_submitit/%j_%t_log.out --error /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/submitit_logs/mc_baseline_submitit/%j_%t_log.err /gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/envs/recon_mri/bin/python -u -m submitit.core._submit /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/submitit_logs/mc_baseline_submitit
