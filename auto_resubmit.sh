#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/auto_resubmit.err
#SBATCH --output=logs/auto_resubmit.out
#SBATCH --job-name=auto_resubmit
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=50000
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --time=100
#
# Instruct Slurm to requeue the job if it fails (e.g., due to time limit)
#SBATCH --requeue
#
# Tell Slurm to send the SIGTERM signal 60 seconds before the time limit
#SBATCH --signal=B:TERM@60

# --- Signal Handler ---
# This function will run when Slurm sends the SIGTERM signal
handle_timeout() {
    echo "Caught SIGTERM signal at $(date). Preparing for graceful shutdown and requeue."

    # Signal your Python script to save a checkpoint.
    # Creating a file is a simple way to do this.
    touch SAVE_AND_EXIT

    echo "Gave Python the signal to save. Waiting for 55 seconds..."
    # This sleep gives your Python script time to detect the file,
    # save a checkpoint, and exit cleanly.
    sleep 55

    echo "Time's up. The job will now terminate and be requeued by Slurm."
    # The script will now exit. Because it was terminated by a signal,
    # it will have a non-zero exit code, which will trigger the --requeue directive.
}

# --- Main Script ---

# Register the signal handler to catch the TERM signal
trap 'handle_timeout' TERM

# Load Micromamba
source /gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/etc/profile.d/micromamba.sh
micromamba activate recon_mri

# Slurm's --requeue feature automatically provides SLURM_RESTART_COUNT.
# It is 0 on the first run and increments on subsequent requeues.
if [ -z "$SLURM_RESTART_COUNT" ] || [ "$SLURM_RESTART_COUNT" -eq 0 ]; then
    echo "This is the first run of the job."
    FROM_CHECKPOINT_FLAG=""
else
    echo "This is restart number $SLURM_RESTART_COUNT. Loading from checkpoint."
    FROM_CHECKPOINT_FLAG="--from_checkpoint True"
fi

echo "Starting Python training script..."
# Run the training script in the background
python3 train.py --config configs/config_ei_diffeo_encode_af.yaml --exp_name test_auto_resubmit_job $FROM_CHECKPOINT_FLAG &

# Store the Process ID (PID) of the Python script
PYTHON_PID=$!

# --- Interruptible Wait Loop ---
# Wait for the python process to finish, checking every 5 seconds.
# This loop allows the 'trap' to be triggered by signals.
while kill -0 $PYTHON_PID 2>/dev/null; do
    sleep 5
done

echo "Python script finished on its own."