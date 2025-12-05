#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=1
#SBATCH --error=logs/no_interpolate.err
#SBATCH --output=logs/no_interpolate.out
#SBATCH --gpus-per-node=1
#SBATCH --job-name=no_interpolate
#SBATCH --mem-per-gpu=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general


echo "--- DIAGNOSTICS START ---"

# Check initial state
echo "Initial PATH: $PATH"
echo "Which micromamba before source: $(which micromamba)"

# Load Micromamba using the recommended shell hook method
# This is more reliable for non-interactive scripts
eval "$(micromamba shell hook -s bash)"

# Check after sourcing
echo "Micromamba path after hook: $(which micromamba)"

# Activate your Micromamba environment
micromamba activate mri_recon_new

# --- CRITICAL DIAGNOSTIC STEPS ---
echo "CONDA_PREFIX after activate: $CONDA_PREFIX"
echo "PATH after activate: $PATH"
echo "Which python: $(which python)"
echo "Which pip: $(which pip)"

echo "Attempting to import cupy with python:"
python -c "import cupy; print('CuPy imported successfully. Path:', cupy.__file__)"

echo "--- DIAGNOSTICS END ---"


# If the diagnostics above work, then you can run your script
# If they fail, the problem is in the setup above.
echo "Running the main script..."
bash process_all_data.sh /net/scratch2/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_ /net/scratch2/rachelgordon/non_zf_data_83_slices 83