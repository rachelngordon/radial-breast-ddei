#!/bin/bash

# Directory containing the dataset
SCRATCH_DIR="/ess/scratch/scratch1/rachelgordon/"

# Log file to track when the script runs
LOG_FILE="/gpfs/data/karczmar-lab/workspaces/rachelgordon/touch_files.log"

# Check if the scratch directory exists
if [ ! -d "$SCRATCH_DIR" ]; then
    echo "Error: Scratch directory $SCRATCH_DIR does not exist!" | tee -a "$LOG_FILE"
    exit 1
fi

# Log the start time
echo "$(date): Touching files in $SCRATCH_DIR" | tee -a "$LOG_FILE"

# Find all files and directories in the scratch directory and update their timestamps
find "$SCRATCH_DIR" -exec touch {} \;

# Log the completion time
echo "$(date): Finished touching files in $SCRATCH_DIR" | tee -a "$LOG_FILE"