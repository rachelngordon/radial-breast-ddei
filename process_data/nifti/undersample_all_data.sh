#!/bin/bash

# Base path for the directories
BASE_PATH="/ess/scratch/scratch1/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_"

# Directory range (e.g., 001_010 to 291_300)
START_DIR="001_010"
END_DIR="291_300"

# Convert ranges to numerical values for looping
START_NUM=$(echo $START_DIR | cut -d'_' -f1 | sed 's/^0*//')
END_NUM=$(echo $END_DIR | cut -d'_' -f2 | sed 's/^0*//')

# Function to format numbers with leading zeros
format_number() {
    printf "%03d" $1
}

# Loop through each directory range
for ((i = START_NUM; i <= END_NUM; i += 10)); do

    # Format the current range
    DIR_START=$(format_number $i)
    DIR_END=$(format_number $((i + 9))) # Adjusted to include correct range
    DIR_SUFFIX="${DIR_START}_${DIR_END}"

    # Construct full directory path
    DATA_PATH="${BASE_PATH}${DIR_SUFFIX}"

    # Debug output
    echo "Formatted directory range: ${DIR_START}_${DIR_END}"
    echo "Constructed data path: $DATA_PATH"

    # Check if directory exists
    if [ -d "$DATA_PATH" ]; then
        echo "Directory exists: $DATA_PATH"
    else
        echo "Warning: Directory does not exist: $DATA_PATH"
        continue
    fi

    # Run the command with the current directory path and other arguments
    #bash loop_single_dir_nifti.sh "$DATA_PATH" "/ess/scratch/scratch1/rachelgordon/reconresnet/undersampled_kspace_29sp" 72 0 192 10
    bash loop_single_dir_nifti.sh "$DATA_PATH" /ess/scratch/scratch1/rachelgordon/reconresnet/undersampled_kspace_29sp 2 False 72 0 192 10
done
