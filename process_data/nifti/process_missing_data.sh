#!/bin/bash

# Define the base paths
BASE_PATH="/ess/scratch/scratch1/rachelgordon/fastMRI_breast_data"
OUTPUT_PATH="/ess/scratch/scratch1/rachelgordon/dce-8tf"

# Define the list of directory names
# DIRS=('fastMRI_breast_006_2' 'fastMRI_breast_156_2' 'fastMRI_breast_159_2' 
#       'fastMRI_breast_240_2' 'fastMRI_breast_279_2' 'fastMRI_breast_281_2' 
#       'fastMRI_breast_299_2')
DIRS=('fastMRI_breast_006_2' 'fastMRI_breast_279_2')

# Loop through each directory and run the command
for DIR in "${DIRS[@]}"; do
    # Extract the patient ID and remove leading zeros
    PATIENT_ID=$(echo "$DIR" | awk -F'_' '{print $3}' | sed 's/^0*//')

    # Compute range start and end
    RANGE_START=$(( (PATIENT_ID - 1) / 10 * 10 + 1 ))
    RANGE_END=$(( RANGE_START + 9 ))

    # Format range ID with leading zeros
    RANGE_ID=$(printf "%03d_%03d" "$RANGE_START" "$RANGE_END")

    # Construct the full path to the .h5 file
    H5_PATH="$BASE_PATH/fastMRI_breast_IDS_${RANGE_ID}/${DIR}.h5"

    # Run the command
    bash loop_single_data_nifti.sh "$H5_PATH" "$OUTPUT_PATH" 2 True 36 0 90 100 True
done

