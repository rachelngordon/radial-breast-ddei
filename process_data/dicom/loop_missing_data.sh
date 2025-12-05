#!/bin/bash

# List of patient directories
# need to redownload these files
patients=(
    'fastMRI_breast_004_1' 'fastMRI_breast_014_1' 'fastMRI_breast_032_1' 'fastMRI_breast_040_1' 'fastMRI_breast_050_1' 'fastMRI_breast_053_1' 'fastMRI_breast_057_1' 'fastMRI_breast_073_1' 'fastMRI_breast_091_1' 'fastMRI_breast_096_1'
)

# Base directory
base_dir="/ess/scratch/scratch1/rachelgordon/fastMRI_breast_data"

# Function to determine the parent directory based on the patient ID
get_parent_dir() {
    local patient="$1"
    # Extract the numeric ID from the patient name
    patient_id=$(echo "$patient" | grep -o '[0-9]\+' | head -n 1 | sed 's/^0*//')

    if [[ -z "$patient_id" ]]; then
        echo "Error: Invalid patient name format: $patient"
        return 1
    fi

    # Calculate the start and end ID
    start_id=$(( (patient_id - 1) / 10 * 10 + 1 ))
    end_id=$(( start_id + 9 ))

    # Construct the parent directory name
    printf "fastMRI_breast_IDS_%03d_%03d" "$start_id" "$end_id"
}

# Main loop to process each patient directory
for patient in "${patients[@]}"; do
    echo "Processing patient: $patient"

    # Get the parent directory
    parent_dir=$(get_parent_dir "$patient")
    if [[ $? -ne 0 ]]; then
        echo "Skipping patient: $patient due to error"
        continue
    fi

    # Construct the input file path
    input_file="${base_dir}/${parent_dir}/${patient::-2}_1.h5"

    # Define output directory
    output_dir="/ess/scratch/scratch1/rachelgordon/pre-contrast-1tf"

    # Print the command being executed
    echo "Running: bash loop_single_data_nifti.sh $input_file $output_dir True 1 288 0 192 100"

    # Execute the command
    bash loop_single_data_nifti.sh "$input_file" "$output_dir" "True" "1" "288" "0" "192" "100"

    # Check for errors in execution
    if [[ $? -ne 0 ]]; then
        echo "Error processing $patient"
    else
        echo "Successfully processed $patient"
    fi
done
