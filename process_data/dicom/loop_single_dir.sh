#!/bin/bash

# to run this code use the following command: bash loop_single_dir.sh [DATA_DIR] [SPOKES_PER_FRAME] [SLICE_IDX] [NUM_SLICES] [PER_SPOKES]

# Check if DIR is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <DIR> [SPOKES] [SLICE_IDX] [SLICE_INC] [PER_SPOKES]"
    exit 1
fi

# Set DIR to the provided argument
DIR="$1"

    
# Set SPOKES, SLICE_IDX, and SLICE_INC with default values or user-provided values
if [ -z "$2" ]; then
    echo "> SPOKES set as default: 72"
    SPOKES=72
else
    NUM='^[0-9]+$'
    if [[ $2 =~ $NUM ]]; then
        SPOKES="$2"
    else
        echo "> Input $2 is not a number. SPOKES set as default: 72"
        SPOKES=72
    fi
fi

if [ -z "$3" ]; then
    echo "> SLICE_IDX set as default: 0"
    SLICE_IDX=0
else
    if [[ $3 =~ $NUM ]]; then
        SLICE_IDX="$3"
    else
        echo "> Input $3 is not a number. SLICE_IDX set as default: 0"
        SLICE_IDX=0
    fi
fi

if [ -z "$4" ]; then
    echo "> SLICE_INC set as default: 192"
    SLICE_INC=192
else
    if [[ $4 =~ $NUM ]]; then
        SLICE_INC="$4"
    else
        echo "> Input $4 is not a number. SLICE_INC set as default: 192"
        SLICE_INC=192
    fi
fi

if [ -z "$5" ]; then
    echo "> PER_SPOKES set as default: 100"
    PER_SPOKES=100
else
    NUM='^[0-9]+$'
    if [[ $5 =~ $NUM ]]; then
        if [ "$5" -ge 0 ] && [ "$5" -le 100 ]; then
            PER_SPOKES="$5"
        else
            echo "> Input $5 is not a valid percentage (0-100). PER_SPOKES set as default: 100"
            PER_SPOKES=100
        fi
    else
        echo "> Input $5 is not a number. PER_SPOKES set as default: 100"
        PER_SPOKES=100
    fi
fi


# Loop through all .h5 files ending in _2.h5
for FILE in "${DIR}"/*_2.h5; do
    # Extract the basename of the file
    #DATA=$(basename "$FILE")

    echo "> DIR: ${DIR}"
    echo "> DATA: ${DATA}"
    echo "> SPOKES: ${SPOKES}"
    echo "> SLICE_IDX: ${SLICE_IDX}"
    echo "> SLICE_INC: ${SLICE_INC}"
    echo "> PER_SPOKES: ${PER_SPOKES}"

    # Check if SLICE_IDX is equal to zero
    if [ "$SLICE_IDX" -ne 0 ]; then  
        ((SLICE_IDX--))
    fi 

    # Reconstruct slice by slice
    python dce_recon.py --dir "${DIR}" --data "${DATA}" --spokes_per_frame "${SPOKES}" --slice_idx "${SLICE_IDX}" --slice_inc "${SLICE_INC}" --per_spokes "${PER_SPOKES}"


    # # Extract the base path from DIR
    # BASE_DIR=$(dirname "$DIR")

    # # Determine the ID range from the directory name and set dcm_dir
    # if [[ $DIR =~ fastMRI_breast_IDS_([0-9]+)_([0-9]+) ]]; then
    #     START_ID=${BASH_REMATCH[1]}
    #     END_ID=${BASH_REMATCH[2]}
        
    #     if [ "$START_ID" -le 150 ]; then
    #         DCM_DIR="${BASE_DIR}/fastMRI_breast_IDS_001_150_DCM"
    #     else
    #         DCM_DIR="${BASE_DIR}/fastMRI_breast_IDS_151_300_DCM"
    #     fi
    # else
    #     echo "DIR does not match expected format. Exiting."
    #     exit 1
    # fi

    # Convert the .h5 file to dicom
    FN="${DIR}/${DATA%.*}"
    python dcm_recon.py --dir "${DIR}" --h5py "${FN}" --spokes_per_frame "${SPOKES}"

    # temporarily exit for testing
    exit
done
