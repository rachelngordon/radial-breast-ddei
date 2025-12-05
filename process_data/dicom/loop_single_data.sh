#/bin/bin -l

# to run this code use the following command: bash loop_single_data_nifti.sh [DATA_DIR] [OUT_DIR] [SPOKES_PER_FRAME] [SLICE_IDX] [NUM_SLICES] [PER_SPOKES]
# ex: bash loop_single_data.sh /ess/scratch/scratch1/rachelgordon/fastMRI_breast_IDS_001_010/fastMRI_breast_001_2.h5 72 0 192 10

# Check if DIR is provided as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <PATH> [SPOKES] [SLICE_IDX] [SLICE_INC] [PER_SPOKES]"
    exit 1
fi

# Set DIR to the provided arguments
DIR=$(dirname "$1")
DATA=$(basename "$1")



# if [ -z "$1" ]; then
#     echo "Find raw .h5 files in the current directory: $(pwd)"
#     DIR = "fastMRI_breast_001_1"
# else
#     echo "Find raw .h5 files in: $1"
#     DIR="$1"
# fi

#DATA=$(find "${DIR}" -maxdepth 1 -type f -name ${DIR}".h5" -exec basename {} \;)

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
    NUM='^[0-9]+$'

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
    NUM='^[0-9]+$'

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
        if [ "$5" -ge 0 ] && [ "$6" -le 100 ]; then
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

    echo "> DIR: ${DIR}"
    echo "> DATA: ${DATA}"
    echo "> SPOKES: ${SPOKES}"
    echo "> SLICE_IDX: ${SLICE_IDX}"
    echo "> SLICE_INC: ${SLICE_INC}"
    echo "> PER_SPOKES: ${PER_SPOKES}"

    # Check if SLICE_IDX is equal to zero
    if [ "$SLICE_IDX" -ne 0 ]; then  
        ((SLICE_IDX--))
        echo "SLICE_IDX is now $SLICE_IDX"
    fi 
    # reconstruct slice by slice
    python dce_recon.py --dir "${DIR}" --data "${DATA}" --spokes_per_frame "${SPOKES}" --slice_idx "${SLICE_IDX}" --slice_inc "${SLICE_INC}" --per_spokes "${PER_SPOKES}"

    # convert the .h5 file to nifti
    FN="${DIR}/${DATA%.*}"
    python dcm_recon.py --dir "${DIR}" --h5py "${FN}" --spokes_per_frame "${SPOKES}"