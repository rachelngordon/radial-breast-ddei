import os
import numpy as np
from raw_kspace_eval import raw_grasp_recon, process_kspace
import torch
import sigpy as sp
import json
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train ReconResNet model.")
parser.add_argument(
    "--slices",
    type=int,
    required=False,
    default=1
)
parser.add_argument(
    "--total_spokes",
    type=int,
    required=False,
    default=288
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=False,
    default="/ess/scratch/scratch1/rachelgordon/zf_data_192_slices/zf_kspace",
)
parser.add_argument(
    "--split_file",
    type=str,
    required=False,
    default="/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json",
)
parser.add_argument(
    "--slice_idx",
    type=int,
    required=True,
)
args = parser.parse_args()


spokes_per_frame = [2, 4, 8, 16, 24, 36]
sp_device = sp.Device(0 if torch.cuda.is_available() else -1)


# load data
with open(args.split_file, "r") as fp:
    splits = json.load(fp)


val_patient_ids = splits["val"]


for spf in spokes_per_frame:

    num_frames = int(args.total_spokes / spf)

    for patient_id in val_patient_ids:

        raw_kspace_path = os.path.join(args.data_dir, f'{patient_id}_2.h5')

        dir = os.path.dirname(args.data_dir)
        zf_kspace, binned_kspace, traj = process_kspace(raw_kspace_path, device=sp_device, spokes_per_frame=spf, images_per_slab=args.slices, center_partition=31)

        zf_kspace = np.expand_dims(zf_kspace[args.slice_idx], axis=0)
        binned_kspace = np.expand_dims(binned_kspace[args.slice_idx], axis=0)

        grasp_img_path = os.path.join(dir, f'{patient_id}_2', f'grasp_recon_{spf}spf_{num_frames}frames_slice{args.slice_idx}.npy')

        grasp_img_slices = raw_grasp_recon(zf_kspace, binned_kspace, traj, N_slices=args.slices, spokes_per_frame=spf, device=sp_device)
        np.save(grasp_img_path, grasp_img_slices)
        print(f"GRASP img saved to {grasp_img_path} with {spf} spokes per frame")