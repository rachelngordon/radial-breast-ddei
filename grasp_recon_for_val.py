import os
import numpy as np
from raw_kspace_eval import raw_grasp_recon, process_kspace
import torch
import sigpy as sp
import json

N_slices = 192
spokes_per_frame = 8

sp_device = sp.Device(0 if torch.cuda.is_available() else -1)

split_file = "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json"
data_dir = "/ess/scratch/scratch1/rachelgordon/zf_data_192_slices/zf_kspace"


# load data
with open(split_file, "r") as fp:
    splits = json.load(fp)


val_patient_ids = splits["val"]


for patient_id in val_patient_ids:

    raw_kspace_path = os.path.join(data_dir, f'{patient_id}_2.h5')

    dir = os.path.dirname(data_dir)
    zf_kspace, binned_kspace, traj = process_kspace(raw_kspace_path, device=sp_device, spokes_per_frame=spokes_per_frame, images_per_slab=N_slices, center_partition=31)

    grasp_img_path = os.path.join(dir, f'{patient_id}_2', f'grasp_recon_{spokes_per_frame}spf.npy')

    if not os.path.exists(grasp_img_path):
        grasp_img_slices = raw_grasp_recon(zf_kspace, binned_kspace, traj, N_slices=N_slices, spokes_per_frame=spokes_per_frame, device=sp_device)
        np.save(grasp_img_path, grasp_img_slices)
        print("GRASP img saved to: ", grasp_img_path)
    else:
        print(f"GRASP img path {grasp_img_path} already exists, moving to next patient ID")