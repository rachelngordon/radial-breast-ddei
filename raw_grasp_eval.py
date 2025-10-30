import json
import os
import h5py
import numpy as np
import sigpy as sp
import torch
from sigpy.mri import app
import cupy as cp
from utils import prep_nufft, to_torch_complex, GRASPRecon, sliding_window_inference, load_checkpoint
import yaml
from lsfpnet_encoding import LSFPNet, ArtifactRemovalLSFPNet
import math
from radial_lsfp import MCNUFFT
import time
from einops import rearrange
import matplotlib.pyplot as plt
from eval import calc_dc
import glob
import random


import json
import os
import matplotlib.pyplot as plt
import torch
import yaml
from dataloader import ZFSliceDataset, SimulatedDataset, SimulatedSPFDataset
from einops import rearrange
from torch.utils.data import DataLoader
import numpy as np
from utils import prep_nufft, log_gradient_stats, plot_enhancement_curve, get_cosine_ei_weight, plot_reconstruction_sample, get_git_commit, save_checkpoint, load_checkpoint, to_torch_complex, GRASPRecon, sliding_window_inference, set_seed
import seaborn as sns

import h5py
import numpy as np

def get_coil(ksp, spokes_per_frame, device=sp.Device(-1), traj=None):
    """
    Estimate ESPIRiT coil sensitivity maps from ONE slice k-space.
    Args:
        ksp: (C, Spokes, Samples) complex64
        spokes_per_frame: int
        device: sigpy device
        traj: (T, spf, Samples, 2) float32 OR (Spokes, Samples, 2). If None, it is built.
    Returns:
        mps: (C, H, W) complex64
    """
    assert ksp.ndim == 3, "ksp should be (C, Spokes, Samples) for a single slice."
    N_coils, N_spokes, N_samples = ksp.shape
    base_res = N_samples // 2
    ishape = [N_coils, base_res, base_res]

    # If a (T, spf, S, 2) trajectory was passed, flatten it back to (Spokes, S, 2)
    if traj is not None:
        if traj.ndim == 4:
            T, spf, S, _ = traj.shape
            assert S == N_samples, "traj samples must match k-space samples"
            traj_use = traj.reshape(T * spf, S, 2)
        else:
            traj_use = traj
    else:
        # Build a single-frame traj with all spokes (no binning) for coil maps
        traj_use = get_traj(N_spokes, csmaps=True, N_spokes=N_spokes, N_time=1,
                            base_res=base_res, gind=1).reshape(N_spokes, N_samples, 2).astype(np.float32)

    # Ramp DCF, normalized
    dcf = np.sqrt(traj_use[..., 0]**2 + traj_use[..., 1]**2).astype(np.float32)
    dcf = dcf / max(np.mean(dcf), 1e-8)

    F = sp.linop.NUFFT(ishape, traj_use)

    # Adjoint NUFFT gives coil images already; DO NOT FFT again
    cim = F.H(ksp * dcf)  # (C, H, W), complex

    mps = app.EspiritCalib(cim, device=device).run()
    mps = sp.to_device(mps)  # move off GPU if needed
    return mps


def get_traj(spokes_per_frame, csmaps=False, N_spokes=13, N_time=1, base_res=320, gind=1):
    """
    Return golden-angle radial trajectory as (T, spf, N_samples, 2), float32.
    NOTE: No more np.squeeze() or 'special case' shapes—always consistent.
    """
    N_tot_spokes = N_spokes * N_time
    N_samples = base_res * 2

    base_lin = np.arange(N_samples, dtype=np.float32).reshape(1, -1) - float(base_res)

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = (np.arange(N_tot_spokes, dtype=np.float32).reshape(-1, 1) * base_rad).astype(np.float32)

    traj = np.zeros((N_tot_spokes, N_samples, 2), dtype=np.float32)
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    # keep scaling as in your original; if you later standardize to match prep_nufft, adjust here.
    traj = traj / 2.0

    traj = traj.reshape(N_time, N_spokes, N_samples, 2).astype(np.float32, copy=False)
    return traj


def raw_grasp_recon(ksp_zf, ksp_prep, traj, N_slices, spokes_per_frame, device):
    """
    Single-slice GRASP recon. Ignores N_slices and reconstructs ONE slice
    from (ksp_zf, ksp_prep, traj). Returns (H, W, T) complex64.
    """
    # coil maps from full set of used spokes
    C = get_coil(ksp_zf, spokes_per_frame, device=device, traj=traj)  # (C, H, W)
    C = C[:, None, :, :]  # (C, 1, H, W) for HighDimensionalRecon

    # HighDimensionalRecon expects k-space as (T, 1, C, 1, spf, Samples)
    # We currently have (T, spf, C, Samples)
    k1 = np.transpose(ksp_prep, (0, 2, 1, 3))  # -> (T, C, spf, Samples)
    k1 = k1[:, None, :, None, :, :]            # -> (T, 1, C, 1, spf, Samples)

    # Run recon
    R = app.HighDimensionalRecon(
        k1, C,
        combine_echo=False,
        lamda=0.001,
        coord=traj,               # (T, spf, Samples, 2)
        regu='TV', regu_axes=[0],
        max_iter=10,
        solver='ADMM', rho=0.1,
        device=device,
        show_pbar=False, verbose=False
    ).run()

    # R usually comes back as (T, 1, 1, H, W) complex
    R_cpu = sp.to_device(R, sp.cpu_device)
    R_np = np.asarray(R_cpu)  # keep dtype (usually complex64)

    # Squeeze and arrange to (H, W, T)
    while R_np.ndim > 3:
        R_np = np.squeeze(R_np, axis=1)  # drop singleton dims iteratively

    # Now R_np is (T, H, W) → (H, W, T)
    if R_np.shape[0] != traj.shape[0]:
        # if axes came out differently, try best-effort squeeze again
        R_np = np.squeeze(R_cpu)
        # assume (T, H, W) if size matches
    R_np = np.transpose(R_np, (1, 2, 0)).astype(np.complex64, copy=False)

    return R_np  # (H, W, T) complex64


def process_kspace_slice(
    kspace_path,
    slice_idx,
    spokes_per_frame,
    images_per_slab=192,
    center_partition=31,
):
    """
    Load and process ONE slice of raw radial k-space from fastMRI-breast file.
    Returns:
        zf_kspace_slice     : (coils, spokes, samples)
        binned_kspace_slice : (T, spokes_per_frame, coils, samples)
        traj_slice          : (T, spokes_per_frame, samples, 2)
    """

    # -------- Load only 1 slice from HDF5 --------
    with h5py.File(kspace_path, 'r') as f:
        # dataset is shape (slices, coils, spokes, samples)
        # load only slice_idx instead of [:]
        ksp_slice = f['kspace'][slice_idx]  # (coils, spokes, samples)

    N_coils, N_spokes, N_samples = ksp_slice.shape
    base_res = N_samples // 2

    # -------- Compute # time frames --------
    N_time = N_spokes // spokes_per_frame
    N_spokes_used = N_time * spokes_per_frame

    # -------- Truncate to divisible number of spokes --------
    ksp_slice = ksp_slice[:, :N_spokes_used, :]  # (coils, spokes, samples)

    # -------- Retrospective binning --------
    # reshape to (time, spf, coils, samples)
    ksp_slice_reshaped = ksp_slice.reshape(
        N_coils,
        N_time,
        spokes_per_frame,
        N_samples
    )
    ksp_slice_reshaped = np.transpose(ksp_slice_reshaped, (1, 2, 0, 3))  # (T, spf, coils, samples)

    # -------- Build trajectory --------
    traj = get_traj(
        spokes_per_frame,
        N_spokes=spokes_per_frame,
        N_time=N_time,
        base_res=base_res,
        gind=1
    )  # (T, spf, samples, 2)

    return ksp_slice, ksp_slice_reshaped, traj

import os, glob
import numpy as np

def load_csmap_slice(root_dir, patient_id, slice_idx):
    """
    Load coil-sensitivity maps for ONE slice for a given patient.
    
    Args:
        root_dir (str): directory containing cs_maps folder
        patient_id (str): like 'fastMRI_breast_001_2'
        slice_idx (int): slice index to load
        
    Returns:
        np.ndarray shaped (1, C, H, W)
            batch=1, coils=C, height=H, width=W
    """
    csmap_dir = os.path.join(root_dir, "cs_maps", f"{patient_id}_cs_maps")

    # find all slice files
    slice_paths = sorted(glob.glob(os.path.join(csmap_dir, "cs_map_slice_*.npy")))
    
    if not slice_paths:
        raise FileNotFoundError(f"No cs maps found at {csmap_dir}")

    if slice_idx >= len(slice_paths):
        raise IndexError(f"Requested slice {slice_idx}, but only {len(slice_paths)} available.")

    # load ONLY this slice
    csmap_slice = np.load(slice_paths[slice_idx])  # (H, W, C)

    # ensure consistent axis order: (C, H, W)
    csmap_slice = np.moveaxis(csmap_slice, -1, 0)  # (C, H, W)

    # add batch dimension → (1, C, H, W)
    csmap_slice = csmap_slice[None, ...]

    return csmap_slice.astype(np.complex64)




def eval_raw_kspace_grasp(num_slices_to_eval, val_patient_ids, data_dir, spokes_per_frame, N_slices, num_frames, physics, device):
    
    sp_device = sp.Device(0 if torch.cuda.is_available() else -1)
    dtype = torch.complex64

    # select random slices to evaluate on
    # NOTE: fix after testing
    random_slice_indices = random.sample(range(N_slices), num_slices_to_eval)
    # random_slice_indices = [1]

    # NOTE: temporarily set val_patient_ids for testing
    # val_patient_ids = ['fastMRI_breast_001']
    
    avg_grasp_dc_mses = []
    avg_grasp_dc_maes = []
    with torch.no_grad():
        for patient_id in val_patient_ids:

            raw_kspace_path = os.path.join(data_dir, f'{patient_id}_2.h5')

            dir = os.path.dirname(data_dir)
            zf_kspace, binned_kspace, traj = process_kspace_slice(raw_kspace_path, random_slice_indices[0], spokes_per_frame, images_per_slab=N_slices)

            grasp_img_path = os.path.join(dir, f'{patient_id}_2', f'grasp_recon_{spokes_per_frame}spf.npy')

            if not os.path.exists(grasp_img_path):
                grasp_img_slices = raw_grasp_recon(zf_kspace, binned_kspace, traj, N_slices=1, spokes_per_frame=spokes_per_frame, device=sp_device)
                np.save(grasp_img_path, grasp_img_slices)
            else:
                grasp_img_slices = np.load(grasp_img_path)
            

            csmap = load_csmap_slice(dir, f'{patient_id}_2', random_slice_indices[0])

            grasp_slice_dc_mses = []
            grasp_slice_dc_maes = []

            kspace_slice = torch.tensor(binned_kspace.squeeze())
            grasp_img_slice = torch.tensor(grasp_img_slices)
            csmap_slice = torch.tensor(csmap)

            # for slice_idx in random_slice_indices:

            #     kspace_slice = torch.tensor(binned_kspace[slice_idx].squeeze())
            #     grasp_img_slice = torch.tensor(grasp_img_slices[slice_idx])
            #     csmap_slice = torch.tensor(csmap[..., slice_idx])

            kspace_slice_flat = rearrange(kspace_slice, 't c sp sam -> c (sp sam) t').to(dtype)
            csmap_slice = csmap_slice.to(dtype)


            # calculate data consistency of output with original k-space input
            print("grasp_img: ", grasp_img_slice.shape)
            
            if grasp_img_slice.shape[1] == 2:
                grasp_img_slice = to_torch_complex(grasp_img_slice)

            if grasp_img_slice.shape[-2] == num_frames: 
                grasp_img_slice = rearrange(grasp_img_slice, 'b h t w -> b h w t')


            grasp_img_slice = rearrange(grasp_img_slice, 't h w -> h w t').unsqueeze(0)

            sim_kspace_grasp = physics(False, grasp_img_slice.to(csmap_slice.dtype).to(device), csmap_slice.to(device))

            raw_grasp_dc_mse, raw_grasp_dc_mae = calc_dc(sim_kspace_grasp, kspace_slice_flat, device)


            grasp_slice_dc_mses.append(raw_grasp_dc_mse)
            grasp_slice_dc_maes.append(raw_grasp_dc_mae)



        avg_grasp_dc_mses.append(np.mean(grasp_slice_dc_mses))
        avg_grasp_dc_maes.append(np.mean(grasp_slice_dc_maes))

    

    avg_grasp_mse = np.mean(avg_grasp_dc_mses)
    avg_grasp_mae = np.mean(avg_grasp_dc_maes)

    std_grasp_mse = np.std(avg_grasp_dc_mses)
    std_grasp_mae = np.std(avg_grasp_dc_maes)



    return avg_grasp_mse, avg_grasp_mae, std_grasp_mse, std_grasp_mae





# load data
split_file = "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json"
with open(split_file, "r") as fp:
    splits = json.load(fp)

val_patient_ids = splits["val"]
val_dro_patient_ids = splits["val_dro"]


root_dir = "/ess/scratch/scratch1/rachelgordon/dro_dataset"
model_type = "LSFPNet"

eval_spf_dataset = SimulatedSPFDataset(
    root_dir=root_dir, 
    model_type=model_type, 
    patient_ids=val_dro_patient_ids,
    )


eval_spf_loader = DataLoader(
    eval_spf_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)



device = torch.device("cuda")
N_samples = 640
exp_name = "plot_grasp_metrics"
output_dir = os.path.join("output", exp_name)
eval_dir = os.path.join(output_dir, "eval_results")



MAIN_EVALUATION_PLAN = [
            {
                "spokes_per_frame": 2,
                "num_frames": 144, # 2 * 144 = 288 total spokes
                "description": "Stress test: max temporal points, 2 spokes"
            },
            {
                "spokes_per_frame": 4,
                "num_frames": 72, # 4 * 72 = 288 total spokes
                "description": "Stress test: max temporal points, 4 spokes"
            },
            {
                "spokes_per_frame": 8,
                "num_frames": 36, # 8 * 36 = 288 total spokes
                "description": "High temporal resolution"
            },
            {
                "spokes_per_frame": 16,
                "num_frames": 18, # 16 * 18 = 288 total spokes
                "description": "High temporal resolution"
            },
            {
                "spokes_per_frame": 24,
                "num_frames": 12, # 24 * 12 = 288 total spokes
                "description": "Good temporal resolution"
            },
            {
                "spokes_per_frame": 32,
                "num_frames": 8, # 36 * 8 = 288 total spokes
                "description": "Standard temporal resolution"
            },
        ]


spf_raw_dc_mse = {}
spf_raw_dc_mae = {}
spf_raw_grasp_dc_mse = {}
spf_raw_grasp_dc_mae = {}

data_dir = "/ess/scratch/scratch1/rachelgordon/zf_data_192_slices/zf_kspace"
N_slices = 192
num_slices_to_eval = 1

for eval_config in MAIN_EVALUATION_PLAN:

    spokes = eval_config["spokes_per_frame"]
    num_frames = eval_config["num_frames"]

    eval_spf_dataset.spokes_per_frame = spokes
    eval_spf_dataset.num_frames = num_frames
    eval_spf_dataset._update_sample_paths()

    ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, spokes, num_frames)
    physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))



    # evaluate on raw k-space
    print(f"Evaluating on raw k-space with {num_slices_to_eval} slices...")
    raw_grasp_dc_mse, raw_grasp_dc_mae, raw_dc_std_grasp_mse, raw_dc_std_grasp_mae = eval_raw_kspace_grasp(num_slices_to_eval, val_patient_ids, data_dir, spokes, N_slices, num_frames, physics, device)


    spf_raw_grasp_dc_mse[spokes] = raw_grasp_dc_mse
    spf_raw_grasp_dc_mae[spokes] = raw_grasp_dc_mae
    

# Create the line plot
temp_resolutions = {}
for spf in spf_raw_grasp_dc_mse.keys():
    num_timeframes = round(288 / int(spf), 0)
    temp_res = round(150 / num_timeframes, 0)
    temp_resolutions[temp_res] = spf_raw_grasp_dc_mse[spf]


sns.lineplot(x=list(temp_resolutions.keys()), y=list(temp_resolutions.values()), marker='o')

plt.title('Raw k-space MSE of GRASP Reconstruction vs. Temporal Resolution', fontsize=16)
plt.xlabel("Temporal Resolution (seconds/frame)", fontsize=14)
plt.ylabel("Raw k-space MSE", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True) # Add a grid for better readability

plt.savefig('/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/raw_grasp_mse.png')
