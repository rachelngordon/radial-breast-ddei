# raw_grasp_eval_streamed.py
# Streams one slice at a time; per-slice caching; no giant arrays in RAM.

import os
import json
import glob
import gc
import random
import h5py
import math
import time
import numpy as np
import torch
import sigpy as sp
from einops import rearrange
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

# plotting (headless)
import matplotlib
matplotlib.use("Agg")  # >>> CHANGED: non-interactive backend to save memory
import matplotlib.pyplot as plt

# your utilities
from utils import (
    prep_nufft,
    to_torch_complex,
)
from radial_lsfp import MCNUFFT
from eval import calc_dc

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

    traj = traj.reshape(N_time, N_spokes, N_samples, 2).astype(np.float32)
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
    R_np = np.transpose(R_np, (1, 2, 0)).astype(np.complex64)

    return R_np  # (H, W, T) complex64


# ----------------------------
# Slice-streaming helpers
# ----------------------------

def process_kspace_slice(kspace_path, slice_idx, spokes_per_frame, images_per_slab=192, center_partition=31):
    """
    Load and process ONE slice of raw radial k-space.

    Returns:
        zf_kspace_slice     : (coils, spokes, samples)  complex64 (as np.array)
        binned_kspace_slice : (T, spokes_per_frame, coils, samples) complex64
        traj_slice          : (T, spokes_per_frame, samples, 2) float32
    """
    # Load only 1 slice from HDF5
    with h5py.File(kspace_path, 'r') as f:
        ksp_slice = f['kspace'][slice_idx]  # expected (coils, spokes, samples), real/imag already combined
        # If your file stores real/imag separately, adapt here before continuing.

    # Ensure dtype complex64 to cut RAM by 2x versus complex128
    ksp_slice = np.asarray(ksp_slice, dtype=np.complex64)

    n_coils, n_spokes, n_samples = ksp_slice.shape
    base_res = n_samples // 2

    # Time frames / spokes per timeframe
    n_time = n_spokes // spokes_per_frame
    n_spokes_used = n_time * spokes_per_frame
    ksp_slice = ksp_slice[:, :n_spokes_used, :]  # (coils, spokes, samples)

    # Retrospective binning → (T, spf, coils, samples)
    ksp_slice_reshaped = ksp_slice.reshape(
        n_coils, n_time, spokes_per_frame, n_samples
    ).transpose(1, 2, 0, 3).astype(np.complex64)

    # Build matching trajectory
    traj = get_traj(
        spokes_per_frame,
        N_spokes=spokes_per_frame,
        N_time=n_time,
        base_res=base_res,
        gind=1
    ).astype(np.float32)

    return ksp_slice, ksp_slice_reshaped, traj


def load_csmap_slice(root_dir, patient_id, slice_idx):
    """
    Load coil-sensitivity maps for ONE slice for a given patient.

    Returns:
        np.ndarray (1, C, H, W) complex64
    """
    csmap_dir = os.path.join(root_dir, "cs_maps", f"{patient_id}_cs_maps")
    slice_paths = sorted(glob.glob(os.path.join(csmap_dir, "cs_map_slice_*.npy")))
    if not slice_paths:
        raise FileNotFoundError(f"No cs maps found at {csmap_dir}")
    if slice_idx >= len(slice_paths):
        raise IndexError(f"Requested slice {slice_idx}, but only {len(slice_paths)} available.")

    csmap_slice = np.load(slice_paths[slice_idx])  # (H, W, C) complex or real pair
    # If stored as real/imag in the last dim, make complex here.

    # (H, W, C) -> (C, H, W)
    csmap_slice = np.moveaxis(csmap_slice, -1, 0)
    # Add batch → (1, C, H, W)
    csmap_slice = csmap_slice[None, ...]
    return csmap_slice.astype(np.complex64)


def shape_grasp_for_physics(arr, num_frames):
    """
    Accepts a GRASP slice array saved from cache and returns (1, H, W, T) complex64.

    Acceptable inputs:
    - (H, W, T)
    - (T, H, W)
    - (2, H, W, T) real/imag  (will be converted to complex)
    - (1, H, W, T) or (1, T, H, W) → leading singleton removed
    """
    g = np.asarray(arr)
    if g.ndim == 4 and g.shape[0] == 1:
        g = np.squeeze(g, 0)

    # If real/imag packed as first dim
    if g.ndim == 4 and g.shape[0] == 2:
        # (2, H, W, T) → complex
        gr = torch.from_numpy(g).contiguous()
        g = to_torch_complex(gr).cpu().numpy()

    # Now handle axis orders
    if g.ndim == 3:
        H, W = g.shape[0], g.shape[1]
        if g.shape[-1] == num_frames:      # (H, W, T)
            pass
        elif g.shape[0] == num_frames:     # (T, H, W) → (H, W, T)
            g = np.transpose(g, (1, 2, 0))
        elif g.shape[-2] == num_frames:    # (H, T, W) → (H, W, T)
            g = np.transpose(g, (0, 2, 1))
        else:
            # Best effort: assume (H, W, T)
            pass
    else:
        # Best effort: try to get to (H, W, T)
        if g.ndim == 4 and g.shape[-1] == num_frames:
            # maybe (C,H,W,T) with C=1 already removed; if more dims remain, pick last 3
            g = g[..., -num_frames:]
            if g.shape[1] == num_frames:  # (H, T, W, T?) weird case; ignore
                pass

    g = np.asarray(g, dtype=np.complex64)
    return g[None, ...]  # (1, H, W, T)


# ----------------------------
# Evaluation (per-slice streaming, per-slice cache)
# ----------------------------

def eval_raw_kspace_grasp(
    num_slices_to_eval,
    val_patient_ids,
    data_dir,
    spokes_per_frame,
    N_slices,
    num_frames,
    physics,
    device,
):
    sp_device = sp.Device(0 if torch.cuda.is_available() else -1)

    # Pick slice indices (small set)
    random_slice_indices = random.sample(range(N_slices), num_slices_to_eval)

    # online aggregation (no big lists)
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0

    with torch.no_grad():
        for patient_id in val_patient_ids:
            # evaluate each requested slice for this patient
            for slice_idx in random_slice_indices:
                raw_kspace_path = os.path.join(data_dir, f'{patient_id}_2.h5')
                root_dir = os.path.dirname(data_dir)

                # -------- per-slice k-space --------
                zf_kspace, binned_kspace, traj = process_kspace_slice(
                    raw_kspace_path, slice_idx, spokes_per_frame, images_per_slab=N_slices
                )

                # -------- per-slice GRASP cache --------
                grasp_patient_dir = os.path.join(root_dir, f'{patient_id}_2')
                os.makedirs(grasp_patient_dir, exist_ok=True)
                grasp_img_path = os.path.join(
                    grasp_patient_dir,
                    f'grasp_recon_{spokes_per_frame}spf_slice{slice_idx:03d}.npy'
                )

                if not os.path.exists(grasp_img_path):
                    # raw_grasp_recon MUST accept single-slice inputs and return (T,H,W) or (H,W,T)
                    grasp_img_slice = raw_grasp_recon(
                        zf_kspace,            # (C, spokes, samples)
                        binned_kspace,        # (T, spf, C, samples)
                        traj,                 # (T, spf, samples, 2)
                        N_slices=1,           # >>> CHANGED: call with 1 slice
                        spokes_per_frame=spokes_per_frame,
                        device=sp_device
                    )
                    np.save(grasp_img_path, np.squeeze(grasp_img_slice))
                    # Free recon arrays as soon as possible
                    del grasp_img_slice
                    gc.collect()

                # Lazy map the cached slice into memory
                grasp_img_slice = np.load(grasp_img_path, mmap_mode='r')

                # -------- per-slice csmaps --------
                csmap = load_csmap_slice(root_dir, f'{patient_id}_2', slice_idx)  # (1,C,H,W)

                # -------- tensor conversions (zero-copy where possible) --------
                # binned_kspace: (T, spf, C, samples) → flat (C, spf*samples, T)
                kspace_slice = np.asarray(binned_kspace, dtype=np.complex64)
                kspace_slice_t = torch.from_numpy(kspace_slice)  # zero-copy
                kspace_slice_flat = rearrange(kspace_slice_t, 't sp c sam -> c (sp sam) t').to(device)

                csmap_slice_t = torch.from_numpy(csmap).to(device)  # (1,C,H,W)

                # GRASP image → (1,H,W,T)
                grasp_img_slice_b = shape_grasp_for_physics(grasp_img_slice, num_frames)  # np
                grasp_img_slice_t = torch.from_numpy(grasp_img_slice_b).to(device)

                # -------- simulate + DC --------
                sim_kspace_grasp = physics(False, grasp_img_slice_t, csmap_slice_t)  # (C, sp*sam, T) complex
                raw_grasp_dc_mse, raw_grasp_dc_mae = calc_dc(sim_kspace_grasp, kspace_slice_flat, device)

                mse_sum += float(raw_grasp_dc_mse)
                mae_sum += float(raw_grasp_dc_mae)
                n += 1

                # -------- cleanup this slice --------
                del zf_kspace, binned_kspace, traj
                del kspace_slice, kspace_slice_t, kspace_slice_flat
                del csmap, csmap_slice_t
                del grasp_img_slice, grasp_img_slice_b, grasp_img_slice_t
                torch.cuda.empty_cache()
                gc.collect()

    avg_grasp_mse = mse_sum / max(n, 1)
    avg_grasp_mae = mae_sum / max(n, 1)

    # For simplicity, we’re not tracking per-slice std online here.
    # If you need std, accumulate values in small Python lists (they’re tiny).
    std_grasp_mse = 0.0
    std_grasp_mae = 0.0

    return avg_grasp_mse, avg_grasp_mae, std_grasp_mse, std_grasp_mae


# ----------------------------
# Main
# ----------------------------

def main():
    # --- config / splits ---
    split_file = "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json"
    with open(split_file, "r") as fp:
        splits = json.load(fp)

    val_patient_ids = splits["val"]

    # hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # constants
    N_samples = 640
    exp_name = "plot_grasp_metrics_streamed"
    output_dir = os.path.join("output", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = "/ess/scratch/scratch1/rachelgordon/zf_data_192_slices/zf_kspace"
    N_slices = 192
    num_slices_to_eval = 1  # evaluate this many random slices per patient

    MAIN_EVALUATION_PLAN = [
        {"spokes_per_frame": 2,  "num_frames": 144, "description": "Stress: 2 spokes"},
        {"spokes_per_frame": 4,  "num_frames": 72,  "description": "Stress: 4 spokes"},
        {"spokes_per_frame": 8,  "num_frames": 36,  "description": "High temporal"},
        {"spokes_per_frame": 16, "num_frames": 18,  "description": "High temporal"},
        {"spokes_per_frame": 24, "num_frames": 12,  "description": "Good temporal"},
        {"spokes_per_frame": 32, "num_frames": 9,   "description": "Standard temporal"},  # >>> CHANGED: 288/32 = 9, not 8
    ]

    spf_raw_grasp_dc_mse = {}

    for cfg in MAIN_EVALUATION_PLAN:
        spf = cfg["spokes_per_frame"]
        n_frames = cfg["num_frames"]

        # NUFFT physics (per SPF)
        ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, spf, n_frames)
        physics = MCNUFFT(
            nufft_ob.to(device), adjnufft_ob.to(device),
            ktraj.to(device), dcomp.to(device)
        )

        print(f"Evaluating SPF={spf} ({n_frames} frames) on {len(val_patient_ids)} patients, {num_slices_to_eval} slice(s) each...")
        avg_mse, avg_mae, _, _ = eval_raw_kspace_grasp(
            num_slices_to_eval,
            val_patient_ids,
            data_dir,
            spf,
            N_slices,
            n_frames,
            physics,
            device,
        )
        spf_raw_grasp_dc_mse[spf] = avg_mse

        # free physics objects before next SPF
        del physics, ktraj, dcomp, nufft_ob, adjnufft_ob
        torch.cuda.empty_cache(); gc.collect()

    # --- Plot (tiny arrays only) ---
    temp_resolutions = {}
    for spf, mse in spf_raw_grasp_dc_mse.items():
        num_timeframes = int(round(288 / int(spf)))
        temp_res = 150.0 / num_timeframes  # seconds/frame
        temp_resolutions[temp_res] = mse

    xs = sorted(temp_resolutions.keys())
    ys = [temp_resolutions[x] for x in xs]

    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.title('Raw k-space MSE of GRASP Reconstruction vs. Temporal Resolution', fontsize=16)
    plt.xlabel("Temporal Resolution (seconds/frame)", fontsize=14)
    plt.ylabel("Raw k-space MSE", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    out_png = '/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/raw_grasp_mse_streamed.png'
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
