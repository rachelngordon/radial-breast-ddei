import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

# --- Assumes your refactored radial.py is in the same directory or accessible ---
from radial import DynamicRadialPhysics


def plot_reconstruction_sample(x_recon, title, filename, output_dir, batch_idx=0):
    """
    Plot reconstruction sample showing magnitude images across timeframes.

    Args:
        x_recon: Reconstructed image tensor of shape (B, C, T, H, W)
        title: Title for the plot
        filename: Filename for saving (without extension)
        output_dir: Directory to save the plot
        batch_idx: Which batch element to plot (default: 0)
    """
    if not isinstance(x_recon, torch.Tensor):
        raise TypeError("x_recon must be a torch.Tensor")

    # compute magnitude from complex-like real tensor
    x_recon_mag = torch.sqrt(x_recon[:, 0, ...] ** 2 + x_recon[:, 1, ...] ** 2)

    n_timeframes = x_recon_mag.shape[1]
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_timeframes,
        figsize=(n_timeframes * 3, 4),
        squeeze=False,
    )
    for t in range(n_timeframes):
        img = x_recon_mag[batch_idx, t, :, :].cpu().numpy()
        ax = axes[0, t]
        ax.imshow(
            img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5)
        )  # Use percentile for better contrast
        ax.set_title(f"t = {t}")
        ax.axis("off")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close(fig)
    print(f"--- Saved plot to {os.path.join(output_dir, f'{filename}.png')} ---")


# --- Configuration ---
DATA_FILE_PATH = (
    "/ess/scratch/scratch1/rachelgordon/dce-8tf/binned_kspace/fastMRI_breast_168_2.h5"
)
DATASET_KEY = "ktspace"

# Image and acquisition parameters from the raw data file
H, W = 320, 320
N_partitions_raw = 83
N_coils_raw = 16
N_time_raw = 8  # Original number of time frames from the pre-processing
N_spokes_raw = 36  # Original spokes per frame
N_samples_raw = 640

# Choose which slice of the 3D volume and which coil to reconstruct
PARTITION_TO_RECON = N_partitions_raw // 2
COIL_TO_RECON = 0

# --- Experiment Parameters: How to bin the time points ---
# We want to create a video with this many final frames
N_FRAMES_OUT = 8  # Options: 8, 4, 2, 1

# --- Setup ---
output_dir = "debug_output"
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running Dynamic Reconstruction Debug Script on device: {device} ---")
print(f"--- Target Output: {N_FRAMES_OUT} frames ---")
print(f"--- Analyzing Partition={PARTITION_TO_RECON}, Coil={COIL_TO_RECON} ---")


# --- 1. Load and Prepare Data ---
print(f"\nLoading data from: {DATA_FILE_PATH}")
with h5py.File(DATA_FILE_PATH, "r") as f:
    kspace_full_data = torch.from_numpy(f[DATASET_KEY][()]).to(device)

# Select the specific partition and coil we want to reconstruct
# Original shape: (partitions, time, coils, spokes, samples)
kspace_single_coil_dyn = kspace_full_data[PARTITION_TO_RECON, :, COIL_TO_RECON, :, :]
# Shape is now (T_raw, S_raw, I_raw) -> (8, 36, 640)
print(f"Selected k-space data for processing. Shape: {kspace_single_coil_dyn.shape}")

if not torch.is_complex(kspace_single_coil_dyn):
    raise TypeError("The selected k-space slice is not complex. Check data loading.")


# --- 2. Re-bin the data into the desired number of output frames ---
if N_time_raw % N_FRAMES_OUT != 0:
    raise ValueError(
        "N_FRAMES_OUT must be a divisor of the original number of time frames (8)"
    )

# This calculates how many of the original frames get binned into one new frame
frames_to_bin = N_time_raw // N_FRAMES_OUT
# This calculates how many spokes will be in each new, combined frame
spokes_per_binned_frame = N_spokes_raw * frames_to_bin

print(
    f"Binning {frames_to_bin} original frames into each of the {N_FRAMES_OUT} output frames."
)
print(f"Each new frame will have {spokes_per_binned_frame} spokes.")

# Add a batch dimension, then perform the re-binning
kspace_binned_complex = rearrange(
    kspace_single_coil_dyn.unsqueeze(0),
    "b (t_out t_inner) s i -> b t_out (t_inner s) i",
    t_out=N_FRAMES_OUT,
)
print(f"Re-binned k-space shape: {kspace_binned_complex.shape}")

# Convert to real tensor format (B, C=2, T_out, S_binned, I) for the physics operator
kspace_binned_real = rearrange(
    torch.view_as_real(kspace_binned_complex), "b t s i c -> b c t s i"
)


# --- 3. Create a new DYNAMIC Physics Operator for the binned data ---
print("\nCreating a dynamic physics operator...")
# This now uses the refactored DynamicRadialPhysics that handles per-frame trajectories
dynamic_physics = DynamicRadialPhysics(
    im_size=(H, W),  # The 2D image size
    N_spokes=spokes_per_binned_frame,  # Spokes PER binned frame
    N_samples=N_samples_raw,
    N_time=N_FRAMES_OUT,  # Number of output time frames
).to(device)


# --- 4. Perform the High-Quality Reconstruction ---
print("Performing A_adjoint reconstruction...")
with torch.no_grad():
    # Pass the re-binned k-space data to the corrected dynamic operator
    reconstructed_image_sequence = dynamic_physics.A_adjoint(kspace_binned_real)


# --- 5. Visualize and Save the Result ---
print("Saving the output image...")
filename = f"dynamic_recon_{N_FRAMES_OUT}frames_p{PARTITION_TO_RECON}_c{COIL_TO_RECON}"
title = f"Dynamic Recon ({N_FRAMES_OUT} Binned Frames)"

plot_reconstruction_sample(
    reconstructed_image_sequence, title, filename, output_dir, batch_idx=0
)

print("\n--- DONE. ---")
print("Check the output image for rotation. If gone, the dynamic physics is correct.")
print(
    "You can now change N_FRAMES_OUT to 4, 2, or 1 to see the effect of having more spokes per frame."
)
