import os

import h5py
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from radial import RadialPhysics

DATA_FILE_PATH = (
    "/ess/scratch/scratch1/rachelgordon/dce-8tf/binned_kspace/fastMRI_breast_168_2.h5"
)
DATASET_KEY = "ktspace"

H, W = 320, 320
N_time = 8
N_samples = 640
N_partitions = 83
N_coils = 16
N_spokes_per_frame = 36

PARTITION_TO_RECON = N_partitions // 2
COIL_TO_RECON = 0

output_dir = "debug_output"
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Running Best Static Reconstruction Script on device: {device} ---")
print(f"--- Reconstructing Partition={PARTITION_TO_RECON}, Coil={COIL_TO_RECON} ---")

# --- 1. Load and Prepare Data ---
print(f"Loading data from: {DATA_FILE_PATH}")
with h5py.File(DATA_FILE_PATH, "r") as f:
    # Load the entire 5D dataset
    kspace_5d = torch.from_numpy(f[DATASET_KEY][()]).to(device)

# --- Select the specific partition and coil we want to reconstruct ---
# kspace_5d has shape (partitions, time, coils, spokes, samples)
kspace_to_process = kspace_5d[PARTITION_TO_RECON, :, COIL_TO_RECON, :, :]
print(
    f"Selected k-space data for processing. Shape: {kspace_to_process.shape}"
)  # Should be (T, S, I)

# The selected data should already be complex. Let's verify.
if not torch.is_complex(kspace_to_process):
    raise TypeError("The selected k-space slice is not complex. Check data loading.")

# Add a batch dimension of 1 for the physics operator
kspace_complex = kspace_to_process.unsqueeze(0)  # -> (B=1, T, S, I)

# --- 2. Combine all spokes into a single k-space frame ---
# Pattern: b t s i -> b (t s) i
total_spokes = N_time * N_spokes_per_frame
combined_kspace_complex = rearrange(kspace_complex, "b t s i -> b (t s) i")

print(f"\nCombined k-space shape: {combined_kspace_complex.shape}")
print(f"Total spokes in combined frame: {combined_kspace_complex.shape[1]}")

# Convert to real tensor format (B, C=2, S_total, I) for the physics operator
combined_kspace_real = rearrange(
    torch.view_as_real(combined_kspace_complex), "b s_total i c -> b c s_total i"
)

# --- 3. Create a new STATIC Physics Operator for the combined data ---
print("\nCreating a high-density static physics operator...")
# This part is now correct as it operates on a single coil's data
static_high_density_physics = RadialPhysics(
    im_size=(H, W),
    N_spokes=total_spokes,
    N_samples=N_samples,
).to(device)

# --- 4. Perform the High-Quality Reconstruction ---
print("Performing A_adjoint reconstruction...")
with torch.no_grad():
    best_static_image = static_high_density_physics.A_adjoint(combined_kspace_real)

# --- 5. Visualize and Save the Result ---
print("Saving the output image...")
best_static_mag = (
    torch.sqrt(best_static_image[0, 0, ...] ** 2 + best_static_image[0, 1, ...] ** 2)
    .cpu()
    .numpy()
)

plt.figure(figsize=(10, 10))
plt.imshow(best_static_mag, cmap="gray")
plt.title(
    f"Best Static Recon (Partition {PARTITION_TO_RECON}, Coil {COIL_TO_RECON})",
    fontsize=16,
)
plt.axis("off")
filename = f"best_static_recon_p{PARTITION_TO_RECON}_c{COIL_TO_RECON}.png"
plt.savefig(os.path.join(output_dir, filename))
plt.close()

print(f"\n--- DONE. Check the file '{output_dir}/{filename}' ---")
