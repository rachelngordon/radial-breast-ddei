import torch
import numpy as np
import matplotlib.pyplot as plt
from radial_dclayer_singlecoil import RadialDCLayer, RadialPhysics
from crnn import CRNN, ArtifactRemovalCRNN
import h5py
from einops import rearrange

# -------------------------------------------------------------------
# 2. Device setup
# -------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------
# 3. Load one “real” radial k‐space example and its trajectory
#    • y_raw should have shape [C, T, S, L] or [C, T, S, L, 2] (if stored as complex)
#    • ktraj should have shape [S, L, 2] (normalized kx, ky per sample)
#
#    Adjust the np.load lines below to point to your actual .npy/.pt files.
# -------------------------------------------------------------------
# Example: assume your radial k‐space was saved as a NumPy array “path_to_y.npy”
# with dtype complex64 and shape (C, T, S, L). We’ll convert to a “real+imag” tensor.

def get_kspace(patient_key):
    """load raw k-space data from h5 file"""
    patient_idx = int(patient_key.split("_")[2])

    idx = (patient_idx - 1) // 10
    start = idx * 10 + 1
    end = start + 9
    dir_suffix = f"{start:03d}_{end:03d}"

    kspace_path = (
        f"/ess/scratch/scratch1/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_{dir_suffix}/{patient_key}.h5"
    )

    # kspace_path = (
    #     f"/ess/scratch/scratch1/rachelgordon/dce-12tf/binned_kspace/{patient_key}.h5"
    # )
    with h5py.File(kspace_path, "r") as f:
        original_kspace = torch.tensor(f["kspace"][:].T, dtype=torch.float32)
    return original_kspace

patient_id = '001'
slice_idx = 40
raw_kspace = get_kspace(f"fastMRI_breast_{patient_id}_1")

# Select first slice
y = raw_kspace[slice_idx][0]  # first slice

 
# y_np = np.load('path_to_y.npy')         # shape (C, T, S, L), dtype=complex64
# ktraj_np = np.load('path_to_ktraj.npy') # shape (S, L, 2), dtype=float32

# Convert complex k‐space → real+imag channels
# After view_as_real: shape becomes (C, T, S, L, 2)
# y_realim = torch.view_as_real(y)  # Tensor[ C, T, S, L, 2 ]

# Move to torch.Tensor and add batch dimension
y = y.unsqueeze(0).to(device)       # [1, C, T, S, L, 2]
# ktraj = torch.from_numpy(ktraj_np).unsqueeze(0).to(device)  # [1, S, L, 2]

# -------------------------------------------------------------------
# 4. Instantiate your “radial” Physics module
#    - This class must implement:
#       A_forward(x)   ↔ NUFFT onto radial trajectory
#       A_adjoint(y)   ↔ adjoint NUFFT back to image space
#    - physics.ktraj should return the same trajectory you pass into it.
#
#    Here im_size=(H, W) is the target image dimension (e.g. 320×320).
# -------------------------------------------------------------------
H, W = 320, 320
N_time, N_spokes, N_samples, N_coils = 1, 288, 640, 1
physics = RadialPhysics(im_size=(H, W, N_time), N_spokes=N_spokes, N_samples=N_samples, N_time=N_time, N_coils=N_coils)
# (If your RadialPhysics requires density‐compensation weights, pass them here.)

# -------------------------------------------------------------------
# 5. Instantiate the RadialDCLayer and CRNN “backbone”
#    - datalayer = RadialDCLayer(im_size, n_coils, n_spokes, n_samples, device)
#    - backbone = CRNN(num_cascades, chans, datalayer)
# -------------------------------------------------------------------
y = y.unsqueeze(1).unsqueeze(1)

_, C, T, S, L, _ = y.shape  # C = #coils, T = timeframes, S = spokes, L = samples

y = rearrange(y, 'b c t sam sp i -> b c i (sam sp) t')

datalayer = RadialDCLayer(
    im_size=(H, W, N_time)
)

backbone = CRNN(
    num_cascades=5,    # or whatever number of cascades you prefer
    chans=64,          # hidden‐channel size (tune as needed)
    datalayer=datalayer
).to(device)

# -------------------------------------------------------------------
# 6. Wrap the backbone in ArtifactRemovalCRNN
#    - model(y, physics) will:
#       • compute x_init = physics.A_adjoint(y)   # → [B, C, T, H, W]
#       • permute to [B, H, W, T, C]
#       • feed through CRNN with RadialDCLayer
#       • return x_hat → [B, C, T, H, W]
# -------------------------------------------------------------------
model = ArtifactRemovalCRNN(backbone_net=backbone).to(device)
model.eval()

# -------------------------------------------------------------------
# 7. Single forward pass (no gradient) on your one example
# -------------------------------------------------------------------
print("---- input shape: ", y.shape)
with torch.no_grad():
    x_recon = model(y, physics)  # → Tensor of shape [1, C, T, H, W]

print(f"\nReconstructed image tensor shape: {tuple(x_recon.shape)}")
# Expect: (1, C, T, H, W)

# -------------------------------------------------------------------
# 8. Visualize one coil/timepoint
#    For instance: coil=0, time=0 → x_recon[0,0,0,:,:]
# -------------------------------------------------------------------
recon_slice = x_recon[0, 0, 0, :, :].cpu().numpy()  # [H, W]
plt.figure(figsize=(5, 5))
plt.imshow(np.abs(recon_slice), cmap='gray')
plt.title('Magnitude of Recon (coil 0, time 0)')
plt.axis('off')
plt.savefig('test_recon_network.png')
