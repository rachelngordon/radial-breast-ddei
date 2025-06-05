import torch
import numpy as np 
from einops import rearrange
import nibabel as nib
import h5py
import matplotlib.pyplot as plt
import os


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):

    N_tot_spokes = N_spokes * N_time

    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))

    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    return traj

def normalize_tensor(tensor):
    scale_factor = torch.mean(torch.abs(tensor))
    return tensor / scale_factor


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
    with h5py.File(kspace_path, "r") as f:
        original_kspace = torch.tensor(f["kspace"][:].T, dtype=torch.float32)

    print("kspace shape before fft: ", original_kspace.shape)
    kspace_spatial_z = torch.fft.ifft(original_kspace, dim=0, norm='ortho')

    return kspace_spatial_z



# 1) Replace these with your actual dimensions:
H        = 320    # image height
W        = 320    # image width
N_time   = 1      # number of time‐frames
N_samples = 184320  # total radial samples per frame (spokes * samples_per_spoke)

# 2) Import your RadialDCLayer implementation.
#    Adjust the import path as needed so that this points to where
#    your class is defined.
from radial_dclayer_singlecoil import RadialDCLayer

# 3) Instantiate the layer (im_size = (H, W, N_time)):
radial_dc = RadialDCLayer(im_size=(H, W, N_time))

# 4) Build “dummy” inputs of the correct shape.
#    In practice, replace these random tensors with your actual data.

patient_id = '001'
scan_num = '1'
slice_idx = 40

dtype = torch.float

# obtain dcf and k-space trajectory
N_COILS, N_SPOKES, N_SAMPLES, N_TIME = 1, 288, 640, 1
BASE_RES = N_SAMPLES // 2
traj = get_traj(N_spokes=N_SPOKES, N_time=N_TIME, base_res=BASE_RES, gind=1)

# reshape the trajectory and normalize
traj = traj.reshape(N_TIME, N_SPOKES * N_SAMPLES, 2).transpose(1, 0, 2)
traj = normalize_tensor(torch.tensor(traj))
traj = torch.tensor(traj)*torch.tensor([1, -1])


# compute the density compensation function from trajectory
dcf = np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2)  # shape: (N_TIME, N_SPOKES)
dcf_tensor = (
    torch.tensor(dcf).unsqueeze(0).unsqueeze(0).unsqueeze(0)
)  # shape: (1, 1, 1, N_TIME, N_SPOKES)
print("dcf: ", dcf_tensor.shape)


# combine real and imaginary components in k-space trajectory
traj = rearrange(traj, "t r i -> i t r")  # shape: (2, N_TIME, N_SPOKES)
traj = traj[0] + 1j * traj[1]  # shape: (N_TIME, N_SPOKES) complex
ktraj_tensor = torch.stack(
    [torch.tensor(traj.real), torch.tensor(traj.imag)], dim=0
).unsqueeze(0)  # shape: (1, 2, N_TIME, N_SPOKES)
print("ktraj: ", ktraj_tensor.shape)



# load combined image
nifti_file = f"/ess/scratch/scratch1/rachelgordon/complex_fully_sampled/fastMRI_breast_{patient_id}_{scan_num}/slice_{slice_idx:03}_frame_000.nii"
img = nib.load(nifti_file)
combined_img_data = img.get_fdata()

# Fix orientation to match raw kspace orientation, and make contiguous with copy()
combined_img_data = np.flip(combined_img_data, axis=-2).copy()
combined_img_data = np.expand_dims(combined_img_data, axis=[0, -1])

im_size = combined_img_data.shape[2:]
print("combined image shape:", combined_img_data.shape)
print("image size:", im_size)

combined_img_complex = rearrange(combined_img_data, "b i h w c -> b c h w i")
combined_img_complex_time = torch.tensor(combined_img_complex).unsqueeze(-1)

# load ground truth k-space 
raw_kspace = get_kspace(f"fastMRI_breast_{patient_id}_1")
print("raw k-space shape:", raw_kspace.shape)

# Select first slice and first coil
kspace_partition = raw_kspace[slice_idx][0].unsqueeze(0)  # first slice
print(
    "k-space slice shape:", kspace_partition.shape
)  # (N_COILS, N_SAMPLES, N_SPOKES, 2)

# Rearrange to the expected input format for apply_Adag
# apply_Adag expects (batch, coil, component, point, time)
y_meas = rearrange(kspace_partition, "c sam sp i -> 1 c (sp sam) i 1")



# 5) (Optional) Move everything to GPU if available:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_est    = combined_img_complex_time.to(device)
y_meas   = y_meas.to(device)
ktraj    = ktraj_tensor.to(device)
dcf      = dcf_tensor.to(device)
radial_dc = radial_dc.to(device)

# verify correct input shapes
print("x shape: ", x_est.shape)
print("y shape: ", y_meas.shape)
print("traj shape: ", ktraj.shape)
print("dcf shape: ", dcf.shape)

# 6) Run a single forward pass:
with torch.no_grad():
    x_dc_out = radial_dc(x_est, y_meas, ktraj, dcf)

# 7) Verify the output shape:
print("Output shape:", x_dc_out.shape)


# 8) plot output image
x_dc_out = x_dc_out.squeeze()
x_mag = torch.abs(x_dc_out[..., 0] + 1j* x_dc_out[..., 1])

gt_image = combined_img_data.squeeze()
gt_image_mag = np.abs(gt_image[0] + 1j * gt_image[1])

# Compare the three methods by plotting them side by side
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(gt_image_mag, cmap="gray", origin="lower")
ax[0].set_title("input image")
ax[1].imshow(x_mag.cpu().numpy(), cmap="gray", origin="lower")
ax[1].set_title("dc output")
plt.tight_layout()
plt.savefig(os.path.join("dc_plots", "radial_dclayer_test.png"), dpi=300)
plt.close()