
import torch
from deepinv.loss.mc import MCLoss
from einops import rearrange
import h5py
from radial_dclayer_singlecoil import RadialPhysics
import numpy as np
import nibabel as nib


## test loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss = MCLoss()

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
scan_num = 1
raw_kspace = get_kspace(f"fastMRI_breast_{patient_id}_{scan_num}")

# Select slice and first coil
y = raw_kspace[slice_idx][0]
y = y.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device) # add batch, coil, and time dimensions
y = rearrange(y, 'b c t sam sp i -> b c (sam sp) i t')
print("y_meas shape: ", y.shape)


H, W = 320, 320
N_time, N_spokes, N_samples, N_coils = 1, 288, 640, 1
physics = RadialPhysics(im_size=(H, W, N_time), N_spokes=N_spokes, N_samples=N_samples, N_time=N_time, N_coils=N_coils)


# load image 
nifti_file = f"/ess/scratch/scratch1/rachelgordon/complex_fully_sampled/fastMRI_breast_{patient_id}_{scan_num}/slice_{slice_idx:03}_frame_000.nii"
img = nib.load(nifti_file)
combined_img_data = img.get_fdata()

# Fix orientation to match raw kspace orientation, and make contiguous with copy()
combined_img_data = np.flip(combined_img_data, axis=-2).copy()
combined_img_data = np.expand_dims(combined_img_data, axis=[0, -1])

combined_img_complex = rearrange(combined_img_data, "b i h w c -> b c h w i")
combined_img_complex_time = torch.tensor(combined_img_complex).unsqueeze(-1)
print("image shape: ", combined_img_complex_time.shape)


mse = loss(y, combined_img_complex_time, physics)
print("MSE: ", mse)