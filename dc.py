import torch
from torchkbnufft import KbNufft
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
import torch
import torch.fft
import nibabel as nib
import os
import h5py

def get_ktraj(N_spokes, N_time, base_res, device):
    """
    Precompute k-space trajectory for efficiency.
    """
    N_tot_spokes = N_spokes * N_time
    N_samples = base_res * 2

    base_lin = torch.arange(N_samples, dtype=torch.float32).to(device) - base_res
    tau = 0.5 * (1 + 5**0.5)
    base_rad = torch.pi / (1 + tau - 1)
    base_rot = torch.arange(N_tot_spokes, dtype=torch.float32).to(device) * base_rad

    traj_x = torch.cos(base_rot).unsqueeze(1) @ base_lin.unsqueeze(0)
    traj_y = torch.sin(base_rot).unsqueeze(1) @ base_lin.unsqueeze(0)
    traj = torch.stack([traj_x, traj_y], dim=-1) / 2  # Shape: (N_tot_spokes, N_samples, 2)

    # reshape the trajectory to be compatible with torchkbnufft
    traj = traj.reshape(N_time, N_spokes * N_samples, 2)#.transpose(1, 0, 2)
    # traj = rearrange(traj, 't len i -> t i len')
    traj = rearrange(traj, 't len i -> len t i')
    
    # normalize
    traj /= torch.mean(torch.abs(traj))

    traj = traj*torch.tensor([1, -1]).to(device)

    traj = rearrange(traj, "len t i -> t i len")  # shape: (2, N_TIME, N_SPOKES)

    return traj

def load_nii(path):
    """
    Load a NIfTI image from the specified path.
    Args:
    - path (str): File path to the NIfTI image.
    """
    nii_image = nib.load(path)
    return nii_image.get_fdata()


def get_cs_map(patient_id, path, ground_truth_dir):

    # extract slice info from path
    filename = os.path.basename(path)
    slice_id = int(filename.split('_')[1])

    # load coil sensitivity maps
    base_dir = os.path.basename(ground_truth_dir)
    cs_maps_path = f"/ess/scratch/scratch1/rachelgordon/{base_dir}/cs_maps/{patient_id}_cs_maps/cs_map_slice_{slice_id:03d}.npy"
    # cs_maps_path = f"{ground_truth_dir}/{patient_id}_cs_maps/cs_map_slice_{slice_id:03d}.npy"

    # load file
    csmap_slice = np.load(cs_maps_path)
    csmap_slice = np.squeeze(csmap_slice, axis=1)

    # split real/imaginary components
    csmap_slice = torch.stack([torch.tensor(csmap_slice.real),torch.tensor(csmap_slice.imag)],dim=1).unsqueeze(0)

    # reshape
    csmap_slice = rearrange(csmap_slice, 'b c i h w -> b c h w i')

    return csmap_slice



def apply_A(NUFFT_op, ktraj_tensor, x_tensor, norm="ortho", csmap=None):
        """
        Apply NUFFT to obtain k-space.
        Vectorized operation to improve efficiency.
        """
        if csmap is not None:
            kdat_tensor = NUFFT_op(x_tensor.contiguous(), ktraj_tensor.contiguous(), smaps=csmap.contiguous(), norm=norm)
        else:
            kdat_tensor = NUFFT_op(x_tensor.contiguous(), ktraj_tensor.contiguous(), norm=norm)

        return kdat_tensor  # Shape: (batch_size, coils, samples, spokes, time)


def simulate_kspace(NUFFT_op, ktraj_tensor, combined_image, csmaps, device):
        """
        Simulate k-space from image.
        """

        print("image shape: ", combined_image.shape)
        print("csmaps shape: ", csmaps.shape)

        dtype = torch.float

        # Ensure tensors are on the same device
        combined_image = combined_image.to(device)
        csmaps = csmaps.to(device).to(dtype).contiguous()

        # Prepare image tensor
        # Fix orientation to match raw kspace orientation, and make contiguous with copy()
        combined_image = torch.flip(combined_image, dims=[-2]).clone()
        combined_image = combined_image.unsqueeze(0).unsqueeze(0)

        # multiply by coil sensitivity maps to get individual coil images
        combined_image = rearrange(combined_image, "b c i h w -> b c h w i")

        multi_coil_images = combined_image * csmaps
        print("multi coil images shape: ", multi_coil_images.shape)

        # forward transform: image to k-space
        sim_kspace = apply_A(
            NUFFT_op, ktraj_tensor, multi_coil_images.to(dtype)
        )
        print("initial sim kspace shape: ", sim_kspace.shape)

        # reshape
        sim_kspace = rearrange(sim_kspace, "b c r i -> b c i r ").to(dtype)
        sim_kspace = torch.reshape(sim_kspace, (1, 16, 2, 288, 640, 1)).squeeze()
        sim_kspace = rearrange(sim_kspace, 'c i sp sam -> c sam sp i')
        print("sim kspace shape after second rearrange: ", sim_kspace.shape)

        return sim_kspace


def simulate_kspace_single_coil(NUFFT_op, ktraj_tensor, combined_image, device):
        """
        Simulate k-space from image.
        """

        print("image shape: ", combined_image.shape)

        dtype = torch.float

        # Ensure tensors are on the same device
        combined_image = combined_image.to(device)
        # csmaps = csmaps.to(device).to(dtype).contiguous()

        # Prepare image tensor
        # Fix orientation to match raw kspace orientation, and make contiguous with copy()
        combined_image = torch.flip(combined_image, dims=[-2]).clone()
        combined_image = combined_image.unsqueeze(0).unsqueeze(0)

        # multiply by coil sensitivity maps to get individual coil images
        combined_image = rearrange(combined_image, "b c i h w -> b c h w i")
        print("combined image shape: ", combined_image.shape)

        # multi_coil_images = combined_image * csmaps

        # forward transform: image to k-space
        sim_kspace = apply_A(
            NUFFT_op, ktraj_tensor, combined_image.to(dtype)
        )
        print("initial sim kspace shape: ", sim_kspace.shape)

        # reshape
        sim_kspace = rearrange(sim_kspace, "b c r i -> b c i r ").to(dtype)
        sim_kspace = torch.reshape(sim_kspace, (1, 1, 2, 288, 640, 1)).squeeze()
        sim_kspace = rearrange(sim_kspace, 'i sp sam -> sam sp i')
        print("sim kspace after reshape: ", sim_kspace.shape)

        return sim_kspace



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define NUFFT operator
im_size = (320, 320)
grid_size = (im_size[1] * 2, im_size[1] * 2)

NUFFT_op = KbNufft(im_size=im_size[:2], grid_size=grid_size).to(device)


# define trajectory
N_spokes = 288
N_time = 1
N_samples = 640
ktraj_tensor = get_ktraj(N_spokes, N_time, N_samples // 2, device).to(device)


# load image and sensitivity maps
ground_truth_dir = "/ess/scratch/scratch1/rachelgordon/complex_fully_sampled/"
patient_id = "fastMRI_breast_001_1"
image_path = f"{ground_truth_dir}{patient_id}/slice_040_frame_000.nii"

image = load_nii(image_path)
image = torch.from_numpy(image)
csmap = get_cs_map(patient_id, image_path, ground_truth_dir)

sim_kspace = simulate_kspace(NUFFT_op, ktraj_tensor, image, csmap, device)
print("final simulate k-space shape: ", sim_kspace.shape)


# # use single coil
# print("Traj shape: ", ktraj_tensor.shape)
# sim_kspace_single_coil = simulate_kspace_single_coil(NUFFT_op, ktraj_tensor, image, device)
# print("final single coil simulate k-space shape: ", sim_kspace_single_coil.shape)



# save k-space to a file
# f = h5py.File(f"fastMRI_breast_001_1_sim_kspace.h5", "w")
# dset = f.create_dataset('kspace', data=sim_kspace)
# f.close()