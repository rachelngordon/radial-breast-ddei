import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from einops import rearrange
import random
import sigpy as sp
from utils import prep_nufft
from radial_lsfp import MCNUFFT
import time
from typing import Union, List, Optional
import re
import csv


def target_positions_centered(Z_in: int, S_out: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Evenly spaced target slice positions covering the same 'z-FOV' as the input partitions.

    Shapes:
        return: (S_out,) float tensor of target slice indices in *input-partition units*
                centered so that 0 corresponds to the mid-partition

    Example:
        Z_in=83, S_out=192  -> returns 192 positions spanning [-(Z_in-1)/2, +(Z_in-1)/2]
    """
    if Z_in <= 0 or S_out <= 0:
        raise ValueError("Z_in and S_out must be positive.")
    half = (Z_in - 1) / 2.0
    return torch.linspace(-half, +half, S_out, device=device, dtype=dtype)


def _kz_grid_centered(Z_in: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Centered, unitless frequency grid along partitions (kz), length Z_in.

    Shapes:
        return: (Z_in,) with values ((p - (Z_in-1)/2) / Z_in), i.e., cycles per input-partition

    This pairs naturally with target_positions_centered so that W = exp(i 2π kz * z_target).
    """
    p = torch.arange(Z_in, device=device, dtype=dtype)
    return (p - (Z_in - 1) / 2.0) / Z_in


def build_reslice_weights(Z_in: int, S_out: int, device=None) -> torch.Tensor:
    """
    Build the complex phase matrix W that maps partitions -> slices in one matmul.

    Shapes:
        return: W with shape (S_out, Z_in), complex64
                W[s, z] = exp(i * 2π * kz[z] * z_targets[s])

    Where:
        kz        = centered frequency grid length Z_in
        z_targets = centered target slice indices length S_out
    """
    z_targets = target_positions_centered(Z_in, S_out, device=device)        # (S_out,)
    kz = _kz_grid_centered(Z_in, device=device)                               # (Z_in,)
    phase = 2.0 * torch.pi * (z_targets[:, None] * kz[None, :])               # (S_out, Z_in)
    W = torch.polar(torch.ones_like(phase), phase)                             # (S_out, Z_in) complex
    return W.to(torch.complex64)


def collapse_partitions_to_slices(K3d: torch.Tensor, S_out: int) -> torch.Tensor:
    """
    Collapse the partition axis into S_out slices by a single complex matrix multiply.

    Shapes:
        K3d:            (Z_in, T, C, Sp, Sa) complex64/complex128
        return Keff:    (S_out, T, C, Sp, Sa) complex64

    Notes:
        - O( S_out * Z_in * T*C*Sp*Sa ) but executed as one GEMM
        - no Python loops; everything is batched and vectorized
    """
    if K3d.dim() != 5:
        raise ValueError(f"expected (Z_in, T, C, Sp, Sa), got {tuple(K3d.shape)}")
    Z_in, T, C, Sp, Sa = K3d.shape
    device = K3d.device
    W = build_reslice_weights(Z_in, S_out, device=device)                      # (S_out, Z_in)
    K3d_flat = rearrange(K3d.to(torch.complex64), 'z t c sp sa -> z (t c sp sa)')  # (Z_in, M)
    Y_flat = W @ K3d_flat                                                       # (S_out, M)
    Keff = rearrange(Y_flat, 's (t c sp sa) -> s t c sp sa', t=T, c=C, sp=Sp, sa=Sa)
    return Keff


def resample_csmaps_along_partitions(S3d: torch.Tensor, S_out: int) -> torch.Tensor:
    """
    Linearly resample 3D coil maps along the partition axis to the same target positions.

    Shapes:
        S3d:                (1, C, H, W, Z_in) complex64/complex128
        return Seff_stack:  (S_out, 1, C, H, W) complex64

    Notes:
        - uses straight linear interpolation in partition index space
        - if you only have 2D maps, replace this with nearest-slice selection for the chosen target
    """
    if S3d.dim() != 5:
        raise ValueError(f"expected (1, C, H, W, Z_in), got {tuple(S3d.shape)}")
    _, C, H, W, Z_in = S3d.shape
    device = S3d.device
    z_targets = target_positions_centered(Z_in, S_out, device=device)          # (S_out,)
    # map centered positions to absolute indices in [0, Z_in-1]
    z_cont = z_targets + (Z_in - 1) / 2.0                                      # (S_out,)
    z0 = torch.floor(z_cont).to(torch.long)                                     # (S_out,)
    z1 = torch.clamp(z0 + 1, max=Z_in - 1)                                      # (S_out,)
    alpha = (z_cont - z0.to(z_cont.dtype)).view(1, 1, 1, 1, -1)                 # (1,1,1,1,S_out)

    idx0 = z0.view(1, 1, 1, 1, -1).expand(1, C, H, W, -1)                       # (1,C,H,W,S_out)
    idx1 = z1.view(1, 1, 1, 1, -1).expand(1, C, H, W, -1)

    S_lo = torch.take_along_dim(S3d.to(torch.complex64), idx0, dim=-1)          # (1,C,H,W,S_out)
    S_hi = torch.take_along_dim(S3d.to(torch.complex64), idx1, dim=-1)          # (1,C,H,W,S_out)
    Seff = (1.0 - alpha) * S_lo + alpha * S_hi                                  # (1,C,H,W,S_out)
    Seff_stack = rearrange(Seff, 'b c h w s -> s b c h w')                       # (S_out,1,C,H,W)
    return Seff_stack


def pack_complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    """
    Pack complex tensor to 2-channel real representation.

    Shapes:
        x:       (...,) complex
        return:  (...,)-> adds a real/imag channel at the front
                 if x is (T,C,Sp,Sa) -> returns (2,T,C,Sp,Sa)
                 if x is (H,W,T)     -> returns (2,H,W,T)
    """
    xr = x.real
    xi = x.imag
    return torch.stack([xr, xi], dim=0)




class SliceDataset(Dataset):
    """
    A Dataset that:
      - Looks for all .h5/.hdf5 files under `root_dir`.
      - Each file is assumed to contain a dataset at `dataset_key`, with shape (... Z),
        where Z is the number of slices/partitions.
      - Can either use a fixed set of slices or randomly sample N slices per volume
        at the start of each epoch.
      - Returns each slice as a torch.Tensor.
    """

    def __init__(
        self,
        root_dir,
        patient_ids,
        dataset_key="kspace",
        file_pattern="*.h5",
        slice_idx: Optional[Union[int, range]] = 41,
        num_random_slices: Optional[int] = None,  # New parameter for random sampling
        N_time=8,
        N_coils=16,
        spf_aug=False,
        spokes_per_frame=None,
        weight_accelerations=False, 
        initial_spokes_range=[8, 16, 24, 36],
        interpolate_kspace=False,
        slices_to_interpolate=192,
        cluster="Randi"
    ):
        """
        Args:
            root_dir (str): Path to the folder containing all HDF5 k-space files.
            patient_ids (list): List of patient IDs to filter the files.
            dataset_key (str): The key/path inside each .h5 file to the k-space dataset.
            file_pattern (str): Glob pattern to match your HDF5 files.
            slice_idx (int, range, optional): A fixed slice index or range of indices to use.
                                              This is ignored if num_random_slices is set.
            num_random_slices (int, optional): If provided, the dataset will randomly sample
                                               this many slices from each volume at the beginning
                                               of each epoch.
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_key = dataset_key
        self.slice_idx = slice_idx
        self.num_random_slices = num_random_slices
        self.N_time = N_time
        self.N_coils = N_coils
        self.spf_aug = spf_aug
        self.weight_acc = weight_accelerations
        self.interpolate_kspace = interpolate_kspace
        self.slices_to_interpolate = slices_to_interpolate
        self.cluster=cluster

        # Find all matching HDF5 files under root_dir
        all_files = sorted(glob.glob(os.path.join(root_dir, file_pattern)))
        print("Number of files in root directory: ", len(all_files))

        if len(all_files) == 0:
            raise RuntimeError(
                f"No files found in {root_dir} matching pattern {file_pattern}"
            )

        # filter file list by patient ID substring
        filtered = []
        for fp in all_files:
            fname = os.path.basename(fp)
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)

        self.file_list = filtered

        if len(self.file_list) == 0:
            raise RuntimeError("No files matched the provided patient_ids filter.")

        # Logic for random slice sampling
        if self.num_random_slices is not None:
            print(f"Initializing in random slice sampling mode with N={self.num_random_slices} slices per volume.")
            self.volume_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    num_slices = f[self.dataset_key].shape[0]
                    self.volume_map.append((fp, num_slices))
            
            # Perform the initial random sampling for the first epoch
            self.resample_slices()
        
        # Original logic for fixed slices, executed only if not in random mode
        else:
            print(f"Initializing in fixed slice mode with slice_idx={self.slice_idx}.")
            self.slice_index_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    ds = f[self.dataset_key]
                    num_slices = ds.shape[0]

                slices_to_add = []
                if isinstance(self.slice_idx, int):
                    if self.slice_idx < num_slices:
                        slices_to_add = [self.slice_idx]
                    else:
                        print(f"Warning: slice_idx {self.slice_idx} is out of bounds for {fp} "
                              f"(size {num_slices}). Skipping this file for this slice.")
                elif isinstance(self.slice_idx, range):
                    slices_to_add = [s for s in self.slice_idx if s < num_slices]
                    if len(slices_to_add) < len(self.slice_idx):
                        print(f"Warning: Some requested slices were out of bounds for {fp}. "
                              f"Using only the valid slice indices from the provided range.")
                else:
                    raise TypeError(f"slice_idx must be an int, range, or None, but got {type(self.slice_idx)}")

                for z in slices_to_add:
                    self.slice_index_map.append((fp, z))

        print(f"Dataset initialized with {len(self.slice_index_map)} total slice examples.")

        self.spokes_per_frame = spokes_per_frame

        # NOTE: removed ultra-high accelerations until curriculum learning is implemented
        # self.spokes_range = [2, 4, 8, 16, 24, 36]
        # self.spokes_range = [8, 16, 24, 36]
        self.spokes_range = initial_spokes_range
        self.update_spokes_weights()
    
    def update_spokes_weights(self):

        if self.weight_acc:
            self.spf_weights = [1.0 / spf for spf in self.spokes_range]
        else:
            self.spf_weights = [1.0 for spf in self.spokes_range]


    def resample_slices(self):
        """
        Resamples N unique slices from each volume. This should be called at the
        beginning of each training epoch to ensure the model sees different data.
        """
        if self.num_random_slices is None:
            # If not in random sampling mode, do nothing.
            return

        self.slice_index_map = []
        for file_path, num_slices in self.volume_map:
            if num_slices >= self.num_random_slices:
                # Randomly sample N unique slices without replacement
                selected_slices = random.sample(range(num_slices), self.num_random_slices)
            else:
                # If the volume has fewer than N slices, take all of them.
                print(f"Warning: Volume {os.path.basename(file_path)} has only {num_slices} slices, "
                      f"which is less than the requested {self.num_random_slices}. Using all available slices.")
                selected_slices = list(range(num_slices))

            for z in selected_slices:
                self.slice_index_map.append((file_path, z))

    def load_dynamic_img(self, patient_id, slice):
        # This method remains unchanged
        H = W = 320
        data = np.empty((2, self.N_time, H, W), dtype=np.float32)
        
        for t in range(self.N_time):
            if self.cluster == "Randi":
                img_path = f'/ess/scratch/scratch1/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii'
            elif self.cluster == "DSI":
                img_path = f'/net/scratch2/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii'
            else:
                raise ValueError("Undefined cluster name.")
            img = nib.load(img_path)
            img_data = img.get_fdata()

            if img_data.shape != (2, H, W):
                raise ValueError(f"{img_path} has shape {img_data.shape}; expected (2, {H}, {W})")

            data[:, t] = img_data.astype(np.float32)
            
        return torch.from_numpy(data)

    def load_csmaps(self, patient_id, slice):
        # This method remains unchanged
        ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
        csmap_path = os.path.join(ground_truth_dir, patient_id + '_cs_maps', f'cs_map_slice_{slice:03d}.npy')
        csmap = np.load(csmap_path)
        return csmap.squeeze()
    
    def load_all_csmaps(self, patient_id):
        """
        Loads all csmap slices for a given patient and stacks them into a single array.

        Args:
            patient_id (str): The ID of the patient.

        Returns:
            numpy.ndarray: A NumPy array containing all the stacked csmap slices
                        with the shape (1, C, H, W, Z_in).
        """
        ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
        patient_csmap_dir = os.path.join(ground_truth_dir, patient_id + '_cs_maps')

        # Find all slice files and sort them to ensure correct order
        slice_paths = sorted(glob.glob(os.path.join(patient_csmap_dir, 'cs_map_slice_*.npy')))

        if not slice_paths:
            raise FileNotFoundError(f"No csmap slices found for patient {patient_id} in {patient_csmap_dir}")

        # Load each slice and store it in a list
        all_slices = [np.load(path) for path in slice_paths]

        # Stack the slices along a new axis (the last axis)
        # Assuming each slice has a shape of (H, W, C)
        stacked_csmaps = np.stack(all_slices, axis=-1)

        print("stacked_csmaps: ", stacked_csmaps.shape)

        final_csmaps = rearrange(stacked_csmaps, 'c b h w z -> b c h w z')

        print("final_csmaps: ", final_csmaps.shape)

        # At this point, the shape is likely (H, W, C, Z_in)
        # We need to rearrange the axes to (C, H, W, Z_in)
        # The axes are indexed as H=0, W=1, C=2, Z_in=3
        # transposed_csmaps = np.transpose(stacked_csmaps, (2, 0, 1, 3))

        # # Add a new dimension at the beginning to get the final shape (1, C, H, W, Z_in)
        # final_csmaps = np.expand_dims(transposed_csmaps, axis=0)

        return final_csmaps

    def __len__(self):
        return len(self.slice_index_map)

    def __getitem__(self, idx):
        # This method remains unchanged as it relies on self.slice_index_map
        file_path, current_slice_idx = self.slice_index_map[idx]
        current_slice_idx = int(current_slice_idx)
        patient_id = file_path.split('/')[-1].strip('.h5')

        # grasp_img = self.load_dynamic_img(patient_id, current_slice_idx)

        start = time.time()
        
        if self.interpolate_kspace:
            csmap_stack = self.load_all_csmaps(patient_id)
            Seff_stack = resample_csmaps_along_partitions(torch.tensor(csmap_stack), S_out=self.slices_to_interpolate)        # (192, 1, C, H, W)
            csmap = Seff_stack[current_slice_idx].squeeze()                                            # (1, C, H, W)
        else:
            csmap = self.load_csmaps(patient_id, current_slice_idx)



        with h5py.File(file_path, "r") as f:

            if self.interpolate_kspace:
                ds = torch.tensor(f[self.dataset_key][:])
                print("shape before interpolation: ", ds.shape)
                Keff_stack = collapse_partitions_to_slices(ds, S_out=self.slices_to_interpolate)           # (192, T, C, Sp, Sa)
                # kspace_slice = pack_complex_to_2ch(Keff_stack[current_slice_idx])                 # (2, T, C, Sp, Sa)
                kspace_slice = Keff_stack[current_slice_idx]                 # (2, T, C, Sp, Sa)

            else:
                kspace_slice = torch.tensor(f[self.dataset_key][current_slice_idx])

        end = time.time()

        print("time for interpolation: ", end-start)


        if self.spf_aug or self.spokes_per_frame:
            total_spokes = kspace_slice.shape[0] * kspace_slice.shape[2]
            N_samples = kspace_slice.shape[-1]
            kspace = rearrange(kspace_slice, 't c sp sam -> t sp c sam')
            kspace_flat = kspace.contiguous().view(total_spokes, self.N_coils, N_samples)
            # kspace_flat = kspace.contiguous().reshape(total_spokes, self.N_coils, N_samples)

            if self.spf_aug:
                print("setting random spokes per frame...")
                spokes_per_frame = random.choices(self.spokes_range, self.spf_weights, k=1)[0]
            else:
                spokes_per_frame = self.spokes_per_frame
                print(f"training with fixed spokes per frame ({spokes_per_frame})")

            N_time = total_spokes // spokes_per_frame
            kspace_binned = kspace_flat.view(N_time, spokes_per_frame, self.N_coils, N_samples)
            # kspace_binned = kspace_flat.reshape(N_time, spokes_per_frame, self.N_coils, N_samples)
            kspace_slice = rearrange(kspace_binned, 't sp c sam -> t c sp sam')
        else:
            N_time = self.N_time
            N_samples = kspace_slice.shape[-1]
            spokes_per_frame = kspace_slice.shape[-2]

        real_part = kspace_slice.real
        imag_part = kspace_slice.imag
        kspace_final = torch.stack([real_part, imag_part], dim=0).float()
        kspace_final = torch.flip(kspace_final, dims=[-1])

        if self.interpolate_kspace == False:
            csmap = torch.from_numpy(csmap)
            
        csmap_tensor = torch.rot90(csmap, k=2, dims=[-2, -1])
        csmap = csmap_tensor.numpy()

        return kspace_final, csmap, N_samples, spokes_per_frame, N_time
    


# class SimulatedDataset(Dataset):
#     """
#     Dataset for loading the simulated data generated by your script.
#     It loads the simulated k-space, coil sensitivity maps, and the
#     ground truth dynamic image (DRO).
#     """
#     def __init__(self, root_dir, model_type, patient_ids, spokes_per_frame=36, num_frames=22):
#         self.model_type = model_type
#         self.spokes_per_frame = spokes_per_frame
#         self.num_frames = num_frames

#         dro_dir = os.path.join(root_dir, f'dro_{num_frames}frames')

#         # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
#         self.sample_paths = sorted(glob.glob(os.path.join(dro_dir, 'sample_*')))
#         if not self.sample_paths:
#             raise FileNotFoundError(f"No sample directories found in {root_dir}. "
#                                     "Please check the path to your simulated dataset.")
        
#         # filter file list by patient ID substring
#         filtered = []
#         for fp in self.sample_paths:
#             fname = os.path.basename(fp)
#             # Check if any patient_id appears in the filename
#             if any(pid in fname for pid in patient_ids):
#                 filtered.append(fp)

#         self.sample_paths = filtered

#         print(f"Found {len(self.sample_paths)} simulated samples in {root_dir} for this dataset.")

#         self.TISSUE_NAMES = [
#             'glandular', 'benign', 'malignant', 'muscle',
#             'skin', 'liver', 'heart', 'vascular'
#         ]

#     def __len__(self):
#         return len(self.sample_paths)

#     def __getitem__(self, idx):
#         sample_dir = self.sample_paths[idx]

#         # Load the data from .npy files
#         csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
#         dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))
#         # grasp_recon = np.load(os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy'))

#         grasp_path = os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')
        
#         if os.path.exists(grasp_path):
#             print("loading grasp image from ", grasp_path)
#             grasp_recon = np.load(grasp_path)

#             # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
#             grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
#             grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

#             grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
#             grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

#         else:
#             print("setting grasp img to zero")
#             grasp_recon_torch = 0

#         kspace_path = os.path.join(sample_dir, f'simulated_kspace_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')

#         if os.path.exists(kspace_path):
#             kspace_complex = np.load(kspace_path, allow_pickle=True)
#             kspace_torch = torch.from_numpy(kspace_complex)
#         else:
#             kspace_torch = kspace_path


#         ground_truth_complex = dro['ground_truth_images']

#         parMap = dro['parMap']
#         aif = dro['aif']
#         S0 = dro['S0']
#         T10 = dro['T10']
#         # mask = dro['mask']

#         # ==========================================================
#         # --- RECONSTRUCT THE MASK DICTIONARY ---
#         # ==========================================================
#         mask_dictionary_rebuilt = {}
#         for tissue_name in self.TISSUE_NAMES:
#             # Check if the key for this tissue (e.g., 'malignant') exists in the file
#             if tissue_name in dro:
#                 # Load the boolean array and add it to the dictionary
#                 mask_dictionary_rebuilt[tissue_name] = dro[tissue_name]
        
#         # 'mask' is now the dictionary of boolean arrays, just like your functions expect
#         mask = mask_dictionary_rebuilt


#         # --- Convert to PyTorch Tensors ---
#         # Ground truth: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
#         ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) # T, H, W
#         ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)

#         # CSMaps: (H, W, C) -> (1, C, H, W) [batch, coils, h, w]
#         csmaps_torch = torch.from_numpy(csmaps).permute(2, 0, 1).unsqueeze(0)

#         return kspace_torch, csmaps_torch, ground_truth_torch, grasp_recon_torch, mask, grasp_path #, parMap, aif, S0, T10, mask


class SimulatedDataset(Dataset):
    """
    Dataset for loading the simulated data generated by your script.
    It loads the simulated k-space, coil sensitivity maps, and the
    ground truth dynamic image (DRO).
    """
    def __init__(self, root_dir, model_type, patient_ids, spokes_per_frame=36, num_frames=8):

        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.model_type = model_type
        self.spokes_per_frame = spokes_per_frame
        self.num_frames = num_frames

        self._update_sample_paths()


        self.TISSUE_NAMES = [
            'glandular', 'benign', 'malignant', 'muscle',
            'skin', 'liver', 'heart', 'vascular'
        ]

    def _update_sample_paths(self):
        self.dro_dir = os.path.join(self.root_dir, f'dro_{self.num_frames}frames')

        # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
        self.sample_paths = sorted(glob.glob(os.path.join(self.dro_dir, 'sample_*')))
        if not self.sample_paths:
            raise FileNotFoundError(f"No sample directories found in {self.dro_dir}. "
                                    "Please check the path to your simulated dataset.")
        
        # filter file list by patient ID substring
        filtered = []
        for fp in self.sample_paths:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in self.patient_ids):
                filtered.append(fp)

        self.sample_paths = filtered

        print(f"Found {len(self.sample_paths)} simulated samples in {self.dro_dir} for {self.num_frames} frames.")
  
            
    def __len__(self):
        return len(self.sample_paths)
    


    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]
        # print("sample dir: ", sample_dir) # /ess/scratch/scratch1/rachelgordon/dro_dataset/dro_36frames/sample_021_sub21
        # sample_id = os.path.basename(sample_dir)#.split("_")[-1].strip("sub")
        # print("sample_id: ", sample_id)


        # Load the data from .npy files
        csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
        dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))
        # grasp_recon = np.load(os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy'))

        grasp_path = os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')
        
        if os.path.exists(grasp_path):
            print("loading grasp image from ", grasp_path)
            grasp_recon = np.load(grasp_path)

            # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
            grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
            grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

            grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
            grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

        else:
            print("setting grasp img to zero")
            grasp_recon_torch = 0

        kspace_path = os.path.join(sample_dir, f'simulated_kspace_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')

        if os.path.exists(kspace_path):
            kspace_complex = np.load(kspace_path, allow_pickle=True)
            kspace_torch = torch.from_numpy(kspace_complex)
        else:
            kspace_torch = kspace_path


        ground_truth_complex = dro['ground_truth_images']

        parMap = dro['parMap']
        aif = dro['aif']
        S0 = dro['S0']
        T10 = dro['T10']
        # mask = dro['mask']

        # ==========================================================
        # --- RECONSTRUCT THE MASK DICTIONARY ---
        # ==========================================================
        mask_dictionary_rebuilt = {}
        for tissue_name in self.TISSUE_NAMES:
            # Check if the key for this tissue (e.g., 'malignant') exists in the file
            if tissue_name in dro:
                # Load the boolean array and add it to the dictionary
                mask_dictionary_rebuilt[tissue_name] = dro[tissue_name]
        
        # 'mask' is now the dictionary of boolean arrays, just like your functions expect
        mask = mask_dictionary_rebuilt


        # --- Convert to PyTorch Tensors ---
        # Ground truth: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) # T, H, W
        ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)

        # CSMaps: (H, W, C) -> (1, C, H, W) [batch, coils, h, w]
        csmaps_torch = torch.from_numpy(csmaps).permute(2, 0, 1).unsqueeze(0)

        # csmaps_torch = torch.rot90(csmaps_torch, k=2, dims=[-2, -1])
        # ground_truth_torch = torch.rot90(ground_truth_torch, k=2, dims=[-2, -1])
        # grasp_recon_torch = torch.rot90(grasp_recon_torch, k=2, dims=[-3, -1])

        return kspace_torch, csmaps_torch, ground_truth_torch, grasp_recon_torch, mask, grasp_path #, parMap, aif, S0, T10, mask
    


class SimulatedSPFDataset(Dataset):
    """
    Dataset for loading the simulated data generated by your script.
    It loads the simulated k-space, coil sensitivity maps, and the
    ground truth dynamic image (DRO).
    """
    def __init__(self, root_dir, model_type, patient_ids):
        self.model_type = model_type
        self.root_dir = root_dir
        self.patient_ids = patient_ids

        # set default parameters to be changed before each call
        self.spokes_per_frame = 16
        self.num_frames = 18

        # Initialize sample paths based on default parameters
        self._update_sample_paths()
        

        self.TISSUE_NAMES = [
            'glandular', 'benign', 'malignant', 'muscle',
            'skin', 'liver', 'heart', 'vascular'
        ]

    def _update_sample_paths(self):
        self.dro_dir = os.path.join(self.root_dir, f'dro_{self.num_frames}frames')

        # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
        self.sample_paths = sorted(glob.glob(os.path.join(self.dro_dir, 'sample_*')))
        if not self.sample_paths:
            raise FileNotFoundError(f"No sample directories found in {self.dro_dir}. "
                                    "Please check the path to your simulated dataset.")
        
        # filter file list by patient ID substring
        filtered = []
        for fp in self.sample_paths:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in self.patient_ids):
                filtered.append(fp)

        self.sample_paths = filtered

        print(f"Found {len(self.sample_paths)} simulated samples in {self.dro_dir} for {self.num_frames} frames.")


    def load_kspace_from_csv_mapping(self, sample_id: str, mapping_file_path: str, data_dir: str) -> np.ndarray:
        """
        Parses a sample ID to get the DRO ID, uses a CSV mapping file to find the fastMRI ID,
        constructs the fastMRI HDF5 file path, and loads the k-space data.

        Args:
            sample_id (str): The sample ID string (e.g., "sample_020_sub20").
                            Expected format: "sample_XXX_subYY", where XXX is the DRO ID.
            mapping_file_path (str): The file path to the CSV file containing the
                                    DRO to fastMRIbreast ID mapping.
                                    Expected header: "DRO,fastMRIbreast".
            data_dir (str): The base directory where fastMRI HDF5 files are stored.
                            E.g., if files are in '/path/to/data/fastMRI_breast_157_2.h5',
                            then data_dir should be '/path/to/data'.

        Returns:
            numpy.ndarray: The complex k-space data from the corresponding fastMRI file.

        Raises:
            ValueError: If the sample_id format is invalid, DRO ID not found in mapping,
                        or the CSV mapping file is malformed.
            FileNotFoundError: If the mapping CSV or the constructed fastMRI HDF5 file does not exist.
            KeyError: If the 'kspace' dataset is not found within the HDF5 file.
            RuntimeError: For other issues encountered during file loading.
        """

        # --- 1. Parse the mapping CSV file into a dictionary ---
        dro_to_fastmri_map = {}
        try:
            with open(mapping_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                # Read and validate header
                try:
                    header = [h.strip() for h in next(reader)]
                except StopIteration:
                    raise ValueError(f"Mapping file is empty: {mapping_file_path}")

                if header != ['DRO', 'fastMRIbreast']:
                    raise ValueError(f"Mapping file header is invalid. Expected ['DRO', 'fastMRIbreast'], but got {header}.")

                # Parse data rows
                for i, row in enumerate(reader):
                    if not row:  # Skip empty lines
                        continue
                    try:
                        dro_id = int(row[0].strip())
                        fastmri_id = int(row[1].strip())
                        dro_to_fastmri_map[dro_id] = fastmri_id
                    except (ValueError, IndexError):
                        raise ValueError(f"Invalid mapping data in row {i+2} of {mapping_file_path}: {row}. Expected two integers.")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Mapping CSV file not found at: {mapping_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading or parsing mapping file {mapping_file_path}: {e}") from e


        # --- 2. Extract DRO ID from sample_id ---
        match = re.match(r"sample_(\d+)_sub\d+", sample_id)
        if not match:
            raise ValueError(
                f"Invalid sample_id format: '{sample_id}'. "
                f"Expected 'sample_XXX_subYY' where XXX is the DRO ID."
            )
        
        dro_id_from_sample = int(match.group(1))

        # --- 3. Get fastMRI ID using the mapping ---
        fastmri_id = dro_to_fastmri_map.get(dro_id_from_sample)
        if fastmri_id is None:
            raise ValueError(
                f"DRO ID {dro_id_from_sample} from sample_id '{sample_id}' "
                f"not found in the mapping file."
            )

        # --- 4. Construct the fastMRI file path ---
        file_name = f"fastMRI_breast_{fastmri_id}_2.h5"
        file_path = os.path.join(data_dir, file_name)

        # --- 5. Load k-space data from the HDF5 file ---
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"fastMRI HDF5 file not found at: {file_path}")

        try:
            with h5py.File(file_path, 'r') as f:
                if 'ktspace' not in f:
                    raise KeyError(f"'ktspace' dataset not found in file: {file_path}. "
                                f"Available keys: {list(f.keys())}")
                kspace_data = f['ktspace'][()] 
                return kspace_data
        except Exception as e:
            raise RuntimeError(f"Error loading k-space from {file_path}: {e}") from e
      


    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]

        # patient_id = os.path.basename(sample_dir)
        # print("patient id: ", patient_id)

        # # get fastMRI mapping and load real k-space
        # real_kspace = self.load_kspace_from_csv_mapping(patient_id, mapping_file_path="/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/DROSubID_vs_fastMRIbreastID.csv", data_dir="/ess/scratch/scratch1/rachelgordon/dce-8tf/binned_kspace")
        # print("ksapce shape: ", real_kspace.shape)

        print(f"  Testing {self.spokes_per_frame} spokes/frame with {self.num_frames} frames.")

        print("loading data from ", sample_dir)

        # Load the data from .npy files
        # kspace_complex = np.load(os.path.join(sample_dir, 'simulated_kspace.npy'))
        csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
        dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))

        grasp_path = os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')
        
        if os.path.exists(grasp_path):
            # print("loading grasp image from ", grasp_path)
            grasp_recon = np.load(grasp_path)

            # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
            grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
            grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)


            grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
            grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

        else:
            # print("setting grasp img to zero")
            grasp_recon_torch = 0


        ground_truth_complex = dro['ground_truth_images']

        # SELECT TIME WINDOW
        # ground_truth_complex = ground_truth_complex[..., self.window]

        smap_torch = rearrange(torch.tensor(csmaps), 'h w c -> c h w').unsqueeze(0)
        simImg_torch = torch.tensor(ground_truth_complex).to(torch.cfloat)



        parMap = dro['parMap']
        aif = dro['aif']
        S0 = dro['S0']
        T10 = dro['T10']
        # mask = dro['mask']

        # ==========================================================
        # --- RECONSTRUCT THE MASK DICTIONARY ---
        # ==========================================================
        mask_dictionary_rebuilt = {}
        for tissue_name in self.TISSUE_NAMES:
            # Check if the key for this tissue (e.g., 'malignant') exists in the file
            if tissue_name in dro:
                # Load the boolean array and add it to the dictionary
                mask_dictionary_rebuilt[tissue_name] = dro[tissue_name]
        
        # 'mask' is now the dictionary of boolean arrays, just like your functions expect
        mask = mask_dictionary_rebuilt


        # --- Convert to PyTorch Tensors ---
        # Ground truth: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) # T, H, W
        ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)

        # smap_torch = torch.rot90(smap_torch, k=2, dims=[-2, -1])
        # simImg_torch = torch.rot90(simImg_torch, k=2, dims=[0, 1])
        # grasp_recon_torch = torch.rot90(grasp_recon_torch, k=2, dims=[-3, -1])


        return smap_torch, simImg_torch, grasp_recon_torch, mask, grasp_path #, parMap, aif, S0, T10, mask


