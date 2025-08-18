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


class SliceDataset(Dataset):
    """
    A Dataset that:
      - Looks for all .h5/.hdf5 files under `root_dir`.
      - Each file is assumed to contain a dataset at `dataset_key`, with shape (... Z),
        where Z is the number of slices/partitions.
      - Splits each volume into Z separate examples (one per slice).
      - Returns each slice as a torch.Tensor.
    """

    def __init__(
        self,
        root_dir,
        patient_ids,
        dataset_key="kspace",
        file_pattern="*.h5",
        slice_idx: Optional[Union[int, range]] = 41,
        N_time = 8,
        N_coils=16,
        spf_aug=False,
    ):
        """
        Args:
            root_dir (str): Path to the folder containing all HDF5 k-space files.
            dataset_key (str): The key/path inside each .h5 file to the k-space dataset (e.g. "kspace").
            file_pattern (str): Glob pattern to match your HDF5 files (default "*.h5").
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_key = dataset_key
        self.slice_idx = slice_idx
        self.N_time = N_time
        self.N_coils = N_coils
        self.spf_aug = spf_aug

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
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)

        self.file_list = filtered

        if len(self.file_list) == 0:
            raise RuntimeError("No files matched the provided patient_ids filter.")

        # Build a list of (file_path, slice_index) for every slice in every volume
        self.slice_index_map = []
        for fp in self.file_list:
            with h5py.File(fp, "r") as f:
                if self.dataset_key not in f:
                    raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                ds = f[self.dataset_key]
                num_slices = ds.shape[0]


            slices_to_add = []

            # Case 1: Use a single, specific slice
            if isinstance(self.slice_idx, int):
                if self.slice_idx < num_slices:
                    slices_to_add = [self.slice_idx]
                else:
                    print(f"Warning: slice_idx {self.slice_idx} is out of bounds for {fp} "
                        f"(size {num_slices}). Skipping this file for this slice.")

            # Case 2: Use a specific range of slices
            elif isinstance(self.slice_idx, range):
                # Find the intersection of requested slices and available slices
                slices_to_add = [s for s in self.slice_idx if s < num_slices]
                if len(slices_to_add) < len(self.slice_idx):
                    print(f"Warning: Some requested slices were out of bounds for {fp}. "
                        f"Using only the valid slice indices from the provided list.")
            else:
                raise TypeError(f"slice_idx must be an int, range, or None, but got {type(self.slice_idx)}")

            for z in slices_to_add:
                self.slice_index_map.append((fp, z))


        print(f"Dataset initialized with {len(self.slice_index_map)} total slice examples.")


        self.spokes_range = [2, 4, 8, 12, 16, 24, 32, 36, 48]
        self.spf_weights = [1.0 / spf for spf in self.spokes_range]

    def load_dynamic_img(self, patient_id, slice):

        H = W = 320
        data = np.empty((2, self.N_time, H, W), dtype=np.float32)
        
        for t in range(self.N_time):
            # load image 
            img_path = f'/ess/scratch/scratch1/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii'

            # if os.path.exists(img_path):

            img = nib.load(img_path)
            img_data = img.get_fdata()

            if img_data.shape != (2, H, W):
                raise ValueError(f"{img_path} has shape {img_data.shape}; "
                                f"expected (2, {H}, {W})")

            data[:, t] = img_data.astype(np.float32)
            
            # else:
            #     return None

        return torch.from_numpy(data) 
    
    def load_csmaps(self, patient_id, slice):

        ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
        csmap_path = os.path.join(ground_truth_dir, patient_id + '_cs_maps', f'cs_map_slice_{slice:03d}.npy')

        csmap = np.load(csmap_path)

        return csmap.squeeze()


    def __len__(self):
        return len(self.slice_index_map)

    def __getitem__(self, idx):
        """
        Returns a single slice of k-space as a torch.Tensor.
        The output shape will be the standard (C=2, T, S, I) where C is [real, imag].
        """

        # if self.slice_idx is None or self.slice_idx == "None":
        file_path, current_slice_idx = self.slice_index_map[idx]
        current_slice_idx = int(current_slice_idx)
        # else:
        #     file_path = self.file_list[idx]
        #     current_slice_idx = self.slice_idx

        patient_id = file_path.split('/')[-1].strip('.h5')

        grasp_img = self.load_dynamic_img(patient_id, current_slice_idx)
        csmap = self.load_csmaps(patient_id, current_slice_idx)

        with h5py.File(file_path, "r") as f:
            ds = torch.tensor(f[self.dataset_key][:])
            kspace_slice = ds[current_slice_idx]


        if self.spf_aug:
            # flatten and re-bin k-space with desired spokes/frame
            total_spokes = kspace_slice.shape[0] * kspace_slice.shape[2]
            N_samples = kspace_slice.shape[-1]

            kspace = rearrange(kspace_slice, 't c sp sam -> t sp c sam')
            kspace_flat = kspace.contiguous().view(total_spokes, self.N_coils, N_samples)

            # spokes_per_frame = random.choice(self.spokes_range)
            spokes_per_frame = random.choices(self.spokes_range, self.spf_weights, k=1)[0]
            N_time = total_spokes // spokes_per_frame

            kspace_binned = kspace_flat.view(N_time, spokes_per_frame, self.N_coils, N_samples)
            kspace_slice = rearrange(kspace_binned, 't sp c sam -> t c sp sam')
        else:
            N_time = self.N_time
            N_samples = kspace_slice.shape[-1]
            spokes_per_frame = kspace_slice.shape[-2]



        # Select the first coil
        # if self.N_coils == 1:
        #     kspace_slice = kspace_slice[:, 0, :, :]  # Shape: (T, S, I)

        # Separate real and imaginary components
        real_part = kspace_slice.real
        imag_part = kspace_slice.imag

        # Stack them along a new 'channel' dimension (dim=0).
        # This creates the final, standard (C=2, T, S, I) format.
        kspace_final = torch.stack([real_part, imag_part], dim=0).float()


        # rotate to match orientation of validation set 
        # kspace_final = torch.rot90(kspace_final, k=2, dims=[-2, -1])
        kspace_final = torch.flip(kspace_final, dims=[-1])

        # You MUST also rotate the corresponding csmaps and grasp_img
        # csmap shape is likely (H, W, C) or (C, H, W). Assuming (H, W, C).
        # Convert to tensor, rotate, and convert back if needed, or rotate numpy array.
        csmap_tensor = torch.from_numpy(csmap)
        csmap_tensor = torch.rot90(csmap_tensor, k=2, dims=[-2, -1]) # Assuming dims 0,1 are H,W
        csmap = csmap_tensor.numpy()

        # The final shape is (2, num_timeframes, num_spokes, num_samples)
        # e.g., (2, 8, 16, 36, 640)
        return kspace_final, csmap, grasp_img, N_samples, spokes_per_frame, N_time




class SimulatedDataset(Dataset):
    """
    Dataset for loading the simulated data generated by your script.
    It loads the simulated k-space, coil sensitivity maps, and the
    ground truth dynamic image (DRO).
    """
    def __init__(self, root_dir, model_type, patient_ids):
        self.model_type = model_type
        # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
        self.sample_paths = sorted(glob.glob(os.path.join(root_dir, 'sample_*')))
        if not self.sample_paths:
            raise FileNotFoundError(f"No sample directories found in {root_dir}. "
                                    "Please check the path to your simulated dataset.")
        
        # filter file list by patient ID substring
        filtered = []
        for fp in self.sample_paths:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)

        self.sample_paths = filtered

        print(f"Found {len(self.sample_paths)} simulated samples in {root_dir} for this dataset.")

        self.TISSUE_NAMES = [
            'glandular', 'benign', 'malignant', 'muscle',
            'skin', 'liver', 'heart', 'vascular'
        ]

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]

        # Load the data from .npy files
        kspace_complex = np.load(os.path.join(sample_dir, 'simulated_kspace.npy'))
        csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
        dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))
        grasp_recon = np.load(os.path.join(sample_dir, 'grasp_recon.npy'))

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

        # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
        grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

        # CSMaps: (H, W, C) -> (1, C, H, W) [batch, coils, h, w]
        csmaps_torch = torch.from_numpy(csmaps).permute(2, 0, 1).unsqueeze(0)

        # --- Prepare k-space based on model type ---
        if self.model_type == "CRNN":
            # k-space: (C, Samples, T) -> (1, 2, C*Samples*T) -> reshape to training format
            # The training code expects k-space as (B, C, 2, Samples, T)
            # Your simulated k-space is (Coils, N_samples_per_frame * N_spokes, N_frames)
            kspace_real_imag = np.stack([kspace_complex.real, kspace_complex.imag]) # (2, C, Samples, T)
            kspace_torch = torch.from_numpy(kspace_real_imag)
            # Add batch and coil dimensions to match trainer. Assuming C=1 from trainer code.
            kspace_torch = kspace_torch.permute(2, 0, 1, 3).unsqueeze(0) # (B, C, 2, Samples, T)
        
        elif self.model_type == "LSFPNet":
            # LSFPNet expects complex k-space of shape (coils, samples, time)
            kspace_torch = torch.from_numpy(kspace_complex)
        
        else:
            raise ValueError(f"Unsupported model_type for SimulatedDataset: {self.model_type}")
        
        grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
        grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

        # return kspace_torch.float(), csmaps_torch.cfloat(), ground_truth_torch.float(), grasp_recon_torch.float(), parMap, aif, S0, T10, mask
        return kspace_torch, csmaps_torch, ground_truth_torch, grasp_recon_torch, mask#, parMap, aif, S0, T10, mask




class SimulatedSPFDataset(Dataset):
    """
    Dataset for loading the simulated data generated by your script.
    It loads the simulated k-space, coil sensitivity maps, and the
    ground truth dynamic image (DRO).
    """
    def __init__(self, root_dir, model_type, patient_ids):
        self.model_type = model_type
        # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
        self.sample_paths = sorted(glob.glob(os.path.join(root_dir, 'sample_*')))
        if not self.sample_paths:
            raise FileNotFoundError(f"No sample directories found in {root_dir}. "
                                    "Please check the path to your simulated dataset.")
        
        # filter file list by patient ID substring
        filtered = []
        for fp in self.sample_paths:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)

        self.sample_paths = filtered

        print(f"Found {len(self.sample_paths)} simulated samples in {root_dir} for this dataset.")

        self.TISSUE_NAMES = [
            'glandular', 'benign', 'malignant', 'muscle',
            'skin', 'liver', 'heart', 'vascular'
        ]

        # set default parameters to be changed before each call
        self.spokes_per_frame = 16
        self.num_frames = 20
        self.window = slice(1, 21)


    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]

        print(f"  Testing {self.spokes_per_frame} spokes/frame with {self.num_frames} frames.")

        # Load the data from .npy files
        # kspace_complex = np.load(os.path.join(sample_dir, 'simulated_kspace.npy'))
        csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
        dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))

        grasp_path = os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')
        
        if os.path.exists(grasp_path):
            grasp_recon = np.load(grasp_path)

            # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
            grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
            grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)


            grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
            grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

        else:
            print("setting grasp img to zero")
            grasp_recon_torch = 0


        ground_truth_complex = dro['ground_truth_images']

        # SELECT TIME WINDOW
        ground_truth_complex = ground_truth_complex[..., self.window]



        # SIMULATE KSPACE WITH DESIRED SPOKES PER FRAME
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


        return smap_torch, simImg_torch, grasp_recon_torch, mask, grasp_path #, parMap, aif, S0, T10, mask





# class SliceDatasetAug(Dataset):
#     """
#     A Dataset that:
#       - Looks for all .h5/.hdf5 files under `root_dir`.
#       - Each file is assumed to contain a dataset at `dataset_key`, with shape (... Z),
#         where Z is the number of slices/partitions.
#       - Splits each volume into Z separate examples (one per slice).
#       - Returns each slice as a torch.Tensor.
#     """

#     def __init__(
#         self,
#         root_dir,
#         patient_ids,
#         dataset_key="kspace",
#         file_pattern="*.h5",
#         slice_idx=41,
#         N_time = 8,
#         N_coils=16
#     ):
#         """
#         Args:
#             root_dir (str): Path to the folder containing all HDF5 k-space files.
#             dataset_key (str): The key/path inside each .h5 file to the k-space dataset (e.g. "kspace").
#             file_pattern (str): Glob pattern to match your HDF5 files (default "*.h5").
#         """
#         super().__init__()
#         self.root_dir = root_dir
#         self.dataset_key = dataset_key
#         self.slice_idx = slice_idx
#         self.N_time = N_time
#         self.N_coils = N_coils

#         # Find all matching HDF5 files under root_dir
#         all_files = sorted(glob.glob(os.path.join(root_dir, file_pattern)))
#         print("Number of files in root directory: ", len(all_files))

#         if len(all_files) == 0:
#             raise RuntimeError(
#                 f"No files found in {root_dir} matching pattern {file_pattern}"
#             )

#         # filter file list by patient ID substring
#         filtered = []
#         for fp in all_files:
#             fname = os.path.basename(fp)
#             # Check if any patient_id appears in the filename
#             if any(pid in fname for pid in patient_ids):
#                 filtered.append(fp)

#         self.file_list = filtered

#         if len(self.file_list) == 0:
#             raise RuntimeError("No files matched the provided patient_ids filter.")
        

#         self.spokes_range = [12, 16, 24, 32, 36, 48]


#     def load_dynamic_img(self, patient_id):

#         H = W = 320
#         data = np.empty((2, self.N_time, H, W), dtype=np.float32)

#         for t in range(self.N_time):
#             # load image 
#             img_path = f'/ess/scratch/scratch1/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{self.slice_idx:03d}_frame_{t:03d}.nii'

#             # if os.path.exists(img_path):

#             img = nib.load(img_path)
#             img_data = img.get_fdata()

#             if img_data.shape != (2, H, W):
#                 raise ValueError(f"{img_path} has shape {img_data.shape}; "
#                                 f"expected (2, {H}, {W})")

#             data[:, t] = img_data.astype(np.float32)
            
#             # else:
#             #     return None

#         return torch.from_numpy(data) 
    
#     def load_csmaps(self, patient_id):

#         ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
#         csmap_path = os.path.join(ground_truth_dir, patient_id + '_cs_maps', f'cs_map_slice_{self.slice_idx:03d}.npy')

#         csmap = np.load(csmap_path)

#         return csmap.squeeze()


#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         """
#         Returns a single slice of k-space as a torch.Tensor.
#         The output shape will be the standard (C=2, T, S, I) where C is [real, imag].
#         """

#         # load GRASP recon image
#         file_path = self.file_list[idx]
#         patient_id = file_path.split('/')[-1].strip('.h5')

#         grasp_img = self.load_dynamic_img(patient_id)
#         csmap = self.load_csmaps(patient_id)

#         with h5py.File(file_path, "r") as f:
#             ds = torch.tensor(f[self.dataset_key][:])
#             kspace_slice = ds[self.slice_idx]



#         # flatten and re-bin k-space with desired spokes/frame
#         total_spokes = kspace_slice.shape[0] * kspace_slice.shape[2]
#         N_samples = kspace_slice.shape[-1]

#         kspace = rearrange(kspace_slice, 't c sp sam -> t sp c sam')
#         kspace_flat = kspace.contiguous().view(total_spokes, self.N_coils, N_samples)

#         spokes_per_frame = random.choice(self.spokes_range)
#         N_time = total_spokes // spokes_per_frame

#         kspace_binned = kspace_flat.view(N_time, spokes_per_frame, self.N_coils, N_samples)
#         kspace_binned = rearrange(kspace_binned, 't sp c sam -> t c sp sam')


#         # Separate real and imaginary components
#         real_part = kspace_binned.real
#         imag_part = kspace_binned.imag

#         # Stack them along a new 'channel' dimension (dim=0).
#         # This creates the final, standard (C=2, T, S, I) format.
#         kspace_final = torch.stack([real_part, imag_part], dim=0).float()


#         # rotate to match orientation of validation set 
#         # kspace_final = torch.rot90(kspace_final, k=2, dims=[-2, -1])
#         kspace_final = torch.flip(kspace_final, dims=[-1])

#         # You MUST also rotate the corresponding csmaps and grasp_img
#         # csmap shape is likely (H, W, C) or (C, H, W). Assuming (H, W, C).
#         # Convert to tensor, rotate, and convert back if needed, or rotate numpy array.
#         csmap_tensor = torch.from_numpy(csmap)
#         csmap_tensor = torch.rot90(csmap_tensor, k=2, dims=[-2, -1]) # Assuming dims 0,1 are H,W
#         csmap = csmap_tensor.numpy()

#         # The final shape is (2, num_timeframes, num_spokes, num_samples)
#         # e.g., (2, 8, 16, 36, 640)
#         return kspace_final, csmap, grasp_img, N_samples, spokes_per_frame, N_time



