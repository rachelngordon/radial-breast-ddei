import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KSpaceSliceDataset(Dataset):
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
        slice_idx=41,
        dataset_key="kspace",
        file_pattern="*.h5",
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

        # Find all matching HDF5 files under root_dir
        all_files = sorted(glob.glob(os.path.join(root_dir, file_pattern)))
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
        # self.slice_index_map = []
        # for fp in self.file_list:
        #     with h5py.File(fp, "r") as f:
        #         if self.dataset_key not in f:
        #             raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
        #         ds = f[self.dataset_key]
        #         num_slices = ds.shape[0]

        #     for z in range(num_slices):
        #         self.slice_index_map.append((fp, z))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Returns a single slice of k-space as a torch.Tensor.
        The output shape will be the standard (C=2, T, S, I) where C is [real, imag].
        """
        file_path = self.file_list[idx]

        with h5py.File(file_path, "r") as f:
            ds = torch.tensor(f[self.dataset_key][:])
            kspace_spatial_z = torch.fft.ifft(ds, dim=0, norm="ortho")
            kspace_slice = kspace_spatial_z[self.slice_idx]

        # Select the first coil
        kspace_single_coil = kspace_slice[:, 0, :, :]  # Shape: (T, S, I)

        # Separate real and imaginary components
        real_part = kspace_single_coil.real
        imag_part = kspace_single_coil.imag

        # Stack them along a new 'channel' dimension (dim=0).
        # This creates the final, standard (C=2, T, S, I) format.
        kspace_final = torch.stack([real_part, imag_part], dim=0).float()

        # The final shape is (2, num_timeframes, num_spokes, num_samples)
        # e.g., (2, 8, 36, 640)
        return kspace_final


# ----------------------------
# Example usage:
# ----------------------------
if __name__ == "__main__":
    # 1) Point this to wherever your HDF5 k-space files live
    root_dir = "/ess/scratch/scratch1/rachelgordon/dce-12tf/binned_kspace"

    # 2) If your HDF5 file stores k-space under a different key path, adjust dataset_key:
    dataset_key = "ktspace"  # change if your HDF5 group/dataset is named differently

    # 3) (Optional) Example transform: convert two‐channel real/imag → complex64
    def to_complex(x_np: "np.ndarray") -> "np.ndarray":
        """
        If x_np.shape = (C, H, W, 2) or similar where the last dim is [real, imag],
        convert to complex64 with shape (C, H, W).
        Adjust slicing logic if your real/imag channels are elsewhere.
        """
        real = x_np[..., 0].astype("float32")
        imag = x_np[..., 1].astype("float32")
        return (real + 1j * imag).astype("complex64")

    dataset = KSpaceSliceDataset(
        root_dir=root_dir, dataset_key=dataset_key, file_pattern="*.h5"
    )

    # 4) Wrap in DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 5) Iterate and inspect
    for batch_idx, kspace_batch in enumerate(loader):
        # kspace_batch.dtype could be torch.float32 or torch.complex64, depending on transform
        print(
            f"Batch {batch_idx}: k-space batch shape = {kspace_batch.shape}, dtype = {kspace_batch.dtype}"
        )
        # Now feed `kspace_batch` into your DDEI model or DC layer, etc.
        break
