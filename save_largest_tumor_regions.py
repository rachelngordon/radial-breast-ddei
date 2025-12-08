from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label

# Source and destination directories
SEG_ROOT = Path("/net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations")
OUT_ROOT = SEG_ROOT.parent / "tumor_segmentations_lcr"


def largest_component(mask_path: Path) -> tuple[np.ndarray, np.ndarray, nib.nifti1.Nifti1Header]:
    nii = nib.load(str(mask_path))
    data = nii.get_fdata()
    bin_mask = data > 0

    labeled, n = label(bin_mask)
    if n == 0:
        return np.zeros_like(bin_mask, dtype=np.uint8), nii.affine, nii.header

    counts = np.bincount(labeled.ravel())[1:]  # skip background
    largest_label = counts.argmax() + 1
    largest = (labeled == largest_label).astype(np.uint8)
    return largest, nii.affine, nii.header


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    masks = sorted(SEG_ROOT.glob("*.nii.gz"))
    print(f"Found {len(masks)} segmentation files in {SEG_ROOT}")

    for p in masks:
        largest, affine, header = largest_component(p)
        out_path = OUT_ROOT / p.name
        nib.save(nib.Nifti1Image(largest, affine, header), str(out_path))
        print(f"Saved largest component to {out_path}")


if __name__ == "__main__":
    main()
