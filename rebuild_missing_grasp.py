#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild missing GRASP recons from raw k-space (val set).

- Input raw k-space: {data_dir}/{fastMRI_breast_XXX}_2.h5
- Output image: {out_root}/{fastMRI_breast_XXX}_2/grasp_recon_{S}spf.npy
  where out_root is the parent of the raw k-space dir, typically:
  out_root = os.path.dirname(data_dir) == /.../zf_data_192_slices

Optionally pass a missing CSV (from the checker) to rebuild only what’s missing.
"""

import gc, os
import csv
import json
import argparse
from typing import List, Optional

import numpy as np
import torch
import sigpy as sp

from raw_kspace_eval import raw_grasp_recon, process_kspace

try:
    import cupy as cp
except Exception:
    cp = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct GRASP images for missing raw-k-space entries."
    )
    p.add_argument("--slices", type=int, required=True,
                   help="Number of images per slab (N_slices) for process_kspace/raw_grasp_recon.")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing raw k-space H5 files (e.g., .../zf_data_192_slices/zf_kspace).")
    p.add_argument("--split_json", type=str, required=True,
                   help="Path to data_split.json (will use the 'val' list).")
    p.add_argument("--spokes", type=str, default="2,4,8,16,24,36",
                   help="Comma-separated spokes/frame to (re)build, e.g. '2,4,16,24'.")
    p.add_argument("--missing_csv", type=str, default="",
                   help="Optional: CSV from checker to restrict builds only to missing rows.")
    p.add_argument("--center_partition", type=int, default=31,
                   help="center_partition arg for process_kspace.")
    p.add_argument("--id_suffix", type=str, default="_2",
                   help="Sample directory suffix (default: _2).")
    p.add_argument("--overwrite", action="store_true",
                   help="If set, overwrite existing recon files.")
    p.add_argument("--device", type=int, default=0,
                   help="GPU id (ignored if CUDA not available). Use -1 for CPU.")
    return p.parse_args()


def load_val_ids(split_json_path: str) -> List[str]:
    with open(split_json_path, "r") as f:
        data = json.load(f)
    if "val" not in data or not isinstance(data["val"], list):
        raise ValueError("data_split.json is missing 'val' list.")
    return data["val"]


def read_missing_csv(missing_csv: str) -> set:
    """
    Returns a set of (sample_dir_name, S) pairs that need building.
    CSV schema (from checker): ["mode","S_spf","sample_dir","status","expected_path_or_dir"]
    """
    need = set()
    with open(missing_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            # Only act on true missing (ignore 'symlink_created' etc.)
            status = row.get("status", "")
            if status.startswith("symlink_created"):
                continue
            sample_dir = row.get("sample_dir", "")
            try:
                S = int(row.get("S_spf", ""))
            except Exception:
                continue
            if sample_dir and S:
                need.add((sample_dir, S))
    return need


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()

    # Keep OpenMP threads small; large values can inflate temp buffers
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Decide SigPy device
    use_gpu = (args.device >= 0 and torch.cuda.is_available())
    sp_device = sp.Device(args.device if use_gpu else -1)

    val_ids = load_val_ids(args.split_json)
    spokes_list = [int(s.strip()) for s in args.spokes.split(",") if s.strip()]

    data_dir  = os.path.abspath(args.data_dir)   # .../zf_kspace
    out_root  = os.path.dirname(data_dir)        # .../zf_data_192_slices

    restrict_to_missing = read_missing_csv(args.missing_csv) if args.missing_csv else None

    # ---- helper to aggressively free memory per step
    def _free_all():
        # Python/CPU
        gc.collect()
        # Torch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # CuPy pools (used by SigPy when running on GPU)
        if cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    for pid in val_ids:
        sample_dir_name = f"{pid}{args.id_suffix}"
        raw_h5 = os.path.join(data_dir, f"{pid}{args.id_suffix}.h5")
        out_dir = os.path.join(out_root, sample_dir_name)
        ensure_dir(out_dir)

        if not os.path.isfile(raw_h5):
            print(f"[SKIP] raw k-space not found: {raw_h5}")
            continue

        # Process each S separately to cap peak memory
        for S in spokes_list:
            if restrict_to_missing is not None and (sample_dir_name, S) not in restrict_to_missing:
                continue

            out_path = os.path.join(out_dir, f"grasp_recon_{S}spf.npy")
            if os.path.exists(out_path) and not args.overwrite:
                # Already present; skip
                continue

            try:
                # ---- reconstruct in single precision
                # Prepare k-space inputs for this S
                zf_kspace, binned_kspace, traj = process_kspace(
                    raw_h5,
                    device=sp_device,
                    spokes_per_frame=S,
                    images_per_slab=args.slices,
                    center_partition=args.center_partition,
                )

                # enforce complex64/float32 to halve memory footprint
                def _to_c64(x):
                    # SigPy arrays can be numpy/cupy; use .astype if available
                    xp = sp.backend.get_array_module(x)
                    if xp.iscomplexobj(x):
                        return x.astype(xp.complex64, copy=False)
                    else:
                        return x.astype(xp.float32, copy=False)

                zf_kspace     = _to_c64(zf_kspace)
                binned_kspace = _to_c64(binned_kspace)
                traj          = _to_c64(traj)

                _free_all()

                # Reconstruct
                recon = raw_grasp_recon(
                    zf_kspace,
                    binned_kspace,
                    traj,
                    N_slices=args.slices,
                    spokes_per_frame=S,
                    device=sp_device,
                )

                # ---- Optional: save smaller (magnitude float16). Comment out if you need complex.
                # If you need complex, keep complex64 and skip magnitude+float16.
                # if np.iscomplexobj(recon):
                #     # move to host as needed
                #     try:
                #         # SigPy may return cupy; ensure numpy
                #         recon_host = sp.to_device(recon, sp.Device(-1))
                #     except Exception:
                #         recon_host = np.asarray(recon)
                #     # magnitude + float16 cuts disk & RAM
                #     mag = np.abs(recon_host).astype(np.float16, copy=False)
                #     tmp = out_path + ".tmp.npy"
                #     np.save(tmp, mag)        # smallest practical write
                #     os.replace(tmp, out_path)
                #     del mag, recon_host
                # else:
                #     # real array: cast to float32 to avoid float64 spikes
                #     recon = np.asarray(recon, dtype=np.float32)
                #     tmp = out_path + ".tmp.npy"
                #     np.save(tmp, recon)
                #     os.replace(tmp, out_path)

                print(f"[SAVE] {out_path}")

            except MemoryError as e:
                print(f"[OOM] {sample_dir_name} S={S}: {e} — try running fewer S or add --mem in Slurm")
            except Exception as e:
                print(f"[ERROR] {sample_dir_name} S={S}: {e}")
            finally:
                # ---- Drop references and free caches aggressively
                for name in ["zf_kspace", "binned_kspace", "traj", "recon"]:
                    if name in locals():
                        del locals()[name]
                _free_all()


    # args = parse_args()

    # # Decide SigPy device
    # if args.device >= 0 and torch.cuda.is_available():
    #     sp_device = sp.Device(args.device)
    # else:
    #     sp_device = sp.Device(-1)

    # val_ids = load_val_ids(args.split_json)
    # spokes_list = [int(s.strip()) for s in args.spokes.split(",") if s.strip()]

    # # Paths
    # data_dir = os.path.abspath(args.data_dir)  # .../zf_data_192_slices/zf_kspace
    # out_root = os.path.dirname(data_dir)       # .../zf_data_192_slices

    # restrict_to_missing: Optional[set] = None
    # if args.missing_csv:
    #     restrict_to_missing = read_missing_csv(args.missing_csv)

    # # Iterate val IDs and spokes
    # for pid in val_ids:
    #     sample_dir_name = f"{pid}{args.id_suffix}"
    #     raw_h5 = os.path.join(data_dir, f"{pid}{args.id_suffix}.h5")
    #     out_dir = os.path.join(out_root, sample_dir_name)
    #     ensure_dir(out_dir)

    #     if not os.path.isfile(raw_h5):
    #         print(f"[SKIP] raw k-space not found: {raw_h5}")
    #         continue

    #     # Load once per patient (we can reuse zf_kspace/binned_kspace/traj across S)
    #     # process_kspace depends on spokes_per_frame, so we must call it per S.
    #     for S in spokes_list:
    #         # If a missing CSV was provided, skip pairs not in it
    #         if restrict_to_missing is not None:
    #             if (sample_dir_name, S) not in restrict_to_missing:
    #                 continue

    #         out_path = os.path.join(out_dir, f"grasp_recon_{S}spf.npy")
    #         if os.path.exists(out_path) and not args.overwrite:
    #             # Already present; skip
    #             # print(f"[OK] exists: {out_path}")
    #             continue

    #         try:
    #             # Prepare k-space inputs for this S
    #             zf_kspace, binned_kspace, traj = process_kspace(
    #                 raw_h5,
    #                 device=sp_device,
    #                 spokes_per_frame=S,
    #                 images_per_slab=args.slices,
    #                 center_partition=args.center_partition,
    #             )

    #             # Reconstruct
    #             recon = raw_grasp_recon(
    #                 zf_kspace,
    #                 binned_kspace,
    #                 traj,
    #                 N_slices=args.slices,
    #                 spokes_per_frame=S,
    #                 device=sp_device,
    #             )

    #             # Atomic-ish save (to avoid half-written files on interrupts)
    #             tmp = out_path + ".tmp.npy"
    #             np.save(tmp, recon)
    #             os.replace(tmp, out_path)

    #             print(f"[SAVE] {out_path}")

    #             # Free per-iteration buffers if on GPU
    #             del zf_kspace, binned_kspace, traj, recon

    #         except Exception as e:
    #             print(f"[ERROR] {sample_dir_name} S={S}: {e}")
    #             # Continue to next item without crashing the whole run
    #             continue


if __name__ == "__main__":
    main()
