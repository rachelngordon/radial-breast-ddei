#!/usr/bin/env python3
import os
import re
import csv
import argparse
from typing import Dict, Any, Optional
import torch
import numpy as np
from einops import rearrange
from radial import to_torch_complex
from radial_lsfp import MCNUFFT
from utils import prep_nufft

FNAME_RE = re.compile(r"spf(?P<spf>\d+)_frames(?P<frames>\d+)\.npy$")

def parse_missing_filename(fname: str) -> Dict[str, int]:
    """
    Parse strings like 'simulated_kspace_spf2_frames144.npy' → {'spf': 2, 'frames': 144}.
    """
    m = FNAME_RE.search(fname)
    if not m:
        raise ValueError(f"Could not parse SPF/frames from filename: {fname}")
    return {"spf": int(m.group("spf")), "frames": int(m.group("frames"))}


def main():
    ap = argparse.ArgumentParser(description="Load and parse missing DRO entries.")
    ap.add_argument("--csv", required=True, help="Path to missing_dro_val.csv")
    args = ap.parse_args()

    device = torch.device("cuda")

    rows = []
    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:

            filename = row["missing_file"]
            context_dir = row["context_dir"]
            try:
                parsed = parse_missing_filename(filename)
            except Exception as e:
                print(f"[WARN] {e}")
                continue

            gt_path = os.path.join(context_dir, 'dro_ground_truth.npz')
            csmaps_path = os.path.join(context_dir, 'csmaps.npy')

            csmaps = np.load(csmaps_path)

            dro = np.load(gt_path)
            ground_truth_complex = dro['ground_truth_images']

            ground_truth = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) 
            ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=0)

            csmap = torch.from_numpy(csmaps).permute(2, 0, 1).unsqueeze(0)

            csmap = csmap.to(device)   # Remove batch dim
            ground_truth = ground_truth.unsqueeze(0).to(device) # Shape: (1, 2, T, H, W)

            # simulate k-space for validation 
            ground_truth_for_physics = rearrange(to_torch_complex(ground_truth), 'b t h w -> b h w t')

            # SIMULATE KSPACE
            # define physics object for evaluation
            N_samples = 640
            N_spokes_eval = parsed['spf']
            N_time_eval = parsed['frames']

            eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, N_spokes_eval, N_time_eval)
            eval_ktraj = eval_ktraj.to(device)
            eval_dcomp = eval_dcomp.to(device)
            eval_nufft_ob = eval_nufft_ob.to(device)
            eval_adjnufft_ob = eval_adjnufft_ob.to(device)

            eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

            measured_kspace = eval_physics(False, ground_truth_for_physics, csmap)

            # save k-space 
            kspace_path = os.path.join(context_dir, filename)
            np.save(kspace_path, measured_kspace.cpu().numpy())

            print(f"simulated k-space of shape {measured_kspace.shape} saved to {kspace_path}")




if __name__ == "__main__":
    main()
