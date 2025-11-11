#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check presence of GRASP recon files for:
  (A) DRO-simulated dataset (mode: dro)    -> expects grasp_spf{S}_frames{T}.npy
  (B) Raw k-space GRASP recons (mode: raw) -> expects grasp_recon_{S}spf.npy

Usage examples:

# DRO mode (previous behavior):
python check_grasp_presence.py \
  --mode dro \
  --root /ess/scratch/scratch1/rachelgordon/dro_dataset \
  --split-json /ess/scratch/scratch1/rachelgordon/dro_dataset/data_split.json \
  --out /ess/scratch/scratch1/rachelgordon/dro_dataset/missing_grasp_val_dro.csv

# RAW mode (NEW):
python check_grasp_presence.py \
  --mode raw \
  --root /ess/scratch/scratch1/rachelgordon/zf_data_192_slices \
  --split-json /ess/scratch/scratch1/rachelgordon/dro_dataset/data_split.json \
  --out /ess/scratch/scratch1/rachelgordon/zf_data_192_slices/missing_grasp_val_raw.csv \
  --spokes 2,4,8,16,24,36 \
  --id-suffix _2

Notes
- RAW mode reads the "val" list (not "val_dro") from data_split.json.
- RAW mode assumes sample dirs are named fastMRI_breast_{ID}{suffix} (default suffix = "_2").
- DRO mode reads "val_dro" and expects per-T files grasp_spf{S}_frames{T}.npy with S = 288 // T.
"""

import argparse
import json
import os
import re
import csv
from collections import defaultdict

# ---------- Common ----------
DRO_DIR_PATTERN = re.compile(r"^dro_(\d+)frames$")

def parse_args():
    ap = argparse.ArgumentParser(description="Identify missing GRASP files (DRO or RAW).")
    ap.add_argument("--mode", choices=["dro", "raw"], required=True,
                    help="'dro' to check DRO-simulated recon files; 'raw' to check raw k-space recons.")
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory. DRO mode: dro_dataset root. RAW mode: zf_data_192_slices root.")
    ap.add_argument("--split-json", type=str, required=True,
                    help="Path to data_split.json.")
    ap.add_argument("--out", type=str, default="missing_grasp_report.csv",
                    help="CSV path to write report.")
    # RAW-mode options
    ap.add_argument("--spokes", type=str, default="2,4,8,16,24,36",
                    help="RAW mode: comma-separated list of spokes/frame to check (e.g., '2,4,8,16,24,36').")
    ap.add_argument("--id-suffix", type=str, default="_2",
                    help="RAW mode: directory suffix after sample id (default: '_2').")
    return ap.parse_args()

# ---------- DRO mode ----------
def load_val_dro(split_json_path):
    with open(split_json_path, "r") as f:
        data = json.load(f)
    if "val_dro" not in data or not isinstance(data["val_dro"], list):
        raise ValueError("data_split.json is missing 'val_dro' list.")
    return data["val_dro"]

def discover_temporal_dirs(root):
    """Return dict {T:int -> abs_path_to_dro_Tframes} for all dro_*frames folders."""
    mapping = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root directory not found: {root}")
    for name in os.listdir(root):
        m = DRO_DIR_PATTERN.match(name)
        if not m:
            continue
        T = int(m.group(1))
        mapping[T] = os.path.join(root, name)
    return dict(sorted(mapping.items()))

def expected_dro_filename(T):
    if T <= 0:
        raise ValueError(f"Invalid T={T}")
    if 288 % T != 0:
        raise ValueError(f"288 not divisible by T={T}; expected integer S=288//T.")
    S = 288 // T
    return f"grasp_spf{S}_frames{T}.npy"

def check_dro_mode(root, split_json_path, out_csv):
    val_dro = load_val_dro(split_json_path)
    dro_dirs = discover_temporal_dirs(root)

    missing = defaultdict(list)      # {T: [(sample, reason, expected_path)]}
    present_count = defaultdict(int) # {T: count}
    total_expected = defaultdict(int) # {T: len(val_dro)}

    for T, dro_path in dro_dirs.items():
        try:
            fname = expected_dro_filename(T)
        except ValueError as e:
            missing[T].append(("[ALL val_dro]", f"invalid_T: {e}", ""))
            continue

        for sample in val_dro:
            total_expected[T] += 1
            sample_dir = os.path.join(dro_path, sample)
            if not os.path.isdir(sample_dir):
                missing[T].append((sample, "sample_dir_missing", sample_dir))
                continue

            target = os.path.join(sample_dir, fname)
            if os.path.isfile(target):
                present_count[T] += 1
            else:
                missing[T].append((sample, "file_missing", target))

    # Console summary
    print("=" * 80)
    print("DRO MODE: GRASP reconstruction presence for val_dro samples\n")
    for T in sorted(dro_dirs.keys()):
        total = total_expected[T]
        present = present_count[T]
        miss = len(missing[T])
        print(f"T = {T:>3} frames | expected: {total:>3} | present: {present:>3} | missing: {miss:>3}")
        if miss > 0:
            preview = missing[T][:10]
            for sample, reason, path in preview:
                print(f"  - {sample:>15} | {reason:17} | {path}")
            if miss > len(preview):
                print(f"  ... and {miss - len(preview)} more")
        print("-" * 80)

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "T_frames", "sample_id", "status", "expected_path_or_dir"])
        for T in sorted(missing.keys()):
            for sample, reason, path in missing[T]:
                writer.writerow(["dro", T, sample, reason, path])

    print(f"\nCSV report written to: {out_csv}")
    print("=" * 80)

# ---------- RAW mode ----------
def load_val_ids(split_json_path):
    with open(split_json_path, "r") as f:
        data = json.load(f)
    if "val" not in data or not isinstance(data["val"], list):
        raise ValueError("data_split.json is missing 'val' list.")
    # entries look like "fastMRI_breast_015" → we keep as-is
    return data["val"]

def check_raw_mode(root, split_json_path, out_csv, spokes_list, id_suffix):
    val_ids = load_val_ids(split_json_path)

    # Expect directories named fastMRI_breast_{XXX}{suffix}
    # Files inside each dir: grasp_recon_{S}spf.npy for S in spokes_list
    missing = defaultdict(list)       # {S: [(sample_dir, reason, expected_path)]}
    present_count = defaultdict(int)  # {S: count}
    total_expected = defaultdict(int) # {S: count of val ids}

    # Pre-scan available directories for a quick existence check
    available_dirs = set(os.listdir(root)) if os.path.isdir(root) else set()

    for val_name in val_ids:
        # Convert "fastMRI_breast_015" -> "fastMRI_breast_015_2" (default suffix)
        sample_dir_name = f"{val_name}{id_suffix}"
        if sample_dir_name not in available_dirs:
            # mark all spokes as missing due to dir missing
            for S in spokes_list:
                total_expected[S] += 1
                missing[S].append((sample_dir_name, "sample_dir_missing",
                                   os.path.join(root, sample_dir_name)))
            continue

        sample_dir = os.path.join(root, sample_dir_name)
        for S in spokes_list:
            total_expected[S] += 1
            fname = f"grasp_recon_{S}spf.npy"
            fpath = os.path.join(sample_dir, fname)
            if os.path.isfile(fpath):
                present_count[S] += 1
            else:
                missing[S].append((sample_dir_name, "file_missing", fpath))

    # Console summary
    print("=" * 80)
    print("RAW MODE: GRASP reconstruction presence for raw k-space val samples\n")
    for S in sorted(spokes_list):
        total = total_expected[S]
        present = present_count[S]
        miss = len(missing[S])
        print(f"S = {S:>2} spf | expected: {total:>3} | present: {present:>3} | missing: {miss:>3}")
        if miss > 0:
            preview = missing[S][:10]
            for sample, reason, path in preview:
                print(f"  - {sample:>25} | {reason:17} | {path}")
            if miss > len(preview):
                print(f"  ... and {miss - len(preview)} more")
        print("-" * 80)

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "S_spf", "sample_dir", "status", "expected_path_or_dir"])
        for S in sorted(missing.keys()):
            for sample, reason, path in missing[S]:
                writer.writerow(["raw", S, sample, reason, path])

    print(f"\nCSV report written to: {out_csv}")
    print("=" * 80)

# ---------- Main ----------
def main():
    args = parse_args()
    if args.mode == "dro":
        check_dro_mode(args.root, args.split_json, args.out)
    else:
        # Parse spokes list for RAW mode
        spokes = [int(x) for x in args.spokes.split(",") if x.strip()]
        check_raw_mode(args.root, args.split_json, args.out, spokes, args.id_suffix)

if __name__ == "__main__":
    main()

