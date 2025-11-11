#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check for missing DRO and fastMRI files and estimate storage needed.

Scopes:
  --scope all            : DRO (all 1..53), fastMRI (all 1..300)
  --scope dro_val_test   : DRO (all), fastMRI ids mapped from val_dro+test_dro via CSV
  --scope val_only       : DRO limited to val_dro; fastMRI limited to val list from data_split.json

Meta files (for dro_val_test or val_only):
  {meta_dir}/data_split.json
  {meta_dir}/DROSubID_vs_fastMRIbreastID.csv (only needed for dro_val_test)


python check_missing_data.py --dro-root /ess/scratch/scratch1/rachelgordon/dro_dataset --fastmri-root /ess/scratch/scratch1/rachelgordon/zf_data_192_slices --out-dro-csv /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/dro_missing_val.csv --out-fastmri-csv /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/fastmri_missing_val.csv --scope val_only --meta-dir /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data
"""

import os
import re
import csv
import json
import argparse
from typing import List, Dict, Optional, Tuple, Set

# -----------------------------
# Static configs
# -----------------------------
DRO_T_LIST = [8, 12, 18, 36, 72, 144]
DRO_SAMPLES_ALL = range(1, 54)      # 1..53 inclusive
FASTMRI_ALL_IDS = range(1, 301)     # 1..300 inclusive
FASTMRI_SPOKES = [2, 4, 8, 16, 24, 36]

# fastMRI size map from measurements (GiB)
FASTMRI_SIZE_GIB = {2: 43.0, 4: 22.0, 8: 11.0, 16: 5.3, 24: 3.6, 36: 2.4}
GIB_TO_MB = 1024.0

# DRO anchors (MB) @ T=36
DRO_GRASP_ANCHOR_MB_AT_T36 = 57.0
DRO_SIMK_ANCHOR_MB_AT_T36  = 23.0


# -----------------------------
# Helpers
# -----------------------------
def dro_required_files_for(T: int) -> List[Tuple[str, Optional[float], Optional[float]]]:
    S = 288 // T
    est_grasp_mb = DRO_GRASP_ANCHOR_MB_AT_T36 * (T / 36.0)
    est_simk_mb  = DRO_SIMK_ANCHOR_MB_AT_T36  * (T / 36.0)
    return [
        ("csmaps.npy", None, None),
        ("dro_ground_truth.npz", None, None),
        (f"grasp_spf{S}_frames{T}.npy", est_grasp_mb, est_grasp_mb / GIB_TO_MB),
        (f"simulated_kspace_spf{S}_frames{T}.npy", est_simk_mb, est_simk_mb / GIB_TO_MB),
    ]

def fastmri_required_files_for() -> List[Tuple[str, float, float]]:
    out = []
    for S in FASTMRI_SPOKES:
        gib = FASTMRI_SIZE_GIB[S]
        out.append((f"grasp_recon_{S}spf.npy", gib * GIB_TO_MB, gib))
    return out

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_csv(rows: List[Dict[str, str]], out_csv: str):
    ensure_parent(out_csv)
    fieldnames = ["dataset", "sample_id", "context_dir", "missing_file", "est_size_mb", "est_size_gib"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            for k in fieldnames:
                r.setdefault(k, "")
            w.writerow(r)

def load_data_split(meta_dir: str) -> Dict:
    p = os.path.join(meta_dir, "data_split.json")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing {p}")
    with open(p, "r") as f:
        return json.load(f)

def load_dro_to_fastmri_map(csv_path: str) -> Dict[int, int]:
    mapping = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = int(row["DRO"])
                m = int(row["fastMRIbreast"])
                mapping[d] = m
            except Exception:
                continue
    return mapping

def parse_dro_id(sample_name: str) -> Optional[int]:
    # sample_XXX_subYY -> YY
    m = re.match(r"^sample_(\d+)_sub(\d+)$", sample_name.strip())
    return int(m.group(2)) if m else None

def parse_fastmri_id(name: str) -> Optional[int]:
    # fastMRI_breast_XXX -> XXX
    m = re.match(r"^fastMRI_breast_(\d+)$", name.strip())
    return int(m.group(1)) if m else None


# -----------------------------
# Checks
# -----------------------------
def check_dro(dro_root: str, dro_ids_to_check: Optional[Set[int]] = None) -> Tuple[List[Dict[str, str]], float, float]:
    rows: List[Dict[str, str]] = []
    tot_mb = 0.0
    tot_gib = 0.0

    ids_iter = sorted(dro_ids_to_check) if dro_ids_to_check else DRO_SAMPLES_ALL

    for T in DRO_T_LIST:
        dro_dir = os.path.join(dro_root, f"dro_{T}frames")
        for i in ids_iter:
            sample_dir = os.path.join(dro_dir, f"sample_{i:03d}_sub{i}")
            for fname, est_mb, est_gib in dro_required_files_for(T):
                fpath = os.path.join(sample_dir, fname)
                if not os.path.isfile(fpath):
                    rows.append({
                        "dataset": "DRO",
                        "sample_id": f"{i}",
                        "context_dir": sample_dir,
                        "missing_file": fname,
                        "est_size_mb": f"{est_mb:.1f}" if est_mb is not None else "unknown",
                        "est_size_gib": f"{est_gib:.3f}" if est_gib is not None else "unknown",
                    })
                    if est_mb is not None:
                        tot_mb += est_mb
                    if est_gib is not None:
                        tot_gib += est_gib
    return rows, tot_mb, tot_gib

def check_fastmri(fastmri_root: str, ids_to_check: List[int]) -> Tuple[List[Dict[str, str]], float, float]:
    rows: List[Dict[str, str]] = []
    tot_mb = 0.0
    tot_gib = 0.0
    reqs = fastmri_required_files_for()

    for i in ids_to_check:
        sample_dir = os.path.join(fastmri_root, f"fastMRI_breast_{i:03d}_2")
        for fname, est_mb, est_gib in reqs:
            fpath = os.path.join(sample_dir, fname)
            if not os.path.isfile(fpath):
                rows.append({
                    "dataset": "fastMRI",
                    "sample_id": f"{i}",
                    "context_dir": sample_dir,
                    "missing_file": fname,
                    "est_size_mb": f"{est_mb:.1f}",
                    "est_size_gib": f"{est_gib:.3f}",
                })
                tot_mb += est_mb
                tot_gib += est_gib
    return rows, tot_mb, tot_gib


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Check missing DRO and fastMRI files with optional val-only scope.")
    ap.add_argument("--dro-root", required=True, help="Path to DRO root (e.g., /ess/scratch/scratch1/rachelgordon/dro_dataset)")
    ap.add_argument("--fastmri-root", required=True, help="Path to fastMRI root (e.g., /ess/scratch/scratch1/rachelgordon/zf_data_192_slices)")
    ap.add_argument("--out-dro-csv", required=True, help="Output CSV for missing DRO entries")
    ap.add_argument("--out-fastmri-csv", required=True, help="Output CSV for missing fastMRI entries")
    ap.add_argument("--scope", choices=["all", "dro_val_test", "val_only"], default="all",
                    help="Apply dataset scope to checks (see header docstring).")
    ap.add_argument("--meta-dir", default="/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data",
                    help="Directory containing data_split.json and mapping CSV (needed for val_only/dro_val_test).")
    args = ap.parse_args()

    # Decide DRO ids and fastMRI ids per scope
    dro_ids_to_check: Optional[Set[int]] = None
    fastmri_ids: List[int]

    if args.scope == "all":
        dro_ids_to_check = None                               # all DRO (1..53)
        fastmri_ids = list(FASTMRI_ALL_IDS)                   # all fastMRI (1..300)
        scope_desc = "ALL (DRO 1..53, fastMRI 1..300)"

    elif args.scope == "dro_val_test":
        ds = load_data_split(args.meta_dir)
        val_dro  = ds.get("val_dro", [])
        test_dro = ds.get("test_dro", [])
        # DRO: still check ALL (as in previous behavior)
        dro_ids_to_check = None

        # fastMRI ids: map dro val+test -> fastMRI via CSV
        csv_path = os.path.join(args.meta_dir, "DROSubID_vs_fastMRIbreastID.csv")
        dro2fmri = load_dro_to_fastmri_map(csv_path)

        def sub_ids(names: List[str]) -> Set[int]:
            out = set()
            for n in names:
                sid = parse_dro_id(n)
                if sid is not None:
                    out.add(sid)
            return out

        dro_ids = sub_ids(val_dro) | sub_ids(test_dro)
        fastmri_ids = sorted({dro2fmri[d] for d in dro_ids if d in dro2fmri})
        scope_desc = f"fastMRI mapped from DRO val+test ({len(fastmri_ids)} ids); DRO all"

    else:  # val_only
        ds = load_data_split(args.meta_dir)

        # DRO ids: from val_dro
        val_dro = ds.get("val_dro", [])
        dro_val_ids: Set[int] = set()
        for name in val_dro:
            sid = parse_dro_id(name)
            if sid is not None:
                dro_val_ids.add(sid)
        dro_ids_to_check = dro_val_ids

        # fastMRI ids: directly from 'val' list
        val_list = ds.get("val", [])
        fastmri_ids = []
        for name in val_list:
            fid = parse_fastmri_id(name)
            if fid is not None:
                fastmri_ids.append(fid)
        fastmri_ids.sort()
        scope_desc = f"VAL ONLY (DRO val_dro {len(dro_val_ids)} ids; fastMRI val {len(fastmri_ids)} ids)"

    # Run checks
    dro_rows, dro_mb, dro_gib = check_dro(args.dro_root, dro_ids_to_check)
    fmri_rows, fmri_mb, fmri_gib = check_fastmri(args.fastmri_root, fastmri_ids)

    # Write CSVs
    write_csv(dro_rows, args.out_dro_csv)
    write_csv(fmri_rows, args.out_fastmri_csv)

    # Totals
    total_mb = dro_mb + fmri_mb
    print("=== Summary ===")
    print(f"Scope: {scope_desc}")
    print(f"DRO missing entries: {len(dro_rows)}")
    print(f"fastMRI missing entries: {len(fmri_rows)}")
    print("\nEstimated storage for missing files (excluding unknown-size DRO files):")
    print(f"  DRO:     {dro_mb:.1f} MB  (~{dro_mb/1024:.2f} GiB)")
    print(f"  fastMRI: {fmri_mb:.1f} MB  (~{fmri_mb/1024:.2f} GiB)")
    print(f"  TOTAL:   {total_mb:.1f} MB  (~{total_mb/1024:.2f} GiB, ~{total_mb/1000:.2f} GB)")

    if any(r["est_size_mb"] == "unknown" for r in dro_rows):
        print("\nNote: Some DRO files (csmaps.npy, dro_ground_truth.npz) have unknown sizes and were excluded from totals.")

if __name__ == "__main__":
    main()

