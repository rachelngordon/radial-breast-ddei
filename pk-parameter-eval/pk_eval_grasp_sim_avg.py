#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch PK evaluation on GRASP reconstructions inside DRO sample directories,
with frames folder selected from spf via frames = total_spokes / spf.

Example:
python pk_eval_grasp_batch.py \
  --dro_root /ess/scratch/scratch1/rachelgordon/dro_dataset \
  --split_json /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json \
  --section val_dro \
  --spf 8
"""

from __future__ import annotations
import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.optimize import least_squares

@dataclass
class DCEConfig:
    TR: float = 0.0045
    flip_deg: float = 12.0
    r1: float = 4.5
    baseline_frames: int = 3
    use_extended_tofts: bool = True
    bounds_Ktrans: Tuple[float, float] = (0.0, 2.0)
    bounds_ve: Tuple[float, float] = (0.01, 0.99)
    bounds_vp: Tuple[float, float] = (0.0, 0.2)
    init_Ktrans: float = 0.2
    init_ve: float = 0.3
    init_vp: float = 0.02
    assume_concentration_input: bool = False

TOTAL_SCAN_TIME_SEC = 150.0  # ~2.5 minutes

def resample_to_length(y: np.ndarray, new_len: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == new_len:
        return y
    x_old = np.linspace(0.0, 1.0, num=y.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    return np.interp(x_new, x_old, y)

def ensure_time_axis_first(vol: np.ndarray, T_expected: Optional[int] = None) -> np.ndarray:
    v = np.asarray(vol)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D dynamic array, got {v.shape}")
    if T_expected is not None:
        if v.shape[0] == T_expected: return v
        if v.shape[-1] == T_expected: return np.moveaxis(v, -1, 0)
    if v.shape[0] <= min(v.shape[1], v.shape[2]):
        return v
    return np.moveaxis(v, -1, 0)

def spgr_invert_to_concentration(S: np.ndarray, T10_map: np.ndarray, cfg: DCEConfig) -> np.ndarray:
    S = S.astype(np.float64, copy=False)
    T10_map = T10_map.astype(np.float64, copy=False)
    TR = cfg.TR
    alpha = np.deg2rad(cfg.flip_deg)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    R10 = 1.0 / np.clip(T10_map, 1e-3, None)
    S_pre = S[:cfg.baseline_frames].mean(axis=0)
    denom = sin_a * (1 - np.exp(-TR * R10))
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    M0 = S_pre * (1 - np.exp(-TR * R10) * cos_a) / denom
    M0 = np.clip(M0, 1e-6, None)

    def invert_R1_frame(Sf):
        E1 = np.full_like(Sf, 0.9, dtype=np.float64)
        for _ in range(25):
            den = (1 - E1 * cos_a)
            den = np.where(np.abs(den) < 1e-8, 1e-8, den)
            S_pred = M0 * sin_a * (1 - E1) / den
            dS_dE1 = M0 * sin_a * (-(den) - (1 - E1) * (-cos_a)) / (den ** 2)
            E1 = np.clip(E1 - (S_pred - Sf) / (dS_dE1 + 1e-8), 1e-6, 0.999999)
        return -np.log(E1) / TR

    R1_t = np.stack([invert_R1_frame(S[k]) for k in range(S.shape[0])], axis=0)
    dR1 = np.clip(R1_t - R10[None, ...], 0.0, None)
    C = dR1 / cfg.r1
    C[~np.isfinite(C)] = 0.0
    return C

def tofts_conv(Cp: np.ndarray, t: np.ndarray, Ktrans: float, ve: float) -> np.ndarray:
    kep = Ktrans / max(ve, 1e-6)
    dt = np.diff(t, prepend=t[0])
    kern = np.exp(-kep * (t[:, None] - t[None, :]))
    kern = np.triu(kern)
    return Ktrans * (kern * (Cp[None, :] * dt[None, :])).sum(axis=1)

def extended_tofts(Cp: np.ndarray, t: np.ndarray, Kt: float, ve: float, vp: float) -> np.ndarray:
    return vp * Cp + tofts_conv(Cp, t, Kt, ve)

def fit_pk(Ct: np.ndarray, Cp: np.ndarray, t: np.ndarray, cfg: DCEConfig):
    m = np.isfinite(Ct) & np.isfinite(Cp) & np.isfinite(t)
    m[0:1] = True
    Ct, Cp, t = Ct[m], Cp[m], t[m]
    if Ct.size < 4:
        raise ValueError("Not enough finite timepoints to fit PK (need ≥4).")

    if cfg.use_extended_tofts:
        x0 = np.array([cfg.init_Ktrans, cfg.init_ve, cfg.init_vp], float)
        lb = np.array([cfg.bounds_Ktrans[0], cfg.bounds_ve[0], cfg.bounds_vp[0]], float)
        ub = np.array([cfg.bounds_Ktrans[1], cfg.bounds_ve[1], cfg.bounds_vp[1]], float)
        def resid(x): return extended_tofts(Cp, t, x[0], x[1], x[2]) - Ct
    else:
        x0 = np.array([cfg.init_Ktrans, cfg.init_ve], float)
        lb = np.array([cfg.bounds_Ktrans[0], cfg.bounds_ve[0]], float)
        ub = np.array([cfg.bounds_Ktrans[1], cfg.bounds_ve[1]], float)
        def resid(x): return tofts_conv(Cp, t, x[0], x[1]) - Ct

    r0 = resid(x0)
    if not np.isfinite(r0).all():
        raise ValueError("Residuals not finite at initial point.")
    res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=400)
    if cfg.use_extended_tofts:
        Kt, ve, vp = [float(v) for v in res.x]
        fit_curve = extended_tofts(Cp, t, Kt, ve, vp)
    else:
        Kt, ve = [float(v) for v in res.x]; vp = 0.0
        fit_curve = tofts_conv(Cp, t, Kt, ve)
    return Kt, ve, vp, fit_curve

def robust_roi_curve(vol_t: np.ndarray, mask: np.ndarray, how: str = "median") -> np.ndarray:
    vox = vol_t[:, mask]
    if vox.size == 0:
        raise ValueError("ROI mask is empty (no True voxels).")
    curve = np.median(vox, axis=1) if how == "median" else np.mean(vox, axis=1)
    if not np.isfinite(curve).all():
        alt = np.mean(vox, axis=1) if how == "median" else np.median(vox, axis=1)
        curve = alt
    if not np.isfinite(curve).all():
        T = vol_t.shape[0]
        out = np.empty(T, dtype=np.float64)
        for k in range(T):
            v = vox[k]; v = v[np.isfinite(v)]
            out[k] = np.median(v) if v.size else np.nan
        curve = out
    if not np.isfinite(curve).all():
        raise ValueError("ROI curve contains NaN/inf after aggregation.")
    return curve

def pk_from_parmap(parMap: np.ndarray) -> Dict[str, np.ndarray]:
    if parMap.ndim != 3:
        raise ValueError(f"parMap must be (H,W,C), got {parMap.shape}")
    H, W, C = parMap.shape
    if C < 2:
        raise ValueError(f"parMap needs ≥2 channels, got {C}.")
    Kt = parMap[..., 0].astype(np.float64)
    ve = parMap[..., 1].astype(np.float64)
    vp = np.zeros_like(Kt, dtype=np.float64)
    if C > 2:
        vp = np.clip(parMap[..., 2].astype(np.float64), 0.0, 1.0)
    return {"Ktrans": Kt, "ve": ve, "vp": vp}

def run_one_sample(sample_dir: str, spf: int, target_frames: int, cfg: DCEConfig):
    dro_npz = os.path.join(sample_dir, "dro_ground_truth.npz")
    if not os.path.exists(dro_npz):
        return None, "missing_dro_npz"

    # exact filename first, then fallback wildcard
    exact = os.path.join(sample_dir, f"grasp_spf{spf}_frames{target_frames}.npy")
    if os.path.exists(exact):
        grasp_path = exact
    else:
        pattern = os.path.join(sample_dir, f"grasp_spf{spf}_frames*.npy")
        matches = sorted(glob.glob(pattern))
        if not matches:
            return None, "missing_grasp_for_spf"
        grasp_path = matches[0]

    try:
        grasp = np.load(grasp_path)
        grasp = ensure_time_axis_first(grasp, T_expected=target_frames)
        if np.iscomplexobj(grasp):
            grasp = np.abs(grasp)
        Tg, H, W = grasp.shape

        dro = np.load(dro_npz)
        for k in ['parMap', 'aif', 'T10']:
            if k not in dro.files:
                return None, f"missing_{k}"
        parMap = dro['parMap']
        aif_raw = dro['aif']
        T10 = dro['T10']

        dt = TOTAL_SCAN_TIME_SEC / float(Tg)
        t = np.arange(Tg, dtype=np.float64) * dt
        aif = aif_raw if aif_raw.size == Tg else resample_to_length(aif_raw, Tg)

        C = grasp.astype(np.float64, copy=False) if cfg.assume_concentration_input \
            else spgr_invert_to_concentration(grasp, T10_map=T10, cfg=cfg)

        gt = pk_from_parmap(parMap)

        keys = ['malignant','benign','glandular','muscle','vascular','skin','liver','heart']
        masks = {k: dro[k].astype(bool) for k in keys if k in dro.files}
        masks = {k:v for k,v in masks.items() if v.shape == (H,W) and np.count_nonzero(v) > 0}
        if not masks:
            return None, "no_valid_masks"

        roi_plan = {}
        if 'malignant' in masks:
            roi_plan['lesion_malignant'] = masks['malignant']
            if 'glandular' in masks:
                roi_plan['background_glandular'] = masks['glandular']
        elif 'benign' in masks:
            roi_plan['lesion_benign'] = masks['benign']
            if 'glandular' in masks:
                roi_plan['background_glandular'] = masks['glandular']
        else:
            if 'glandular' in masks: roi_plan['parenchyma'] = masks['glandular']
            if 'muscle' in masks:    roi_plan['muscle'] = masks['muscle']

        if not roi_plan:
            return None, "no_roi_selected"

        abs_errs = []
        for name, m in roi_plan.items():
            Ct = robust_roi_curve(C, m, how="median")
            Kt, ve, vp, _ = fit_pk(Ct, aif, t, cfg)
            Kt_gt = float(np.median(gt['Ktrans'][m]))
            ve_gt = float(np.median(gt['ve'][m]))
            vp_gt = float(np.median(gt['vp'][m]))
            abs_errs.append({
                "roi": name,
                "Ktrans": abs(Kt - Kt_gt),
                "ve": abs(ve - ve_gt),
                "vp": abs(vp - vp_gt),
            })
        return abs_errs, None

    except Exception as e:
        return None, f"exception:{type(e).__name__}:{str(e)[:120]}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch PK MAE over val_dro for fixed GRASP spokes/frame.")
    parser.add_argument("--dro_root", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--section", default="val_dro")
    parser.add_argument("--spf", type=int, required=True)
    parser.add_argument("--total_spokes", type=int, default=288,
                        help="Total spokes in the simulated scan (default: 288).")
    parser.add_argument("--assume_conc", action="store_true")
    args = parser.parse_args()

    target_frames = args.total_spokes // args.spf
    if target_frames * args.spf != args.total_spokes:
        raise ValueError(f"total_spokes ({args.total_spokes}) must be divisible by spf ({args.spf}).")

    with open(args.split_json, "r") as f:
        split = json.load(f)
    if args.section not in split:
        raise KeyError(f"Section '{args.section}' not found in {args.split_json}. Keys: {list(split.keys())}")
    sample_ids: List[str] = split[args.section]

    def locate_sample_dir(sample_id: str) -> Optional[str]:
        pattern = os.path.join(args.dro_root, f"dro_{target_frames}frames", sample_id)
        matches = sorted(glob.glob(pattern))
        print("sample dir: ", matches[0] if matches else f"(not found) {pattern}")
        return matches[0] if matches else None

    cfg = DCEConfig(assume_concentration_input=args.assume_conc)

    all_abs_errs_Kt: List[float] = []
    all_abs_errs_ve: List[float] = []
    all_abs_errs_vp: List[float] = []

    n_samples_total = 0
    n_samples_used = 0
    fail_logs = []

    for sid in sample_ids:
        n_samples_total += 1
        sample_dir = locate_sample_dir(sid)
        if sample_dir is None:
            fail_logs.append((sid, "sample_dir_not_found"))
            continue

        abs_errs, reason = run_one_sample(sample_dir, args.spf, target_frames, cfg)
        if abs_errs is None:
            fail_logs.append((sid, reason))
            continue

        n_samples_used += 1
        for d in abs_errs:
            all_abs_errs_Kt.append(float(d["Ktrans"]))
            all_abs_errs_ve.append(float(d["ve"]))
            all_abs_errs_vp.append(float(d["vp"]))

    def robust_mae(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        vals = np.array(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        return float(np.median(vals))

    mae_Kt = robust_mae(all_abs_errs_Kt)
    mae_ve = robust_mae(all_abs_errs_ve)
    mae_vp = robust_mae(all_abs_errs_vp)

    print("\n================= PK Median Absolute Error (pooled over ROIs & samples) =================")
    print(f"Acceleration (spf): {args.spf}  ->  frames: {target_frames} (from total_spokes={args.total_spokes})")
    print(f"Samples used: {n_samples_used}/{n_samples_total}")
    print(f"MAE Ktrans: {mae_Kt:.4f}")
    print(f"MAE ve    : {mae_ve:.4f}")
    print(f"MAE vp    : {mae_vp:.4f}")
    print("=========================================================================================")

    if fail_logs:
        print("\nSkipped/failed samples:")
        for sid, why in fail_logs:
            print(f"  - {sid}: {why}")

if __name__ == "__main__":
    main()
