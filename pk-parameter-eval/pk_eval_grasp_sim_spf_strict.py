#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-SPF PK evaluation on GRASP reconstructions embedded in DRO directories.

Key features:
- Common subject set across all SPF values (intersection of available malignant cases).
- Malignant-only ROI for sensitivity (skip subject/spf if no malignant mask).
- AIF↔Ct alignment (±2 frames, minimizes early-time SSE).
- Optional scale factor on Cp during PK fit (default ON).
- Dynamic baseline detection for SPGR inversion from AIF (pre-bolus frames).
- Outputs per-SPF MAE (Ktrans, ve, vp) and a plot vs temporal resolution.

Assumed layout per spf/frames:
  dro_root/dro_{frames}frames/sample_xxx_subyy/
    - grasp_spf{spf}_frames{frames}.npy
    - dro_ground_truth.npz  (must contain parMap, aif, T10, malignant mask)

Run example:
python pk_eval_grasp_multi_spf_strict.py \
  --dro_root /ess/scratch/scratch1/rachelgordon/dro_dataset \
  --split_json /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json \
  --section val_dro \
  --spf_list 2 4 8 16 24 36
"""

import os, json, argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ------------------------------- Config -------------------------------

@dataclass
class DCEConfig:
    TR: float = 0.0045
    flip_deg: float = 12.0
    r1: float = 4.5
    baseline_frames_min: int = 3
    baseline_frames_max: int = 12
    use_extended_tofts: bool = True
    bounds_Ktrans: Tuple[float, float] = (0.0, 2.0)
    bounds_ve: Tuple[float, float] = (0.01, 0.99)
    bounds_vp: Tuple[float, float] = (0.0, 0.2)
    init_Ktrans: float = 0.2
    init_ve: float = 0.3
    init_vp: float = 0.02

TOTAL_SCAN_TIME_SEC = 150.0     # total exam duration
TOTAL_SPOKES = 288              # simulated spokes per exam for all spfs

# --------------------------- Small utilities --------------------------

def robust_mae(vals: List[float]) -> float:
    if not vals: return float("nan")
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")

def ensure_T_first(vol: np.ndarray, T_expected: Optional[int] = None) -> np.ndarray:
    v = np.asarray(vol)
    if v.ndim != 3: raise ValueError(f"Expected (T,H,W)/(H,W,T), got {v.shape}")
    if T_expected is not None:
        if v.shape[0] == T_expected: return v
        if v.shape[-1] == T_expected: return np.moveaxis(v, -1, 0)
    return v if v.shape[0] <= v.shape[-1] else np.moveaxis(v, -1, 0)

def resample_to_length(y: np.ndarray, new_len: int) -> np.ndarray:
    y = np.asarray(y, float)
    if y.size == new_len: return y
    x_old = np.linspace(0.0, 1.0, y.size)
    x_new = np.linspace(0.0, 1.0, new_len)
    return np.interp(x_new, x_old, y)

def parse_parmap(pm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pm.ndim != 3 or pm.shape[-1] < 2:
        raise ValueError(f"parMap must be (H,W,C>=2), got {pm.shape}")
    Kt = pm[..., 0].astype(np.float64)
    ve = pm[..., 1].astype(np.float64)
    vp = pm[..., 2].astype(np.float64) if pm.shape[-1] > 2 else np.zeros_like(Kt)
    return Kt, ve, vp

# --------------------- SPGR inversion with dynamic baseline ---------------------

def choose_baseline_from_aif(aif: np.ndarray, pct: float = 0.05,
                             min_frames: int = 3, max_frames: int = 12) -> slice:
    """First frames where Cp < pct * max(Cp). Guarantees [min_frames, max_frames]."""
    aif = np.asarray(aif, float)
    thr = pct * float(np.max(aif)) if aif.size else 0.0
    idx = np.where(aif < thr)[0]
    if idx.size == 0:
        end = min_frames
    else:
        end = max(min_frames, min(idx[-1] + 1, max_frames))
    return slice(0, end)

def spgr_invert_to_conc(S: np.ndarray, T10: np.ndarray, cfg: DCEConfig,
                        baseline_slice: slice) -> np.ndarray:
    """Estimate concentration from SPGR signal via iterative R1 inversion."""
    S = S.astype(np.float64, copy=False)
    T10 = T10.astype(np.float64, copy=False)
    TR = cfg.TR
    alpha = np.deg2rad(cfg.flip_deg)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    R10 = 1.0 / np.clip(T10, 1e-3, None)
    S_pre = S[baseline_slice].mean(axis=0)
    denom = sin_a * (1 - np.exp(-TR * R10))
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    M0 = S_pre * (1 - np.exp(-TR * R10) * cos_a) / denom
    M0 = np.clip(M0, 1e-6, None)

    def invert_R1_frame(Sf):
        E1 = np.full_like(Sf, 0.9)
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

# --------------------------- PK models & fitting ---------------------------

def tofts_conv(Cp: np.ndarray, t: np.ndarray, Kt: float, ve: float) -> np.ndarray:
    kep = Kt / max(ve, 1e-6)
    dt = np.diff(t, prepend=t[0])
    kern = np.exp(-kep * (t[:, None] - t[None, :]))
    kern = np.triu(kern)
    return Kt * (kern * (Cp[None, :] * dt[None, :])).sum(axis=1)

def extended_tofts(Cp: np.ndarray, t: np.ndarray, Kt: float, ve: float, vp: float) -> np.ndarray:
    return vp * Cp + tofts_conv(Cp, t, Kt, ve)

def fit_pk_scaled(Ct: np.ndarray, Cp: np.ndarray, t: np.ndarray, cfg: DCEConfig,
                  use_scale: bool = True) -> Tuple[float, float, float, float]:
    """
    Return (Ktrans, ve, vp, s), where s is Cp scale factor (1.0 if use_scale=False).
    """
    Ct = np.asarray(Ct, float); Cp = np.asarray(Cp, float); t = np.asarray(t, float)
    m = np.isfinite(Ct) & np.isfinite(Cp) & np.isfinite(t); m[0:1] = True
    Ct, Cp, t = Ct[m], Cp[m], t[m]
    if Ct.size < 4: raise ValueError("Too few timepoints for PK fit")

    if use_scale:
        x0 = np.array([cfg.init_Ktrans, cfg.init_ve, cfg.init_vp, 1.0], float)
        lb = np.array([cfg.bounds_Ktrans[0], cfg.bounds_ve[0], cfg.bounds_vp[0], 0.2], float)
        ub = np.array([cfg.bounds_Ktrans[1], cfg.bounds_ve[1], cfg.bounds_vp[1], 5.0], float)
        def resid(x): return extended_tofts(x[3] * Cp, t, x[0], x[1], x[2]) - Ct
    else:
        x0 = np.array([cfg.init_Ktrans, cfg.init_ve, cfg.init_vp], float)
        lb = np.array([cfg.bounds_Ktrans[0], cfg.bounds_ve[0], cfg.bounds_vp[0]], float)
        ub = np.array([cfg.bounds_Ktrans[1], cfg.bounds_ve[1], cfg.bounds_vp[1]], float)
        def resid(x): return extended_tofts(Cp, t, x[0], x[1], x[2]) - Ct

    r0 = resid(x0)
    if not np.isfinite(r0).all(): raise ValueError("Non-finite residuals at init")
    res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=400)

    if use_scale:
        Kt, ve, vp, s = [float(v) for v in res.x]
    else:
        Kt, ve, vp = [float(v) for v in res.x]; s = 1.0
    return Kt, ve, vp, s

# ---------------------- AIF ↔ Ct tiny alignment (±2) ----------------------

def align_aif_to_ct(Ct: np.ndarray, Cp: np.ndarray, max_shift: int = 2) -> np.ndarray:
    """
    Integer shift Cp to minimize early-time SSE up to Ct peak. Returns shifted Cp (same length).
    """
    Ct = np.asarray(Ct, float); Cp = np.asarray(Cp, float)
    k_peak = int(np.argmax(Ct)) if Ct.size else 0
    best = (0, np.inf, Cp.copy())

    for sh in range(-max_shift, max_shift + 1):
        if sh < 0:
            Cp_s = Cp[-sh:]
            Ct_s = Ct[:Cp_s.size]
        elif sh > 0:
            Cp_s = Cp[:-sh]
            Ct_s = Ct[sh:]
        else:
            Cp_s, Ct_s = Cp, Ct
        k_use = min(k_peak, Ct_s.size)
        if k_use < 3: continue
        sse = np.sum((Ct_s[:k_use] - Cp_s[:k_use]) ** 2)
        if sse < best[1]:
            best = (sh, sse, Cp_s.copy())
    # Put back to original length by simple interpolation on index
    Cp_opt = best[2]
    return resample_to_length(Cp_opt, Cp.size)

# ------------------------------- Core eval -------------------------------

def eval_one_sample_spf(sample_dir: str, spf: int, frames: int, cfg: DCEConfig,
                        use_scale: bool) -> Optional[Tuple[float, float, float]]:
    """
    Evaluate malignant ROI for a single sample at one SPF.
    Returns (abs_err_Kt, abs_err_ve, abs_err_vp) or None if not evaluable.
    """
    dro_npz = os.path.join(sample_dir, "dro_ground_truth.npz")
    grasp_npy = os.path.join(sample_dir, f"grasp_spf{spf}_frames{frames}.npy")
    if not (os.path.exists(dro_npz) and os.path.exists(grasp_npy)):
        return None

    dro = np.load(dro_npz)
    if "malignant" not in dro:  # enforce malignant-only
        return None
    malignant = dro["malignant"]
    grasp = np.load(grasp_npy)
    grasp = ensure_T_first(grasp, frames)
    if np.iscomplexobj(grasp): grasp = np.abs(grasp)

    T, H, W = grasp.shape
    if malignant.shape != (H, W) or malignant.sum() == 0:
        return None

    aif = dro["aif"]
    if aif.size != T:
        aif = resample_to_length(aif, T)

    # timebase
    t = np.arange(T, dtype=np.float64) * (TOTAL_SCAN_TIME_SEC / T)

    # SPGR inversion with dynamic baseline from AIF
    cfg_local = cfg
    bsl = choose_baseline_from_aif(aif, pct=0.05,
                                   min_frames=cfg_local.baseline_frames_min,
                                   max_frames=cfg_local.baseline_frames_max)
    C = spgr_invert_to_conc(grasp, dro["T10"], cfg_local, bsl)

    # ROI median curve
    Ct = np.median(C[:, malignant.astype(bool)], axis=1)

    # AIF alignment (small shifts)
    Cp_aligned = align_aif_to_ct(Ct, aif, max_shift=2)

    # Fit
    Kt_hat, ve_hat, vp_hat, s_hat = fit_pk_scaled(Ct, Cp_aligned, t, cfg_local, use_scale=use_scale)

    # GT medians from parMap
    Kt_gt, ve_gt, vp_gt = parse_parmap(dro["parMap"])
    Kt_gt_m = float(np.median(Kt_gt[malignant]))
    ve_gt_m = float(np.median(ve_gt[malignant]))
    vp_gt_m = float(np.median(vp_gt[malignant]))

    return abs(Kt_hat - Kt_gt_m), abs(ve_hat - ve_gt_m), abs(vp_hat - vp_gt_m)

# --------------------------------- Main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dro_root", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--section", default="val_dro")
    ap.add_argument("--spf_list", nargs="+", type=int, default=[2,4,8,16,24,36])
    ap.add_argument("--no_scale", action="store_true", help="Disable Cp scale factor in PK fitting")
    args = ap.parse_args()

    with open(args.split_json, "r") as f:
        split = json.load(f)
    samples: List[str] = split[args.section]
    cfg = DCEConfig()
    use_scale = (not args.no_scale)

    # First pass: figure out which subjects are evaluable (malignant + GRASP) for ALL spfs
    evaluable_by_spf: Dict[int, set] = {}
    for spf in args.spf_list:
        frames = TOTAL_SPOKES // spf
        ok = set()
        for sid in samples:
            sd = os.path.join(args.dro_root, f"dro_{frames}frames", sid)
            dro_npz = os.path.join(sd, "dro_ground_truth.npz")
            grasp_npy = os.path.join(sd, f"grasp_spf{spf}_frames{frames}.npy")
            if not (os.path.exists(dro_npz) and os.path.exists(grasp_npy)):
                continue
            try:
                dro = np.load(dro_npz)
                if "malignant" in dro and dro["malignant"].sum() > 0:
                    ok.add(sid)
            except Exception:
                pass
        evaluable_by_spf[spf] = ok

    common_subjects = set(samples)
    for spf, ok in evaluable_by_spf.items():
        common_subjects &= ok

    print(f"Subjects in split: {len(samples)}")
    for spf in args.spf_list:
        print(f"  SPF {spf:2d}: evaluable malignant subjects = {len(evaluable_by_spf[spf])}")
    print(f"Common malignant subject set across all spfs: {len(common_subjects)}")
    if len(common_subjects) == 0:
        print("No common malignant subjects across all SPF values — aborting.")
        return

    # Second pass: compute MAE per spf on the common set only
    results = {}  # spf -> (MAE_Kt, MAE_ve, MAE_vp, dt, n_subjects)
    for spf in args.spf_list:
        frames = TOTAL_SPOKES // spf
        errs_Kt, errs_ve, errs_vp = [], [], []
        used = 0
        for sid in sorted(common_subjects):
            sd = os.path.join(args.dro_root, f"dro_{frames}frames", sid)
            out = eval_one_sample_spf(sd, spf, frames, cfg, use_scale=use_scale)
            if out is None:
                continue  # shouldn’t happen for common set, but be safe
            eK, eVe, eVp = out
            errs_Kt.append(eK); errs_ve.append(eVe); errs_vp.append(eVp)
            used += 1
        dt = TOTAL_SCAN_TIME_SEC / frames
        results[spf] = (robust_mae(errs_Kt), robust_mae(errs_ve), robust_mae(errs_vp), dt, used)

    # Print table
    print("\n=== PK Median Absolute Error (malignant ROI; common set) ===")
    print("spf | n | Δt(s/frame) |  MAE_Ktrans  |    MAE_ve    |   MAE_vp")
    for spf in args.spf_list:
        mK, mVe, mVp, dt, n = results[spf]
        print(f"{spf:3d} | {n:2d} | {dt:10.3f} | {mK:11.4f} | {mVe:11.4f} | {mVp:8.4f}")

    # Plot MAE vs temporal resolution
    dts = [results[spf][3] for spf in args.spf_list]
    maeK = [results[spf][0] for spf in args.spf_list]
    maeVe = [results[spf][1] for spf in args.spf_list]
    maeVp = [results[spf][2] for spf in args.spf_list]

    plt.figure(figsize=(6,4))
    plt.plot(dts, maeK, marker='o', label="Ktrans MAE")
    plt.plot(dts, maeVe, marker='o', label="ve MAE")
    plt.plot(dts, maeVp, marker='o', label="vp MAE")
    plt.xlabel("Temporal resolution (sec/frame)")
    plt.ylabel("Median Absolute Error")
    plt.title("PK MAE vs Temporal Resolution (malignant ROI, common subjects)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    out_png = "pk_mae_vs_temporal_resolution_malignant_common.png"
    plt.savefig(out_png)
    print(f"\nSaved plot: {out_png}")

if __name__ == "__main__":
    main()
