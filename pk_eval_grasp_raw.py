#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PK evaluation on a GRASP reconstruction by matching frame count to DRO variant.

- Loads GRASP (.npy) -> infer T from array shape (T,H,W).
- Maps fastMRI_breast ID -> DRO ID via CSV.
- Chooses dro_dataset/dro_{T}frames/sample_*_sub{DRO:02d}/dro_ground_truth.npz
  (falls back to closest dro_*frames if exact T not found).
- Uses DRO AIF/T10/S0/masks to convert GRASP signal -> concentration.
- Fits Extended Tofts per ROI and compares ROI-median PK to DRO ground-truth parMap.

Example:
  grasp path .../fastMRI_breast_015_2/grasp_recon_8spf.npy
  -> fastMRI ID = 15 -> DRO ID via CSV e.g. 30
  -> GRASP T inferred from array (or 288/8 in your workflow) ≈ 36
  -> load dro_dataset/dro_36frames/sample_*_sub30/dro_ground_truth.npz
"""

from __future__ import annotations
import os, re, glob, csv, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ------------------------- Config -------------------------

@dataclass
class DCEConfig:
    TR: float = 0.0045            # seconds
    flip_deg: float = 12.0        # degrees
    r1: float = 4.5               # 1/(mM*s)
    baseline_frames: int = 3
    use_extended_tofts: bool = True
    bounds_Ktrans: Tuple[float, float] = (0.0, 2.0)
    bounds_ve: Tuple[float, float] = (0.01, 0.99)
    bounds_vp: Tuple[float, float] = (0.0, 0.2)
    init_Ktrans: float = 0.2
    init_ve: float = 0.3
    init_vp: float = 0.02
    assume_concentration_input: bool = False  # set True if GRASP input is already concentration (mM)

TOTAL_SCAN_TIME_SEC = 150.0  # fastMRI breast ~2.5 minutes

# ------------------------- Helpers -------------------------

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
    if v.shape[0] <= min(v.shape[1], v.shape[2]):  # time usually smallest
        return v
    return np.moveaxis(v, -1, 0)

def spgr_invert_to_concentration(S: np.ndarray, S0_map: np.ndarray, T10_map: np.ndarray, cfg: DCEConfig) -> np.ndarray:
    # S0_map kept for API compatibility; M0 is estimated from baseline
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
        return -np.log(E1) / TR  # R1

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
        raise ValueError("Residuals not finite at initial point: check Ct/Cp/t and SI→C inversion.")

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

def pk_from_parmap(parMap: np.ndarray, verbose: bool = True) -> Dict[str, np.ndarray]:
    if parMap.ndim != 3:
        raise ValueError(f"parMap must be (H,W,C), got {parMap.shape}")
    H, W, C = parMap.shape
    if C < 2:
        raise ValueError(f"parMap needs ≥2 channels, got {C}.")
    Kt = parMap[..., 0].astype(np.float64)
    ve = parMap[..., 1].astype(np.float64)
    vp = np.zeros_like(Kt, dtype=np.float64)

    extras = [(c, parMap[..., c].astype(np.float64)) for c in range(2, C)]
    kep_idx = None
    if extras:
        eps = 1e-6
        kep_theory = Kt / (ve + eps)
        m = np.isfinite(kep_theory)
        if np.any(m):
            best_r = -np.inf
            for (ci, ch) in extras:
                x, y = kep_theory[m].ravel(), ch[m].ravel()
                if x.size > 10 and np.std(x) > 0 and np.std(y) > 0:
                    r = np.corrcoef(x, y)[0, 1]
                    if r > best_r: best_r, kep_idx = r, ci
            if verbose and kep_idx is not None:
                print(f"parMap: detected kep at channel {kep_idx} (corr≈{best_r:.3f})")
    remaining = [ch for ch in extras if ch[0] != kep_idx]
    vp_idx = None; best_score = -np.inf
    for (ci, ch) in remaining:
        ch_clip = np.clip(ch, 0.0, 1.0)
        frac_mean = np.mean(ch_clip); frac99 = np.quantile(ch_clip, 0.99)
        score = (0.3 - abs(frac99 - 0.15)) + (0.3 - abs(frac_mean - 0.05))
        if score > best_score:
            best_score = score; vp_idx = ci; vp = ch_clip
    if verbose:
        if vp_idx is not None: print(f"parMap: treated channel {vp_idx} as vp.")
        ignored = [ci for (ci, _) in extras if ci not in (kep_idx, vp_idx)]
        if ignored: print(f"parMap: ignored extra channel(s): {ignored}.")
    return {"Ktrans": Kt, "ve": ve, "vp": vp}

def read_mapping_csv(csv_path: str) -> Dict[int, int]:
    """Return dict fastMRIbreast_id -> DRO_id."""
    mapping = {}
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                dro_id = int(row['DRO'])
                fmri_id = int(row['fastMRIbreast'])
                mapping[fmri_id] = dro_id
            except Exception:
                continue
    if not mapping:
        raise RuntimeError(f"No mappings parsed from {csv_path}")
    return mapping

def extract_fastmri_id_from_path(path: str) -> Optional[int]:
    # Matches ...fastMRI_breast_015_2... -> 15
    m = re.search(r'fastMRI_breast_(\d+)', path)
    if m:
        return int(m.group(1).lstrip('0') or '0')
    return None

def frames_from_dirname(p: str) -> int:
    # p contains dro_XXXframes
    m = re.search(r'dro_(\d+)frames', p)
    return int(m.group(1)) if m else 0

def find_dro_sample_dir_exact_or_closest(dro_root: str, dro_id: int, T_target: int) -> str:
    """
    Prefer dro_{T_target}frames/sample_*_sub{dro_id:02d}; if not found,
    fall back to closest dro_*frames for that sub.
    """
    sub_tag = f"sub{dro_id:02d}"
    exact_root = os.path.join(dro_root, f"dro_{T_target}frames")
    exact_candidates = sorted(glob.glob(os.path.join(exact_root, f"sample_*_{sub_tag}")))
    if exact_candidates:
        return exact_candidates[0]

    # fallback: search all dro_*frames and pick closest in T
    all_candidates = sorted(glob.glob(os.path.join(dro_root, "dro_*frames", f"sample_*_{sub_tag}")))
    if not all_candidates:
        raise FileNotFoundError(f"No DRO sample dirs for DRO ID {dro_id} under {dro_root}")
    best = min(all_candidates, key=lambda p: abs(frames_from_dirname(p) - T_target))
    print(f"[WARN] Exact dro_{T_target}frames not found; using closest: {best}")
    return best

# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(description="PK eval on GRASP recon; match frame count to DRO variant.")
    parser.add_argument("--grasp_path", required=True,
                        help="Path to GRASP .npy, e.g., /.../fastMRI_breast_015_2/grasp_recon_8spf.npy")
    parser.add_argument("--mapping_csv", default="/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/DROSubID_vs_fastMRIbreastID.csv")
    parser.add_argument("--dro_root", default="/ess/scratch/scratch1/rachelgordon/dro_dataset",
                        help="Root containing dro_*frames/sample_XXX_subYY/")
    parser.add_argument("--out_png", default="pk_eval_grasp_matched.png")
    parser.add_argument("--fastmri_id", type=int, default=None, help="Override fastMRI ID if not parsable from path")
    parser.add_argument("--assume_conc", action="store_true", help="Set if GRASP input is already concentration (mM)")
    args = parser.parse_args()

    # Load GRASP volume and infer T from the array itself
    grasp = np.load(args.grasp_path)
    grasp = ensure_time_axis_first(grasp)
    if np.iscomplexobj(grasp):
        grasp = np.abs(grasp)
    Tg, H, W = grasp.shape
    print(f"GRASP shape: (T={Tg}, H={H}, W={W})")

    # Determine fastMRI ID
    fmri_id = args.fastmri_id if args.fastmri_id is not None else extract_fastmri_id_from_path(args.grasp_path)
    if fmri_id is None:
        raise RuntimeError("Could not infer fastMRI ID from grasp_path; pass --fastmri_id explicitly.")
    print(f"fastMRI ID: {fmri_id}")

    # Map to DRO ID
    mapping = read_mapping_csv(args.mapping_csv)
    if fmri_id not in mapping:
        raise KeyError(f"fastMRI ID {fmri_id} not found in mapping CSV {args.mapping_csv}")
    dro_id = mapping[fmri_id]
    print(f"Mapped DRO ID: {dro_id}")

    # Locate DRO sample dir for matched frames (dro_{Tg}frames)
    dro_sample_dir = find_dro_sample_dir_exact_or_closest(args.dro_root, dro_id, T_target=Tg)
    dro_npz = os.path.join(dro_sample_dir, "dro_ground_truth.npz")
    print(f"DRO sample dir: {dro_sample_dir}")
    dro = np.load(dro_npz)

    # Required keys from DRO
    for k in ['parMap', 'aif', 'S0', 'T10']:
        if k not in dro.files:
            raise KeyError(f"Missing key '{k}' in {dro_npz}. Found: {dro.files}")

    parMap = dro['parMap']
    aif_raw = dro['aif']
    S0 = dro['S0']          # retained for API parity
    T10 = dro['T10']

    # Build time vector from GRASP frames (Δt = 150 / T)
    dt = TOTAL_SCAN_TIME_SEC / float(Tg)
    t = np.arange(Tg, dtype=np.float64) * dt

    # AIF: expect len == Tg for matched dro_{Tg}frames; resample if not
    if aif_raw.size != Tg:
        aif = resample_to_length(aif_raw, Tg)
        print(f"AIF resampled: {aif_raw.size} -> {aif.size}")
    else:
        aif = aif_raw

    # Config + SI->Concentration
    cfg = DCEConfig(TR=0.0045, flip_deg=12.0, r1=4.5, baseline_frames=3,
                    use_extended_tofts=True, assume_concentration_input=args.assume_conc)

    C = grasp.astype(np.float64, copy=False) if cfg.assume_concentration_input \
        else spgr_invert_to_concentration(grasp, S0, T10, cfg)

    # Ground-truth PK from parMap (for ROI-median comparison)
    gt = pk_from_parmap(parMap, verbose=True)

    # Collect available ROI masks from DRO
    mask_keys = ['malignant','benign','glandular','muscle','vascular','skin','liver','heart']
    masks = {k: dro[k].astype(bool) for k in mask_keys if k in dro.files}
    masks = {k:v for k,v in masks.items() if v.shape == (H,W) and np.count_nonzero(v) > 0}
    print("Available non-empty ROIs:", list(masks.keys()))

    # ROI plan
    roi_plan = {}
    pair_for_delta = None
    if 'malignant' in masks:
        roi_plan['lesion_malignant'] = masks['malignant']
        if 'glandular' in masks:
            roi_plan['background_glandular'] = masks['glandular']
            pair_for_delta = ('lesion_malignant','background_glandular')
    elif 'benign' in masks:
        roi_plan['lesion_benign'] = masks['benign']
        if 'glandular' in masks:
            roi_plan['background_glandular'] = masks['glandular']
            pair_for_delta = ('lesion_benign','background_glandular')
    else:
        if 'glandular' in masks: roi_plan['parenchyma'] = masks['glandular']
        if 'muscle' in masks:    roi_plan['muscle'] = masks['muscle']

    if not roi_plan:
        print("No valid ROIs in this case; exiting.")
        return

    # Fit PK per ROI
    results: Dict[str, Dict] = {}
    for name, m in roi_plan.items():
        Ct = robust_roi_curve(C, m, how="median")
        Kt, ve, vp, fit_curve = fit_pk(Ct, aif, t, cfg)
        # ROI-median GT from parMap
        Kt_gt = float(np.median(gt['Ktrans'][m]))
        ve_gt = float(np.median(gt['ve'][m]))
        vp_gt = float(np.median(gt['vp'][m]))
        results[name] = {
            "Ct": Ct, "fit_curve": fit_curve, "nvox": int(np.count_nonzero(m)),
            "fit": {"Ktrans": Kt, "ve": ve, "vp": vp},
            "gt":  {"Ktrans": Kt_gt, "ve": ve_gt, "vp": vp_gt},
        }

    # Optional lesion–background deltas
    if pair_for_delta is not None:
        L, B = pair_for_delta
        if L in results and B in results:
            dK = results[L]['fit']['Ktrans'] - results[B]['fit']['Ktrans']
            dTTP = (t[int(np.argmax(results[L]['Ct']))] - t[int(np.argmax(results[B]['Ct']))])
            print(f"ΔKtrans ({L} - {B}) = {dK:.3f}")
            print(f"ΔTTP ({L} - {B}) = {dTTP:.2f} s")

    # Console table
    print("\n=== GRASP ROI PK (median) vs DRO Ground Truth ===")
    for name, r in results.items():
        print(f"[{name:18s}] nvox={r['nvox']:6d}  "
              f"Ktrans: {r['fit']['Ktrans']:.3f} (gt {r['gt']['Ktrans']:.3f})  "
              f"ve: {r['fit']['ve']:.3f} (gt {r['gt']['ve']:.3f})  "
              f"vp: {r['fit']['vp']:.3f} (gt {r['gt']['vp']:.3f})")

    # Plots
    out_base = os.path.splitext(args.out_png)[0]

    # 1) Ct curves + fits
    plt.figure(figsize=(6, 4))
    for name in results:
        plt.plot(t, results[name]["Ct"], label=f"{name} data")
        plt.plot(t, results[name]["fit_curve"], linestyle='--', label=f"{name} fit")
    plt.xlabel("Time (s)"); plt.ylabel("Concentration (mM)")
    plt.title("GRASP ROI enhancement & PK fits (frames matched to DRO)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_base + "_curves.png")

    # 2) Fit vs GT scatter
    keys = ["Ktrans", "ve", "vp"]
    plt.figure(figsize=(10, 3))
    for i, key in enumerate(keys, 1):
        plt.subplot(1, 3, i)
        x = np.array([r["gt"][key] for r in results.values()], dtype=np.float64)
        y = np.array([r["fit"][key] for r in results.values()], dtype=np.float64)
        plt.scatter(x, y)
        lim = max(1e-6, float(np.max([x.max() if x.size else 0, y.max() if y.size else 0])))
        lim = lim * 1.1 if lim > 0 else 1.0
        plt.plot([0, lim], [0, lim], linestyle="--")
        plt.xlabel(f"GT {key}"); plt.ylabel(f"Fitted {key}")
        plt.title(f"{key}: fit vs GT"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_base + "_scatter.png")

    print(f"Saved figures to: {out_base}_curves.png and {out_base}_scatter.png")

if __name__ == "__main__":
    main()
