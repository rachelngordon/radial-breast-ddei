#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PK evaluation on a GRASP reconstruction stored inside a DRO sample directory.

Expected layout (example):
  /.../dro_dataset/dro_36frames/sample_030_sub30/
      grasp_spf8_frames36.npy
      dro_ground_truth.npz

The NPZ must contain (at least): parMap, aif, T10
Optional masks: malignant, benign, glandular, muscle, vascular, skin, liver, heart

Pipeline:
  - Load GRASP (T,H,W) and the matching DRO NPZ from the same folder
  - Timebase: Δt = 150 / T seconds (fastMRI breast total ~150 s)
  - AIF: resampled to T if needed
  - SI → concentration via SPGR inversion (R1 model, M0 from baseline frames)
  - Fit Extended Tofts per available ROI; compare to DRO parMap medians
  - Save: curves+fits and fit-vs-GT scatter
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

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
    assume_concentration_input: bool = False  # set True if GRASP is already in mM

TOTAL_SCAN_TIME_SEC = 150.0  # ~2.5 minutes

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
    """
    Spoiled GRE inversion: estimate R1(t) from signal, then C(t) = ΔR1 / r1.
    Here M0 is estimated from the mean of the first `baseline_frames` timepoints.
    """
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

    # Heuristic picking of vp if extra channels exist
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
    remaining = [ch for ch in extras if ch[0] != kep_idx]
    for (ci, ch) in remaining:
        ch_clip = np.clip(ch, 0.0, 1.0)
        vp = ch_clip  # simplest choice; adjust if you know exact channel order
        break
    if verbose:
        print(f"parMap channels: {parMap.shape[-1]} (assumed [Ktrans, ve{', vp' if C>2 else ''}, ...])")
    return {"Ktrans": Kt, "ve": ve, "vp": vp}

# ------------------------- Main -------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PK eval on GRASP recon inside a DRO sample directory.")
    parser.add_argument("--grasp_path", required=True,
                        help="Path to GRASP .npy, e.g., /.../dro_36frames/sample_030_sub30/grasp_spf8_frames36.npy")
    parser.add_argument("--out_prefix", default="pk_eval_grasp_in_dro")
    parser.add_argument("--assume_conc", action="store_true",
                        help="Set if GRASP input is already concentration (mM)")
    args = parser.parse_args()

    # Resolve paths
    sample_dir = os.path.dirname(args.grasp_path)
    dro_npz = os.path.join(sample_dir, "dro_ground_truth.npz")
    if not os.path.exists(dro_npz):
        raise FileNotFoundError(f"Could not find {dro_npz}")

    # Load GRASP and DRO
    grasp = np.load(args.grasp_path)
    grasp = ensure_time_axis_first(grasp)
    if np.iscomplexobj(grasp):
        grasp = np.abs(grasp)
    Tg, H, W = grasp.shape
    dro = np.load(dro_npz)

    # Required keys from DRO
    for k in ['parMap', 'aif', 'T10']:
        if k not in dro.files:
            raise KeyError(f"Missing key '{k}' in {dro_npz}. Found: {dro.files}")

    parMap = dro['parMap']
    aif_raw = dro['aif']
    T10 = dro['T10']

    # Timebase
    dt = TOTAL_SCAN_TIME_SEC / float(Tg)
    t = np.arange(Tg, dtype=np.float64) * dt

    # AIF → match T
    aif = aif_raw if aif_raw.size == Tg else resample_to_length(aif_raw, Tg)

    # Config + SI→Concentration
    cfg = DCEConfig(TR=0.0045, flip_deg=12.0, r1=4.5, baseline_frames=3,
                    use_extended_tofts=True, assume_concentration_input=args.assume_conc)

    C = grasp.astype(np.float64, copy=False) if cfg.assume_concentration_input \
        else spgr_invert_to_concentration(grasp, S0_map=np.zeros_like(T10), T10_map=T10, cfg=cfg)
    # (S0_map not used in current inversion; M0 comes from baseline frames)

    # Ground-truth PK maps (for ROI-median comparison)
    gt = pk_from_parmap(parMap, verbose=True)

    # Collect masks
    keys = ['malignant','benign','glandular','muscle','vascular','skin','liver','heart']
    masks = {k: dro[k].astype(bool) for k in keys if k in dro.files}
    masks = {k:v for k,v in masks.items() if v.shape == (H,W) and np.count_nonzero(v) > 0}
    print("Available non-empty ROIs:", list(masks.keys()))

    if not masks:
        print("No valid ROI masks found; exiting.")
        return

    # ROI plan
    roi_plan = {}
    pair_for_delta = None
    if 'malignant' in masks:
        roi_plan['lesion_malignant'] = masks['malignant']
        if 'glandular' in masks:
            roi_plan['background_glandular'] = masks['glandular']
            pair_for_delta = ('lesion_malignant', 'background_glandular')
    elif 'benign' in masks:
        roi_plan['lesion_benign'] = masks['benign']
        if 'glandular' in masks:
            roi_plan['background_glandular'] = masks['glandular']
            pair_for_delta = ('lesion_benign', 'background_glandular')
    else:
        if 'glandular' in masks: roi_plan['parenchyma'] = masks['glandular']
        if 'muscle' in masks:    roi_plan['muscle'] = masks['muscle']

    if not roi_plan:
        print("No lesion/benign/parenchyma/muscle ROI to evaluate; exiting.")
        return

    # Fit PK per ROI
    results: Dict[str, Dict] = {}
    for name, m in roi_plan.items():
        Ct = robust_roi_curve(C, m, how="median")
        Kt, ve, vp, fit_curve = fit_pk(Ct, aif, t, cfg)
        # ROI-median GT from maps
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
    out_base = args.out_prefix
    # 1) Curves+fits
    plt.figure(figsize=(6, 4))
    for name in results:
        plt.plot(t, results[name]["Ct"], label=f"{name} data")
        plt.plot(t, results[name]["fit_curve"], linestyle='--', label=f"{name} fit")
    plt.xlabel("Time (s)"); plt.ylabel("Concentration (mM)")
    plt.title("GRASP ROI enhancement & PK fits (matched DRO)")
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

    print(f"Saved: {out_base}_curves.png  and  {out_base}_scatter.png")

if __name__ == "__main__":
    main()
