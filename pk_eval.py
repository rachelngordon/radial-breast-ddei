#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PK evaluation on a single DRO sample.
- Infers temporal resolution from DRO frame count (Δt = 150/T).
- Resamples AIF to match T.
- SI -> concentration using S0, T10 (SPGR inversion).
- Fits (extended) Tofts on ROI median curves.
- Compares to ground-truth parMap.

Edit `sample_dir` below.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ============================== Config ==============================

@dataclass
class DCEConfig:
    TR: float = 0.0045            # seconds (for SPGR inversion)
    flip_deg: float = 12.0        # degrees
    r1: float = 4.5               # 1/(mM*s) (tissue relaxivity)
    baseline_frames: int = 3      # frames used to estimate pre-contrast baseline
    use_extended_tofts: bool = True
    bounds_Ktrans: Tuple[float, float] = (0.0, 2.0)
    bounds_ve: Tuple[float, float] = (0.01, 0.99)
    bounds_vp: Tuple[float, float] = (0.0, 0.2)
    init_Ktrans: float = 0.2
    init_ve: float = 0.3
    init_vp: float = 0.02
    assume_concentration_input: bool = False  # set True if ground_truth_images are already concentration (mM)

TOTAL_SCAN_TIME_SEC = 150.0  # fastMRI breast ~2.5 minutes


# ============================ Utilities =============================

def resample_to_length(y: np.ndarray, new_len: int) -> np.ndarray:
    """Linear resample a 1D signal y to new_len samples."""
    y = np.asarray(y, dtype=np.float64)
    if y.size == new_len:
        return y
    x_old = np.linspace(0.0, 1.0, num=y.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    return np.interp(x_new, x_old, y)


def ensure_time_axis_first(vol: np.ndarray, T_expected: Optional[int] = None) -> np.ndarray:
    """
    Accepts (T,H,W) or (H,W,T). If T_expected is provided, use it to disambiguate.
    Otherwise, returns as-is if first dim seems like time.
    """
    v = np.asarray(vol)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D array for dynamic volume, got {v.shape}")
    if T_expected is not None:
        if v.shape[0] == T_expected:
            return v
        if v.shape[-1] == T_expected:
            return np.moveaxis(v, -1, 0)
    # Heuristic: time dimension is the smallest for most DROs
    if v.shape[0] <= min(v.shape[1], v.shape[2]):
        return v
    return np.moveaxis(v, -1, 0)


# ===================== SPGR inversion: SI -> C(t) ===================

def spgr_invert_to_concentration(
    S: np.ndarray, S0_map: np.ndarray, T10_map: np.ndarray, cfg: DCEConfig
) -> np.ndarray:
    """
    Invert spoiled GRE signal to concentration (mM).
    S: (T,H,W), S0_map, T10_map: (H,W)
    """
    S = S.astype(np.float64, copy=False)
    S0_map = S0_map.astype(np.float64, copy=False)
    T10_map = T10_map.astype(np.float64, copy=False)

    TR = cfg.TR
    alpha = np.deg2rad(cfg.flip_deg)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    # R10 and approximate M0 from pre-contrast frames
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


# ============================ PK models =============================

def tofts_conv(Cp: np.ndarray, t: np.ndarray, Ktrans: float, ve: float) -> np.ndarray:
    kep = Ktrans / max(ve, 1e-6)
    dt = np.diff(t, prepend=t[0])
    kern = np.exp(-kep * (t[:, None] - t[None, :]))
    kern = np.triu(kern)  # causal
    return Ktrans * (kern * (Cp[None, :] * dt[None, :])).sum(axis=1)


def extended_tofts(Cp: np.ndarray, t: np.ndarray, Kt: float, ve: float, vp: float) -> np.ndarray:
    return vp * Cp + tofts_conv(Cp, t, Kt, ve)

def fit_pk(Ct: np.ndarray, Cp: np.ndarray, t: np.ndarray, cfg: DCEConfig):
    m = np.isfinite(Ct) & np.isfinite(Cp) & np.isfinite(t)
    m[0:1] = True  # keep at least one early sample
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

    # sanity on initial residuals:
    r0 = resid(x0)
    if not np.isfinite(r0).all():
        raise ValueError("Residuals not finite at initial point: check Ct/Cp/t for NaNs and SI→C inversion.")

    res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=400)
    if cfg.use_extended_tofts:
        Kt, ve, vp = [float(v) for v in res.x]
        fit_curve = extended_tofts(Cp, t, Kt, ve, vp)
    else:
        Kt, ve = [float(v) for v in res.x]; vp = 0.0
        fit_curve = tofts_conv(Cp, t, Kt, ve)
    return Kt, ve, vp, fit_curve



# =========================== ROI helpers ===========================

def robust_roi_curve(vol_t: np.ndarray, mask: np.ndarray, how: str = "median") -> np.ndarray:
    vox = vol_t[:, mask]
    if vox.size == 0:
        raise ValueError("ROI mask is empty (no True voxels) for this case.")
    curve = np.median(vox, axis=1) if how == "median" else np.mean(vox, axis=1)
    if not np.isfinite(curve).all():
        # try alternate aggregation then fallback to finite-only
        alt = np.mean(vox, axis=1) if how == "median" else np.median(vox, axis=1)
        curve = alt
    if not np.isfinite(curve).all():
        # use finite voxel filtering per timepoint
        T = vol_t.shape[0]
        out = np.empty(T, dtype=np.float64)
        for k in range(T):
            v = vox[k]
            v = v[np.isfinite(v)]
            out[k] = np.median(v) if v.size else np.nan
        curve = out
    if not np.isfinite(curve).all():
        raise ValueError("ROI curve contains NaN/inf after aggregation.")
    return curve



def pk_from_parmap(parMap: np.ndarray, verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Accepts parMap with shape (H,W,C), where channels are typically:
      C=2: [Ktrans, ve]
      C=3: [Ktrans, ve, vp]  or [Ktrans, ve, kep]
      C=4: [Ktrans, ve, vp, kep] or similar
    Returns dict with Ktrans, ve, vp (vp zeros if unavailable).
    """
    if parMap.ndim != 3:
        raise ValueError(f"Unexpected parMap ndim {parMap.ndim}, expected 3 (H,W,C).")
    H, W, C = parMap.shape
    if C < 2:
        raise ValueError(f"parMap must have at least 2 channels, got {C}.")

    # Always take channel 0 → Ktrans, channel 1 → ve (this matches most DROs)
    Kt = parMap[..., 0].astype(np.float64)
    ve = parMap[..., 1].astype(np.float64)
    vp = np.zeros_like(Kt, dtype=np.float64)

    extras = []
    if C > 2:
        for c in range(2, C):
            extras.append((c, parMap[..., c].astype(np.float64)))

    # Heuristic: identify kep ≈ Ktrans/ve among extras (highest correlation)
    kep_idx = None
    if extras:
        eps = 1e-6
        kep_theory = Kt / (ve + eps)
        # Flatten masks to valid finite entries
        m = np.isfinite(kep_theory)
        if np.any(m):
            best_r = -np.inf
            for (ci, ch) in extras:
                x = kep_theory[m].ravel()
                y = ch[m].ravel()
                if x.size > 10 and np.std(x) > 0 and np.std(y) > 0:
                    r = np.corrcoef(x, y)[0, 1]
                    if r > best_r:
                        best_r = r
                        kep_idx = ci
            if verbose and kep_idx is not None:
                print(f"parMap: detected kep at channel {kep_idx} (corr≈{best_r:.3f})")

    # If a kep channel was found, ignore it (redundant)
    remaining = [ch for ch in extras if ch[0] != kep_idx]

    # Try to find vp among remaining: small positive fraction range is a hint
    vp_idx = None
    best_score = -np.inf
    for (ci, ch) in remaining:
        ch_clip = np.clip(ch, 0.0, 1.0)
        frac_mean = np.mean(ch_clip)
        frac99 = np.quantile(ch_clip, 0.99)
        # A simple score: prefer channels sitting mostly in [0, 0.3]
        score = (0.3 - abs(frac99 - 0.15)) + (0.3 - abs(frac_mean - 0.05))
        if score > best_score:
            best_score = score
            vp_idx = ci
            vp = ch_clip  # keep clipped to [0,1]

    if verbose:
        if vp_idx is not None:
            print(f"parMap: treated channel {vp_idx} as vp.")
        ignored = [ci for (ci, _) in extras if ci not in (kep_idx, vp_idx)]
        if ignored:
            print(f"parMap: ignored extra channel(s): {ignored}.")

    return {"Ktrans": Kt, "ve": ve, "vp": vp}



# ============================== Main ===============================

def main():
    # ---- edit this path ----
    sample_dir = '/ess/scratch/scratch1/rachelgordon/dro_dataset/dro_144frames/sample_021_sub21'
    out_png = '/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/pk_eval_test.png'

    dro_path = os.path.join(sample_dir, 'dro_ground_truth.npz')
    dro = np.load(dro_path)
    print("DRO keys: ", dro.files)

    required_keys = ['ground_truth_images', 'parMap', 'aif', 'S0', 'T10']
    for k in required_keys:
        if k not in dro.files:
            raise KeyError(f"Missing key '{k}' in {dro_path}. Found: {dro.files}")

    # --- load arrays ---
    imgs = dro['ground_truth_images']
    imgs = ensure_time_axis_first(imgs)  # (T,H,W)
    if np.iscomplexobj(imgs):
        imgs = np.abs(imgs)
    T = imgs.shape[0]

    parMap = dro['parMap']
    aif_raw = dro['aif']
    S0 = dro['S0']
    T10 = dro['T10']

    # --- timing from frames ---
    dt = TOTAL_SCAN_TIME_SEC / float(T)
    t = np.arange(T, dtype=np.float64) * dt
    # aif = resample_to_length(aif_raw, T)
    aif = aif_raw

    print("------------------------------------------------------")
    print(f"DRO sample: {sample_dir}")
    print(f"Frames (T): {T}")
    print(f"Estimated temporal resolution Δt ≈ {dt:.3f} s/frame (150 s / {T})")
    print(f"AIF original length: {len(aif_raw)} -> resampled to: {len(aif)}")

    # --- config & SI->Concentration ---
    cfg = DCEConfig(TR=0.0045, flip_deg=12.0, r1=4.5, baseline_frames=3,
                    use_extended_tofts=True, assume_concentration_input=False)

    C = imgs.astype(np.float64, copy=False) if cfg.assume_concentration_input \
        else spgr_invert_to_concentration(imgs, S0, T10, cfg)

    # --- parse ground-truth PK maps BEFORE using `gt` ---
    gt = pk_from_parmap(parMap, verbose=True)

    # --- gather masks and keep non-empty ones matching (H,W) ---
    masks = {k: dro[k].astype(bool) for k in dro.files
             if k in ['malignant','benign','glandular','muscle','vascular','skin','liver','heart']}
    def non_empty(mask):
        return (mask.shape == C.shape[1:]) and (np.count_nonzero(mask) > 0)
    available = {k: v for k, v in masks.items() if non_empty(v)}

    print("Available non-empty ROIs:", list(available.keys()))
    print("------------------------------------------------------")

    # --- decide ROIs to evaluate ---
    roi_plan = {}
    pair_for_delta = None
    if 'malignant' in available:
        roi_plan['lesion_malignant'] = available['malignant']
        if 'glandular' in available:
            roi_plan['background_glandular'] = available['glandular']
            pair_for_delta = ('lesion_malignant', 'background_glandular')
    elif 'benign' in available:
        roi_plan['lesion_benign'] = available['benign']
        if 'glandular' in available:
            roi_plan['background_glandular'] = available['glandular']
            pair_for_delta = ('lesion_benign', 'background_glandular')
    else:
        # no-lesion fallback
        if 'glandular' in available:
            roi_plan['parenchyma'] = available['glandular']
        if 'muscle' in available:
            roi_plan['muscle'] = available['muscle']

    # --- run PK on planned ROIs ---
    results: Dict[str, Dict] = {}
    for name, m in roi_plan.items():
        Ct = robust_roi_curve(C, m, how="median")
        Kt, ve, vp, fit_curve = fit_pk(Ct, aif, t, cfg)

        # GT ROI medians
        Kt_gt = float(np.median(gt['Ktrans'][m]))
        ve_gt = float(np.median(gt['ve'][m]))
        vp_gt = float(np.median(gt['vp'][m]))

        results[name] = {
            "Ct": Ct, "fit_curve": fit_curve, "nvox": int(np.count_nonzero(m)),
            "fit": {"Ktrans": Kt, "ve": ve, "vp": vp},
            "gt":  {"Ktrans": Kt_gt, "ve": ve_gt, "vp": vp_gt},
        }

    if not results:
        print("No non-empty ROIs to evaluate for this sample—try a different sample or ROI.")
        return

    # Optional lesion–background deltas
    if pair_for_delta is not None:
        L, B = pair_for_delta
        if L in results and B in results:
            dK = results[L]['fit']['Ktrans'] - results[B]['fit']['Ktrans']
            dTTP = (t[np.argmax(results[L]['Ct'])] - t[np.argmax(results[B]['Ct'])])
            print(f"ΔKtrans ({L} - {B}) = {dK:.3f}")
            print(f"ΔTTP ({L} - {B}) = {dTTP:.2f} s")

    # --- print table ---
    print("\n=== ROI PK (median) vs Ground Truth ===")
    for name, r in results.items():
        print(f"[{name:18s}] nvox={r['nvox']:6d}  "
              f"Ktrans: {r['fit']['Ktrans']:.3f} (gt {r['gt']['Ktrans']:.3f})  "
              f"ve: {r['fit']['ve']:.3f} (gt {r['gt']['ve']:.3f})  "
              f"vp: {r['fit']['vp']:.3f} (gt {r['gt']['vp']:.3f})")

    # --- plots ---
    plt.figure(figsize=(6, 4))
    for name in results:
        plt.plot(t, results[name]["Ct"], label=f"{name} data")
        plt.plot(t, results[name]["fit_curve"], linestyle='--', label=f"{name} fit")
    plt.xlabel("Time (s)"); plt.ylabel("Concentration (mM)")
    plt.title("ROI enhancement & PK fits (DRO)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

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

    plt.savefig(out_png)
    print(f"Saved figure to: {out_png}")


if __name__ == "__main__":
    main()
