#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voxelwise PK + kinetics evaluation for GRASP-in-DRO experiments.

Adds:
  • Per-voxel PK fit inside ROIs → error maps vs parMap + failure rates
  • Temporal kinetics: wash-in slope & TTP (interpolated)
  • Summary table across SPF values (median absolute errors, failure rates)
  • One-figure "case study" panel (maps, curves, error overlays)

Assumed layout per SPF:
  dro_root/dro_{frames}frames/sample_xxx_subyy/
      grasp_spf{spf}_frames{frames}.npy
      dro_ground_truth.npz   (parMap, aif, T10, ROI masks: malignant/glandular/etc.)

Run example:
  python pk_eval_voxelwise_plus_case_study.py \
    --dro_root /ess/scratch/scratch1/rachelgordon/dro_dataset \
    --split_json /gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json \
    --section val_dro \
    --spf_list 2 4 8 16 24 36 \
    --case_study sample_030_sub30 --case_study_spf 8 \
    --outdir results_pk_voxel

Notes:
  • Voxelwise fitting is done ONLY inside requested ROIs (default: malignant & glandular).
  • To keep runtime sane, you can sub-sample voxels via --roi_stride.
  • Total scan time is assumed 150 s, total spokes 288 → frames = 288 / SPF.
"""

import os, json, argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ------------------ constants & config ------------------

TOTAL_SCAN_TIME_SEC = 150.0
TOTAL_SPOKES = 288

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
    bounds_vp: Tuple[float, float] = (0.0, 0.3)
    init_Ktrans: float = 0.2
    init_ve: float = 0.3
    init_vp: float = 0.02

# ------------------ small utils ------------------

def robust_mae(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x))) if x.size else float("nan")

def ensure_T_first(vol: np.ndarray, T_expected: Optional[int] = None) -> np.ndarray:
    v = np.asarray(vol)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D dynamic (T,H,W)/(H,W,T), got {v.shape}")
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

# -------- SPGR inversion (dynamic baseline from AIF) --------

def choose_baseline_from_aif(aif: np.ndarray, pct: float = 0.05,
                             min_frames: int = 3, max_frames: int = 12) -> slice:
    aif = np.asarray(aif, float)
    thr = pct * float(np.max(aif)) if aif.size else 0.0
    idx = np.where(aif < thr)[0]
    if idx.size == 0:
        end = min_frames
    else:
        end = max(min_frames, min(idx[-1] + 1, max_frames))
    return slice(0, end)

def spgr_invert_to_conc(S: np.ndarray, T10: np.ndarray, cfg: DCEConfig, baseline_slice: slice) -> np.ndarray:
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

# ---------------- PK models & AIF alignment ----------------

def tofts_conv(Cp: np.ndarray, t: np.ndarray, Kt: float, ve: float) -> np.ndarray:
    kep = Kt / max(ve, 1e-6)
    dt = np.diff(t, prepend=t[0])
    kern = np.exp(-kep * (t[:, None] - t[None, :]))
    kern = np.triu(kern)
    return Kt * (kern * (Cp[None, :] * dt[None, :])).sum(axis=1)

def extended_tofts(Cp: np.ndarray, t: np.ndarray, Kt: float, ve: float, vp: float) -> np.ndarray:
    return vp * Cp + tofts_conv(Cp, t, Kt, ve)

def fit_pk_scaled(Ct: np.ndarray, Cp: np.ndarray, t: np.ndarray, cfg: DCEConfig,
                  use_scale: bool = True) -> Tuple[float, float, float, float, bool]:
    """
    Returns (Ktrans, ve, vp, s, hit_bounds_flag)
    """
    Ct = np.asarray(Ct, float); Cp = np.asarray(Cp, float); t = np.asarray(t, float)
    m = np.isfinite(Ct) & np.isfinite(Cp) & np.isfinite(t); m[0:1] = True
    Ct, Cp, t = Ct[m], Cp[m], t[m]
    if Ct.size < 4:
        raise ValueError("Too few timepoints for PK fit")

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
    if not np.isfinite(r0).all():
        raise ValueError("Non-finite residuals at init")
    res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=400)
    x = res.x
    if use_scale:
        Kt, ve, vp, s = [float(v) for v in x]
    else:
        Kt, ve, vp = [float(v) for v in x]; s = 1.0
    hit_bounds = np.any(np.isclose(x, lb, atol=1e-6)) or np.any(np.isclose(x, ub, atol=1e-6))
    return Kt, ve, vp, s, bool(hit_bounds)

def align_aif_to_ct(Ct: np.ndarray, Cp: np.ndarray, max_shift: int = 2) -> np.ndarray:
    Ct = np.asarray(Ct, float); Cp = np.asarray(Cp, float)
    k_peak = int(np.argmax(Ct)) if Ct.size else 0
    best = (0, np.inf, Cp.copy())
    for sh in range(-max_shift, max_shift + 1):
        if sh < 0:
            Cp_s = Cp[-sh:]; Ct_s = Ct[:Cp_s.size]
        elif sh > 0:
            Cp_s = Cp[:-sh]; Ct_s = Ct[sh:]
        else:
            Cp_s, Ct_s = Cp, Ct
        k_use = min(k_peak, Ct_s.size)
        if k_use < 3: continue
        sse = np.sum((Ct_s[:k_use] - Cp_s[:k_use]) ** 2)
        if sse < best[1]: best = (sh, sse, Cp_s.copy())
    Cp_opt = best[2]
    return resample_to_length(Cp_opt, Cp.size)

# ---------------- Temporal kinetics (wash-in & TTP) ----------------

def wash_in_slope(t: np.ndarray, c: np.ndarray, t_win: float = 60.0) -> float:
    """Slope from first sample to the sample nearest t_win (least-squares on [0, t_win])."""
    t = np.asarray(t, float); c = np.asarray(c, float)
    m = (t <= t_win) & np.isfinite(c)
    if np.count_nonzero(m) < 2: return float("nan")
    tt = t[m]; cc = c[m]
    A = np.vstack([tt - tt[0], np.ones_like(tt)]).T
    sol, *_ = np.linalg.lstsq(A, cc, rcond=None)  # slope wrt time (mM/s)
    return float(sol[0])

def ttp_interpolated(t: np.ndarray, c: np.ndarray, upsample: int = 10) -> float:
    """Interpolated TTP in seconds using linear upsampling."""
    t = np.asarray(t, float); c = np.asarray(c, float)
    if t.size < 2 or not np.isfinite(c).any(): return float("nan")
    t_f = np.linspace(t[0], t[-1], num=upsample * t.size)
    c_f = np.interp(t_f, t, c)
    return float(t_f[int(np.argmax(c_f))])

# ---------------- Core routines ----------------

def voxelwise_pk_and_errors(grasp: np.ndarray, aif: np.ndarray, T10: np.ndarray,
                            parMap: np.ndarray, mask: np.ndarray, cfg: DCEConfig,
                            use_scale: bool, dt: float, roi_stride: int = 1
                            ) -> Dict[str, np.ndarray]:
    """
    Fit PK per-voxel inside mask (optionally subsampled by roi_stride).
    Returns dict with voxelwise estimates, errors vs GT, and failure maps.
    """
    T, H, W = grasp.shape
    t = np.arange(T, dtype=np.float64) * dt
    # SI->conc
    bsl = choose_baseline_from_aif(aif, pct=0.05,
                                   min_frames=cfg.baseline_frames_min,
                                   max_frames=cfg.baseline_frames_max)
    C = spgr_invert_to_conc(grasp, T10, cfg, bsl)

    Kt_gt, ve_gt, vp_gt = parse_parmap(parMap)
    ms = mask.astype(bool)
    if roi_stride > 1:
        # light subsample for speed
        ms_sub = np.zeros_like(ms, bool)
        ms_sub[::roi_stride, ::roi_stride] = ms[::roi_stride, ::roi_stride]
        ms = ms_sub

    # outputs
    Kt_hat = np.full((H, W), np.nan, float)
    ve_hat = np.full((H, W), np.nan, float)
    vp_hat = np.full((H, W), np.nan, float)
    fail = np.zeros((H, W), bool)
    hitb = np.zeros((H, W), bool)

    Cp0 = aif.copy()

    # iterate voxels
    ys, xs = np.where(ms)
    for (y, x) in zip(ys, xs):
        Ct = C[:, y, x]
        if not np.isfinite(Ct).any():
            fail[y, x] = True
            continue
        Cp = align_aif_to_ct(Ct, Cp0, max_shift=2)
        try:
            Kt, ve, vp, s, hb = fit_pk_scaled(Ct, Cp, t, cfg, use_scale=use_scale)
            Kt_hat[y, x] = Kt; ve_hat[y, x] = ve; vp_hat[y, x] = vp
            hitb[y, x] = hb
        except Exception:
            fail[y, x] = True

    # errors (only where both est & GT finite)
    m_valid = np.isfinite(Kt_hat) & np.isfinite(Kt_gt)
    err_Kt = np.full((H, W), np.nan, float); err_Kt[m_valid] = np.abs(Kt_hat[m_valid] - Kt_gt[m_valid])
    m_valid = np.isfinite(ve_hat) & np.isfinite(ve_gt)
    err_ve = np.full((H, W), np.nan, float); err_ve[m_valid] = np.abs(ve_hat[m_valid] - ve_gt[m_valid])
    m_valid = np.isfinite(vp_hat) & np.isfinite(vp_gt)
    err_vp = np.full((H, W), np.nan, float); err_vp[m_valid] = np.abs(vp_hat[m_valid] - vp_gt[m_valid])

    return dict(
        Kt_hat=Kt_hat, ve_hat=ve_hat, vp_hat=vp_hat,
        err_Kt=err_Kt, err_ve=err_ve, err_vp=err_vp,
        fail=fail, hit_bounds=hitb,
        C=C,  # pass back for ROI curves if desired
    )

def roi_curves_and_kinetics(C: np.ndarray, mask: np.ndarray, aif: np.ndarray, dt: float) -> Dict[str, float]:
    """Median ROI curve → wash-in slope & interpolated TTP (seconds)."""
    t = np.arange(C.shape[0], dtype=np.float64) * dt
    m = mask.astype(bool)
    if np.count_nonzero(m) == 0:
        return dict(washin=np.nan, ttp=np.nan)
    Ct = np.median(C[:, m], axis=1)
    return dict(
        washin=wash_in_slope(t, Ct, t_win=60.0),
        ttp=ttp_interpolated(t, Ct, upsample=10),
    )

# ---------------- Case study figure ----------------

def case_study_figure(out_png: str, grasp: np.ndarray, T10: np.ndarray, aif: np.ndarray,
                      parMap: np.ndarray, masks: Dict[str, np.ndarray], dt: float,
                      voxel_out: Dict[str, np.ndarray], title: str = ""):
    """
    Panel: one mid-time GRASP frame, GT Kt/ve, estimated Kt/ve, error maps, curves (malignant & glandular).
    """
    T, H, W = grasp.shape
    mid = min(T - 1, T // 2)

    Kt_gt, ve_gt, vp_gt = parse_parmap(parMap)
    Kt_hat = voxel_out["Kt_hat"]; ve_hat = voxel_out["ve_hat"]
    err_Kt = voxel_out["err_Kt"]; err_ve = voxel_out["err_ve"]

    # ROI curves
    C = voxel_out["C"]
    t = np.arange(T, dtype=np.float64) * dt
    curves = []
    for k in ["malignant", "glandular"]:
        if k in masks and np.count_nonzero(masks[k]) > 0:
            m = masks[k].astype(bool)
            curves.append((k, np.median(C[:, m], axis=1)))

    # figure
    plt.figure(figsize=(11, 8))
    gs = plt.GridSpec(3, 4, height_ratios=[1,1,1], width_ratios=[1,1,1,1])

    # frame
    ax = plt.subplot(gs[0,0])
    ax.imshow(grasp[mid], cmap="gray")
    ax.set_title("GRASP frame (mid)"); ax.axis("off")

    # GT maps
    ax = plt.subplot(gs[0,1]); im = ax.imshow(Kt_gt, vmin=0, vmax=np.nanpercentile(Kt_gt, 99))
    ax.set_title("GT Ktrans"); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(gs[0,2]); im = ax.imshow(ve_gt, vmin=0, vmax=1.0)
    ax.set_title("GT ve"); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)

    # Est maps
    ax = plt.subplot(gs[1,1]); im = ax.imshow(Kt_hat, vmin=0, vmax=np.nanpercentile(Kt_gt, 99))
    ax.set_title("Est Ktrans"); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(gs[1,2]); im = ax.imshow(ve_hat, vmin=0, vmax=1.0)
    ax.set_title("Est ve"); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)

    # Error maps
    ax = plt.subplot(gs[2,1]); im = ax.imshow(err_Kt, vmin=0, vmax=np.nanpercentile(err_Kt, 99))
    ax.set_title("|Ktrans error|"); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = plt.subplot(gs[2,2]); im = ax.imshow(err_ve, vmin=0, vmax=np.nanpercentile(err_ve, 99))
    ax.set_title("|ve error|"); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)

    # Curves
    ax = plt.subplot(gs[:,3])
    for name, ct in curves:
        ax.plot(t, ct, label=f"{name} Ct")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Concentration (mM)")
    ax.set_title("ROI enhancement (median)"); ax.grid(alpha=0.3); ax.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_png, dpi=180)
    plt.close()

# ---------------- Main driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dro_root", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--section", default="val_dro")
    ap.add_argument("--spf_list", nargs="+", type=int, default=[2,4,8,16,24,36])
    ap.add_argument("--rois", nargs="+", default=["malignant","glandular"])
    ap.add_argument("--roi_stride", type=int, default=1, help="Subsample ROI voxels for speed (1=no skip, 2=every 2nd, etc.)")
    ap.add_argument("--no_scale", action="store_true", help="Disable Cp scale factor in PK fit")
    ap.add_argument("--outdir", default="pk_voxel_results")
    # case study options
    ap.add_argument("--case_study", default=None, help="sample_xxx_subyy to visualize")
    ap.add_argument("--case_study_spf", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.split_json, "r") as f:
        split = json.load(f)
    samples: List[str] = split[args.section]

    cfg = DCEConfig()
    use_scale = (not args.no_scale)

    # Summary accumulators per SPF
    summary = {}  # spf -> dict of pooled voxel errors/failures over all subjects & ROIs
    for spf in args.spf_list:
        frames = TOTAL_SPOKES // spf
        dt = TOTAL_SCAN_TIME_SEC / frames

        pooled_err_Kt = []
        pooled_err_ve = []
        pooled_err_vp = []
        pooled_fail = 0
        pooled_count = 0
        pooled_hitb = 0

        missing = []

        for sid in samples:
            sd = os.path.join(args.dro_root, f"dro_{frames}frames", sid)
            dro_npz = os.path.join(sd, "dro_ground_truth.npz")
            grasp_npy = os.path.join(sd, f"grasp_spf{spf}_frames{frames}.npy")
            if not (os.path.exists(dro_npz) and os.path.exists(grasp_npy)):
                missing.append((sid, "missing_files"))
                continue

            try:
                dro = np.load(dro_npz)
                grasp = ensure_T_first(np.load(grasp_npy), frames)
                if np.iscomplexobj(grasp): grasp = np.abs(grasp)

                aif = dro["aif"]
                if aif.size != frames:
                    aif = resample_to_length(aif, frames)

                parMap = dro["parMap"]
                T10 = dro["T10"]

                # Build ROI mask (union of requested ROIs present)
                masks = {k: dro[k].astype(bool) for k in args.rois if k in dro.files}
                roi_union = np.zeros(grasp.shape[1:], bool)
                for m in masks.values():
                    if m.shape == roi_union.shape:
                        roi_union |= m
                if np.count_nonzero(roi_union) == 0:
                    missing.append((sid, "no_requested_roi"))
                    continue

                # Voxelwise fitting + errors inside ROIs
                out = voxelwise_pk_and_errors(
                    grasp=grasp, aif=aif, T10=T10, parMap=parMap,
                    mask=roi_union, cfg=cfg, use_scale=use_scale, dt=dt,
                    roi_stride=args.roi_stride
                )

                # Collect pooled stats
                mv = roi_union & np.isfinite(out["err_Kt"]) & np.isfinite(out["err_ve"]) & np.isfinite(out["err_vp"])
                pooled_err_Kt.extend(out["err_Kt"][mv].ravel())
                pooled_err_ve.extend(out["err_ve"][mv].ravel())
                pooled_err_vp.extend(out["err_vp"][mv].ravel())
                pooled_count += int(np.count_nonzero(roi_union))
                pooled_fail += int(np.count_nonzero(out["fail"] & roi_union))
                pooled_hitb += int(np.count_nonzero(out["hit_bounds"] & roi_union))

                # Save per-subject error maps (quicklook)
                for name, arr in [("err_Ktrans", out["err_Kt"]), ("err_ve", out["err_ve"])]:
                    plt.figure(figsize=(4,3))
                    vmax = np.nanpercentile(arr[roi_union], 99) if np.any(np.isfinite(arr[roi_union])) else 1.0
                    plt.imshow(arr, vmin=0, vmax=vmax); plt.axis("off"); plt.title(f"{sid} SPF{spf} {name}")
                    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"{sid}_spf{spf}_{name}.png"), dpi=150); plt.close()

                # Save per-subject kinetics table (wash-in/TTP) for malignant & glandular if available
                kinetics_rows = []
                for k in ["malignant", "glandular"]:
                    if k in masks and np.count_nonzero(masks[k]) > 0:
                        kin = roi_curves_and_kinetics(out["C"], masks[k], aif, dt)
                        kinetics_rows.append((sid, spf, k, kin["washin"], kin["ttp"]))
                if kinetics_rows:
                    with open(os.path.join(args.outdir, f"{sid}_spf{spf}_kinetics.tsv"), "w") as f:
                        f.write("subject\tspf\troi\twashin(mM/s)\tttp(s)\n")
                        for row in kinetics_rows:
                            f.write("\t".join([str(x) for x in row]) + "\n")

            except Exception as e:
                missing.append((sid, f"exception:{type(e).__name__}"))

        # Summaries for this SPF
        mae_Kt = robust_mae(np.array(pooled_err_Kt))
        mae_ve = robust_mae(np.array(pooled_err_ve))
        mae_vp = robust_mae(np.array(pooled_err_vp))
        fail_rate = (pooled_fail / max(1, pooled_count))
        hitb_rate = (pooled_hitb / max(1, pooled_count))

        summary[spf] = dict(
            dt=dt, n_vox=pooled_count, mae_Kt=mae_Kt, mae_ve=mae_ve, mae_vp=mae_vp,
            fail_rate=fail_rate, hit_bounds_rate=hitb_rate, missing=missing
        )

    # -------- print & save summary table --------
    print("\n=== Voxelwise PK summary over requested ROIs (pooled across subjects) ===")
    print("SPF | Δt(s/frame) | Voxels | MAE Ktrans |  MAE ve  | MAE vp |  Fail% | HitBounds%")
    lines = []
    for spf in args.spf_list:
        s = summary[spf]
        print(f"{spf:3d} | {s['dt']:11.3f} | {s['n_vox']:6d} | {s['mae_Kt']:10.4f} | {s['mae_ve']:8.4f} | {s['mae_vp']:7.4f} | {100*s['fail_rate']:6.2f} | {100*s['hit_bounds_rate']:9.2f}")
        lines.append([spf, s['dt'], s['n_vox'], s['mae_Kt'], s['mae_ve'], s['mae_vp'], s['fail_rate'], s['hit_bounds_rate']])

    # write TSV
    out_tsv = os.path.join(args.outdir, "summary_voxelwise.tsv")
    with open(out_tsv, "w") as f:
        f.write("spf\tdt_s\tvoxels\tmae_ktrans\tmae_ve\tmae_vp\tfail_rate\thit_bounds_rate\n")
        for r in lines:
            f.write("\t".join([str(x) for x in r]) + "\n")
    print(f"\nSaved table: {out_tsv}")

    # -------- plot MAE & failures vs temporal resolution --------
    dts = [summary[s]['dt'] for s in args.spf_list]
    maeK = [summary[s]['mae_Kt'] for s in args.spf_list]
    maeVe = [summary[s]['mae_ve'] for s in args.spf_list]
    maeVp = [summary[s]['mae_vp'] for s in args.spf_list]
    fr   = [summary[s]['fail_rate'] for s in args.spf_list]

    plt.figure(figsize=(6,4))
    plt.plot(dts, maeK, marker='o', label="Ktrans MAE")
    plt.plot(dts, maeVe, marker='o', label="ve MAE")
    plt.plot(dts, maeVp, marker='o', label="vp MAE")
    plt.xlabel("Temporal resolution (sec/frame)"); plt.ylabel("Voxelwise Median Abs Error")
    plt.title("Voxelwise PK error vs temporal resolution"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    out_png = os.path.join(args.outdir, "voxelwise_pk_mae_vs_dt.png")
    plt.savefig(out_png, dpi=180); plt.close()
    print(f"Saved plot: {out_png}")

    plt.figure(figsize=(6,4))
    plt.plot(dts, np.array(fr)*100.0, marker='o', label="PK fit failure rate")
    plt.xlabel("Temporal resolution (sec/frame)"); plt.ylabel("Failure rate (%)")
    plt.title("Voxelwise PK fit failures vs temporal resolution"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    out_png = os.path.join(args.outdir, "voxelwise_fail_rate_vs_dt.png")
    plt.savefig(out_png, dpi=180); plt.close()
    print(f"Saved plot: {out_png}")

    # -------- list missing / skipped per SPF --------
    miss_report = os.path.join(args.outdir, "missing_per_spf.txt")
    with open(miss_report, "w") as f:
        for spf in args.spf_list:
            misses = summary[spf]["missing"]
            if misses:
                f.write(f"SPF {spf} (frames={TOTAL_SPOKES//spf}):\n")
                for sid, why in misses:
                    f.write(f"  - {sid}: {why}\n")
    print(f"Saved missing report: {miss_report}")

    # -------- Case study panel --------
    if args.case_study and args.case_study_spf:
        sid = args.case_study; spf = args.case_study_spf
        frames = TOTAL_SPOKES // spf; dt = TOTAL_SCAN_TIME_SEC / frames
        sd = os.path.join(args.dro_root, f"dro_{frames}frames", sid)
        dro_npz = os.path.join(sd, "dro_ground_truth.npz")
        grasp_npy = os.path.join(sd, f"grasp_spf{spf}_frames{frames}.npy")
        if os.path.exists(dro_npz) and os.path.exists(grasp_npy):
            dro = np.load(dro_npz)
            grasp = ensure_T_first(np.load(grasp_npy), frames)
            if np.iscomplexobj(grasp): grasp = np.abs(grasp)
            aif = dro["aif"];  aif = aif if aif.size == frames else resample_to_length(aif, frames)
            parMap = dro["parMap"]; T10 = dro["T10"]
            masks = {k: dro[k].astype(bool) for k in ["malignant","glandular"] if k in dro.files}

            out = voxelwise_pk_and_errors(
                grasp=grasp, aif=aif, T10=T10, parMap=parMap,
                mask=(masks["malignant"] | masks.get("glandular", np.zeros_like(masks["malignant"])) if "malignant" in masks else list(masks.values())[0]),
                cfg=cfg, use_scale=use_scale, dt=dt, roi_stride=args.roi_stride
            )

            title = f"Case study: {sid}  |  SPF={spf}  (Δt≈{dt:.2f}s)"
            fig_path = os.path.join(args.outdir, f"case_{sid}_spf{spf}.png")
            case_study_figure(fig_path, grasp, T10, aif, parMap, masks, dt, out, title=title)
            print(f"Saved case study: {fig_path}")
        else:
            print("Case study files not found; skipping.")

if __name__ == "__main__":
    main()
