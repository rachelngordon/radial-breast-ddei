#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loop over multiple spokes/frame values (e.g., 2,4,8,16,24,36),
run PK fits on GRASP reconstructions inside the DRO directories,
compute MAE for Ktrans, ve, vp per acceleration,
and plot MAE vs temporal resolution (sec/frame).
"""

import os, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from scipy.optimize import least_squares

# =========================================================
# Config
# =========================================================

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

TOTAL_SCAN_TIME_SEC = 150.0
TOTAL_SPOKES = 288  # Important: fastMRI-derived simulation

# =========================================================
# Utility functions (unchanged from the single-spf version)
# =========================================================

def robust_mae(vals):
    if len(vals) == 0:
        return float("nan")
    vals = np.array(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) > 0 else float("nan")

def resample_to_length(y, new_len):
    y = np.asarray(y, float)
    if len(y) == new_len: return y
    old = np.linspace(0,1,len(y))
    new = np.linspace(0,1,new_len)
    return np.interp(new, old, y)

def ensure_T_first(vol, T_expected=None):
    v = np.asarray(vol)
    if v.ndim != 3: raise ValueError(v.shape)
    if T_expected and v.shape[0]==T_expected: return v
    if T_expected and v.shape[-1]==T_expected: return np.moveaxis(v,-1,0)
    return v if v.shape[0] < v.shape[-1] else np.moveaxis(v,-1,0)

def spgr_invert(S, T10, cfg):
    S = np.asarray(S,float)
    T10 = np.asarray(T10,float)
    TR = cfg.TR; a = np.deg2rad(cfg.flip_deg)
    s,c = np.sin(a), np.cos(a)
    R10 = 1/np.clip(T10,1e-3,None)
    Spre = S[:cfg.baseline_frames].mean(0)
    denom = s*(1-np.exp(-TR*R10)); denom=np.where(np.abs(denom)<1e-8,1e-8,denom)
    M0 = Spre*(1-np.exp(-TR*R10)*c)/denom; M0=np.clip(M0,1e-6,None)

    def inv_frame(Sf):
        E = np.full_like(Sf,0.9)
        for _ in range(25):
            den = (1-E*c); den=np.where(np.abs(den)<1e-8,1e-8,den)
            Spred = M0*s*(1-E)/den
            d = M0*s*(-den - (1-E)*(-c))/(den**2)
            E = np.clip(E - (Spred-Sf)/(d+1e-8),1e-6,0.999999)
        return -np.log(E)/TR

    R = np.stack([inv_frame(S[k]) for k in range(S.shape[0])])
    dR = np.clip(R - R10[None,...],0,None)
    C = dR / cfg.r1
    C[~np.isfinite(C)] = 0
    return C

def tofts(Cp,t,Kt,ve):
    kep = Kt/max(ve,1e-6)
    dt = np.diff(t,prepend=t[0])
    kern = np.exp(-kep*(t[:,None]-t[None,:]))*(np.triu(np.ones((len(t),len(t)))))
    return Kt*(kern*(Cp[None,:]*dt[None,:])).sum(1)

def ext_tofts(Cp,t,Kt,ve,vp): return vp*Cp + tofts(Cp,t,Kt,ve)

def fit_pk(Ct,Cp,t,cfg):
    m = np.isfinite(Ct)&np.isfinite(Cp)&np.isfinite(t); m[0]=True
    Ct,Cp,t = Ct[m],Cp[m],t[m]
    x0 = np.array([cfg.init_Ktrans,cfg.init_ve,cfg.init_vp])
    lb = np.array([cfg.bounds_Ktrans[0],cfg.bounds_ve[0],cfg.bounds_vp[0]])
    ub = np.array([cfg.bounds_Ktrans[1],cfg.bounds_ve[1],cfg.bounds_vp[1]])
    fun = lambda x: ext_tofts(Cp,t,x[0],x[1],x[2]) - Ct
    r0=fun(x0); 
    if not np.isfinite(r0).all(): raise ValueError("nonfinite residuals init")
    r=least_squares(fun,x0,bounds=(lb,ub),max_nfev=400)
    Kt,ve,vp = r.x
    return Kt, ve, vp

def roi_curve(vol,mask):
    vox = vol[:,mask]
    if vox.size==0: raise ValueError
    return np.median(vox,1)

def parse_parmap(pm):
    Kt = pm[...,0]; ve = pm[...,1]
    vp = pm[...,2] if pm.shape[-1]>2 else np.zeros_like(Kt)
    return Kt,ve,vp

# =========================================================
# Run PK eval for one sample @ one SPF
# =========================================================

def eval_one(sample_dir, spf, frames, cfg):
    dro_file = os.path.join(sample_dir,"dro_ground_truth.npz")
    if not os.path.exists(dro_file): return None,"no_dro"
    dro = np.load(dro_file)

    # grasp file
    exact = os.path.join(sample_dir,f"grasp_spf{spf}_frames{frames}.npy")
    if not os.path.exists(exact):
        return None,"no_grasp"

    grasp = np.load(exact)
    grasp = ensure_T_first(grasp,frames)
    if np.iscomplexobj(grasp): grasp=np.abs(grasp)

    T,H,W = grasp.shape
    t = np.arange(T)*TOTAL_SCAN_TIME_SEC/T

    aif = dro["aif"]; 
    if len(aif)!=T: aif=resample_to_length(aif,T)

    T10 = dro["T10"]
    C = spgr_invert(grasp,T10,cfg)

    pm = dro["parMap"]
    Ktgt,vegt,vpgt = parse_parmap(pm)

    rois = [r for r in ["malignant","benign","glandular","muscle"]
            if r in dro and dro[r].shape==(H,W) and dro[r].sum()>0]

    if not rois: return None,"no_rois"

    errs=[]
    for roiname in rois:
        m = dro[roiname].astype(bool)
        Ct = roi_curve(C,m)
        Kt,ve,vp = fit_pk(Ct,aif,t,cfg)
        errs.append((abs(Kt-np.median(Ktgt[m])),
                     abs(ve-np.median(vegt[m])),
                     abs(vp-np.median(vpgt[m]))))
    return errs,None

# =========================================================
# Main loop over SPF list
# =========================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dro_root",required=True)
    p.add_argument("--split_json",required=True)
    p.add_argument("--section",default="val_dro")
    p.add_argument("--spf_list",nargs="+",type=int,default=[2,4,8,16,24,36])
    args=p.parse_args()

    split=json.load(open(args.split_json))
    samples = split[args.section]
    cfg=DCEConfig()

    results = {}   # spf -> dict(Kt=[], ve=[], vp=[])
    missing = []   # list of (sample, spf)

    for spf in args.spf_list:
        
        print(f"Evaluating at {spf} spokes per frame...")

        frames = TOTAL_SPOKES//spf
        spf_errs = {"Kt":[], "ve":[], "vp":[]}

        for sid in samples:
            sample_dir = os.path.join(args.dro_root,f"dro_{frames}frames",sid)
            if not os.path.exists(sample_dir):
                missing.append((sid,spf))
                continue

            out,reason = eval_one(sample_dir,spf,frames,cfg)
            if out is None:
                missing.append((sid,spf))
                continue

            for e in out:
                spf_errs["Kt"].append(e[0])
                spf_errs["ve"].append(e[1])
                spf_errs["vp"].append(e[2])

        results[spf] = (
            robust_mae(spf_errs["Kt"]),
            robust_mae(spf_errs["ve"]),
            robust_mae(spf_errs["vp"]),
            TOTAL_SCAN_TIME_SEC/frames  # sec/frame
        )

    # ===================================================
    # Print table
    # ===================================================
    print("\n=== MAE vs acceleration ===")
    print("spf | Δt(s/frame) | Ktrans | ve | vp")
    for spf,(maeK,maeve,maevp,dt) in results.items():
        print(f"{spf:3d} | {dt:6.2f} | {maeK:.4f} | {maeve:.4f} | {maevp:.4f}")

    # ===================================================
    # Missing list
    # ===================================================
    if missing:
        print("\nMissing GRASP files:")
        for sid,spf in missing:
            print(f"  {sid} @ spf={spf}")

    # ===================================================
    # Plot MAE vs Δt
    # ===================================================
    spfs, maeK, maeVe, maeVp, dts = [],[],[],[],[]
    for spf,(mK,mVe,mVp,dt) in results.items():
        spfs.append(spf); dts.append(dt)
        maeK.append(mK); maeVe.append(mVe); maeVp.append(mVp)

    plt.figure(figsize=(6,4))
    plt.plot(dts, maeK,marker='o',label="Ktrans MAE")
    plt.plot(dts, maeVe,marker='o',label="ve MAE")
    plt.plot(dts, maeVp,marker='o',label="vp MAE")
    plt.xlabel("Temporal resolution (sec/frame)")
    plt.ylabel("Median Absolute Error")
    plt.title("PK MAE vs Temporal Resolution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pk_mae_vs_temporal_resolution.png")
    print("\nSaved: pk_mae_vs_temporal_resolution.png")

if __name__=="__main__":
    main()
