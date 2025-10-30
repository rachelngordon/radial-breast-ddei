#!/usr/bin/env python3
import argparse, os, math, yaml, torch, numpy as np
from einops import rearrange

# --- project imports (match train_zf.py) ---
from dataloader import ZFSliceDataset
from torch.utils.data import DataLoader
from radial_lsfp import MCNUFFT
from lsfpnet_encoding import LSFPNet, ArtifactRemovalLSFPNet
from utils import (
    prep_nufft, to_torch_complex, plot_reconstruction_sample,
    get_git_commit, load_checkpoint
)

# ------------------------
# Hann-windowed blending
# ------------------------
def hann1d(L: int, device):
    n = torch.arange(L, device=device, dtype=torch.float32)
    denom = max(L - 1, 1)  # avoid divide-by-zero for L=1
    return 0.5 * (1.0 - torch.cos(2.0 * torch.pi * n / float(denom)))

@torch.no_grad()
def sliding_window_inference_hann(
    H, W, N_time_eval,
    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob,
    chunk_size, chunk_overlap,
    measured_kspace, csmap, acceleration_encoding, model, device,
    norm_mode: str = "frame",
):
    """
    Stitch T-length sequence using Hann-weighted overlap-add.
    Input shapes:
        measured_kspace: (co, I, T)
        csmap:           (co, H, W)
    Output:
        x_recon: (1, 2, H, W, T)
        adj_loss: torch.tensor(0.)  # (placeholder for API parity)
    """
    assert chunk_overlap < chunk_size, "overlap must be < chunk_size"
    stride = chunk_size - chunk_overlap
    starts = list(range(0, N_time_eval - chunk_size + 1, stride))
    if starts[-1] + chunk_size < N_time_eval:
        starts.append(N_time_eval - chunk_size)  # ensure coverage

    # Accumulators
    recon_sum = torch.zeros((1, 2, H, W, N_time_eval), device=device)
    weight_sum = torch.zeros((1, 1, 1, 1, N_time_eval), device=device)

    # Static physics objects for each chunk are built per-slice in time
    window = hann1d(chunk_size, device=device).view(1, 1, 1, 1, -1)  # broadcast

    for s in starts:
        e = s + chunk_size
        ktraj_chunk  = eval_ktraj[..., s:e]
        dcomp_chunk  = eval_dcomp[..., s:e]
        physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, ktraj_chunk, dcomp_chunk)

        # Select k-space time slab
        ks_chunk = measured_kspace[..., s:e]  # (co, I, tc)
        # time index encoding: pass the chunk's absolute start
        start_timepoint_index = torch.tensor([s], dtype=torch.float, device=device)

        # Forward pass (match your model call signature)
        x_chunk, *_ = model(
            ks_chunk.to(device), physics, csmap,
            acceleration_encoding, start_timepoint_index,
            epoch="infer", norm=norm_mode
        )  # (1, 2, H, W, tc)

        # Hann blend
        recon_sum[..., s:e]  += x_chunk * window
        weight_sum[..., s:e] += window

    # Normalize by the sum of windows
    x_recon = recon_sum / torch.clamp_min(weight_sum, 1e-8)
    adj_loss = torch.tensor(0.0, device=device)
    return x_recon, adj_loss


def build_model_from_config(config, block_dir, device):
    # Lambdas and backbone match train_zf.py
    initial_lambdas = {
        'lambda_L': config['model']['lambda_L'],
        'lambda_S': config['model']['lambda_S'],
        'lambda_spatial_L': config['model']['lambda_spatial_L'],
        'lambda_spatial_S': config['model']['lambda_spatial_S'],
        'gamma': config['model']['gamma'],
        'lambda_step': config['model']['lambda_step'],
    }
    backbone = LSFPNet(
        LayerNo=config["model"]["num_layers"],
        lambdas=initial_lambdas,
        channels=config['model']['channels'],
        style_dim=config['model']['style_dim'],
        svd_mode=config['model']['svd_mode'],
        use_lowk_dc=config['model']['use_lowk_dc'],
        lowk_frac=config['model']['lowk_frac'],
        lowk_alpha=config['model']['lowk_alpha'],
        film_bounded=config['model']['film_bounded'],
        film_gain=config['model']['film_gain'],
        film_identity_init=config['model']['film_identity_init'],
        svd_noise_std=config['model']['svd_noise_std'],
        film_L=config['model']['film_L'],
    )
    # channel count depends on encodings (acceleration + time-index)
    if config['model']['encode_acceleration'] and config['model']['encode_time_index']:
        model = ArtifactRemovalLSFPNet(backbone, block_dir, channels=2).to(device)
    else:
        model = ArtifactRemovalLSFPNet(backbone, block_dir, channels=1).to(device)
    return model


def main():
    parser = argparse.ArgumentParser("Inference for LSFPNet + Dynamic EI (ZF input)")
    parser.add_argument("--config", type=str, default="output/EXPERIMENT/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved model .pth from training")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (used for output dir)")
    parser.add_argument("--root_dir", type=str, help="Data root (overrides config)")
    parser.add_argument("--split_file", type=str, help="Path to split .json (optional)")
    parser.add_argument("--patient_id", type=str, required=True,
                        help="Patient id present in your dataset")
    parser.add_argument("--slice_idx", type=int, default=0, help="Slice index to run")
    parser.add_argument("--spokes_per_frame", type=int, default=None,
                        help="Override eval spokes/frame")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Override eval number of frames")
    parser.add_argument("--chunk_size", type=int, default=24)
    parser.add_argument("--chunk_overlap", type=int, default=12)
    parser.add_argument("--norm", type=str, default=None,
                        choices=["none", "frame", "both"],
                        help="Normalization used inside model at eval; default = config value")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    # -------------------
    # Load config
    # -------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_type = config["model"]["name"]
    H, W = config["data"]["height"], config["data"]["width"]
    N_samples = config["data"]["samples"]
    total_spokes = config["data"]["total_spokes"]

    # Eval spokes/frames (overrides)
    N_time_eval = args.num_frames if args.num_frames is not None else config["data"]["eval_timeframes"]
    N_spokes_eval = args.spokes_per_frame if args.spokes_per_frame is not None else config["data"]["eval_spokes"]
    Ng = config["data"]["fpg"]  # frames per group in training; not used directly here

    # Output dirs
    output_root = args.out_dir or os.path.join(config["experiment"]["output_dir"], args.exp_name)
    os.makedirs(output_root, exist_ok=True)
    block_dir = os.path.join(output_root, "block_outputs")
    os.makedirs(block_dir, exist_ok=True)
    print(f"Git commit: {get_git_commit()}")

    # -------------------
    # Build model + load weights
    # -------------------
    model = build_model_from_config(config, block_dir, device)
    # Create a dummy optimizer just to satisfy load_checkpoint signature
    dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, _, _, _, _, _, _, _, _ = load_checkpoint(model, dummy_opt, args.checkpoint)
    model.eval()

    # -------------------
    # Physics for eval
    # -------------------
    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, N_spokes_eval, N_time_eval)
    eval_ktraj, eval_dcomp = eval_ktraj.to(device), eval_dcomp.to(device)
    eval_nufft_ob, eval_adjnufft_ob = eval_nufft_ob.to(device), eval_adjnufft_ob.to(device)
    eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

    # -------------------
    # Dataset (single patient / slice)
    # -------------------
    data_root = args.root_dir or config["data"]["root_dir"]
    dataset = ZFSliceDataset(
        root_dir=data_root,
        patient_ids=[args.patient_id],
        dataset_key=config["data"]["dataset_key"],
        file_pattern="*.h5",
        slice_idx=args.slice_idx,
        num_random_slices=None,
        N_time=N_time_eval,              # restrict/time-match to eval length
        N_coils=config["data"]["coils"],
        spf_aug=False,
        spokes_per_frame=N_spokes_eval,  # fixed eval spokes
        weight_accelerations=False,
        initial_spokes_range=[N_spokes_eval, N_spokes_eval],
        cluster=config["experiment"].get("cluster", "Randi"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # -------------------
    # Inference loop
    # -------------------
    for measured_kspace, csmap, N_samp, N_spokes, N_time in loader:
        # Shapes to match train_zf.py
        measured_kspace = to_torch_complex(measured_kspace).squeeze()          # (T, Co, Sp, Sam)
        measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')  # (Co, I, T)
        csmap = csmap.to(device).to(measured_kspace.dtype).squeeze(0)          # (Co, H, W)

        # Acceleration & encodings
        N_full = config['data']['height'] * math.pi / 2
        accel = torch.tensor([N_full / int(N_spokes_eval)], dtype=torch.float, device=device)
        acceleration_encoding = accel if config['model']['encode_acceleration'] else None

        # Choose normalization used inside the model
        norm_mode = args.norm if args.norm is not None else config['model']['norm']

        csmap = csmap.unsqueeze(0)
        print("csmap shape: ", csmap.shape)

        if N_time_eval > args.chunk_size:
            print(f"[Infer] Sliding window with Hann blending | T={N_time_eval} | chunk={args.chunk_size} | overlap={args.chunk_overlap}")
            x_recon, _ = sliding_window_inference_hann(
                H, W, N_time_eval,
                eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob,
                args.chunk_size, args.chunk_overlap,
                measured_kspace.to(device), csmap,
                acceleration_encoding, model, device,
                norm_mode=norm_mode,
            )
        else:
            # One-shot (no chunking)
            start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device) if config['model']['encode_time_index'] else None
            x_recon, *_ = model(
                measured_kspace.to(device), eval_physics, csmap,
                acceleration_encoding, start_timepoint_index,
                epoch="infer", norm=norm_mode
            )

        # Save outputs
        npy_path = os.path.join(output_root, f"{args.patient_id}_slice{args.slice_idx}_spf{N_spokes_eval}_T{N_time_eval}.npy")
        np.save(npy_path, x_recon.squeeze(0).cpu().numpy())  # (2, H, W, T)
        print(f"[Infer] Saved reconstruction array to {npy_path}")

        # Optional: quick montage
        try:
            plot_reconstruction_sample(
                x_recon, f"Inference Sample (AF={round(accel.item(),1)}, SPF={int(N_spokes_eval)})",
                f"infer_sample_slice{args.slice_idx}_spf{N_spokes_eval}_T{N_time_eval}",
                output_root
            )
        except Exception as e:
            print(f"[Infer] Plotting skipped: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
