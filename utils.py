import os
import subprocess
import matplotlib.pyplot as plt
import torch
import numpy as np
from einops import rearrange
import torchkbnufft as tkbn
import csv
import sigpy as sp
from sigpy.mri import app

def log_gradient_stats(model, epoch, iteration, output_dir, log_filename="gradient_stats.csv"):
    """
    Computes, prints, and logs the L2 norm of gradients for each parameter and the total gradient norm.
    
    Args:
        model (torch.nn.Module): The model being trained.
        epoch (int): The current epoch.
        iteration (int): The current global iteration/step count.
        output_dir (str): The main experiment output directory.
        log_filename (str): The CSV filename for storing detailed logs.
    """
    total_norm = 0.0
    param_norms = []
    
    # Iterate over all named parameters
    for name, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            # Calculate the L2 norm of the gradient for this parameter
            param_norm = p.grad.data.norm(2)
            # Handle potential inf/nan values gracefully
            if not torch.isfinite(param_norm):
                param_norm_item = float('inf')
            else:
                param_norm_item = param_norm.item()
            
            param_norms.append((name, param_norm_item))
            total_norm += param_norm_item ** 2
            
    total_norm = total_norm ** 0.5
    
    # --- Logging to Console ---
    print(f"--- Gradient Stats (Epoch {epoch}, Iter {iteration}) ---")
    
    # Use :.4e for exponential notation with 4 digits of precision
    print(f"Total Gradient Norm: {total_norm:.4e}")
    
    # Sort parameters by gradient norm (descending) to see the largest ones
    # Use a lambda that is safe for potential 'inf' values
    param_norms.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 layers with largest gradients:")
    for name, norm in param_norms[:5]:
        # Use :.4e here as well
        print(f"  - {name}: {norm:.4e}")
        
    print("Top 5 layers with smallest gradients:")
    # To print the smallest non-zero gradients, we filter out zeros
    non_zero_norms = [p for p in param_norms if p[1] > 0]
    for name, norm in non_zero_norms[-5:]:
        # Use :.4e here as well
        print(f"  - {name}: {norm:.4e}")
    print("-------------------------------------------------")

    # --- Logging to CSV File for later analysis ---
    # No changes needed here, as CSV should ideally store full precision numbers.
    # The exponential formatting is mainly for console readability.
    log_path = os.path.join(output_dir, log_filename)
    file_exists = os.path.isfile(log_path)
    
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'iteration', 'total_norm', 'param_name', 'param_norm'])
            
        # The writer will handle inf/nan correctly
        writer.writerow([epoch, iteration, total_norm, '---TOTAL---', total_norm])
        for name, norm in param_norms:
            writer.writerow([epoch, iteration, total_norm, name, norm])



def trajGR(Nkx, Nspokes):
    '''
    function for generating golden-angle radial sampling trajectory
    :param Nkx: spoke length
    :param Nspokes: number of spokes
    :return: ktraj: golden-angle radial sampling trajectory
    '''
    # ga = np.deg2rad(180 / ((np.sqrt(5) + 1) / 2))
    ga = np.pi * ((1 - np.sqrt(5)) / 2)
    kx = np.zeros(shape=(Nkx, Nspokes))
    ky = np.zeros(shape=(Nkx, Nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, Nkx)
    for i in range(1, Nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    ky = np.transpose(ky)
    kx = np.transpose(kx)

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)

    return ktraj

################### prepare NUFFT ################
def prep_nufft(Nsample, Nspokes, Ng):

    overSmaple = 2
    im_size = (int(Nsample/overSmaple), int(Nsample/overSmaple))
    grid_size = (Nsample, Nsample)

    ktraj = trajGR(Nsample, Nspokes * Ng)
    ktraj = torch.tensor(ktraj, dtype=torch.float)
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)
    dcomp = dcomp.squeeze()

    ktraju = np.zeros([2, Nspokes * Nsample, Ng], dtype=float)
    dcompu = np.zeros([Nspokes * Nsample, Ng], dtype=complex)

    for ii in range(0, Ng):
        ktraju[:, :, ii] = ktraj[:, (ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]
        dcompu[:, ii] = dcomp[(ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]

    ktraju = torch.tensor(ktraju, dtype=torch.float)
    dcompu = torch.tensor(dcompu, dtype=torch.complex64)

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)  # forward nufft
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)  # backward nufft

    return ktraju, dcompu, nufft_ob, adjnufft_ob


def _calculate_top_percentile_curve(dynamic_slice: torch.Tensor, percentile: float) -> list[float]:
    """Helper function to calculate the enhancement curve for a single dynamic slice."""
    
    if dynamic_slice.dim() != 5 or dynamic_slice.shape[0] != 1 or dynamic_slice.shape[1] != 2:
        raise ValueError(f"Expected input shape (1, 2, T, H, W), but got {dynamic_slice.shape}")

    # Calculate magnitude: sqrt(real^2 + imag^2) and remove batch/channel dims
    magnitude_video = torch.sqrt(dynamic_slice[:, 0, ...] ** 2 + dynamic_slice[:, 1, ...] ** 2).squeeze(0)
    
    num_time_frames = magnitude_video.shape[0]
    top_percentile_means = []
    
    q = percentile / 100.0

    for t in range(num_time_frames):
        frame_t = magnitude_video[t, :, :]
        
        if frame_t.max() == 0:
            top_percentile_means.append(0)
            continue
            
        threshold = torch.quantile(frame_t.flatten(), q)
        bright_pixels = frame_t[frame_t > threshold]
        
        mean_val = torch.mean(bright_pixels) if bright_pixels.numel() > 0 else threshold
        top_percentile_means.append(mean_val.item())
        
    return top_percentile_means


def plot_enhancement_curve(
    model_output: torch.Tensor,
    percentile: float = 99.0,
    title: str = "Enhancement Curve Comparison",
    output_filename: str = None
):
    """
    Calculates and plots the enhancement curves for a model output and a benchmark
    image on the same graph for direct comparison.

    Args:
        model_output (torch.Tensor): The model's reconstructed dynamic slice.
                                     Shape (1, 2, T, H, W).
        benchmark_image (torch.Tensor): The ground truth or benchmark dynamic slice.
                                        Shape (1, 2, T, H, W).
        percentile (float, optional): The percentile for defining the brightest pixels.
                                      Defaults to 99.0.
        title (str, optional): The title for the plot. Defaults to "Enhancement Curve Comparison".
        output_filename (str, optional): If provided, saves the plot to this file path.
                                         Defaults to None (displays plot).
    """

    # --- 1. Input Validation ---
    if not 0 < percentile < 100:
        raise ValueError("Percentile must be between 0 and 100.")

    # --- 2. Calculate Curves for Both Images ---
    model_curve = _calculate_top_percentile_curve(model_output.detach(), percentile)
    # benchmark_curve = _calculate_top_percentile_curve(benchmark_image.detach(), percentile)
    
    # Ensure time axis is consistent
    num_time_frames = model_output.shape[2]
    time_axis = np.arange(num_time_frames)

    # --- 3. Plotting ---
    plt.figure(figsize=(12, 7))
    
    # Plot model output curve
    plt.plot(time_axis, model_curve, label='Model Output', marker='o', linestyle='-', color='tab:blue')
    
    # Plot benchmark curve
    # plt.plot(time_axis, benchmark_curve, label='GRASP Benchmark', marker='x', linestyle='--', color='tab:orange')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Time Frame", fontsize=12)
    plt.ylabel(f"Mean Signal of Top {100-percentile:.1f}% Pixels", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_filename:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plt.savefig(output_filename)
    else:
        plt.show()
        
    plt.close()

    

def get_cosine_ei_weight(
    current_epoch,
    warmup_epochs,
    schedule_duration,
    target_weight
):
    """
    Calculates the EI loss weight for the current epoch using a cosine schedule.

    This implements a curriculum learning strategy:
    1. For `warmup_epochs`, the weight is 0 (MC loss only).
    2. Over the next `schedule_duration` epochs, the weight smoothly ramps
       up from 0 to `target_weight` following a cosine curve.
    3. After the schedule is complete, the weight stays at `target_weight`.

    Args:
        current_epoch (int): The current training epoch (starting from 1).
        warmup_epochs (int): Number of epochs to train with only MC loss.
        schedule_duration (int): Number of epochs for the ramp-up.
        target_weight (float): The final EI loss weight to reach.

    Returns:
        float: The EI loss weight for the current epoch.
    """
    # Phase 1: Warm-up phase (MC loss only)
    if current_epoch <= warmup_epochs:
        return 0.0

    # Calculate progress within the scheduling phase
    schedule_progress_epoch = current_epoch - warmup_epochs

    # Phase 3: Schedule is complete, hold at target weight
    if schedule_progress_epoch >= schedule_duration:
        return target_weight

    # Phase 2: Cosine ramp-up phase
    # This creates a value that goes from 0 to 1 along a cosine curve.
    cosine_multiplier = 0.5 * (1 - np.cos(np.pi * schedule_progress_epoch / schedule_duration))
    
    return target_weight * cosine_multiplier





def plot_reconstruction_sample(x_recon, title, filename, output_dir, grasp_img=None, batch_idx=0, transform=False):
    """
    Plot reconstruction sample showing magnitude images across timeframes.

    Args:
        x_recon: Reconstructed image tensor of shape (B, C, T, H, W)
        title: Title for the plot
        filename: Filename for saving (without extension)
        output_dir: Directory to save the plot
        batch_idx: Which batch element to plot (default: 0)
    """
    os.makedirs(output_dir, exist_ok=True)

    # compute magnitude from complex reconstruction
    if x_recon.shape[1] == 2:
        x_recon_mag = torch.sqrt(x_recon[:, 0, ...] ** 2 + x_recon[:, 1, ...] ** 2)
    else:
        x_recon_mag = x_recon

    grasp_img_mag = torch.sqrt(grasp_img[:, 0, ...] ** 2 + grasp_img[:, 1, ...] ** 2)

    if grasp_img_mag.shape[-1] == 320 and grasp_img_mag.shape[-2] == 320:
        n_timeframes = grasp_img_mag.shape[1]
    elif grasp_img_mag.shape[-1] == 320 and grasp_img_mag.shape[1] == 320:
        n_timeframes = grasp_img_mag.shape[-2]
    else:
        n_timeframes = grasp_img_mag.shape[-1]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_timeframes,
        figsize=(n_timeframes * 3, 8),
        squeeze=False,
    )

    if transform:
        axes[0, 0].set_ylabel("Transformed Image", fontsize=14, labelpad=10)
        axes[1, 0].set_ylabel("Model Output", fontsize=14, labelpad=10)

        os.makedirs(os.path.join(output_dir, "transforms"), exist_ok=True)

    else:
        
        axes[0, 0].set_ylabel("Model Output", fontsize=14, labelpad=10)
        axes[1, 0].set_ylabel("GRASP Benchmark", fontsize=14, labelpad=10)
    
    for t in range(n_timeframes):

        if x_recon_mag.shape[1] == n_timeframes:
            img = x_recon_mag[batch_idx, t, :, :].cpu().detach().numpy()
        else:
            img = x_recon_mag[batch_idx, ..., t].cpu().detach().numpy()

        if grasp_img_mag.shape[1] == n_timeframes:
            grasp_img = grasp_img_mag[batch_idx, t, :, :].cpu().detach().numpy()
        elif grasp_img_mag.shape[-1] == n_timeframes:
            grasp_img = grasp_img_mag[batch_idx, :, :, t].cpu().detach().numpy()
        else:
            grasp_img = grasp_img_mag[batch_idx, :, t, :].cpu().detach().numpy()

        ax1 = axes[0, t]
        # ax1.imshow(np.rot90(img, 2), cmap="gray")
        ax1.imshow(img, cmap="gray")
        ax1.set_title(f"t = {t}")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2 = axes[1, t]
        ax2.imshow(grasp_img, cmap="gray")
        ax2.set_title(f"t = {t}")
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close(fig)


def get_git_commit():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except Exception as e:
        print(f"Error retrieving Git commit: {e}")
        return "unknown"
    

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict


def save_checkpoint(model, optimizer, epoch,
                    train_curves, val_curves, eval_curves, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **train_curves,   # unpack the dicts
        **val_curves,
        **eval_curves,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} to {filename}")


def load_checkpoint(model, optimizer, filename):
    ckpt = torch.load(filename, map_location="cpu")

    model.load_state_dict(remove_module_prefix(ckpt["model_state_dict"]))
    optimizer.load_state_dict(remove_module_prefix(ckpt["optimizer_state_dict"]))

    # curves come back as Python lists (or start empty if key not found)
    train_curves = {
        "train_mc_losses": ckpt.get("train_mc_losses", []),
        "train_ei_losses": ckpt.get("train_ei_losses", []),
        "train_adj_losses": ckpt.get("train_adj_losses", []),
        "weighted_train_mc_losses": ckpt.get("weighted_train_mc_losses", []),
        "weighted_train_ei_losses": ckpt.get("weighted_train_ei_losses", []),
        "weighted_train_adj_losses": ckpt.get("weighted_train_adj_losses", []),
    }
    val_curves = {
        "val_mc_losses": ckpt.get("val_mc_losses", []),
        "val_ei_losses": ckpt.get("val_ei_losses", []),
        "val_adj_losses": ckpt.get("val_adj_losses", []),
    }

    eval_curves = {
        "eval_ssims": ckpt.get("eval_ssims", []),
        "eval_psnrs": ckpt.get("eval_psnrs", []),
        "eval_mses": ckpt.get("eval_mses", []),
        "eval_lpipses": ckpt.get("eval_lpipses", []),
        "eval_dc_mses": ckpt.get("eval_dc_mses", []),
        "eval_dc_maes": ckpt.get("eval_dc_maes", []),
        "eval_curve_corrs": ckpt.get("eval_curve_corrs", []),
    }

    return model, optimizer, ckpt.get("epoch", 1), train_curves, val_curves, eval_curves

def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    return torch.view_as_complex(rearrange(x, "b c ... -> b ... c").contiguous())



def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):

    N_tot_spokes = N_spokes * N_time

    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    return np.squeeze(traj)



def GRASPRecon(csmaps, kspace, spokes_per_frame, num_frames, grasp_path):

    traj = get_traj(N_spokes=spokes_per_frame, N_time=num_frames)
    device = sp.Device(0 if torch.cuda.is_available() else -1)

    kspace = rearrange(kspace, 'c (sp sam) t -> t c sp sam', sam=640).unsqueeze(1).unsqueeze(3).cpu().numpy()
    csmaps = rearrange(csmaps, 'b c h w -> c b h w').cpu().numpy()

    # reconstruct image
    R1 = app.HighDimensionalRecon(kspace, csmaps,
                            combine_echo=False,
                            lamda=0.001,
                            coord=traj,
                            regu='TV', regu_axes=[0],
                            max_iter=10,
                            solver='ADMM', rho=0.1,
                            device=device,
                            show_pbar=False,
                            verbose=False).run()

    R1 = np.squeeze(R1.get())

    np.save(grasp_path, R1)
    print(f"GRASP Recon with {spokes_per_frame} spokes/frame and {num_frames} timeframes saved to {grasp_path}")

    return R1