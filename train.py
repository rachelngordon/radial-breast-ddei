import argparse
import json
import os
import subprocess

import deepinv as dinv
import matplotlib.pyplot as plt
import torch
import yaml
from crnn import CRNN, ArtifactRemovalCRNN
from dataloader import SliceDataset
# from deepinv.loss import MCLoss#, EILoss
from deepinv.transform import Transform
from einops import rearrange
from radial import DynamicRadialPhysics, RadialDCLayer, to_torch_complex, MCNUFFT_CRNN
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import numpy as np
from transform import VideoRotate, VideoDiffeo, SubsampleTime, MonophasicTimeWarp, TemporalNoise, TimeReverse
from ei import EILoss
from mc import MCLoss
from lsfpnet import LSFPNet, ArtifactRemovalLSFPNet
from radial_lsfp import MCNUFFT, MCNUFFT_pure
import torchkbnufft as tkbn
from torch.amp import GradScaler, autocast
import csv


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


# def log_gradient_stats(model, epoch, iteration, output_dir, log_filename="gradient_stats.csv"):
#     """
#     Computes, prints, and logs the L2 norm of gradients for each parameter and the total gradient norm.
    
#     Args:
#         model (torch.nn.Module): The model being trained.
#         epoch (int): The current epoch.
#         iteration (int): The current global iteration/step count.
#         output_dir (str): The main experiment output directory.
#         log_filename (str): The CSV filename for storing detailed logs.
#     """
#     total_norm = 0.0
#     param_norms = []
    
#     # Iterate over all named parameters
#     for name, p in model.named_parameters():
#         if p.grad is not None and p.requires_grad:
#             # Calculate the L2 norm of the gradient for this parameter
#             param_norm = p.grad.data.norm(2)
#             param_norms.append((name, param_norm.item()))
#             total_norm += param_norm.item() ** 2
            
#     total_norm = total_norm ** 0.5
    
#     # --- Logging to Console ---
#     print(f"--- Gradient Stats (Epoch {epoch}, Iter {iteration}) ---")
#     print(f"Total Gradient Norm: {total_norm:.10f}")
    
#     # Sort parameters by gradient norm (descending) to see the largest ones
#     param_norms.sort(key=lambda x: x[1], reverse=True)
    
#     print("Top 5 layers with largest gradients:")
#     for name, norm in param_norms[:5]:
#         print(f"  - {name}: {norm:.10f}")
        
#     print("Top 5 layers with smallest gradients:")
#     for name, norm in param_norms[-5:]:
#         print(f"  - {name}: {norm:.10f}")
#     print("-------------------------------------------------")

#     # --- Logging to CSV File for later analysis ---
#     log_path = os.path.join(output_dir, log_filename)
#     file_exists = os.path.isfile(log_path)
    
#     with open(log_path, 'a', newline='') as csvfile:
#         # Create a writer and write the header row if the file is new
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             writer.writerow(['epoch', 'iteration', 'total_norm', 'param_name', 'param_norm'])
            
#         # Write the total norm
#         writer.writerow([epoch, iteration, total_norm, '---TOTAL---', total_norm])
#         # Write individual parameter norms
#         for name, norm in param_norms:
#             writer.writerow([epoch, iteration, total_norm, name, norm])

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

    if grasp_img_mag.shape[1] == 320:
        n_timeframes = grasp_img_mag.shape[-1]
    else:
        n_timeframes = grasp_img_mag.shape[1]

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


        grasp_img = grasp_img_mag[batch_idx, t, :, :].cpu().detach().numpy()
        ax1 = axes[0, t]
        ax1.imshow(np.rot90(img, 2), cmap="gray")
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


def save_checkpoint(model, optimizer, scheduler, epoch,
                    train_curves, val_curves, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **train_curves,   # unpack the dicts
        **val_curves,
    }

    # Add scheduler state if it exists
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

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
        "learning_rates": ckpt.get("learning_rates", []),
        
    }
    val_curves = {
        "val_mc_losses": ckpt.get("val_mc_losses", []),
        "val_ei_losses": ckpt.get("val_ei_losses", []),
        "val_adj_losses": ckpt.get("val_adj_losses", []),
    }

    return model, optimizer, ckpt.get("epoch", 1), train_curves, val_curves

    

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train ReconResNet model.")
parser.add_argument(
    "--config",
    type=str,
    required=False,
    default="config.yaml",
    help="Path to the configuration file",
)
parser.add_argument(
    "--exp_name", type=str, required=True, help="Name of the experiment"
)
parser.add_argument(
    "--from_checkpoint",
    type=bool,
    required=False,
    default=False,
    help="Whether to load from a checkpoint",
)
args = parser.parse_args()

# print experiment name and git commit
commit_hash = get_git_commit()
print(f"Running experiment on Git commit: {commit_hash}")

exp_name = args.exp_name
print(f"Experiment: {exp_name}")

# Load the configuration file
if args.from_checkpoint == True:
    with open(f"output/{exp_name}/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    with open(args.config, "r") as file:
        new_config = yaml.safe_load(file)
    
    epochs = new_config['training']["epochs"]
else:
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    epochs = config['training']["epochs"]


output_dir = os.path.join(config["experiment"]["output_dir"], exp_name)
os.makedirs(output_dir, exist_ok=True)


# Save the configuration file
if args.from_checkpoint == False:
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)


# load params
split_file = config["data"]["split_file"]

batch_size = config["dataloader"]["batch_size"]
max_subjects = config["dataloader"]["max_subjects"]

mc_loss_weight = config["model"]["losses"]["mc_loss"]["weight"]
use_ei_loss = config["model"]["losses"]["use_ei_loss"]
target_weight = config["model"]["losses"]["ei_loss"]["weight"]
warmup = config["model"]["losses"]["ei_loss"]["warmup"]
duration = config["model"]["losses"]["ei_loss"]["duration"]
lambda_adj = config["model"]["losses"]["adj_loss"]["weight"]

save_interval = config["training"]["save_interval"]
plot_interval = config["training"]["plot_interval"]
device = torch.device(config["training"]["device"])
start_epoch = 1

model_type = config["model"]["name"]

H, W = config["data"]["height"], config["data"]["width"]
N_time, N_samples, N_coils = (
    config["data"]["timeframes"],
    config["data"]["spokes_per_frame"],
    config["data"]["coils"],
)
N_spokes = int(config["data"]["total_spokes"] / N_time)

os.makedirs(os.path.join(output_dir, 'enhancement_curves'), exist_ok=True)

# load data
with open(split_file, "r") as fp:
    splits = json.load(fp)


# NOTE: need to look into why I am only loading 88 training samples and not 192
if max_subjects < 300:
    max_train = int(max_subjects * (1 - config["data"]["val_split_ratio"]))
    max_val =int(max_subjects * config["data"]["val_split_ratio"])

    train_patient_ids = splits["train"][:max_train]
    val_patient_ids = splits["val"][:max_val]
else:
    train_patient_ids = splits["train"]
    val_patient_ids = splits["val"]


train_dataset = SliceDataset(
    root_dir=config["data"]["root_dir"],
    patient_ids=train_patient_ids,
    dataset_key=config["data"]["dataset_key"],
    file_pattern="*.h5",
    slice_idx=config["dataloader"]["slice_idx"],
    N_coils=N_coils
)

val_dataset = SliceDataset(
    root_dir=config["data"]["root_dir"],
    patient_ids=val_patient_ids,
    dataset_key=config["data"]["dataset_key"],
    file_pattern="*.h5",
    slice_idx=config["dataloader"]["slice_idx"],
    N_coils=N_coils
)


train_loader = DataLoader(
    train_dataset,
    batch_size=config["dataloader"]["batch_size"],
    shuffle=config["dataloader"]["shuffle"],
    num_workers=config["dataloader"]["num_workers"],
)


val_loader = DataLoader(
    val_dataset,
    batch_size=config["dataloader"]["batch_size"],
    shuffle=config["dataloader"]["shuffle"],
    num_workers=config["dataloader"]["num_workers"],
)


# NOTE: currently processing all 8 timeframes as one group, can be changed later
ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
ktraj = ktraj.to(device)
dcomp = dcomp.to(device)
nufft_ob = nufft_ob.to(device)
adjnufft_ob = adjnufft_ob.to(device)


if model_type == "CRNN":
    # physics = DynamicRadialPhysics(
    # im_size=(H, W, N_time),
    # N_spokes=N_spokes,
    # N_samples=N_samples,
    # N_time=N_time,
    # N_coils=N_coils,
    # )

    physics = MCNUFFT_CRNN(nufft_ob, adjnufft_ob, ktraj, dcomp, N_time, N_spokes, N_samples, N_coils)

    datalayer = RadialDCLayer(physics=physics)
    backbone = CRNN(
        num_cascades=config["model"]["cascades"],
        chans=config["model"]["channels"],
        datalayer=datalayer,
    ).to(device)

    model = ArtifactRemovalCRNN(backbone_net=backbone).to(device)

elif model_type == "LSFPNet":

    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)
    physics_pure = MCNUFFT_pure(nufft_ob, adjnufft_ob, ktraj)

    lsfp_backbone = LSFPNet(LayerNo=config["model"]["num_layers"], channels=config['model']['channels'])
    model = ArtifactRemovalLSFPNet(lsfp_backbone).to(device)

else:
    raise(ValueError("Unsupported model."))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["model"]["optimizer"]["lr"],
    betas=(config["model"]["optimizer"]["b1"], config["model"]["optimizer"]["b2"]),
    eps=config["model"]["optimizer"]["eps"],
    weight_decay=config["model"]["optimizer"]["weight_decay"],
)


scheduler = None
if config["model"]["scheduler"]["enable"]:
    print("INFO: Initializing CosineAnnealingLR scheduler.")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],  # The number of epochs for one cycle
        eta_min=config["model"]["scheduler"]["min_lr"] # The minimum learning rate
    )


# Load the checkpoint to resume training
if args.from_checkpoint == True:
    checkpoint_file = f'output/{exp_name}/{exp_name}_model.pth'
    model, optimizer, start_epoch, train_curves, val_curves = load_checkpoint(model, optimizer, checkpoint_file)
    print("start epoch: ", start_epoch)

    if scheduler is not None:
        ckpt = torch.load(checkpoint_file, map_location="cpu")
        if "scheduler_state_dict" in ckpt:
            print("INFO: Loading scheduler state from checkpoint.")
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # If resuming an old checkpoint without a scheduler, fast-forward it
            print("INFO: No scheduler state in checkpoint. Fast-forwarding scheduler.")
            for _ in range(start_epoch - 1):
                scheduler.step()
else:
    start_epoch = 1


# define transformations and loss functions
mc_loss_fn = MCLoss(model_type=model_type)

if use_ei_loss:
    # rotate = VideoRotate(n_trans=1, interpolation_mode=InterpolationMode.BILINEAR)
    rotate = VideoRotate(n_trans=1, interpolation_mode="bilinear")
    diffeo = VideoDiffeo(n_trans=1, device=device)

    subsample = SubsampleTime(n_trans=1, subsample_ratio_range=(config['model']['losses']['ei_loss']['subsample_ratio_min'], config['model']['losses']['ei_loss']['subsample_ratio_max']))
    monophasic_warp = MonophasicTimeWarp(n_trans=1, warp_ratio_range=(config['model']['losses']['ei_loss']['warp_ratio_min'], config['model']['losses']['ei_loss']['warp_ratio_max']))
    temp_noise = TemporalNoise(n_trans=1)
    time_reverse = TimeReverse(n_trans=1)

    # NOTE: set apply_noise = FALSE for now multi coil implementation
    if config['model']['losses']['ei_loss']['temporal_transform'] == "subsample":
        ei_loss_fn = EILoss(subsample | (diffeo | rotate), model_type=model_type, dcomp=dcomp)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "monophasic":
        ei_loss_fn = EILoss(monophasic_warp | (diffeo | rotate), model_type=model_type, dcomp=dcomp)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "noise":
        ei_loss_fn = EILoss(temp_noise | (diffeo | rotate), model_type=model_type, dcomp=dcomp)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "reverse":
        ei_loss_fn = EILoss(time_reverse | (diffeo | rotate), model_type=model_type, dcomp=dcomp)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "all":
        ei_loss_fn = EILoss((subsample | monophasic_warp | temp_noise | time_reverse) | (diffeo | rotate), model_type=model_type, dcomp=dcomp)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "noise_monophasic":
        ei_loss_fn = EILoss((monophasic_warp | temp_noise) | (diffeo | rotate), model_type=model_type, dcomp=dcomp)
    else:
        raise(ValueError, "Unsupported Temporal Transform.")


print(
    "--- Generating and saving a Zero-Filled (ZF) reconstruction sample before training ---"
)
# Use the validation loader to get a sample without affecting the training loader's state
with torch.no_grad():
    # Get a single batch of validation k-space data
    val_kspace_sample, csmap, grasp_img = next(iter(val_loader))
    val_kspace_sample = val_kspace_sample.to(device)

    # Perform the simplest reconstruction: A_adjoint(y)
    # This is the "zero-filled" image (or more accurately, the gridded image)
    if model_type == "CRNN":
        x_zf = physics.A_adjoint(val_kspace_sample, csmap)
    elif model_type == "LSFPNet":
        val_kspace_sample = to_torch_complex(val_kspace_sample).squeeze()
        val_kspace_sample = rearrange(val_kspace_sample, 't co sp sam -> co (sp sam) t')
        
        x_zf = physics(inv=True, data=val_kspace_sample, smaps=csmap.to(device))

        # compute magnitude and add batch dimx
        x_zf = torch.abs(x_zf).unsqueeze(0)

    # Plot and save the image using your existing function
    plot_reconstruction_sample(
        x_zf,
        "Zero-Filled (ZF) Reconstruction (Before Training)",
        "zf_reconstruction_baseline",
        output_dir,
        grasp_img
    )
print("--- ZF baseline image saved to output directory. Starting training. ---")

if args.from_checkpoint:
    train_mc_losses = train_curves["train_mc_losses"]
    val_mc_losses = val_curves["val_mc_losses"]
    train_ei_losses = train_curves["train_ei_losses"]
    val_ei_losses = val_curves["val_ei_losses"]
    train_adj_losses = train_curves["train_adj_losses"]
    val_adj_losses = train_curves["val_adj_losses"]
    weighted_train_mc_losses = train_curves["weighted_train_mc_losses"]
    weighted_train_ei_losses = train_curves["weighted_train_ei_losses"]
    weighted_train_adj_losses = train_curves["weighted_train_adj_losses"]
    learning_rates = train_curves["learning_rates"]
else:
    train_mc_losses = []
    val_mc_losses = []
    train_ei_losses = []
    val_ei_losses = []
    train_adj_losses = []
    val_adj_losses = []
    weighted_train_mc_losses = []
    weighted_train_ei_losses = []
    weighted_train_adj_losses = []
    learning_rates = []

iteration_count = 0

# scaler = GradScaler()


# Step 0: Evaluate the untrained model
if config["debugging"]["calc_step_0"]:
    if args.from_checkpoint == False:
        model.eval()
        initial_train_mc_loss = 0.0
        initial_val_mc_loss = 0.0
        initial_train_ei_loss = 0.0
        initial_val_ei_loss = 0.0
        initial_train_adj_loss = 0.0
        initial_val_adj_loss = 0.0


        with torch.no_grad():
            # Evaluate on training data
            for measured_kspace, csmap, grasp_img in tqdm(train_loader, desc="Step 0 Training Evaluation"):

                # with autocast(config["training"]["device"]):

                if model_type == "LSFPNet":

                    measured_kspace = to_torch_complex(measured_kspace).squeeze()
                    measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

                    csmap = csmap.to(device).to(measured_kspace.dtype)

                    # 1. --- NEW NORMALIZATION STEP ---
                    # Calculate a stable scaling factor from the k-space data itself.
                    with torch.no_grad():
                        # Use the 99th percentile for robustness against a single hot pixel/outlier
                        scale = torch.quantile(measured_kspace.abs(), 0.99)
                        # Add epsilon for safety
                        scale = scale + 1e-8
                    
                    # Normalize the input k-space
                    measured_kspace_norm = measured_kspace / scale

                    x_recon, total_adj_loss = model(
                        measured_kspace_norm.to(device), physics, csmap, dcomp
                    )

                    initial_train_adj_loss += total_adj_loss.item()

                    mc_loss = mc_loss_fn(measured_kspace_norm.to(device), x_recon, physics_pure, csmap)
                    initial_train_mc_loss += mc_loss.item()

                else:

                    x_recon = model(
                        measured_kspace.to(device), physics, csmap
                    )  # model output shape: (B, C, T, H, W)

                    mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics_pure, csmap)
                    initial_train_mc_loss += mc_loss.item()

                if use_ei_loss:
                    # x_recon: reconstructed image
                    ei_loss, t_img = ei_loss_fn(
                        x_recon, physics, model, csmap
                    )

                    initial_train_ei_loss += ei_loss.item()

            step0_train_mc_loss = initial_train_mc_loss / len(train_loader)
            train_mc_losses.append(step0_train_mc_loss)

            step0_train_ei_loss = initial_train_ei_loss / len(train_loader)
            train_ei_losses.append(step0_train_ei_loss)

            step0_train_adj_loss = initial_train_adj_loss / len(train_loader)
            train_adj_losses.append(step0_train_adj_loss)


            # Evaluate on validation data
            for measured_kspace, csmap, grasp_img in tqdm(val_loader, desc="Step 0 Validation Evaluation"):

                # with autocast(config["training"]["device"]):

                if model_type == "LSFPNet":

                    measured_kspace = to_torch_complex(measured_kspace).squeeze()
                    measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

                    csmap = csmap.to(device).to(measured_kspace.dtype)

                    # 1. --- NEW NORMALIZATION STEP ---
                    # Calculate a stable scaling factor from the k-space data itself.
                    with torch.no_grad():
                        # Use the 99th percentile for robustness against a single hot pixel/outlier
                        scale = torch.quantile(measured_kspace.abs(), 0.99)
                        # Add epsilon for safety
                        scale = scale + 1e-8
                    
                    # Normalize the input k-space
                    measured_kspace_norm = measured_kspace / scale

                    x_recon, total_adj_loss = model(
                        measured_kspace_norm.to(device), physics, csmap, dcomp
                    )

                    initial_val_adj_loss += total_adj_loss.item()

                    mc_loss = mc_loss_fn(measured_kspace_norm.to(device), x_recon, physics_pure, csmap)
                    initial_val_mc_loss += mc_loss.item()
                
                else:

                    x_recon = model(
                        measured_kspace.to(device), physics, csmap
                    )  # model output shape: (B, C, T, H, W)

                    mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics_pure, csmap)
                    initial_val_mc_loss += mc_loss.item()

                if use_ei_loss:
                    # x_recon: reconstructed image
                    ei_loss, t_img = ei_loss_fn(
                        x_recon, physics, model, csmap
                    )

                    initial_val_ei_loss += ei_loss.item()

            step0_val_mc_loss = initial_val_mc_loss / len(val_loader)
            val_mc_losses.append(step0_val_mc_loss)

            step0_val_ei_loss = initial_val_ei_loss / len(val_loader)
            val_ei_losses.append(step0_val_ei_loss)

            step0_val_adj_loss = initial_val_adj_loss / len(val_loader)
            val_adj_losses.append(step0_val_adj_loss)


            print(f"Step 0 Training MC Loss: {step0_train_mc_loss}, Validation MC Loss: {step0_val_mc_loss}")
            print(f"Step 0 Training Adj Loss: {step0_train_adj_loss}, Validation Adj Loss: {step0_val_adj_loss}")

            if use_ei_loss:
                print(f"Step 0 Training EI Loss: {step0_train_ei_loss}, Validation EI Loss: {step0_val_ei_loss}")

# Training Loop
if (epochs + 1) == start_epoch:
    raise(ValueError("Full training epochs already complete."))

else: 

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_mc_loss = 0.0
        running_ei_loss = 0.0
        running_adj_loss = 0.0
        # turn on anomaly detection for debugging but slows down training
        # with torch.autograd.set_detect_anomaly(False):
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch"
        )
        # measured_kspace shape: (B, C, I, S, T) = 1, 1, 2, 23040, 8
        for measured_kspace, csmap, grasp_img in train_loader_tqdm:  # measured_kspace shape: (B, C, I, S, T)

            # with autocast(config["training"]["device"]):

            iteration_count += 1
            optimizer.zero_grad()


            if model_type == "LSFPNet":

                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

                csmap = csmap.to(device).to(measured_kspace.dtype)


                # 1. --- NEW NORMALIZATION STEP ---
                # Calculate a stable scaling factor from the k-space data itself.
                with torch.no_grad():
                    # Use the 99th percentile for robustness against a single hot pixel/outlier
                    scale = torch.quantile(measured_kspace.abs(), 0.99)
                    # Add epsilon for safety
                    scale = scale + 1e-8
                
                # Normalize the input k-space
                measured_kspace_norm = measured_kspace / scale

                x_recon, total_adjoint_loss  = model(
                    measured_kspace_norm.to(device), physics, csmap, dcomp
                )

                running_adj_loss += total_adjoint_loss.item()

                mc_loss = mc_loss_fn(measured_kspace_norm.to(device), x_recon, physics_pure, csmap)
                running_mc_loss += mc_loss.item()

            else:

                x_recon = model(
                    measured_kspace.to(device), physics, csmap
                )  # model output shape: (B, C, T, H, W)

                mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics_pure, csmap)
                running_mc_loss += mc_loss.item()

            if use_ei_loss and epoch > warmup:
                # x_recon: reconstructed image
                ei_loss, t_img = ei_loss_fn(
                    x_recon, physics, model, csmap
                )

                ei_loss_weight = get_cosine_ei_weight(
                    current_epoch=epoch,
                    warmup_epochs=warmup,
                    schedule_duration=duration,
                    target_weight=target_weight
                )

                
                running_ei_loss += ei_loss.item()

                if model_type == "LSFPNET":
                    total_loss = mc_loss * mc_loss_weight + ei_loss * ei_loss_weight + lambda_adj * total_adjoint_loss
                else:
                    total_loss = mc_loss * mc_loss_weight + ei_loss * ei_loss_weight
                train_loader_tqdm.set_postfix(
                    mc_loss=mc_loss.item(), ei_loss=ei_loss.item()
                )

            else:

                total_loss = mc_loss * mc_loss_weight + lambda_adj * total_adjoint_loss
                train_loader_tqdm.set_postfix(mc_loss=mc_loss.item())

            if torch.isnan(total_loss):
                print(
                    "!!! ERROR: total_loss is NaN before backward pass. Aborting. !!!"
                )
                raise RuntimeError("total_loss is NaN")

            # scaler.scale(total_loss).backward()
            total_loss.backward()

            if config["debugging"]["enable_gradient_monitoring"] == True and iteration_count % config["debugging"]["monitoring_interval"] == 0:
            
                log_gradient_stats(
                    model=model,
                    epoch=epoch,
                    iteration=iteration_count,
                    output_dir=output_dir,
                    log_filename="gradient_stats.csv"
                )


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

            if epoch % save_interval == 0:
                plot_reconstruction_sample(
                    x_recon,
                    f"Training Sample - Epoch {epoch}",
                    f"train_sample_epoch_{epoch}",
                    output_dir,
                    grasp_img
                )

                x_recon_reshaped = rearrange(x_recon, 'b c h w t -> b c t h w')

                plot_enhancement_curve(
                    x_recon_reshaped,
                    output_filename = os.path.join(output_dir, 'enhancement_curves', f'train_sample_enhancement_curve_epoch_{epoch}.png'))
                
                plot_enhancement_curve(
                    grasp_img,
                    output_filename = os.path.join(output_dir, 'enhancement_curves', f'grasp_sample_enhancement_curve_epoch_{epoch}.png'))

                if use_ei_loss and epoch > warmup:

                    plot_reconstruction_sample(
                        t_img,
                        f"Transformed Train Sample - Epoch {epoch}",
                        f"transforms/transform_train_sample_epoch_{epoch}",
                        output_dir,
                        x_recon,
                        transform=True
                    )

        # Calculate and store average epoch losses
        epoch_train_mc_loss = running_mc_loss / len(train_loader)
        train_mc_losses.append(epoch_train_mc_loss)
        weighted_train_mc_losses.append(epoch_train_mc_loss*mc_loss_weight)

        if use_ei_loss and epoch > warmup:
            epoch_train_ei_loss = running_ei_loss / len(train_loader)
            train_ei_losses.append(epoch_train_ei_loss)
            weighted_train_ei_losses.append(epoch_train_ei_loss*ei_loss_weight)
        else:
            # Append 0 if EI loss is not used to keep lists aligned
            train_ei_losses.append(0.0)
            weighted_train_ei_losses.append(0.0)

        if model_type == "LSFPNet":
            epoch_train_adj_loss = running_adj_loss / len(train_loader)
            train_adj_losses.append(epoch_train_adj_loss)
            weighted_train_adj_losses.append(epoch_train_adj_loss*lambda_adj)
        else:
            train_adj_losses.append(0.0)
            weighted_train_adj_losses.append(0.0)



        # --- Validation Loop ---
        model.eval()
        val_running_mc_loss = 0.0
        val_running_ei_loss = 0.0
        val_running_adj_loss = 0.0
        val_loader_tqdm = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{epochs}  Validation",
            unit="batch",
            leave=False,
        )
        with torch.no_grad():
            for val_kspace_batch, val_csmap, val_grasp_img in val_loader_tqdm:

                # with autocast(config["training"]["device"]):

                if model_type == "LSFPNet":

                    val_kspace_batch = to_torch_complex(val_kspace_batch).squeeze()
                    val_kspace_batch = rearrange(val_kspace_batch, 't co sp sam -> co (sp sam) t')

                    val_csmap = val_csmap.to(device).to(val_kspace_batch.dtype)

                    # 1. --- NEW NORMALIZATION STEP ---
                    # Calculate a stable scaling factor from the k-space data itself.
                    with torch.no_grad():
                        # Use the 99th percentile for robustness against a single hot pixel/outlier
                        scale = torch.quantile(val_kspace_batch.abs(), 0.99)
                        # Add epsilon for safety
                        scale = scale + 1e-8
                    
                    # Normalize the input k-space
                    val_kspace_batch_norm = val_kspace_batch / scale

                    val_x_recon, val_adj_loss = model(
                        val_kspace_batch_norm.to(device), physics, val_csmap, dcomp
                    )

                    val_running_adj_loss += val_adj_loss.item()

                    # For MCLoss, compare the physics model's output with the measured k-space.
                    val_mc_loss = mc_loss_fn(val_kspace_batch_norm.to(device), val_x_recon, physics_pure, val_csmap)
                    val_running_mc_loss += val_mc_loss.item()

                else:
                    # The model takes the raw k-space and physics operator
                    val_x_recon = model(val_kspace_batch.to(device), physics, val_csmap)

                    # For MCLoss, compare the physics model's output with the measured k-space.
                    val_mc_loss = mc_loss_fn(val_kspace_batch.to(device), val_x_recon, physics_pure, val_csmap)
                    val_running_mc_loss += val_mc_loss.item()

                if use_ei_loss and epoch > warmup:
                    val_ei_loss, val_t_img = ei_loss_fn(
                        val_x_recon, physics, model, val_csmap
                    )
                    val_running_ei_loss += val_ei_loss.item()
                    val_loader_tqdm.set_postfix(
                        val_mc_loss=val_mc_loss.item(), val_ei_loss=val_ei_loss.item()
                    )
                else:
                    val_loader_tqdm.set_postfix(val_mc_loss=val_mc_loss.item())

            # save a sample from the last validation batch of the epoch
            if epoch % save_interval == 0:
                plot_reconstruction_sample(
                    val_x_recon,
                    f"Validation Sample - Epoch {epoch}",
                    f"val_sample_epoch_{epoch}",
                    output_dir,
                    val_grasp_img
                )

                val_x_recon_reshaped = rearrange(val_x_recon, 'b c h w t -> b c t h w')

                plot_enhancement_curve(
                    val_x_recon_reshaped,
                    output_filename = os.path.join(output_dir, 'enhancement_curves', f'val_sample_enhancement_curve_epoch_{epoch}.png'))
                
                plot_enhancement_curve(
                    val_grasp_img,
                    output_filename = os.path.join(output_dir, 'enhancement_curves', f'val_grasp_sample_enhancement_curve_epoch_{epoch}.png'))

                if use_ei_loss and epoch > warmup:
                    plot_reconstruction_sample(
                        val_t_img,
                        f"Transformed Validation Sample - Epoch {epoch}",
                        f"transforms/transform_val_sample_epoch_{epoch}",
                        output_dir,
                        val_x_recon,
                        transform=True
                    )


            # Save the model checkpoint
            # model_save_path = os.path.join(output_dir, f'{exp_name}_model_checkpoint_epoch{epoch}.pth')
            # torch.save(model.state_dict(), model_save_path)
            # print(f'Model saved to {model_save_path}')
            train_curves = dict(
                train_mc_losses=train_mc_losses,
                train_ei_losses=train_ei_losses,
                train_adj_losses=train_adj_losses,
                weighted_train_mc_losses=weighted_train_mc_losses,
                weighted_train_ei_losses=weighted_train_ei_losses,
                weighted_train_adj_losses=weighted_train_adj_losses,
            )
            val_curves = dict(
                val_mc_losses=val_mc_losses,
                val_ei_losses=val_ei_losses,
                val_adj_losses=val_adj_losses,
            )
            model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, train_curves, val_curves, model_save_path)
            print(f'Model saved to {model_save_path}')


        # Calculate and store average validation losses
        epoch_val_mc_loss = val_running_mc_loss / len(val_loader)
        val_mc_losses.append(epoch_val_mc_loss)
        if use_ei_loss and epoch > warmup:
            epoch_val_ei_loss = val_running_ei_loss / len(val_loader)
            val_ei_losses.append(epoch_val_ei_loss)
        else:
            val_ei_losses.append(0.0)

        if model_type == "LSFPNet":
            epoch_val_adj_loss = val_running_adj_loss / len(val_loader)
            val_adj_losses.append(epoch_val_adj_loss)
        else:
            val_adj_losses.append(0.0)


        if scheduler is not None:
            scheduler.step()
            # Optional: Log the learning rate to see it change
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            print(f"Epoch {epoch}: Learning rate updated to {current_lr:.8f}")


        # --- Plotting and Logging ---
        if epoch % save_interval == 0:

            # Plot Learning Rate
            if scheduler is not None and len(learning_rates) > 0:
                plt.figure()
                plt.plot(range(1, len(learning_rates) + 1), learning_rates, label="Learning Rate")
                plt.xlabel("Epoch")
                plt.ylabel("Learning Rate")
                plt.title("Learning Rate Schedule")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "learning_rate.png"))
                plt.close()

            # Plot MC Loss
            plt.figure()
            plt.plot(train_mc_losses, label="Training MC Loss")
            plt.plot(val_mc_losses, label="Validation MC Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MC Loss")
            plt.title("Measurement Consistency Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "mc_losses.png"))
            plt.close()

            if use_ei_loss and epoch > warmup:
                # Plot EI Loss
                plt.figure()
                plt.plot(train_ei_losses, label="Training EI Loss")
                plt.plot(val_ei_losses, label="Validation EI Loss")
                plt.xlabel("Epoch")
                plt.ylabel("EI Loss")
                plt.title("Equivariant Imaging Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "ei_losses.png"))
                plt.close()


            # Plot Weighted Losses
            plt.figure()
            plt.plot(weighted_train_mc_losses, label="MC Loss")
            plt.plot(weighted_train_ei_losses, label="EI Loss")
            plt.plot(weighted_train_adj_losses, label="Adj Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Weighted Training Losses")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "weighted_losses.png"))
            plt.close()

            if model_type == "LSFPNet":
                # Plot EI Loss
                plt.figure()
                plt.plot(train_adj_losses, label="Training Adjoint Loss")
                plt.plot(val_adj_losses, label="Validation EIAdjoint Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Adjoint Loss")
                plt.title("LSFPNet CNN Adjoint Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "adj_losses.png"))
                plt.close()



        # Print epoch summary
        print(
            f"Epoch {epoch}: Training MC Loss: {epoch_train_mc_loss:.6f}, Validation MC Loss: {epoch_val_mc_loss:.6f}"
        )
        if model_type == "LSFPNet":
            print(
                f"Epoch {epoch}: Training Adjoint Loss: {epoch_train_adj_loss:.6f}, Validation Adjoint Loss: {epoch_val_adj_loss:.6f}"
            )
        if use_ei_loss and epoch > warmup:
            print(
                f"Epoch {epoch}: Training EI Loss: {epoch_train_ei_loss:.6f}, Validation EI Loss: {epoch_val_ei_loss:.6f}"
            )
            print(f"Epoch {epoch:2d}: EI Weight = {ei_loss_weight:.8f}")


# Save the model at the end of training

train_curves = dict(
    train_mc_losses=train_mc_losses,
    train_ei_losses=train_ei_losses,
    train_adj_losses=train_adj_losses,
    weighted_train_mc_losses=weighted_train_mc_losses,
    weighted_train_ei_losses=weighted_train_ei_losses,
    weighted_train_adj_losses=weighted_train_adj_losses,
)
val_curves = dict(
    val_mc_losses=val_mc_losses,
    val_ei_losses=val_ei_losses,
    val_adj_losses=val_adj_losses,
)
model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
save_checkpoint(model, optimizer, scheduler, epochs + 1, train_curves, val_curves, model_save_path)
print(f'Model saved to {model_save_path}')



