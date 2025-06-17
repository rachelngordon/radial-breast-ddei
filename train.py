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
from deepinv.loss import MCLoss#, EILoss
from deepinv.transform import Transform
from einops import rearrange
from radial import DynamicRadialPhysics, RadialDCLayer
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import numpy as np
from transform import VideoRotate, VideoDiffeo, SubsampleTime, MonophasicTimeWarp, TemporalNoise, TimeReverse
from ei import EILoss


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
    # compute magnitude from complex reconstruction
    x_recon_mag = torch.sqrt(x_recon[:, 0, ...] ** 2 + x_recon[:, 1, ...] ** 2)
    grasp_img_mag = torch.sqrt(grasp_img[:, 0, ...] ** 2 + grasp_img[:, 1, ...] ** 2)

    n_timeframes = x_recon_mag.shape[1]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_timeframes,
        figsize=(n_timeframes * 3, 8),
        squeeze=False,
    )
    if transform:
        axes[0, 0].set_ylabel("Transformed Image", fontsize=14, labelpad=10)
        axes[1, 0].set_ylabel("Model Output", fontsize=14, labelpad=10)

    if transform:
        axes[0, 0].set_ylabel("Transformed Image", fontsize=14, labelpad=10)
        axes[1, 0].set_ylabel("Model Output", fontsize=14, labelpad=10)

        os.makedirs(os.path.join(output_dir, "transforms"), exist_ok=True)

    else:
        axes[0, 0].set_ylabel("Model Output", fontsize=14, labelpad=10)
        axes[1, 0].set_ylabel("GRASP Benchmark", fontsize=14, labelpad=10)
    
    for t in range(n_timeframes):
        img = x_recon_mag[batch_idx, t, :, :].cpu().detach().numpy()
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
else:
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)


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

epochs = config["training"]["epochs"]
save_interval = config["training"]["save_interval"]
plot_interval = config["training"]["plot_interval"]
device = torch.device(config["training"]["device"])
start_epoch = 1


# load data
with open(split_file, "r") as fp:
    splits = json.load(fp)


# NOTE: need to look into why I am only loading 88 training samples and not 192
if max_subjects < 300:
    max_train = max_subjects * (1 - config["data"]["val_split_ratio"])
    max_val = max_subjects * config["data"]["val_split_ratio"]

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
)

val_dataset = SliceDataset(
    root_dir=config["data"]["root_dir"],
    patient_ids=val_patient_ids,
    dataset_key=config["data"]["dataset_key"],
    file_pattern="*.h5",
    slice_idx=config["dataloader"]["slice_idx"],
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


# define physics operators
H, W = config["data"]["height"], config["data"]["width"]
N_time, N_samples, N_coils = (
    config["data"]["timeframes"],
    config["data"]["spokes_per_frame"],
    config["data"]["coils"],
)
N_spokes = int(config["data"]["total_spokes"] / N_time)

physics = DynamicRadialPhysics(
    im_size=(H, W, N_time),
    N_spokes=N_spokes,
    N_samples=N_samples,
    N_time=N_time,
    N_coils=N_coils,
)

datalayer = RadialDCLayer(physics=physics)

backbone = CRNN(
    num_cascades=config["model"]["cascades"],
    chans=config["model"]["channels"],
    datalayer=datalayer,
).to(device)

model = ArtifactRemovalCRNN(backbone_net=backbone).to(device)


# define loss functions and optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["model"]["optimizer"]["lr"],
    betas=(config["model"]["optimizer"]["b1"], config["model"]["optimizer"]["b2"]),
    eps=config["model"]["optimizer"]["eps"],
    weight_decay=config["model"]["optimizer"]["weight_decay"],
)

# define transformations and loss functions
mc_loss_fn = MCLoss()

if use_ei_loss:
    # rotate = VideoRotate(n_trans=1, interpolation_mode=InterpolationMode.BILINEAR)
    rotate = VideoRotate(n_trans=1, interpolation_mode="bilinear")
    diffeo = VideoDiffeo(n_trans=1, device=device)

    subsample = SubsampleTime(n_trans=1, subsample_ratio_range=(config['model']['losses']['ei_loss']['subsample_ratio_min'], config['model']['losses']['ei_loss']['subsample_ratio_max']))
    monophasic_warp = MonophasicTimeWarp(n_trans=1, warp_ratio_range=(config['model']['losses']['ei_loss']['warp_ratio_min'], config['model']['losses']['ei_loss']['warp_ratio_max']))
    temp_noise = TemporalNoise(n_trans=1)
    time_reverse = TimeReverse(n_trans=1)

    if config['model']['losses']['ei_loss']['temporal_transform'] == "subsample":
        ei_loss_fn = EILoss(subsample | (diffeo | rotate))
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "monophasic":
        ei_loss_fn = EILoss(monophasic_warp | (diffeo | rotate))
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "noise":
        ei_loss_fn = EILoss(temp_noise | (diffeo | rotate))
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "reverse":
        ei_loss_fn = EILoss(time_reverse | (diffeo | rotate))
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "all":
        ei_loss_fn = EILoss((subsample | monophasic_warp | temp_noise | time_reverse) | (diffeo | rotate))
    else:
        raise(ValueError, "Unsupported Temporal Transform.")


print(
    "--- Generating and saving a Zero-Filled (ZF) reconstruction sample before training ---"
)
# Use the validation loader to get a sample without affecting the training loader's state
with torch.no_grad():
    # Get a single batch of validation k-space data
    val_kspace_sample, grasp_img = next(iter(val_loader))
    val_kspace_sample = val_kspace_sample.to(device)

    # Perform the simplest reconstruction: A_adjoint(y)
    # This is the "zero-filled" image (or more accurately, the gridded image)
    x_zf = physics.A_adjoint(val_kspace_sample)

    # Plot and save the image using your existing function
    plot_reconstruction_sample(
        x_zf,
        "Zero-Filled (ZF) Reconstruction (Before Training)",
        "zf_reconstruction_baseline",
        output_dir,
        grasp_img
    )
print("--- ZF baseline image saved to output directory. Starting training. ---")


train_mc_losses = []
val_mc_losses = []
train_ei_losses = []
val_ei_losses = []
weighted_train_mc_losses = []
weighted_train_ei_losses = []

iteration_count = 0

# Step 0: Evaluate the untrained model
if args.from_checkpoint == False:
    model.eval()
    initial_train_mc_loss = 0.0
    initial_val_mc_loss = 0.0
    initial_train_ei_loss = 0.0
    initial_val_ei_loss = 0.0


    with torch.no_grad():
        # Evaluate on training data
        for measured_kspace, grasp_img in tqdm(train_loader, desc="Step 0 Training Evaluation"):

            x_recon = model(
                measured_kspace.to(device), physics
            )  # model output shape: (B, C, T, H, W)

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics)
            initial_train_mc_loss += mc_loss.item()

            if use_ei_loss:
                # x_recon: reconstructed image
                ei_loss, t_img = ei_loss_fn(
                    x_recon, physics, model
                )

                initial_train_ei_loss += ei_loss.item()

        step0_train_mc_loss = initial_train_mc_loss / len(train_loader)
        train_mc_losses.append(step0_train_mc_loss)

        step0_train_ei_loss = initial_train_ei_loss / len(train_loader)
        train_ei_losses.append(step0_train_ei_loss)


        # Evaluate on validation data
        for measured_kspace, grasp_img in tqdm(val_loader, desc="Step 0 Validation Evaluation"):

            x_recon = model(
                measured_kspace.to(device), physics
            )  # model output shape: (B, C, T, H, W)

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics)
            initial_val_mc_loss += mc_loss.item()

            if use_ei_loss:
                # x_recon: reconstructed image
                ei_loss, t_img = ei_loss_fn(
                    x_recon, physics, model
                )

                initial_val_ei_loss += ei_loss.item()

        step0_val_mc_loss = initial_val_mc_loss / len(val_loader)
        val_mc_losses.append(step0_val_mc_loss)

        step0_val_ei_loss = initial_val_ei_loss / len(val_loader)
        val_ei_losses.append(step0_val_ei_loss)

# Training Loop
for epoch in range(start_epoch, epochs + 1):
    model.train()
    running_mc_loss = 0.0
    running_ei_loss = 0.0
    # turn on anomaly detection for debugging but slows down training
    with torch.autograd.set_detect_anomaly(False):
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch"
        )
        # measured_kspace shape: (B, C, I, S, T) = 1, 1, 2, 23040, 8
        for measured_kspace, grasp_img in train_loader_tqdm:  # measured_kspace shape: (B, C, I, S, T)
            iteration_count += 1
            optimizer.zero_grad()

            x_recon = model(
                measured_kspace.to(device), physics
            )  # model output shape: (B, C, T, H, W)

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics)
            running_mc_loss += mc_loss.item()

            if use_ei_loss:
                # x_recon: reconstructed image
                ei_loss, t_img = ei_loss_fn(
                    x_recon, physics, model
                )

                ei_loss_weight = get_cosine_ei_weight(
                    current_epoch=epoch,
                    warmup_epochs=warmup,
                    schedule_duration=duration,
                    target_weight=target_weight
                )

                print(f"Epoch {epoch:2d}: EI Weight = {ei_loss_weight:.8f}")
                
                running_ei_loss += ei_loss.item()
                total_loss = mc_loss * mc_loss_weight + ei_loss * ei_loss_weight
                train_loader_tqdm.set_postfix(
                    mc_loss=mc_loss.item(), ei_loss=ei_loss.item()
                )

            else:
                total_loss = mc_loss
                train_loader_tqdm.set_postfix(mc_loss=mc_loss.item())

            if torch.isnan(total_loss):
                print(
                    "!!! ERROR: total_loss is NaN before backward pass. Aborting. !!!"
                )
                raise RuntimeError("total_loss is NaN")

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch % save_interval == 0:
                plot_reconstruction_sample(
                    x_recon,
                    f"Training Sample - Epoch {epoch}",
                    f"train_sample_epoch_{epoch}",
                    output_dir,
                    grasp_img
                )

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
        if use_ei_loss:
            epoch_train_ei_loss = running_ei_loss / len(train_loader)
            train_ei_losses.append(epoch_train_ei_loss)
            weighted_train_ei_losses.append(epoch_train_ei_loss*ei_loss_weight)
        else:
            # Append 0 if EI loss is not used to keep lists aligned
            train_ei_losses.append(0.0)
            weighted_train_ei_losses.append(0.0)

        # --- Validation Loop ---
        model.eval()
        val_running_mc_loss = 0.0
        val_running_ei_loss = 0.0
        val_loader_tqdm = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{epochs}  Validation",
            unit="batch",
            leave=False,
        )
        with torch.no_grad():
            for val_kspace_batch, val_grasp_img in val_loader_tqdm:
                # The model takes the raw k-space and physics operator
                val_x_recon = model(val_kspace_batch.to(device), physics)

                # For MCLoss, compare the physics model's output with the measured k-space.
                val_y_meas = val_kspace_batch
                val_mc_loss = mc_loss_fn(val_y_meas.to(device), val_x_recon, physics)
                val_running_mc_loss += val_mc_loss.item()

                if use_ei_loss:
                    val_ei_loss, val_t_img = ei_loss_fn(
                        val_x_recon, physics, model
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

                plot_reconstruction_sample(
                    val_t_img,
                    f"Transformed Validation Sample - Epoch {epoch}",
                    f"transforms/transform_val_sample_epoch_{epoch}",
                    output_dir,
                    val_x_recon,
                    transform=True
                )


                # Save the model checkpoint
                model_save_path = os.path.join(output_dir, f'{exp_name}_model_checkpoint_epoch{epoch}.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f'Model saved to {model_save_path}')


        # Calculate and store average validation losses
        epoch_val_mc_loss = val_running_mc_loss / len(val_loader)
        val_mc_losses.append(epoch_val_mc_loss)
        if use_ei_loss:
            epoch_val_ei_loss = val_running_ei_loss / len(val_loader)
            val_ei_losses.append(epoch_val_ei_loss)
        else:
            val_ei_losses.append(0.0)

        # --- Plotting and Logging ---
        if epoch % save_interval == 0:
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

            if use_ei_loss:
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
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Weighted Training Losses")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "weighted_losses.png"))
                plt.close()


        # Print epoch summary
        print(
            f"Epoch {epoch}: Training MC Loss: {epoch_train_mc_loss:.6f}, Validation MC Loss: {epoch_val_mc_loss:.6f}"
        )
        if use_ei_loss:
            print(
                f"Epoch {epoch}: Training EI Loss: {epoch_train_ei_loss:.6f}, Validation EI Loss: {epoch_val_ei_loss:.6f}"
            )


# Save the model at the end of training
model_save_path = os.path.join(output_dir, f'{exp_name}_model_checkpoint.pth')
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')


# delete all other checkpoints for this run after the model is saved


