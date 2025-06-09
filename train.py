import argparse
import json
import os
import subprocess

import deepinv as dinv
import matplotlib.pyplot as plt
import torch
import yaml
from crnn import CRNN, ArtifactRemovalCRNN
from dataloader import KSpaceSliceDataset
from deepinv.loss import EILoss, MCLoss
from deepinv.transform import Transform
from einops import rearrange
from radial import DynamicRadialPhysics, RadialDCLayer
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def plot_reconstruction_sample(x_recon, title, filename, output_dir, batch_idx=0):
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

    n_timeframes = x_recon_mag.shape[1]
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_timeframes,
        figsize=(n_timeframes * 3, 4),
        squeeze=False,
    )
    for t in range(n_timeframes):
        img = x_recon_mag[batch_idx, t, :, :].cpu().numpy()
        ax = axes[0, t]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"t = {t}")
        ax.axis("off")
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


class VideoRotate(dinv.transform.Rotate):
    """A Rotate transform that correctly handles 5D video tensors by flattening time into the batch dimension."""

    def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
        # First, check if we even need to flatten. If it's already 4D, just rotate.
        if not self._check_x_5D(x):
            return super()._transform(x, **params)

        # It's a 5D video tensor. Flatten time into the batch dimension.
        B = x.shape[0]
        x_flat = dinv.physics.TimeMixin.flatten(x)  # (B, C, T, H, W) -> (B*T, C, H, W)

        # The parent's _transform method can now work correctly on the 4D tensor (batch of 2D images).
        # We need to get the right parameters for this new batch size.
        # The `get_params` is usually called before `_transform`, so we should be okay.
        # However, to be safe, let's pass a modified params dictionary.
        flat_params = self.get_params(x_flat)

        transformed_flat = super()._transform(x_flat, **flat_params)

        # Unflatten to restore the original 5D video shape.
        return dinv.physics.TimeMixin.unflatten(transformed_flat, batch_size=B)


class VideoDiffeo(dinv.transform.CPABDiffeomorphism):
    """A Diffeomorphism transform that correctly handles 5D video tensors."""

    def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
        if not self._check_x_5D(x):
            return super()._transform(x, **params)

        B = x.shape[0]
        x_flat = dinv.physics.TimeMixin.flatten(x)
        flat_params = self.get_params(x_flat)
        transformed_flat = super()._transform(x_flat, **flat_params)
        return dinv.physics.TimeMixin.unflatten(transformed_flat, batch_size=B)


class SubsampleTime(Transform):
    r"""
    Augments a video by taking a random contiguous temporal sub-sequence.
    This is suitable for non-cyclical data like contrast enhancement curves,
    as it preserves the local arrow of time.

    :param int n_trans: Number of transformed versions to generate per input image.
    :param float subsample_ratio: The ratio of the total time frames to keep (e.g., 0.8 for 80%).
    :param torch.Generator rng: Random number generator.
    """

    def __init__(self, *args, subsample_ratio: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False  # We operate directly on the 5D tensor
        assert 0.0 < subsample_ratio <= 1.0, "subsample_ratio must be between 0 and 1."
        self.subsample_ratio = subsample_ratio

    def _get_params(self, x: torch.Tensor) -> dict:
        """Generates a random start index for the temporal crop."""
        total_time_frames = x.shape[2]  # Shape is (B, C, T, H, W)
        subsample_length = int(total_time_frames * self.subsample_ratio)
        if subsample_length >= total_time_frames:
            # Handle edge case where ratio is 1.0 or rounds up
            return {"start_indices": torch.zeros(self.n_trans, dtype=torch.long)}

        max_start_index = total_time_frames - subsample_length
        start_indices = torch.randint(
            low=0, high=max_start_index + 1, size=(self.n_trans,), generator=self.rng
        )
        return {"start_indices": start_indices}

    def _transform(
        self, x: torch.Tensor, start_indices: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Performs the temporal subsampling and resizes back to the original length."""
        B, C, total_time_frames, H, W = x.shape
        subsample_length = int(total_time_frames * self.subsample_ratio)

        if subsample_length >= total_time_frames:
            return x.repeat(self.n_trans, 1, 1, 1, 1)

        output_list = []
        for start_idx in start_indices:
            # 1. Take the temporal subsequence
            sub_sequence = x[:, :, start_idx : start_idx + subsample_length, :, :]

            # 2. Flatten all non-time dimensions into one giant "channel" dimension for interpolation.
            # Pattern: (Batch, Channels, Time, Height, Width) -> (Batch, (Channels*Height*Width), Time)
            flat_for_interp = rearrange(sub_sequence, "b c t h w -> b (c h w) t")

            # 3. Interpolate along the time dimension (the last dimension).
            # This is a 1D interpolation.
            resized_flat = torch.nn.functional.interpolate(
                flat_for_interp,
                size=total_time_frames,
                mode="linear",
                align_corners=False,
            )

            # 4. Un-flatten the dimensions back to the original video format.
            # Einops can do this because it knows how (c h w) was constructed.
            # Pattern: (Batch, (Channels*Height*Width), Time) -> (Batch, Channels, Time, Height, Width)
            resized_sequence = rearrange(
                resized_flat, "b (c h w) t -> b c t h w", c=C, h=H, w=W
            )

            output_list.append(resized_sequence)

        return torch.cat(output_list, dim=0)


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


# load params
split_file = config["data"]["split_file"]

output_dir = os.path.join(config["experiment"]["output_dir"], exp_name)
os.makedirs(output_dir, exist_ok=True)

batch_size = config["dataloader"]["batch_size"]
max_subjects = config["dataloader"]["max_subjects"]

mc_loss_weight = config["model"]["losses"]["mc_loss"]["weight"]
ei_loss_weight = config["model"]["losses"]["ei_loss"]["weight"]
use_ei_loss = config["model"]["losses"]["use_ei_loss"]

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


train_dataset = KSpaceSliceDataset(
    root_dir=config["dataloader"]["root_dir"],
    patient_ids=train_patient_ids,
    dataset_key=config["dataloader"]["dataset_key"],
    file_pattern="*.h5",
    slice_idx=config["dataloader"]["slice_idx"],
)

val_dataset = KSpaceSliceDataset(
    root_dir=config["dataloader"]["root_dir"],
    patient_ids=val_patient_ids,
    dataset_key=config["dataloader"]["dataset_key"],
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
    rotate = VideoRotate(n_trans=1, interpolation_mode=InterpolationMode.BILINEAR)
    diffeo = VideoDiffeo(n_trans=1, device=device)
    subsample = SubsampleTime(n_trans=1, subsample_ratio=0.75)

    # NOTE: Not using temporal transforms for now
    ei_loss_fn = EILoss((diffeo | rotate))

print(
    "--- Generating and saving a Zero-Filled (ZF) reconstruction sample before training ---"
)
# Use the validation loader to get a sample without affecting the training loader's state
with torch.no_grad():
    # Get a single batch of validation k-space data
    val_kspace_sample = next(iter(val_loader))
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
    )
print("--- ZF baseline image saved to output directory. Starting training. ---")

# Training Loop
train_mc_losses = []
val_mc_losses = []
train_ei_losses = []
val_ei_losses = []

iteration_count = 0

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
        for (
            measured_kspace
        ) in train_loader_tqdm:  # measured_kspace shape: (B, C, I, S, T)
            iteration_count += 1
            optimizer.zero_grad()

            x_recon = model(
                measured_kspace.to(device), physics
            )  # model output shape: (B, C, T, H, W)

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics)
            running_mc_loss += mc_loss.item()

            if use_ei_loss:
                # x_recon: reconstructed image
                ei_loss = ei_loss_fn(x_recon, physics, model)
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

            if iteration_count % plot_interval == 0:
                with torch.no_grad():
                    plot_reconstruction_sample(
                        x_recon,
                        f"Training Sample - Epoch {epoch}, Iteration {iteration_count}",
                        f"train_sample_epoch_{epoch}_iter_{iteration_count}",
                        output_dir,
                    )

        # Calculate and store average epoch losses
        epoch_train_mc_loss = running_mc_loss / len(train_loader)
        train_mc_losses.append(epoch_train_mc_loss)
        if use_ei_loss:
            epoch_train_ei_loss = running_ei_loss / len(train_loader)
            train_ei_losses.append(epoch_train_ei_loss)
        else:
            # Append 0 if EI loss is not used to keep lists aligned
            train_ei_losses.append(0.0)

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
            for val_kspace_batch in val_loader_tqdm:
                # The model takes the raw k-space and physics operator
                val_x_recon = model(val_kspace_batch.to(device), physics)

                # For MCLoss, compare the physics model's output with the measured k-space.
                val_y_meas = val_kspace_batch
                val_mc_loss = mc_loss_fn(val_y_meas.to(device), val_x_recon, physics)
                val_running_mc_loss += val_mc_loss.item()

                if use_ei_loss:
                    val_ei_loss = ei_loss_fn(val_x_recon, physics, model)
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
                )

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

        # Print epoch summary
        print(
            f"Epoch {epoch}: Training MC Loss: {epoch_train_mc_loss:.6f}, Validation MC Loss: {epoch_val_mc_loss:.6f}"
        )
        if use_ei_loss:
            print(
                f"Epoch {epoch}: Training EI Loss: {epoch_train_ei_loss:.6f}, Validation EI Loss: {epoch_val_ei_loss:.6f}"
            )
