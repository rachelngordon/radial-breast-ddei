import json
import os

import deepinv as dinv
import matplotlib.pyplot as plt
import torch
from crnn import CRNN, ArtifactRemovalCRNN
from dataloader import KSpaceSliceDataset
from deepinv.loss import EILoss, MCLoss
from radial import DynamicRadialPhysics, RadialDCLayer
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# parameters: need to pass as command line arguments later

root_dir = "/ess/scratch/scratch1/rachelgordon/dce-8tf/binned_kspace"
dataset_key = "ktspace"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_epoch = 1
epochs = 25
save_interval = 1
exp_name = "mc_ei_loss_norm"
output_dir = os.path.join("output", exp_name)
os.makedirs(output_dir, exist_ok=True)
use_ei_loss = True
mc_loss_weight = 1

# load data
split_file = "patient_splits.json"
with open(split_file, "r") as fp:
    splits = json.load(fp)

# NOTE: need to look into why I am only loading 88 training samples and not 192
train_patient_ids = splits["train"]
val_patient_ids = splits["val"]


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


train_dataset = KSpaceSliceDataset(
    root_dir=root_dir,
    patient_ids=train_patient_ids,
    dataset_key=dataset_key,
    file_pattern="*.h5",
)

val_dataset = KSpaceSliceDataset(
    root_dir=root_dir,
    patient_ids=val_patient_ids,
    dataset_key=dataset_key,
    file_pattern="*.h5",
)


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)


val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)


# define physics operators
H, W = 320, 320
N_time, N_samples, N_coils = 8, 640, 1
N_spokes = int(288 / N_time)
physics = DynamicRadialPhysics(
    im_size=(H, W, N_time),
    N_spokes=N_spokes,
    N_samples=N_samples,
    N_time=N_time,
    N_coils=N_coils,
)

datalayer = RadialDCLayer(physics=physics)

backbone = CRNN(num_cascades=2, chans=64, datalayer=datalayer).to(device)

model = ArtifactRemovalCRNN(backbone_net=backbone).to(device)


# define loss functions and optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.00005,
    betas=(0.9, 0.999),
    eps=0.000000001,
    weight_decay=0.0,
)
mc_loss_fn = MCLoss()


if use_ei_loss:
    rotate = VideoRotate(n_trans=1, interpolation_mode=InterpolationMode.BILINEAR)
    diffeo = VideoDiffeo(n_trans=1, device=device)

    tempad = dinv.transform.ShiftTime(n_trans=1)

    ei_loss_fn = EILoss(tempad | (diffeo | rotate))

# Training Loop
train_mc_losses = []
val_mc_losses = []
train_ei_losses = []
val_ei_losses = []

for epoch in range(start_epoch, epochs + 1):
    model.train()
    running_mc_loss = 0.0
    running_ei_loss = 0.0
    train_loader_tqdm = tqdm(
        train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch"
    )
    # measured_kspace shape: (B, C, I, S, T) = 1, 1, 2, 23040, 8
    for measured_kspace in train_loader_tqdm:  # measured_kspace shape: (B, C, I, S, T)
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
            total_loss = mc_loss * mc_loss_weight + ei_loss
        else:
            total_loss = mc_loss

        total_loss.backward()
        optimizer.step()

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

        # Save a sample from the last validation batch of the epoch
        if epoch % save_interval == 0:
            val_x_recon_mag = torch.sqrt(
                val_x_recon[:, 0, ...] ** 2 + val_x_recon[:, 1, ...] ** 2
            )
            batch_idx = 0
            n_timeframes = val_x_recon_mag.shape[1]
            fig, axes = plt.subplots(
                nrows=1,
                ncols=n_timeframes,
                figsize=(n_timeframes * 3, 4),
                squeeze=False,
            )
            for t in range(n_timeframes):
                img = val_x_recon_mag[batch_idx, t, :, :].cpu().numpy()
                ax = axes[0, t]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"T = {t}")
                ax.axis("off")
            fig.suptitle(f"Validation Sample - Epoch {epoch}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"val_sample_epoch_{epoch}.png"))
            plt.close(fig)

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
