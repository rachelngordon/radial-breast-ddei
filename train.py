from ddei_data_loader import KSpaceSliceDataset
from torch.utils.data import DataLoader
from radial_dclayer_singlecoil import RadialPhysics, RadialDCLayer
from crnn import CRNN, ArtifactRemovalCRNN
import torch
from tqdm import tqdm
from deepinv.loss.mc import MCLoss
from einops import rearrange
import json
import matplotlib.pyplot as plt
import os


# parameters: need to pass as command line arguments later

root_dir = "/ess/scratch/scratch1/rachelgordon/dce-12tf/binned_kspace"
# root_dir = "/ess/scratch/scratch1/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_001_010"
dataset_key = "ktspace"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_epoch = 1
epochs = 25
save_interval = 1

# load data
split_file = 'patient_splits.json'
with open(split_file, "r") as fp:
        splits = json.load(fp)

# NOTE: select only a subset of the data for now
train_patient_ids = splits["train"][:2]
val_patient_ids   = splits["val"][:1]


train_dataset = KSpaceSliceDataset(
        root_dir=root_dir,
        patient_ids=train_patient_ids, 
        dataset_key=dataset_key,
        file_pattern="*.h5"
    )

val_dataset = KSpaceSliceDataset(
        root_dir=root_dir,
        patient_ids=val_patient_ids, 
        dataset_key=dataset_key,
        file_pattern="*.h5"
    )


train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )


val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )


# define physics operators
H, W = 320, 320
N_time, N_spokes, N_samples, N_coils = 12, 24, 640, 1
physics = RadialPhysics(im_size=(H, W, N_time), N_spokes=N_spokes, N_samples=N_samples, N_time=N_time, N_coils=N_coils)


# define model
datalayer = RadialDCLayer(
    im_size=(H, W, N_time)
)

backbone = CRNN(
    num_cascades=5,    # or whatever number of cascades you prefer
    chans=64,          # hidden‐channel size (tune as needed)
    datalayer=datalayer
).to(device)

model = ArtifactRemovalCRNN(backbone_net=backbone).to(device)


# define loss function and optimizer
loss_fn = MCLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas = (0.9, 0.999), eps=0.000000001, weight_decay=0.0)


# Training Loop
train_mc_losses = []
val_mc_losses = []

for epoch in range(start_epoch, epochs+1):

    # Training step
    model.train()

    running_mc_loss = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch")

    for kspace_batch in train_loader_tqdm:

        optimizer.zero_grad()

        # expected k-space shape: (B C Ch TotSam T)
        x_recon = model(kspace_batch.to(device), physics)

        # expected k-space shape: (B C TotSam Ch T)
        y_meas = rearrange(kspace_batch, 'b c i s t -> b c s i t')
        x_recon = rearrange(x_recon, 'b t i h w -> b h w i t')

        loss = loss_fn(y_meas.to(device), x_recon, physics)

        running_mc_loss += loss.item()

        loss.backward()

        optimizer.step()

        if epoch % save_interval == 0:

                x_recon_mag = torch.abs(x_recon[..., 0, :] + 1j * x_recon[..., 1, :])

                batch_idx = 0
                n_timeframes = x_recon_mag.shape[-1]

                # Create a row of subplots, one for each time‐frame
                fig, axes = plt.subplots(
                    nrows=1,
                    ncols=n_timeframes,
                    figsize=(n_timeframes * 3, 6),
                    squeeze=False
                )

                for t in range(n_timeframes):
                    img = x_recon_mag[batch_idx, :, :, t].detach().cpu().numpy()
                    ax = axes[0, t]
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"T = {t}")
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(f'output/train_sample_epoch_{epoch}.png')
                plt.close()


                # Plot train + val losses
                plt.plot(train_mc_losses, label='Training MC Loss')
                plt.plot(val_mc_losses, label='Validation MC Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join('output', f'losses.png'))
                plt.close()

    # average losses 
    epoch_train_mc_loss = running_mc_loss / len(train_loader)
    train_mc_losses.append(epoch_train_mc_loss)


    # Validation step
    model.eval()
    val_running_mc_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs}  Validation", unit="batch", leave=False)

    with torch.no_grad():
        for val_kspace_batch in val_loader_tqdm:

            # expected k-space shape: (B C Ch TotSam T)
            val_x_recon = model(val_kspace_batch.to(device), physics)

            # expected k-space shape: (B C TotSam Ch T)
            val_y_meas = rearrange(val_kspace_batch, 'b c i s t -> b c s i t')
            val_x_recon = rearrange(val_x_recon, 'b t i h w -> b h w i t')

            val_loss = loss_fn(val_y_meas.to(device), val_x_recon, physics)

            val_running_mc_loss += val_loss.item()

            if epoch % save_interval == 0:

                val_x_recon_mag = torch.abs(val_x_recon[..., 0, :] + 1j * val_x_recon[..., 1, :])

                batch_idx = 0
                n_timeframes = val_x_recon_mag.shape[-1]

                # Create a row of subplots, one for each time‐frame
                fig, axes = plt.subplots(
                    nrows=1,
                    ncols=n_timeframes,
                    figsize=(n_timeframes * 3, 6),
                    squeeze=False
                )

                for t in range(n_timeframes):
                    img = val_x_recon_mag[batch_idx, :, :, t].cpu().numpy()
                    ax = axes[0, t]
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"T = {t}")
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(f'output/val_sample_epoch_{epoch}.png')
                plt.close()


                # Plot train + val losses
                plt.plot(train_mc_losses, label='Training MC Loss')
                plt.plot(val_mc_losses, label='Validation MC Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join('output', f'losses.png'))
                plt.close()

                

    # average losses 
    epoch_val_mc_loss = val_running_mc_loss / len(val_loader)
    val_mc_losses.append(epoch_val_mc_loss)


    print(f'Epoch {epoch}: Training MC Loss: {epoch_train_mc_loss}, Validation SSIM Loss: {epoch_val_mc_loss}')


