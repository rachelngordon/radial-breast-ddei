from dataloader import KSpaceSliceDataset
from torch.utils.data import DataLoader
from radial import RadialPhysics, RadialDCLayer
from crnn import CRNN, ArtifactRemovalCRNN
import torch
from tqdm import tqdm
from deepinv.loss import MCLoss, EILoss
from einops import rearrange
import json
import matplotlib.pyplot as plt
import os
from torchvision.transforms import InterpolationMode
import deepinv as dinv
from utils import OverTime


# parameters: need to pass as command line arguments later

root_dir = "/ess/scratch/scratch1/rachelgordon/dce-12tf/binned_kspace"
dataset_key = "ktspace"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_epoch = 1
epochs = 25
save_interval = 1
exp_name = 'mc1e11_ei_loss_norm'
output_dir = os.path.join('output', exp_name)
os.makedirs(output_dir, exist_ok=True)
use_ei_loss = True
mc_loss_weight = 1e11

# load data
split_file = 'patient_splits.json'
with open(split_file, "r") as fp:
        splits = json.load(fp)

# NOTE: need to look into why I am only loading 88 training samples and not 192
train_patient_ids = splits["train"]
val_patient_ids   = splits["val"]


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



# define loss functions and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas = (0.9, 0.999), eps=0.000000001, weight_decay=0.0)
mc_loss_fn = MCLoss()


if use_ei_loss == True:
    # define transformations
    rotate = dinv.transform.Rotate(n_trans=1, interpolation_mode=InterpolationMode.BILINEAR)
    tempad = dinv.transform.ShiftTime(n_trans=1)
    diffeo = dinv.transform.CPABDiffeomorphism(n_trans=1, device=device)

    spatial_per_frame = OverTime(rotate | diffeo)   # (B,C,T,H,W) in, same out
    T = tempad | spatial_per_frame                  # final 5-D transform

    ei_loss_fn = dinv.loss.EILoss(T, apply_noise=False)

    # ei_loss_fn = EILoss(tempad | (diffeo | rotate))



# Training Loop
train_mc_losses = []
val_mc_losses = []
train_ei_losses = []
val_ei_losses = []



for epoch in range(start_epoch, epochs+1):

    # Training step
    model.train()

    running_mc_loss = 0.0
    running_ei_loss = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch")

    for kspace_batch in train_loader_tqdm:

        optimizer.zero_grad()


        # expected k-space shape: (B C Ch TotSam T)
        print("model input shape: ", kspace_batch.shape)
        x_recon, scale = model(kspace_batch.to(device), physics, return_scale=True)

        # expected k-space shape: (B C TotSam Ch T)
        y_meas = rearrange(kspace_batch, 'b c i s t -> b c s i t')
        x_recon = rearrange(x_recon, 'b t i h w -> b h w i t')

        # unnormalize output
        unnorm_x_recon = x_recon * scale.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        print("mc loss kspace input shape: ", y_meas.shape)
        print("mc loss image input shape: ", unnorm_x_recon.shape)
        mc_loss = mc_loss_fn(y_meas.to(device), unnorm_x_recon, physics)
        running_mc_loss += mc_loss.item()

        if use_ei_loss == True:

            print("ei loss image input shape: ", unnorm_x_recon.shape)
            ei_loss = ei_loss_fn(unnorm_x_recon, physics, model)
            running_ei_loss += ei_loss.item()

            total_loss = mc_loss * mc_loss_weight + ei_loss
        else:
            total_loss = mc_loss

        total_loss.backward()

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
                plt.savefig(os.path.join(output_dir, f'train_sample_epoch_{epoch}.png'))
                plt.close()


    # average losses 
    epoch_train_mc_loss = running_mc_loss / len(train_loader)
    train_mc_losses.append(epoch_train_mc_loss)

    epoch_train_ei_loss = running_ei_loss / len(train_loader)
    train_ei_losses.append(epoch_train_ei_loss)


    # Validation step
    model.eval()
    val_running_mc_loss = 0.0
    val_running_ei_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs}  Validation", unit="batch", leave=False)

    with torch.no_grad():
        for val_kspace_batch in val_loader_tqdm:

            # expected k-space shape: (B C Ch TotSam T)
            val_x_recon, scale = model(val_kspace_batch.to(device), physics, return_scale=True)

            # expected k-space shape: (B C TotSam Ch T)
            val_y_meas = rearrange(val_kspace_batch, 'b c i s t -> b c s i t')
            val_x_recon = rearrange(val_x_recon, 'b t i h w -> b h w i t')

            # unnormalize output
            unnorm_val_x_recon = val_x_recon * scale.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            val_mc_loss = mc_loss_fn(val_y_meas.to(device), unnorm_val_x_recon, physics)
            val_running_mc_loss += val_mc_loss.item()

            if use_ei_loss == True:
                val_ei_loss = ei_loss_fn(unnorm_val_x_recon, physics, model)
                val_running_ei_loss += val_ei_loss.item()

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
                plt.savefig(os.path.join(output_dir, f'val_sample_epoch_{epoch}.png'))
                plt.close()


                # Plot train + val losses
                plt.plot(train_mc_losses, label='Training MC Loss')
                plt.plot(val_mc_losses, label='Validation MC Loss')
                plt.xlabel('Epoch')
                plt.ylabel('MC Loss')
                plt.legend()
                plt.savefig(os.path.join(output_dir, f'mc_losses.png'))
                plt.close()

                if use_ei_loss == True:
                    # Plot train + val losses
                    plt.plot(train_ei_losses, label='Training EI Loss')
                    plt.plot(val_ei_losses, label='Validation EI Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('EI Loss')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, f'ei_losses.png'))
                    plt.close()

                

    # average losses 
    epoch_val_mc_loss = val_running_mc_loss / len(val_loader)
    val_mc_losses.append(epoch_val_mc_loss)

    print(f'Epoch {epoch}: Training MC Loss: {epoch_train_mc_loss}, Validation MC Loss: {epoch_val_mc_loss}')

    if use_ei_loss == True:
        epoch_val_ei_loss = val_running_ei_loss / len(val_loader)
        val_ei_losses.append(epoch_val_ei_loss)

        print(f'Epoch {epoch}: Training EI Loss: {epoch_train_ei_loss}, Validation EI Loss: {epoch_val_ei_loss}')


