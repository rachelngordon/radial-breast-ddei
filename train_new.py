import argparse
import json
import os
import matplotlib.pyplot as plt
import torch
import yaml
from data.dataloader import SliceDataset
from deepinv.transform import Transform
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from losses.transform import VideoRotate, VideoDiffeo, SubsampleTime, MonophasicTimeWarp, TemporalNoise, TimeReverse
from losses.ei import EILoss
from losses.mc import MCLoss
from models.lsfpnet import LSFPNet
from data.radial_lsfp import MCNUFFT
from eval import eval_model
from utils import prep_nufft, log_gradient_stats, plot_enhancement_curve, get_cosine_ei_weight, plot_reconstruction_sample, get_git_commit, save_checkpoint, load_checkpoint, to_torch_complex


    

# Parse command-line arguments
parser = argparse.ArgumentParser()
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

eval_dir = os.path.join(output_dir, "eval_results")
os.makedirs(eval_dir, exist_ok=True)


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
num_layers = config["model"]["num_layers"]

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

eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, N_spokes, 22)
eval_ktraj = eval_ktraj.to(device)
eval_dcomp = eval_dcomp.to(device)
eval_nufft_ob = eval_nufft_ob.to(device)
eval_adjnufft_ob = eval_adjnufft_ob.to(device)





physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)
eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

model = LSFPNet(LayerNo=num_layers, channels=config['model']['channels']).to(device)


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
    model, optimizer, start_epoch, train_curves, val_curves, eval_curves = load_checkpoint(model, optimizer, checkpoint_file)
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
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "none":
        ei_loss_fn = EILoss((rotate | diffeo), model_type=model_type, dcomp=dcomp)
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
    eval_ssims = eval_curves["ssim"]
    eval_psnrs = eval_curves["psnr"]
    eval_mses = eval_curves["mse"]
    eval_dcs = eval_curves["dc"]
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
    eval_ssims = []
    eval_psnrs = []
    eval_mses = []
    eval_dcs = []

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


                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t').to(device)

                csmap = csmap.to(device).to(measured_kspace.dtype)


                # Get the initial ZF recon. This defines our target energy/scale.
                x_init = physics(inv=True, data=measured_kspace, smaps=csmap)

                [L, S, loss_layers_adj_L, loss_layers_adj_S, _] = model(x_init, physics, measured_kspace, csmap)
                # x_recon = torch.abs(L + S)
                x_recon = (L + S)
                x_recon = torch.stack((x_recon.real, x_recon.imag), dim=0).unsqueeze(0)
                # x_recon = x_recon.cpu().data.numpy()

                # aggregate adjoint loss
                loss_constraint_L = torch.square(torch.mean(loss_layers_adj_L[0])) / num_layers
                loss_constraint_S = torch.square(torch.mean(loss_layers_adj_S[0])) / num_layers

                for k in range(num_layers - 1):
                    loss_constraint_S += torch.square(torch.mean(loss_layers_adj_S[k + 1])) / num_layers
                    loss_constraint_L += torch.square(torch.mean(loss_layers_adj_L[k + 1])) / num_layers

                gamma = torch.Tensor([0.01]).to(device)
                total_adj_loss = torch.mul(gamma, loss_constraint_L + loss_constraint_S)
                initial_train_adj_loss += total_adj_loss.item()

                mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
                initial_train_mc_loss += mc_loss.item()


                if use_ei_loss:
                    
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


                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t').to(device)

                csmap = csmap.to(device).to(measured_kspace.dtype)

                # Get the initial ZF recon. This defines our target energy/scale.
                x_init = physics(inv=True, data=measured_kspace, smaps=csmap)

                [L, S, loss_layers_adj_L, loss_layers_adj_S, _] = model(x_init, physics, measured_kspace, csmap)
                # x_recon = torch.abs(L + S)
                x_recon = (L + S)
                x_recon = torch.stack((x_recon.real, x_recon.imag), dim=0).unsqueeze(0)
                # x_recon = x_recon.cpu().data.numpy()

                # aggregate adjoint loss
                loss_constraint_L = torch.square(torch.mean(loss_layers_adj_L[0])) / num_layers
                loss_constraint_S = torch.square(torch.mean(loss_layers_adj_S[0])) / num_layers

                for k in range(num_layers - 1):
                    loss_constraint_S += torch.square(torch.mean(loss_layers_adj_S[k + 1])) / num_layers
                    loss_constraint_L += torch.square(torch.mean(loss_layers_adj_L[k + 1])) / num_layers

                gamma = torch.Tensor([0.01]).to(device)
                total_adj_loss = torch.mul(gamma, loss_constraint_L + loss_constraint_S)

                initial_val_adj_loss += total_adj_loss.item()



                mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
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

        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch"
        )
        # measured_kspace shape: (B, C, I, S, T) = 1, 1, 2, 23040, 8
        for measured_kspace, csmap, grasp_img in train_loader_tqdm:  # measured_kspace shape: (B, C, I, S, T)

            iteration_count += 1
            optimizer.zero_grad()


            measured_kspace = to_torch_complex(measured_kspace).squeeze()
            measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t').to(device)

            csmap = csmap.to(device).to(measured_kspace.dtype)


            # Get the initial ZF recon. This defines our target energy/scale.
            x_init = physics(inv=True, data=measured_kspace, smaps=csmap)

            [L, S, loss_layers_adj_L, loss_layers_adj_S, handles] = model(x_init, physics, measured_kspace, csmap)
            # x_recon = torch.abs(L + S)
            x_recon = (L + S)
            x_recon = torch.stack((x_recon.real, x_recon.imag), dim=0).unsqueeze(0)
            # x_recon = x_recon.cpu().data.numpy()

            # aggregate adjoint loss
            loss_constraint_L = torch.square(torch.mean(loss_layers_adj_L[0])) / num_layers
            loss_constraint_S = torch.square(torch.mean(loss_layers_adj_S[0])) / num_layers

            for k in range(num_layers - 1):
                loss_constraint_S += torch.square(torch.mean(loss_layers_adj_S[k + 1])) / num_layers
                loss_constraint_L += torch.square(torch.mean(loss_layers_adj_L[k + 1])) / num_layers

            gamma = torch.Tensor([0.01]).to(device)
            total_adj_loss = torch.mul(gamma, loss_constraint_L + loss_constraint_S)

            running_adj_loss += total_adj_loss.item()


            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
            running_mc_loss += mc_loss.item()


            if use_ei_loss and epoch > warmup:

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


                total_loss = mc_loss * mc_loss_weight + ei_loss * ei_loss_weight + lambda_adj * total_adj_loss

                train_loader_tqdm.set_postfix(
                    mc_loss=mc_loss.item(), ei_loss=ei_loss.item()
                )

            else:
                total_loss = mc_loss * mc_loss_weight + lambda_adj * total_adj_loss
                train_loader_tqdm.set_postfix(mc_loss=mc_loss.item())

            if torch.isnan(total_loss):
                print(
                    "!!! ERROR: total_loss is NaN before backward pass. Aborting. !!!"
                )
                raise RuntimeError("total_loss is NaN")

            # scaler.scale(total_loss).backward()
            total_loss.backward()

            for handle in handles:
                handle.remove()

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


                val_kspace_batch = to_torch_complex(val_kspace_batch).squeeze()
                val_kspace_batch = rearrange(val_kspace_batch, 't co sp sam -> co (sp sam) t').to(device)

                val_csmap = val_csmap.to(device).to(val_kspace_batch.dtype)


                # Get the initial ZF recon. This defines our target energy/scale.
                x_init = physics(inv=True, data=val_kspace_batch, smaps=val_csmap)

                [L, S, loss_layers_adj_L, loss_layers_adj_S, _] = model(x_init, physics, val_kspace_batch, val_csmap)
                # x_recon = torch.abs(L + S)
                val_x_recon = (L + S)
                val_x_recon = torch.stack((val_x_recon.real, val_x_recon.imag), dim=0).unsqueeze(0)
                # val_x_recon = val_x_recon.cpu().data.numpy()

                # aggregate adjoint loss
                loss_constraint_L = torch.square(torch.mean(loss_layers_adj_L[0])) / num_layers
                loss_constraint_S = torch.square(torch.mean(loss_layers_adj_S[0])) / num_layers

                for k in range(num_layers - 1):
                    loss_constraint_S += torch.square(torch.mean(loss_layers_adj_S[k + 1])) / num_layers
                    loss_constraint_L += torch.square(torch.mean(loss_layers_adj_L[k + 1])) / num_layers

                gamma = torch.Tensor([0.01]).to(device)
                val_adj_loss = torch.mul(gamma, loss_constraint_L + loss_constraint_S)

                val_running_adj_loss += val_adj_loss.item()



                # For MCLoss, compare the physics model's output with the measured k-space.
                val_mc_loss = mc_loss_fn(val_kspace_batch.to(device), val_x_recon, physics, val_csmap)
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
            eval_curves = dict(
                eval_ssims=eval_ssims,
                eval_psnrs=eval_psnrs,
                eval_mses=eval_mses,
                eval_dcs=eval_dcs
            )
            model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, train_curves, val_curves, eval_curves, model_save_path)
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

            # =====================================================================
            # --- EVALUATION ON SIMULATED DATASET ---
            # =====================================================================
            if config['evaluation']['num_samples'] > 0:

                with torch.no_grad():
                    physics_objects = {
                        'physics': eval_physics,
                        'dcomp': eval_dcomp if model_type == "LSFPNet" else None,
                    }

                    if epoch != epochs:
                        eval_ssim, eval_psnr, eval_mse, eval_dc = eval_model(model, device, config, eval_dir, physics_objects, epoch)
                    else:
                        eval_ssim, eval_psnr, eval_mse, eval_dc = eval_model(model, device, config, eval_dir, physics_objects, epoch)#, temporal_eval=True)

                    eval_ssims.append(eval_ssim)
                    eval_psnrs.append(eval_psnr)
                    eval_mses.append(eval_mse)
                    eval_dcs.append(eval_dc)


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

            # Plot Evaluation Stats
            plt.figure()
            plt.plot(eval_ssims)
            plt.xlabel("Epoch")
            plt.ylabel("SSIM")
            plt.title("Evaluation SSIM")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(eval_dir, "eval_ssims.png"))
            plt.close()

            plt.figure()
            plt.plot(eval_psnrs)
            plt.xlabel("Epoch")
            plt.ylabel("PSNR")
            plt.title("Evaluation PSNR")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(eval_dir, "eval_psnrs.png"))
            plt.close()

            plt.figure()
            plt.plot(eval_mses)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title("Evaluation MSE")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(eval_dir, "eval_mses.png"))
            plt.close()

            plt.figure()
            plt.plot(eval_dcs)
            plt.xlabel("Epoch")
            plt.ylabel("DC")
            plt.title("Evaluation DC")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(eval_dir, "eval_dcs.png"))
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
eval_curves = dict(
    eval_ssims=eval_ssims,
    eval_psnrs=eval_psnrs,
    eval_mses=eval_mses,
    eval_dcs=eval_dcs,
)
model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
save_checkpoint(model, optimizer, scheduler, epochs + 1, train_curves, val_curves, eval_curves, model_save_path)
print(f'Model saved to {model_save_path}')



