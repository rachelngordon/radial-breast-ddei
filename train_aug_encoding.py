import argparse
import json
import os
import subprocess
import matplotlib.pyplot as plt
import torch
import yaml
from crnn import CRNN, ArtifactRemovalCRNN
from dataloader import SliceDataset, SimulatedDataset, SimulatedSPFDataset, SliceDatasetAug
from deepinv.transform import Transform
from einops import rearrange
from radial import RadialDCLayer, to_torch_complex, MCNUFFT_CRNN
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transform import VideoRotate, VideoDiffeo, SubsampleTime, MonophasicTimeWarp, TemporalNoise, TimeReverse
from ei import EILoss
from mc import MCLoss
from lsfpnet_encode_acc import LSFPNet, ArtifactRemovalLSFPNet
from radial_lsfp import MCNUFFT
from utils import prep_nufft, log_gradient_stats, plot_enhancement_curve, get_cosine_ei_weight, plot_reconstruction_sample, get_git_commit, save_checkpoint, load_checkpoint, to_torch_complex
from eval import eval_grasp, eval_sample
import csv
import math

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

initial_lambdas = {'lambda_L': config['model']['lambda_L'], 
                   'lambda_S': config['model']['lambda_S'], 
                   'lambda_spatial_L': config['model']['lambda_spatial_L'],
                   'lambda_spatial_S': config['model']['lambda_spatial_S'],
                   'gamma': config['model']['gamma'],
                   'lambda_step': config['model']['lambda_step']}

mc_loss_weight = config["model"]["losses"]["mc_loss"]["weight"]
adj_loss_weight = config["model"]["losses"]["adj_loss"]["weight"]

use_ei_loss = config["model"]["losses"]["use_ei_loss"]
target_weight = config["model"]["losses"]["ei_loss"]["weight"]
warmup = config["model"]["losses"]["ei_loss"]["warmup"]
duration = config["model"]["losses"]["ei_loss"]["duration"]

save_interval = config["training"]["save_interval"]
plot_interval = config["training"]["plot_interval"]
device = torch.device(config["training"]["device"])
start_epoch = 1

N_full = config['data']['height'] * math.pi / 2

model_type = config["model"]["name"]

H, W = config["data"]["height"], config["data"]["width"]
N_time, N_samples, N_coils, N_time_eval = (
    config["data"]["timeframes"],
    config["data"]["samples"],
    config["data"]["coils"],
    config["data"]["eval_timeframes"]
)
N_spokes = int(config["data"]["total_spokes"] / N_time)

# os.makedirs(os.path.join(output_dir, 'enhancement_curves'), exist_ok=True)

# load data
with open(split_file, "r") as fp:
    splits = json.load(fp)



if max_subjects < 300:
    max_train = int(max_subjects * (1 - config["data"]["val_split_ratio"]))

    train_patient_ids = splits["train"][:max_train]
    

else:
    train_patient_ids = splits["train"]

val_patient_ids = splits["val"]
val_dro_patient_ids = splits["val_dro"]


train_dataset = SliceDatasetAug(
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

val_dro_dataset = SimulatedDataset(
    root_dir=config["evaluation"]["simulated_dataset_path"], 
    model_type=model_type, 
    patient_ids=val_dro_patient_ids)


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

val_dro_loader = DataLoader(
    val_dro_dataset,
    batch_size=config["dataloader"]["batch_size"],
    shuffle=config["dataloader"]["shuffle"],
    num_workers=config["dataloader"]["num_workers"],
)


eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time_eval)
eval_ktraj = eval_ktraj.to(device)
eval_dcomp = eval_dcomp.to(device)
eval_nufft_ob = eval_nufft_ob.to(device)
eval_adjnufft_ob = eval_adjnufft_ob.to(device)





eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

lsfp_backbone = LSFPNet(LayerNo=config["model"]["num_layers"], lambdas=initial_lambdas, channels=config['model']['channels'])
model = ArtifactRemovalLSFPNet(lsfp_backbone).to(device)



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["model"]["optimizer"]["lr"],
    betas=(config["model"]["optimizer"]["b1"], config["model"]["optimizer"]["b2"]),
    eps=config["model"]["optimizer"]["eps"],
    weight_decay=config["model"]["optimizer"]["weight_decay"],
)

# Load the checkpoint to resume training
if args.from_checkpoint == True:
    checkpoint_file = f'output/{exp_name}/{exp_name}_model.pth'
    model, optimizer, start_epoch, train_curves, val_curves, eval_curves = load_checkpoint(model, optimizer, checkpoint_file)
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
        ei_loss_fn = EILoss(subsample | (diffeo | rotate), model_type=model_type)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "monophasic":
        ei_loss_fn = EILoss(monophasic_warp | (diffeo | rotate), model_type=model_type)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "noise":
        ei_loss_fn = EILoss(temp_noise | (diffeo | rotate), model_type=model_type)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "reverse":
        ei_loss_fn = EILoss(time_reverse | (diffeo | rotate), model_type=model_type)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "all":
        ei_loss_fn = EILoss((subsample | monophasic_warp | temp_noise | time_reverse) | (diffeo | rotate), model_type=model_type)
    else:
        raise(ValueError, "Unsupported Temporal Transform.")


# print(
#     "--- Generating and saving a Zero-Filled (ZF) reconstruction sample before training ---"
# )
# # Use the validation loader to get a sample without affecting the training loader's state
# with torch.no_grad():
#     # Get a single batch of validation k-space data
#     val_kspace_sample, csmap, grasp_img = next(iter(val_loader))
#     val_kspace_sample = val_kspace_sample.to(device)

#     # Perform the simplest reconstruction: A_adjoint(y)
#     # This is the "zero-filled" image (or more accurately, the gridded image)
#     if model_type == "CRNN":
#         x_zf = physics.A_adjoint(val_kspace_sample, csmap)
#     elif model_type == "LSFPNet":
#         val_kspace_sample = to_torch_complex(val_kspace_sample).squeeze()
#         val_kspace_sample = rearrange(val_kspace_sample, 't co sp sam -> co (sp sam) t')
        
#         x_zf = physics(inv=True, data=val_kspace_sample, smaps=csmap.to(device))

#         # compute magnitude and add batch dimx
#         x_zf = torch.abs(x_zf).unsqueeze(0)

#     # Plot and save the image using your existing function
#     plot_reconstruction_sample(
#         x_zf,
#         "Zero-Filled (ZF) Reconstruction (Before Training)",
#         "zf_reconstruction_baseline",
#         output_dir,
#         grasp_img
#     )
# print("--- ZF baseline image saved to output directory. Starting training. ---")

if args.from_checkpoint:
    train_mc_losses = train_curves["train_mc_losses"]
    val_mc_losses = val_curves["val_mc_losses"]
    train_ei_losses = train_curves["train_ei_losses"]
    val_ei_losses = val_curves["val_ei_losses"]
    train_adj_losses = train_curves["train_adj_losses"]
    val_adj_losses = val_curves["val_adj_losses"]
    weighted_train_mc_losses = train_curves["weighted_train_mc_losses"]
    weighted_train_ei_losses = train_curves["weighted_train_ei_losses"]
    weighted_train_adj_losses = train_curves["weighted_train_adj_losses"]
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
    eval_ssims = []
    eval_psnrs = []
    eval_mses = []
    eval_dcs = []


grasp_ssims = []
grasp_psnrs = []
grasp_mses = []
grasp_dcs = []

iteration_count = 0

# scaler = GradScaler()


# Step 0: Evaluate the untrained model
if args.from_checkpoint == False and config['debugging']['calc_step_0'] == True:
    model.eval()
    initial_train_mc_loss = 0.0
    initial_val_mc_loss = 0.0
    initial_train_ei_loss = 0.0
    initial_val_ei_loss = 0.0
    initial_train_adj_loss = 0.0
    initial_val_adj_loss = 0.0


    with torch.no_grad():
        # Evaluate on training data
        for measured_kspace, csmap, grasp_img, N_samples, N_spokes, N_time in tqdm(train_loader, desc="Step 0 Training Evaluation"):

            acceleration_factor = round(N_full / int(N_spokes), 1)
            print("acceleration: ", acceleration_factor)

            ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
            ktraj = ktraj.to(device)
            dcomp = dcomp.to(device)
            nufft_ob = nufft_ob.to(device)
            adjnufft_ob = adjnufft_ob.to(device)

            physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)

            # with autocast(config["training"]["device"]):

            if model_type == "LSFPNet":

                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

                csmap = csmap.to(device).to(measured_kspace.dtype)

                x_recon, adj_loss = model(
                    measured_kspace.to(device), physics, csmap, acceleration_factor, norm=config['model']['norm']
                )

                initial_train_adj_loss += adj_loss.item()

            else:

                x_recon = model(
                    measured_kspace.to(device), physics, csmap
                )  # model output shape: (B, C, T, H, W)

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
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
        for measured_kspace, csmap, ground_truth, grasp_img, *_ in tqdm(val_dro_loader, desc="Step 0 Validation Evaluation"):
        #for measured_kspace, csmap, grasp_img in tqdm(val_dro_loader, desc="Step 0 Validation Evaluation"):

            # with autocast(config["training"]["device"]):
            N_spokes = eval_ktraj.shape[1] / config['data']['samples']
            acceleration_factor = round(N_full / N_spokes, 1)


            if model_type == "LSFPNet":

                measured_kspace = measured_kspace.to(device)
                csmap = csmap.to(device)

                measured_kspace = measured_kspace.squeeze(0).to(device) # Remove batch dim
                csmap = csmap.squeeze(0).to(device)   # Remove batch dim


                # measured_kspace = to_torch_complex(measured_kspace).squeeze()
                # measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

                # csmap = csmap.to(device).to(measured_kspace.dtype)
                # al_kspace_batch:  torch.Size([16, 23040, 22])                                                                             | 0/15 [00:00<?, ?it/s]
                # x_temp:  torch.Size([1, 1, 16, 320, 320])
                # measured_kspace:  torch.Size([16, 23040, 22])

                x_recon, adj_loss = model(
                    measured_kspace.to(device), eval_physics, csmap, acceleration_factor, norm=config['model']['norm']
                )
                initial_val_adj_loss += adj_loss.item()
            
            else:

                x_recon = model(
                    measured_kspace.to(device), eval_physics, csmap
                )  # model output shape: (B, C, T, H, W)

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, eval_physics, csmap)
            initial_val_mc_loss += mc_loss.item()

            if use_ei_loss:
                # x_recon: reconstructed image
                ei_loss, t_img = ei_loss_fn(
                    x_recon, eval_physics, model, csmap
                )

                initial_val_ei_loss += ei_loss.item()

            ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)
            grasp_recon = grasp_img.to(device) # Shape: (1, 2, H, T, W)

            ssim_grasp, psnr_grasp, mse_grasp, dc_grasp = eval_grasp(measured_kspace, csmap, ground_truth, grasp_recon, eval_physics, device)
            grasp_ssims.append(ssim_grasp)
            grasp_psnrs.append(psnr_grasp)
            grasp_mses.append(mse_grasp)
            grasp_dcs.append(dc_grasp)

        step0_val_mc_loss = initial_val_mc_loss / len(val_dro_loader)
        val_mc_losses.append(step0_val_mc_loss)

        step0_val_ei_loss = initial_val_ei_loss / len(val_dro_loader)
        val_ei_losses.append(step0_val_ei_loss)

        step0_val_adj_loss = initial_val_adj_loss / len(val_dro_loader)
        val_adj_losses.append(step0_val_adj_loss)

    print(f"Step 0 Train Losses: MC: {step0_train_mc_loss}, EI: {step0_train_ei_loss}, Adj: {step0_train_adj_loss}")
    print(f"Step 0 Val Losses: MC: {step0_val_mc_loss}, EI: {step0_val_ei_loss}, Adj: {step0_val_adj_loss}")

# Training Loop
if (epochs + 1) == start_epoch:
    raise(ValueError("Full training epochs already complete."))

else: 

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_mc_loss = 0.0
        running_ei_loss = 0.0
        running_adj_loss = 0.0
        epoch_eval_ssims = []
        epoch_eval_psnrs = []
        epoch_eval_mses = []
        epoch_eval_dcs = []

        # turn on anomaly detection for debugging but slows down training
        # with torch.autograd.set_detect_anomaly(False):
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch"
        )
        for measured_kspace, csmap, grasp_img, N_samples, N_spokes, N_time in train_loader_tqdm:  # measured_kspace shape: (B, C, I, S, T)
            

            acceleration_factor = round(N_full / int(N_spokes), 1)
            print("acceleration: ", acceleration_factor)



            ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
            ktraj = ktraj.to(device)
            dcomp = dcomp.to(device)
            nufft_ob = nufft_ob.to(device)
            adjnufft_ob = adjnufft_ob.to(device)

            physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)

            # with autocast(config["training"]["device"]):

            iteration_count += 1
            optimizer.zero_grad()

            if model_type == "LSFPNet":

                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

                csmap = csmap.to(device).to(measured_kspace.dtype)

                x_recon, adj_loss = model(
                    measured_kspace.to(device), physics, csmap, acceleration_factor, norm=config['model']['norm']
                )
                running_adj_loss += adj_loss.item()

            else:
                x_recon = model(
                    measured_kspace.to(device), physics, csmap
                )  # model output shape: (B, C, T, H, W)

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
            running_mc_loss += mc_loss.item()

            if use_ei_loss:
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

                # print(f"Epoch {epoch:2d}: EI Weight = {ei_loss_weight:.8f}")
                
                running_ei_loss += ei_loss.item()
                total_loss = mc_loss * mc_loss_weight + ei_loss * ei_loss_weight + torch.mul(adj_loss_weight, adj_loss)
                # total_loss = ei_loss * ei_loss_weight + torch.mul(adj_loss_weight, adj_loss)
                train_loader_tqdm.set_postfix(
                    mc_loss=mc_loss.item(), ei_loss=ei_loss.item()
                )

            else:
                total_loss = mc_loss * mc_loss_weight + torch.mul(adj_loss_weight, adj_loss)
                # total_loss = torch.mul(adj_loss_weight, adj_loss)
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

        # NOTE: commented out plotting until figured out how to handle variable numbers of timeframes
        # if epoch % save_interval == 0:

        #     # grasp_img = torch.rot90(grasp_img, dims=[-2, -1])
        #     plot_reconstruction_sample(
        #         x_recon,
        #         f"Training Sample - Epoch {epoch}",
        #         f"train_sample_epoch_{epoch}",
        #         output_dir,
        #         grasp_img
        #     )

        #     x_recon_reshaped = rearrange(x_recon, 'b c h w t -> b c t h w')

        #     plot_enhancement_curve(
        #         x_recon_reshaped,
        #         output_filename = os.path.join(output_dir, 'enhancement_curves', f'train_sample_enhancement_curve_epoch_{epoch}.png'))
            
        #     plot_enhancement_curve(
        #         grasp_img,
        #         output_filename = os.path.join(output_dir, 'enhancement_curves', f'grasp_sample_enhancement_curve_epoch_{epoch}.png'))

        #     if use_ei_loss:

        #         x_recon_flip = torch.flip(x_recon, dims=[2])

        #         plot_reconstruction_sample(
        #             t_img,
        #             f"Transformed Train Sample - Epoch {epoch}",
        #             f"transforms/transform_train_sample_epoch_{epoch}",
        #             output_dir,
        #             x_recon_flip,
        #             transform=True
        #         )

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
        if model_type == "LSFPNet":
            epoch_train_adj_loss = running_adj_loss / len(train_loader)
            train_adj_losses.append(epoch_train_adj_loss)
            weighted_train_adj_losses.append(epoch_train_adj_loss*adj_loss_weight)
        else:
            train_adj_losses.append(0.0)
            weighted_train_adj_losses.append(0.0)

        # --- Validation Loop ---
        model.eval()
        val_running_mc_loss = 0.0
        val_running_ei_loss = 0.0
        val_running_adj_loss = 0.0
        val_loader_tqdm = tqdm(
            val_dro_loader,
            desc=f"Epoch {epoch}/{epochs}  Validation",
            unit="batch",
            leave=False,
        )
        with torch.no_grad():
            for val_kspace_batch, val_csmap, val_ground_truth, val_grasp_img, val_mask in tqdm(val_dro_loader):
            # for val_kspace_batch, val_csmap, val_grasp_img in val_loader_tqdm:

                # with autocast(config["training"]["device"]):

                N_spokes = eval_ktraj.shape[1] / config['data']['samples']
                acceleration_factor = round(N_full / N_spokes, 1)

                if model_type == "LSFPNet":

                    val_kspace_batch = val_kspace_batch.squeeze(0).to(device) # Remove batch dim
                    val_csmap = val_csmap.squeeze(0).to(device)   # Remove batch dim
                    val_ground_truth = val_ground_truth.to(device) # Shape: (1, 2, T, H, W)

                    val_grasp_img = torch.flip(val_grasp_img, dims=[-3])
                    val_grasp_img = torch.rot90(val_grasp_img, k=3, dims=[-3,-1])
                    val_grasp_img_tensor = val_grasp_img.to(device)

                    # val_kspace_batch = to_torch_complex(val_kspace_batch).squeeze()
                    # val_kspace_batch = rearrange(val_kspace_batch, 't co sp sam -> co (sp sam) t')

                    # val_csmap = val_csmap.to(device).to(val_kspace_batch.dtype)

                    val_x_recon, val_adj_loss = model(
                        val_kspace_batch.to(device), eval_physics, val_csmap, acceleration_factor, norm=config['model']['norm']
                    )
                    val_running_adj_loss += val_adj_loss.item()

                else:
                    # The model takes the raw k-space and physics operator
                    val_x_recon = model(val_kspace_batch.to(device), eval_physics, val_csmap)

                # For MCLoss, compare the physics model's output with the measured k-space.
                val_mc_loss = mc_loss_fn(val_kspace_batch.to(device), val_x_recon, eval_physics, val_csmap)
                val_running_mc_loss += val_mc_loss.item()

                if use_ei_loss:
                    val_ei_loss, val_t_img = ei_loss_fn(
                        val_x_recon, eval_physics, model, val_csmap
                    )
                    val_running_ei_loss += val_ei_loss.item()
                    val_loader_tqdm.set_postfix(
                        val_mc_loss=val_mc_loss.item(), val_ei_loss=val_ei_loss.item()
                    )
                else:
                    val_loader_tqdm.set_postfix(val_mc_loss=val_mc_loss.item())


                ## Evaluation
                ssim, psnr, mse, dc = eval_sample(val_kspace_batch, val_csmap, val_ground_truth, val_x_recon, eval_physics, val_mask, val_grasp_img_tensor, eval_dir, f'epoch{epoch}', device)
                epoch_eval_ssims.append(ssim)
                epoch_eval_psnrs.append(psnr)
                epoch_eval_mses.append(mse)
                epoch_eval_dcs.append(dc)



        # Calculate and store average validation evaluation metrics
        epoch_eval_ssim = np.mean(epoch_eval_ssims)
        epoch_eval_psnr = np.mean(epoch_eval_psnrs)
        epoch_eval_mse = np.mean(epoch_eval_mses)
        epoch_eval_dc = np.mean(epoch_eval_dcs)

        eval_ssims.append(epoch_eval_ssim)
        eval_psnrs.append(epoch_eval_psnr)
        eval_mses.append(epoch_eval_mse)
        eval_dcs.append(epoch_eval_dc)    
        
        # save a sample from the last validation batch of the epoch
        if epoch % save_interval == 0:

            val_x_recon_rot = torch.rot90(val_x_recon, k=2, dims=[-3,-2])
            
            plot_reconstruction_sample(
                val_x_recon_rot,
                f"Validation Sample - Epoch {epoch}",
                f"val_sample_epoch_{epoch}",
                output_dir,
                val_grasp_img
            )

            val_x_recon_reshaped = rearrange(val_x_recon, 'b c h w t -> b c t h w')

            # plot_enhancement_curve(
            #     val_x_recon_reshaped,
            #     output_filename = os.path.join(output_dir, 'enhancement_curves', f'val_sample_enhancement_curve_epoch_{epoch}.png'))
            
            # plot_enhancement_curve(
            #     val_grasp_img,
            #     output_filename = os.path.join(output_dir, 'enhancement_curves', f'val_grasp_sample_enhancement_curve_epoch_{epoch}.png'))

            if use_ei_loss:
                plot_reconstruction_sample(
                    val_t_img,
                    f"Transformed Validation Sample - Epoch {epoch}",
                    f"transforms/transform_val_sample_epoch_{epoch}",
                    output_dir,
                    val_x_recon,
                    transform=True
                )


        # Calculate and store average validation losses
        epoch_val_mc_loss = val_running_mc_loss / len(val_dro_loader)
        val_mc_losses.append(epoch_val_mc_loss)
        if use_ei_loss:
            epoch_val_ei_loss = val_running_ei_loss / len(val_dro_loader)
            val_ei_losses.append(epoch_val_ei_loss)
        else:
            val_ei_losses.append(0.0)
        if model_type == "LSFPNet":
            epoch_val_adj_loss = val_running_adj_loss / len(val_dro_loader)
            val_adj_losses.append(epoch_val_adj_loss)
        else:
            val_adj_losses.append(0.0)




        # --- Plotting and Logging ---
        if epoch % save_interval == 0:

            # =====================================================================
            # --- EVALUATION ON SIMULATED DATASET ---
            # =====================================================================
            # if config['evaluation']['num_samples'] > 0:

            # with torch.no_grad():
            #     physics_objects = {
            #         'physics': eval_physics,
            #         'dcomp': eval_dcomp if model_type == "LSFPNet" else None,
            #     }

            #     # if epoch != epochs:
            #     #     eval_ssim, eval_psnr, eval_mse, eval_dc = eval_model(val_dro_dataset, model, device, config, eval_dir, physics_objects, epoch)
            #     # else:
            #     eval_ssim, eval_psnr, eval_mse, eval_dc = eval_model(val_dro_dataset, model, device, config, eval_dir, physics_objects, epoch)#, temporal_eval=True)

            #     eval_ssims.append(eval_ssim)
            #     eval_psnrs.append(eval_psnr)
            #     eval_mses.append(eval_mse)
            #     eval_dcs.append(eval_dc)


            # Save the model checkpoint
            train_curves = dict(
                train_mc_losses=train_mc_losses,
                train_ei_losses=train_ei_losses,
                weighted_train_mc_losses=weighted_train_mc_losses,
                weighted_train_ei_losses=weighted_train_ei_losses,
            )
            val_curves = dict(
                val_mc_losses=val_mc_losses,
                val_ei_losses=val_ei_losses,
            )
            eval_curves = dict(
                eval_ssims=eval_ssims,
                eval_psnrs=eval_psnrs,
                eval_mses=eval_mses,
                eval_dcs=eval_dcs
            )
            model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
            save_checkpoint(model, optimizer, epoch + 1, train_curves, val_curves, eval_curves, model_save_path)
            print(f'Model saved to {model_save_path}')


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

            # Plot Train and Val Losses Individually
            plt.figure()
            plt.plot(train_mc_losses)
            plt.xlabel("Epoch")
            plt.ylabel("MC Loss")
            plt.title("Training Measurement Consistency Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "train_mc_losses.png"))
            plt.close()

            plt.figure()
            plt.plot(val_mc_losses)
            plt.xlabel("Epoch")
            plt.ylabel("MC Loss")
            plt.title("Validation Measurement Consistency Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "val_mc_losses.png"))
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

                plt.figure()
                plt.plot(train_ei_losses)
                plt.xlabel("Epoch")
                plt.ylabel("EI Loss")
                plt.title("Training Equivariant Imaging Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "train_ei_losses.png"))
                plt.close()

                plt.figure()
                plt.plot(val_ei_losses)
                plt.xlabel("Epoch")
                plt.ylabel("EI Loss")
                plt.title("Validation Equivariant Imaging Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "val_ei_losses.png"))
                plt.close()

            if model_type == "LSFPNet":
                plt.figure()
                plt.plot(train_adj_losses, label="Training Adjoint Loss")
                plt.plot(val_adj_losses, label="Validation Adjoint Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Adjoint Loss")
                plt.title("CNN Adjoint Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "adj_losses.png"))
                plt.close()

                plt.figure()
                plt.plot(train_adj_losses)
                plt.xlabel("Epoch")
                plt.ylabel("Adjoint Loss")
                plt.title("Training CNN Adjoint Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "train_adj_losses.png"))
                plt.close()

                plt.figure()
                plt.plot(val_adj_losses)
                plt.xlabel("Epoch")
                plt.ylabel("Adjoint Loss")
                plt.title("Validation CNN Adjoint Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "val_adj_losses.png"))
                plt.close()



            # Plot Weighted Losses
            plt.figure()
            plt.plot(weighted_train_mc_losses, label="MC Loss")
            plt.plot(weighted_train_ei_losses, label="EI Loss")
            plt.plot(weighted_train_adj_losses, label="Adjoint Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Weighted Training Losses")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "weighted_losses.png"))
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
        if use_ei_loss:
            print(
                f"Epoch {epoch}: Training EI Loss: {epoch_train_ei_loss:.6f}, Validation EI Loss: {epoch_val_ei_loss:.6f}"
            )

        if model_type == "LSFPNet":
            print(
                f"Epoch {epoch}: Training Adj Loss: {epoch_train_adj_loss:.6f}, Validation Adj Loss: {epoch_val_adj_loss:.6f}"
            )
        print(f"--- Evaluation Metrics: Epoch {epoch} ---")
        print(f"Recon SSIM: {epoch_eval_ssim:.4f} ± {np.std(epoch_eval_ssims):.4f}")
        print(f"Recon PSNR: {epoch_eval_psnr:.4f} ± {np.std(epoch_eval_psnrs):.4f}")
        print(f"Recon MSE: {epoch_eval_mse:.4f} ± {np.std(epoch_eval_mses):.4f}")
        print(f"Recon DC: {epoch_eval_dc:.4f} ± {np.std(epoch_eval_dcs):.4f}")
        print(f"GRASP SSIM: {np.mean(grasp_ssims):.4f} ± {np.std(grasp_ssims):.4f}")
        print(f"GRASP PSNR: {np.mean(grasp_psnrs):.4f} ± {np.std(grasp_psnrs):.4f}")
        print(f"GRASP MSE: {np.mean(grasp_mses):.4f} ± {np.std(grasp_mses):.4f}")
        print(f"GRASP DC: {np.mean(grasp_dcs):.6f} ± {np.std(grasp_dcs):.4f}")


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
save_checkpoint(model, optimizer, epochs + 1, train_curves, val_curves, eval_curves, model_save_path)
print(f'Model saved to {model_save_path}')


# save final evaluation metrics
metrics_path = os.path.join(eval_dir, "eval_metrics.csv")

with open(metrics_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Recon', 'SSIM', 'PSNR', 'MSE', 'DC'])
    writer.writerow(['DL', f'{epoch_eval_ssim:.4f} ± {np.std(epoch_eval_ssims):.4f}', f'{epoch_eval_psnr:.4f} ± {np.std(epoch_eval_psnrs):.4f}', f'{epoch_eval_mse:.4f} ± {np.std(epoch_eval_mses):.4f}', f'{epoch_eval_dc:.4f} ± {np.std(epoch_eval_dcs):.4f}'])
    writer.writerow(['GRASP', f'{np.mean(grasp_ssims):.4f} ± {np.std(grasp_ssims):.4f}', f'{np.mean(grasp_psnrs):.4f} ± {np.std(grasp_psnrs):.4f}', f'{np.mean(grasp_mses):.4f} ± {np.std(grasp_mses):.4f}', f'{np.mean(grasp_dcs):.4f} ± {np.std(grasp_dcs):.4f}'])



# EVALUATE WITH VARIABLE SPOKES PER FRAME

MAIN_EVALUATION_PLAN = [
    {
        "spokes_per_frame": 16,
        "num_frames": 20, # 16 * 20 = 320 total spokes
        "slice": slice(1, 21), # Wide window
        "description": "High temporal resolution"
    },
    {
        "spokes_per_frame": 20,
        "num_frames": 16, # 20 * 16 = 320 total spokes
        "slice": slice(3, 19), # Medium-wide window
        "description": "Good temporal resolution"
    },
    {
        "spokes_per_frame": 32,
        "num_frames": 10, # 32 * 10 = 320 total spokes
        "slice": slice(5, 15), # Narrow window centered on enhancement
        "description": "Standard temporal resolution"
    },
    {
        "spokes_per_frame": 40,
        "num_frames": 8,  # 40 * 8 = 320 total spokes
        "slice": slice(5, 13), # Very narrow window on peak
        "description": "Low temporal resolution"
    }
]



# --- Stress Test Plan ---
# Designed to push the limits with very few spokes per frame.
# This has a different (lower) total spoke budget.
STRESS_TEST_PLAN = [
    {
        "spokes_per_frame": 8,
        "num_frames": 22, # 8 * 22 = 176 total spokes
        "slice": slice(0, 22), # The entire 22-frame duration
        "description": "Stress test: max temporal points, 8 spokes"
    },
    {
        "spokes_per_frame": 4,
        "num_frames": 22, # 4 * 22 = 88 total spokes
        "slice": slice(0, 22), # The entire 22-frame duration
        "description": "Stress test: max temporal points, 4 spokes"
    },

]


eval_spf_dataset = SimulatedSPFDataset(
    root_dir=config["evaluation"]["simulated_dataset_path"], 
    model_type=model_type, 
    patient_ids=val_dro_patient_ids,
    )

eval_spf_loader = DataLoader(
    eval_spf_dataset,
    batch_size=config["dataloader"]["batch_size"],
    shuffle=config["dataloader"]["shuffle"],
    num_workers=config["dataloader"]["num_workers"],
)

# stress_test_path = os.path.join(os.path.dirname(config["evaluation"]["simulated_dataset_path"]), 'undersampled_sim_8spf')


# stress_test_dataset = SimulatedSPFDataset(
#     root_dir=stress_test_path, 
#     model_type=model_type, 
#     patient_ids=val_dro_patient_ids,
#     )

# stress_test_loader = DataLoader(
#     stress_test_dataset,
#     batch_size=config["dataloader"]["batch_size"],
#     shuffle=config["dataloader"]["shuffle"],
#     num_workers=config["dataloader"]["num_workers"],
# )


# First, run the fair comparisons
with torch.no_grad():

    spf_recon_ssim = {}
    spf_recon_psnr = {}
    spf_recon_mse = {}
    spf_recon_dc = {}
    spf_grasp_ssim = {}
    spf_grasp_psnr = {}
    spf_grasp_mse = {}
    spf_grasp_dc = {}


    print("--- Running Stress Test Evaluation (Budget: 176 spokes) ---")
    for eval_config in STRESS_TEST_PLAN:

        stress_test_ssims = []
        stress_test_psnrs = []
        stress_test_mses = []
        stress_test_dcs = []
        stress_test_grasp_ssims = []
        stress_test_grasp_psnrs = []
        stress_test_grasp_mses = []
        stress_test_grasp_dcs = []

        spokes = eval_config["spokes_per_frame"]
        time_slice = eval_config["slice"]
        num_frames = eval_config["num_frames"]

        eval_spf_dataset.spokes_per_frame = spokes
        eval_spf_dataset.window = time_slice
        eval_spf_dataset.num_frames = num_frames


        for csmap, ground_truth, grasp_img, mask in tqdm(eval_spf_loader, desc="Variable Spokes Per Frame Evaluation"):
            

            acceleration_factor = round(N_full / spokes, 1)


            csmap = csmap.squeeze(0).to(device)   # Remove batch dim
            ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

            grasp_img = grasp_img.to(device)
            grasp_img_plot = torch.flip(grasp_img, dims=[-3])
            grasp_img_plot = torch.rot90(grasp_img_plot, k=3, dims=[-3,-1])



            # SIMULATE KSPACE
            ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(640, spokes, num_frames)
            physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))

            sim_kspace = physics(False, ground_truth, csmap)

            kspace = sim_kspace.squeeze(0).to(device) # Remove batch dim
            


            x_recon, _ = model(
                kspace.to(device), physics, csmap, acceleration_factor, norm=config['model']['norm']
            )

            ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
            ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')


            ## Evaluation
            ssim, psnr, mse, dc = eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img_plot, eval_dir, f"{spokes}spf", device)
            stress_test_ssims.append(ssim)
            stress_test_psnrs.append(psnr)
            stress_test_mses.append(mse)
            stress_test_dcs.append(dc)


            ssim_grasp, psnr_grasp, mse_grasp, dc_grasp = eval_grasp(kspace, csmap, ground_truth, grasp_img, physics, device)
            stress_test_grasp_ssims.append(ssim_grasp)
            stress_test_grasp_psnrs.append(psnr_grasp)
            stress_test_grasp_mses.append(mse_grasp)
            stress_test_grasp_dcs.append(dc_grasp)


            spf_recon_ssim[spokes] = np.mean(stress_test_ssims)
            spf_recon_psnr[spokes] = np.mean(stress_test_psnrs)
            spf_recon_mse[spokes] = np.mean(stress_test_mses)
            spf_recon_dc[spokes] = np.mean(stress_test_dcs)

            spf_grasp_ssim[spokes] = np.mean(stress_test_grasp_ssims)
            spf_grasp_psnr[spokes] = np.mean(stress_test_grasp_psnrs)
            spf_grasp_mse[spokes] = np.mean(stress_test_grasp_mses)
            spf_grasp_dc[spokes] = np.mean(stress_test_grasp_dcs)



        # Save Results
        spf_metrics_path = os.path.join(eval_dir, "eval_metrics.csv")
        with open(spf_metrics_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', 'DC'])

            writer.writerow(['DL', spokes, 
            f'{np.mean(stress_test_ssims):.4f} ± {np.std(stress_test_ssims):.4f}', 
            f'{np.mean(stress_test_psnrs):.4f} ± {np.std(stress_test_psnrs):.4f}', 
            f'{np.mean(stress_test_mses):.4f} ± {np.std(stress_test_mses):.4f}', 
            f'{np.mean(stress_test_dcs):.4f} ± {np.std(stress_test_dcs):.4f}'])

            writer.writerow(['GRASP', spokes, 
            f'{np.mean(stress_test_grasp_ssims):.4f} ± {np.std(stress_test_grasp_ssims):.4f}', 
            f'{np.mean(stress_test_grasp_psnrs):.4f} ± {np.std(stress_test_grasp_psnrs):.4f}', 
            f'{np.mean(stress_test_grasp_mses):.4f} ± {np.std(stress_test_grasp_mses):.4f}', 
            f'{np.mean(stress_test_grasp_dcs):.4f} ± {np.std(stress_test_grasp_dcs):.4f}'])



    print("--- Running Main Evaluation (Budget: 320 spokes) ---")
    for eval_config in MAIN_EVALUATION_PLAN:

        spf_eval_ssims = []
        spf_eval_psnrs = []
        spf_eval_mses = []
        spf_eval_dcs = []
        spf_grasp_ssims = []
        spf_grasp_psnrs = []
        spf_grasp_mses = []
        spf_grasp_dcs = []

        spokes = eval_config["spokes_per_frame"]
        time_slice = eval_config["slice"]
        num_frames = eval_config["num_frames"]

        eval_spf_dataset.spokes_per_frame = spokes
        eval_spf_dataset.window = time_slice
        eval_spf_dataset.num_frames = num_frames


        for csmap, ground_truth, grasp_img, mask in tqdm(eval_spf_loader, desc="Variable Spokes Per Frame Evaluation"):


            acceleration_factor = round(N_full / spokes, 1)


            csmap = csmap.squeeze(0).to(device)   # Remove batch dim
            ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

            grasp_img = grasp_img.to(device)
            grasp_img_plot = torch.flip(grasp_img, dims=[-3])
            grasp_img_plot = torch.rot90(grasp_img_plot, k=3, dims=[-3,-1])



            # SIMULATE KSPACE
            ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(640, spokes, num_frames)
            physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))

            sim_kspace = physics(False, ground_truth, csmap)

            kspace = sim_kspace.squeeze(0).to(device) # Remove batch dim
            


            x_recon, _ = model(
                kspace.to(device), physics, csmap, acceleration_factor, norm=config['model']['norm']
            )

            ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
            ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')


            ## Evaluation
            ssim, psnr, mse, dc = eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img_plot, eval_dir, f'{spokes}spf', device)
            spf_eval_ssims.append(ssim)
            spf_eval_psnrs.append(psnr)
            spf_eval_mses.append(mse)
            spf_eval_dcs.append(dc)


            ssim_grasp, psnr_grasp, mse_grasp, dc_grasp = eval_grasp(kspace, csmap, ground_truth, grasp_img, physics, device)
            spf_grasp_ssims.append(ssim_grasp)
            spf_grasp_psnrs.append(psnr_grasp)
            spf_grasp_mses.append(mse_grasp)
            spf_grasp_dcs.append(dc_grasp)


        spf_recon_ssim[spokes] = np.mean(spf_eval_ssims)
        spf_recon_psnr[spokes] = np.mean(spf_eval_psnrs)
        spf_recon_mse[spokes] = np.mean(spf_eval_mses)
        spf_recon_dc[spokes] = np.mean(spf_eval_dcs)

        spf_grasp_ssim[spokes] = np.mean(spf_grasp_ssims)
        spf_grasp_psnr[spokes] = np.mean(spf_grasp_psnrs)
        spf_grasp_mse[spokes] = np.mean(spf_grasp_mses)
        spf_grasp_dc[spokes] = np.mean(spf_grasp_dcs)


        # Save Results
        spf_metrics_path = os.path.join(eval_dir, "eval_metrics.csv")
        with open(spf_metrics_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', 'DC'])

            writer.writerow(['DL', spokes, 
            f'{np.mean(spf_eval_ssims):.4f} ± {np.std(spf_eval_ssims):.4f}', 
            f'{np.mean(spf_eval_psnrs):.4f} ± {np.std(spf_eval_psnrs):.4f}', 
            f'{np.mean(spf_eval_mses):.4f} ± {np.std(spf_eval_mses):.4f}', 
            f'{np.mean(spf_eval_dcs):.4f} ± {np.std(spf_eval_dcs):.4f}'])

            writer.writerow(['GRASP', spokes, 
            f'{np.mean(spf_grasp_ssims):.4f} ± {np.std(spf_grasp_ssims):.4f}', 
            f'{np.mean(spf_grasp_psnrs):.4f} ± {np.std(spf_grasp_psnrs):.4f}', 
            f'{np.mean(spf_grasp_mses):.4f} ± {np.std(spf_grasp_mses):.4f}', 
            f'{np.mean(spf_grasp_dcs):.4f} ± {np.std(spf_grasp_dcs):.4f}'])

    

    






# # Then, run the stress test to highlight DL model's advantage
# print("\n--- Running Stress Test (Budget: 176 spokes) ---")
# for eval_config in STRESS_TEST_PLAN:

#     spokes = eval_config["spokes_per_frame"]
#     time_slice = eval_config["slice"]
#     num_frames = eval_config["num_frames"]

#     print(f"  Testing {spokes} spokes/frame with {num_frames} frames.")

#     stress_test_ssims = []
#     stress_test_psnrs = []
#     stress_test_mses = []
#     stress_test_dcs = []
#     stress_test_grasp_ssims = []
#     stress_test_grasp_psnrs = []
#     stress_test_grasp_mses = []
#     stress_test_grasp_dcs = []

#     eval_spf_dataset.spokes_per_frame = spokes
#     eval_spf_dataset.window = time_slice
#     eval_spf_dataset.num_frames = num_frames


#     for kspace, csmap, ground_truth, grasp_img, mask, physics in tqdm(val_dro_loader, desc="Stress Test Evaluation"):


#         kspace = kspace.squeeze(0).to(device) # Remove batch dim
#         csmap = csmap.squeeze(0).to(device)   # Remove batch dim
#         ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

#         grasp_img = grasp_img.to(device)
#         grasp_img_plot = torch.flip(grasp_img, dims=[-3])
#         grasp_img_plot = torch.rot90(grasp_img_plot, k=3, dims=[-3,-1])
        

#         print("performing inference...")
#         x_recon, _ = model(
#             kspace.to(device), physics, csmap, norm=config['model']['norm']
#         )


#         print("calculating evaluation metrics...")

#         ## Evaluation
#         ssim, psnr, mse, dc = eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img_plot, eval_dir, epoch, device)
#         stress_test_ssims.append(ssim)
#         stress_test_psnrs.append(psnr)
#         stress_test_mses.append(mse)
#         stress_test_dcs.append(dc)


#         ssim_grasp, psnr_grasp, mse_grasp, dc_grasp = eval_grasp(kspace, csmap, ground_truth, grasp_img, physics, device)
#         stress_test_grasp_ssims.append(ssim_grasp)
#         stress_test_grasp_psnrs.append(psnr_grasp)
#         stress_test_grasp_mses.append(mse_grasp)
#         stress_test_grasp_dcs.append(dc_grasp)


#         spf_recon_ssim[spokes] = np.mean(stress_test_ssims)
#         spf_recon_psnr[spokes] = np.mean(stress_test_psnrs)
#         spf_recon_mse[spokes] = np.mean(stress_test_mses)
#         spf_recon_dc[spokes] = np.mean(stress_test_dcs)

#         spf_grasp_ssim[spokes] = np.mean(stress_test_grasp_ssims)
#         spf_grasp_psnr[spokes] = np.mean(stress_test_grasp_psnrs)
#         spf_grasp_mse[spokes] = np.mean(stress_test_grasp_mses)
#         spf_grasp_dc[spokes] = np.mean(stress_test_grasp_dcs)



#     # Save Results
#     spf_metrics_path = os.path.join(eval_dir, "eval_metrics.csv")
#     with open(spf_metrics_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', 'DC'])

#         writer.writerow(['DL', spokes, 
#         f'{np.mean(stress_test_ssims):.4f} ± {np.std(stress_test_ssims):.4f}', 
#         f'{np.mean(stress_test_psnrs):.4f} ± {np.std(stress_test_psnrs):.4f}', 
#         f'{np.mean(stress_test_mses):.4f} ± {np.std(stress_test_mses):.4f}', 
#         f'{np.mean(stress_test_dcs):.4f} ± {np.std(stress_test_dcs):.4f}'])

#         writer.writerow(['GRASP', spokes, 
#         f'{np.mean(stress_test_grasp_ssims):.4f} ± {np.std(stress_test_grasp_ssims):.4f}', 
#         f'{np.mean(stress_test_grasp_psnrs):.4f} ± {np.std(stress_test_grasp_psnrs):.4f}', 
#         f'{np.mean(stress_test_grasp_mses):.4f} ± {np.std(stress_test_grasp_mses):.4f}', 
#         f'{np.mean(stress_test_grasp_dcs):.4f} ± {np.std(stress_test_grasp_dcs):.4f}'])



# Plot Metrics vs Spokes Per Frame


plt.figure()
plt.plot(list(spf_recon_ssim.keys()), list(spf_recon_ssim.values()), label="DL Recon", marker='o')
plt.plot(list(spf_grasp_ssim.keys()), list(spf_grasp_ssim.values()), label="GRASP Recon", marker='o')
plt.xlabel("Spokes per Frame")
plt.ylabel("SSIM")
plt.title("Evaluation SSIM vs Spokes per Frame")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(eval_dir, "spf_eval_ssim.png"))
plt.close()


plt.figure()
plt.plot(list(spf_recon_psnr.keys()), list(spf_recon_psnr.values()), label="DL Recon", marker='o')
plt.plot(list(spf_grasp_psnr.keys()), list(spf_grasp_psnr.values()), label="GRASP Recon", marker='o')
plt.xlabel("Spokes per Frame")
plt.ylabel("PSNR")
plt.title("Evaluation PSNR vs Spokes per Frame")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(eval_dir, "spf_eval_psnr.png"))
plt.close()


plt.figure()
plt.plot(list(spf_recon_mse.keys()), list(spf_recon_mse.values()), label="DL Recon", marker='o')
plt.plot(list(spf_grasp_mse.keys()), list(spf_grasp_mse.values()), label="GRASP Recon", marker='o')
plt.xlabel("Spokes per Frame")
plt.ylabel("MSE")
plt.title("Evaluation Image MSE vs Spokes per Frame")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(eval_dir, "spf_eval_mse.png"))
plt.close()


plt.figure()
plt.plot(list(spf_recon_dc.keys()), list(spf_recon_dc.values()), label="DL Recon", marker='o')
plt.plot(list(spf_grasp_dc.keys()), list(spf_grasp_dc.values()), label="GRASP Recon", marker='o')
plt.xlabel("Spokes per Frame")
plt.ylabel("k-space MSE")
plt.title("Data Consistency Evaluation (MSE) vs Spokes per Frame")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(eval_dir, "spf_eval_dc.png"))
plt.close()