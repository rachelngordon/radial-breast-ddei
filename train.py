import argparse
import json
import os
import matplotlib.pyplot as plt
import torch
import yaml
from dataloader import SliceDataset, SimulatedDataset, SimulatedSPFDataset
from deepinv.transform import Transform
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transform import VideoRotate, VideoDiffeo, SubsampleTime, MonophasicTimeWarp, TemporalNoise, TimeReverse
from ei import EILoss
from mc import MCLoss
from lsfpnet_encoding import LSFPNet, ArtifactRemovalLSFPNet
from radial_lsfp import MCNUFFT
from utils import prep_nufft, log_gradient_stats, plot_enhancement_curve, get_cosine_ei_weight, plot_reconstruction_sample, get_git_commit, save_checkpoint, load_checkpoint, to_torch_complex, GRASPRecon
from eval import eval_grasp, eval_sample
import csv
import math
import random
import time 
import seaborn as sns

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

block_dir = os.path.join(output_dir, "block_outputs")
os.makedirs(block_dir, exist_ok=True)


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

model_type = config["model"]["name"]

H, W = config["data"]["height"], config["data"]["width"]
N_time, N_samples, N_coils, N_time_eval, N_spokes_eval = (
    config["data"]["timeframes"],
    config["data"]["samples"],
    config["data"]["coils"],
    config["data"]["eval_timeframes"],
    config["data"]["eval_spokes"]
)
Ng = config["data"]["fpg"] 
# Mg = N_time // Ng  
# indG = np.arange(0, N_time, 1, dtype=int)
# indG = np.reshape(indG, [Mg, Ng])


N_spokes = int(config["data"]["total_spokes"] / N_time)

N_full = config['data']['height'] * math.pi / 2

os.makedirs(os.path.join(output_dir, 'enhancement_curves'), exist_ok=True)

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

# if config['data']['spf_aug']:
#     train_dataset = SliceDatasetAug(
#         root_dir=config["data"]["root_dir"],
#         patient_ids=train_patient_ids,
#         dataset_key=config["data"]["dataset_key"],
#         file_pattern="*.h5",
#         slice_idx=config["dataloader"]["slice_idx"],
#         N_coils=N_coils
#     )

if config['dataloader']['slice_range_start'] == "None" or config['dataloader']['slice_range_end'] == "None":
    train_dataset = SliceDataset(
        root_dir=config["data"]["root_dir"],
        patient_ids=train_patient_ids,
        dataset_key=config["data"]["dataset_key"],
        file_pattern="*.h5",
        slice_idx=config["dataloader"]["slice_idx"],
        num_random_slices=config["dataloader"].get("num_random_slices", None),
        N_time=N_time,
        N_coils=N_coils,
        spf_aug=config['data']['spf_aug'],
    )
else:
    train_dataset = SliceDataset(
        root_dir=config["data"]["root_dir"],
        patient_ids=train_patient_ids,
        dataset_key=config["data"]["dataset_key"],
        file_pattern="*.h5",
        slice_idx=range(config['dataloader']['slice_range_start'], config['dataloader']['slice_range_end']),
        num_random_slices=config["dataloader"].get("num_random_slices", None),
        N_time=N_time,
        N_coils=N_coils,
        spf_aug=config['data']['spf_aug'],
    )



# val_dataset = SliceDataset(
#     root_dir=config["data"]["root_dir"],
#     patient_ids=val_patient_ids,
#     dataset_key=config["data"]["dataset_key"],
#     file_pattern="*.h5",
#     slice_idx=config["dataloader"]["slice_idx"],
#     N_coils=N_coils
# )


val_dro_dataset = SimulatedDataset(
    root_dir=config["evaluation"]["simulated_dataset_path"], 
    model_type=model_type, 
    patient_ids=val_dro_patient_ids,
    spokes_per_frame=N_spokes_eval,
    num_frames=N_time_eval)


train_loader = DataLoader(
    train_dataset,
    batch_size=config["dataloader"]["batch_size"],
    shuffle=config["dataloader"]["shuffle"],
    num_workers=config["dataloader"]["num_workers"],
    pin_memory=True,
)


# val_loader = DataLoader(
#     val_dataset,
#     batch_size=config["dataloader"]["batch_size"],
#     shuffle=config["dataloader"]["shuffle"],
#     num_workers=config["dataloader"]["num_workers"],
# )

val_dro_loader = DataLoader(
    val_dro_dataset,
    batch_size=config["dataloader"]["batch_size"],
    shuffle=config["dataloader"]["shuffle"],
    num_workers=config["dataloader"]["num_workers"],
    pin_memory=True,
)


eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, N_spokes_eval, N_time_eval)
eval_ktraj = eval_ktraj.to(device)
eval_dcomp = eval_dcomp.to(device)
eval_nufft_ob = eval_nufft_ob.to(device)
eval_adjnufft_ob = eval_adjnufft_ob.to(device)





eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

lsfp_backbone = LSFPNet(LayerNo=config["model"]["num_layers"], lambdas=initial_lambdas, channels=config['model']['channels'])
model = ArtifactRemovalLSFPNet(lsfp_backbone, block_dir).to(device)



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
if config['model']['losses']['mc_loss']['metric'] == "MSE":
    mc_loss_fn = MCLoss(model_type=model_type)
elif config['model']['losses']['mc_loss']['metric'] == "MAE":
    mc_loss_fn = MCLoss(model_type=model_type, metric=torch.nn.L1Loss())
else:
    raise(ValueError, "Unsupported MC Loss Metric.")


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
        if config['model']['losses']['ei_loss']['spatial_transform'] == "none":
            ei_loss_fn = EILoss(subsample, model_type=model_type)
        else:
            ei_loss_fn = EILoss(subsample | (diffeo | rotate), model_type=model_type)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "warp":
        # ei_loss_fn = EILoss(monophasic_warp | (diffeo | rotate), model_type=model_type)
        ei_loss_fn = EILoss(monophasic_warp, model_type=model_type)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "noise":
        ei_loss_fn = EILoss(temp_noise, model_type=model_type)
    # elif config['model']['losses']['ei_loss']['temporal_transform'] == "reverse":
    #     ei_loss_fn = EILoss(time_reverse | (diffeo | rotate), model_type=model_type)
    # elif config['model']['losses']['ei_loss']['temporal_transform'] == "all":
    #     ei_loss_fn = EILoss((subsample | monophasic_warp | temp_noise | time_reverse) | (diffeo | rotate), model_type=model_type)
    elif config['model']['losses']['ei_loss']['temporal_transform'] == "none":
        if config['model']['losses']['ei_loss']['spatial_transform'] == "rotate":
            ei_loss_fn = EILoss(rotate, model_type=model_type)
        elif config['model']['losses']['ei_loss']['spatial_transform'] == "diffeo":
            ei_loss_fn = EILoss(diffeo, model_type=model_type)
        else:
            ei_loss_fn = EILoss(rotate | diffeo, model_type=model_type)
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
    eval_ssims = eval_curves["eval_ssims"]
    eval_psnrs = eval_curves["eval_psnrs"]
    eval_mses = eval_curves["eval_mses"]
    eval_lpipses = eval_curves["eval_lpipses"]
    eval_dc_mses = eval_curves["eval_dc_mses"]
    eval_dc_maes = eval_curves["eval_dc_maes"]
    eval_curve_corrs = eval_curves["eval_curve_corrs"]
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
    eval_lpipses = []
    eval_psnrs = []
    eval_mses = []
    eval_dc_mses = []
    eval_dc_maes = []
    eval_curve_corrs = []


grasp_ssims = []
grasp_psnrs = []
grasp_mses = []
grasp_lpipses = []
grasp_dc_mses = []
grasp_dc_maes = []
grasp_curve_corrs = []

lambda_Ls = []
lambda_Ss = []
lambda_spatial_Ls = []
lambda_spatial_Ss = []
gammas = []
lambda_steps = []

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
    initial_eval_ssims = []
    initial_eval_psnrs = []
    initial_eval_mses = []
    initial_eval_lpipses = []
    initial_eval_dc_mses = []
    initial_eval_dc_maes = []
    initial_eval_curve_corrs = []


    with torch.no_grad():
        # Evaluate on training data
        for measured_kspace, csmap, grasp_img, N_samples, N_spokes, N_time in tqdm(train_loader, desc="Step 0 Training Evaluation"):

            # prepare inputs
            measured_kspace = to_torch_complex(measured_kspace).squeeze()
            measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')
            
            if N_time > Ng:
                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, Ng)
                ktraj = ktraj.to(device)
                dcomp = dcomp.to(device)
                nufft_ob = nufft_ob.to(device)
                adjnufft_ob = adjnufft_ob.to(device)

                physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)

                max_idx = N_time - Ng
                random_index = random.randint(0, max_idx - 1) 

                measured_kspace = measured_kspace[..., random_index:random_index + Ng]

            else:
                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
                ktraj = ktraj.to(device)
                dcomp = dcomp.to(device)
                nufft_ob = nufft_ob.to(device)
                adjnufft_ob = adjnufft_ob.to(device)

                physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)


            csmap = csmap.to(device).to(measured_kspace.dtype)

            acceleration = torch.tensor([N_full / int(N_spokes)], dtype=torch.float, device=device)

            if config['model']['encode_acceleration']:
                acceleration_encoding = acceleration
            else: 
                acceleration_encoding = None


            x_recon, adj_loss, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step = model(
                measured_kspace.to(device), physics, csmap, acceleration_encoding, epoch="train0", norm=config['model']['norm']
            )

            # calculate losses
            initial_train_adj_loss += adj_loss.item()

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
            initial_train_mc_loss += mc_loss.item()

            if use_ei_loss:
                ei_loss, t_img = ei_loss_fn(
                    x_recon, physics, model, csmap, acceleration_encoding
                )

                initial_train_ei_loss += ei_loss.item()
            

        # record losses
        step0_train_mc_loss = initial_train_mc_loss / len(train_loader)
        train_mc_losses.append(step0_train_mc_loss)

        step0_train_ei_loss = initial_train_ei_loss / len(train_loader)
        train_ei_losses.append(step0_train_ei_loss)

        step0_train_adj_loss = initial_train_adj_loss / len(train_loader)
        train_adj_losses.append(step0_train_adj_loss)


        lambda_Ls.append(lambda_L.item())
        lambda_Ss.append(lambda_S.item())
        lambda_spatial_Ls.append(lambda_spatial_L.item())
        lambda_spatial_Ss.append(lambda_spatial_S.item())
        gammas.append(gamma.item())
        lambda_steps.append(lambda_step.item())


        # Evaluate on validation data
        for measured_kspace, csmap, ground_truth, grasp_img, mask, *_ in tqdm(val_dro_loader, desc="Step 0 Validation Evaluation"):

            csmap = csmap.squeeze(0).to(device)   # Remove batch dim
            ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

            # simulate k-space for validation if path does not exist
            if type(measured_kspace) is list:

                ground_truth_for_physics = rearrange(to_torch_complex(ground_truth), 'b t h w -> b h w t')
                kspace_path = measured_kspace[0]

                # SIMULATE KSPACE
                measured_kspace = eval_physics(False, ground_truth_for_physics, csmap)

                # save k-space 
                np.save(kspace_path, measured_kspace.cpu().numpy())


            measured_kspace = measured_kspace.squeeze(0).to(device) # Remove batch dim

            N_spokes = eval_ktraj.shape[1] / config['data']['samples']
            acceleration = torch.tensor([N_full / int(N_spokes)], dtype=torch.float, device=device)

            if config['model']['encode_acceleration']:
                acceleration_encoding = acceleration
            else: 
                acceleration_encoding = None

            x_recon, adj_loss, *_ = model(
                measured_kspace.to(device), eval_physics, csmap, acceleration_encoding, epoch="val0", norm=config['model']['norm']
            )

            # compute losses
            initial_val_adj_loss += adj_loss.item()
            
            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, eval_physics, csmap)
            initial_val_mc_loss += mc_loss.item()

            if use_ei_loss:
                ei_loss, t_img = ei_loss_fn(
                    x_recon, eval_physics, model, csmap, acceleration_encoding
                )

                initial_val_ei_loss += ei_loss.item()

            ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)
            grasp_recon = grasp_img.to(device) # Shape: (1, 2, H, T, W)

            # calculate grasp metrics
            ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp = eval_grasp(measured_kspace, csmap, ground_truth, grasp_recon, eval_physics, device, eval_dir)
            grasp_ssims.append(ssim_grasp)
            grasp_psnrs.append(psnr_grasp)
            grasp_mses.append(mse_grasp)
            grasp_lpipses.append(lpips_grasp)
            grasp_dc_mses.append(dc_mse_grasp)
            grasp_dc_maes.append(dc_mae_grasp)

            ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr = eval_sample(measured_kspace, csmap, ground_truth, x_recon, eval_physics, mask, grasp_recon, acceleration, eval_dir, label=None, device=device)
            initial_eval_ssims.append(ssim)
            initial_eval_psnrs.append(psnr)
            initial_eval_mses.append(mse)
            initial_eval_lpipses.append(lpips)
            initial_eval_dc_mses.append(dc_mse)
            initial_eval_dc_maes.append(dc_mae)

            if recon_corr is not None:
                initial_eval_curve_corrs.append(recon_corr)
                grasp_curve_corrs.append(grasp_corr)

        step0_val_mc_loss = initial_val_mc_loss / len(val_dro_loader)
        val_mc_losses.append(step0_val_mc_loss)

        step0_val_ei_loss = initial_val_ei_loss / len(val_dro_loader)
        val_ei_losses.append(step0_val_ei_loss)

        step0_val_adj_loss = initial_val_adj_loss / len(val_dro_loader)
        val_adj_losses.append(step0_val_adj_loss)


        # Calculate and store average validation evaluation metrics
        initial_eval_ssim = np.mean(initial_eval_ssims)
        initial_eval_psnr = np.mean(initial_eval_psnrs)
        initial_eval_mse = np.mean(initial_eval_mses)
        initial_eval_lpips = np.mean(initial_eval_lpipses)
        initial_eval_dc_mse = np.mean(initial_eval_dc_mses)
        initial_eval_dc_mae = np.mean(initial_eval_dc_maes)
        initial_eval_curve_corr = np.mean(initial_eval_curve_corrs)

        eval_ssims.append(initial_eval_ssim)
        eval_psnrs.append(initial_eval_psnr)
        eval_mses.append(initial_eval_mse)
        eval_lpipses.append(initial_eval_lpips)
        eval_dc_mses.append(initial_eval_dc_mse) 
        eval_dc_maes.append(initial_eval_dc_mae) 
        eval_curve_corrs.append(initial_eval_curve_corr)

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
        epoch_eval_lpipses = []
        epoch_eval_dc_mses = []
        epoch_eval_dc_maes = []
        epoch_eval_curve_corrs = []


        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch"
        )

        if hasattr(train_dataset, 'resample_slices'):
            print(f"Epoch {epoch}: Resampling training slices...")
            train_dataset.resample_slices()

        if use_ei_loss:

            # --- Check if it's the transition epoch ---
            if epoch < warmup + 1:
                target_w_ei = 0.0
            elif epoch == warmup + 1:
                # Get the last known MC loss value from the previous epoch
                mc_loss_at_transition = epoch_train_mc_loss
                
                print(f"Transitioning at Epoch {epoch}. MC Loss: {mc_loss_at_transition:.4e}")
                
                # Calculate the final target weight ONCE
                if step0_train_ei_loss > 0:
                    target_w_ei = mc_loss_at_transition / step0_train_ei_loss
                else:
                    target_w_ei = 0.0 # Prevent division by zero
                    
                print(f"Dynamically calculated target EI weight: {target_w_ei:.4f}")



        for measured_kspace, csmap, grasp_img, N_samples, N_spokes, N_time in train_loader_tqdm:  # measured_kspace shape: (B, C, I, S, T)
            
            start = time.time()

            # prepare inputs
            measured_kspace = to_torch_complex(measured_kspace).squeeze()
            measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

            if N_time > Ng:

                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, Ng)
                ktraj = ktraj.to(device)
                dcomp = dcomp.to(device)
                nufft_ob = nufft_ob.to(device)
                adjnufft_ob = adjnufft_ob.to(device)

                physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)

                max_idx = N_time - Ng
                random_index = random.randint(0, max_idx - 1) 

                measured_kspace = measured_kspace[..., random_index:random_index + Ng]
                

            else:
                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
                ktraj = ktraj.to(device)
                dcomp = dcomp.to(device)
                nufft_ob = nufft_ob.to(device)
                adjnufft_ob = adjnufft_ob.to(device)

                physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)


            iteration_count += 1
            optimizer.zero_grad()

            csmap = csmap.to(device).to(measured_kspace.dtype)

            # calculate acceleration factor
            acceleration = torch.tensor([N_full / int(N_spokes)], dtype=torch.float, device=device)

            if config['model']['encode_acceleration']:
                acceleration_encoding = acceleration
            else: 
                acceleration_encoding = None


            x_recon, adj_loss, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step = model(
                measured_kspace.to(device), physics, csmap, acceleration_encoding, epoch=f"train{epoch}", norm=config['model']['norm']
            )

            # compute losses
            running_adj_loss += adj_loss.item()

            mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
            running_mc_loss += mc_loss.item()

            if use_ei_loss:
                ei_loss, t_img = ei_loss_fn(
                    x_recon, physics, model, csmap, acceleration_encoding
                )


                ei_loss_weight = get_cosine_ei_weight(
                    current_epoch=epoch,
                    warmup_epochs=warmup,
                    schedule_duration=duration,
                    target_weight=target_w_ei
                )


                running_ei_loss += ei_loss.item()
                total_loss = mc_loss * mc_loss_weight + ei_loss * ei_loss_weight + torch.mul(adj_loss_weight, adj_loss)
                train_loader_tqdm.set_postfix(
                    mc_loss=mc_loss.item(), ei_loss=ei_loss.item()
                )

            else:
                total_loss = mc_loss * mc_loss_weight + torch.mul(adj_loss_weight, adj_loss)
                train_loader_tqdm.set_postfix(mc_loss=mc_loss.item())

            if torch.isnan(total_loss):
                print(
                    "!!! ERROR: total_loss is NaN before backward pass. Aborting. !!!"
                )
                raise RuntimeError("total_loss is NaN")


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

            end = time.time()
            print("time for one iteration: ", end-start)


        # NOTE: commented out plotting until figured out how to handle variable numbers of timeframes
        # if epoch % save_interval == 0:

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

        #         plot_reconstruction_sample(
        #             t_img,
        #             f"Transformed Train Sample - Epoch {epoch}",
        #             f"transforms/transform_train_sample_epoch_{epoch}",
        #             output_dir,
        #             x_recon,
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

        epoch_train_adj_loss = running_adj_loss / len(train_loader)
        train_adj_losses.append(epoch_train_adj_loss)
        weighted_train_adj_losses.append(epoch_train_adj_loss*adj_loss_weight)


        lambda_Ls.append(lambda_L.item())
        lambda_Ss.append(lambda_S.item())
        lambda_spatial_Ls.append(lambda_spatial_L.item())
        lambda_spatial_Ss.append(lambda_spatial_S.item())
        gammas.append(gamma.item())
        lambda_steps.append(lambda_step.item())


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

                val_csmap = val_csmap.squeeze(0).to(device)   # Remove batch dim
                val_ground_truth = val_ground_truth.to(device) # Shape: (1, 2, T, H, W)

                # simulate k-space for validation if path does not exist
                if type(val_kspace_batch) is list:

                    ground_truth_for_physics = rearrange(to_torch_complex(val_ground_truth), 'b t h w -> b h w t')
                    kspace_path = val_kspace_batch[0]

                    # SIMULATE KSPACE
                    val_kspace_batch = eval_physics(False, ground_truth_for_physics, val_csmap)

                    # save k-space 
                    np.save(kspace_path, val_kspace_batch)

                # prepare inputs
                val_kspace_batch = val_kspace_batch.squeeze(0).to(device) # Remove batch dim

                val_grasp_img_tensor = val_grasp_img.to(device)

                # calculate acceleration factor
                N_spokes = eval_ktraj.shape[1] / config['data']['samples']
                acceleration = torch.tensor([N_full / int(N_spokes)], dtype=torch.float, device=device)

                if config['model']['encode_acceleration']:
                    acceleration_encoding = acceleration
                else: 
                    acceleration_encoding = None
                    

                val_x_recon, val_adj_loss, *_ = model(
                    val_kspace_batch.to(device), eval_physics, val_csmap, acceleration_encoding, epoch=f"val{epoch}", norm=config['model']['norm']
                )

                # compute losses
                val_running_adj_loss += val_adj_loss.item()

                val_mc_loss = mc_loss_fn(val_kspace_batch.to(device), val_x_recon, eval_physics, val_csmap)
                val_running_mc_loss += val_mc_loss.item()

                if use_ei_loss:
                    val_ei_loss, val_t_img = ei_loss_fn(
                        val_x_recon, eval_physics, model, val_csmap, acceleration_encoding
                    )

                    # # 1. Apply the image transform
                    # x_net_rearranged = rearrange(val_x_recon, 'b c h w t -> b c t h w')
                    # x2_rearranged = ei_loss_fn.T(x_net_rearranged) # This calls VideoRotate
                    # val_t_img = rearrange(x2_rearranged, 'b c t h w -> b c h w t')
                    # x2_complex = to_torch_complex(val_t_img)

                    # # 2. Get the rotation angle used
                    # #    (assuming you've modified VideoRotate to store it)
                    # angle_deg = ei_loss_fn.T.last_angle # Access it from the composed transform
                    # angle_rad = torch.deg2rad(torch.tensor(angle_deg, device=device))

                    # # 3. Create a 2D rotation matrix
                    # cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
                    # # Note: k-space rotation is often the transpose of image rotation
                    # rot_matrix = torch.tensor([[cos_a, sin_a],
                    #                         [-sin_a, cos_a]], dtype=torch.float32, device=device)

                    # # 4. Rotate the k-space trajectory
                    # # ktraj shape is likely (2, n_points, n_time), so we permute for matmul
                    # ktraj_permuted = eval_ktraj.permute(1, 2, 0) # (n_points, n_time, 2)
                    # ktraj_rot_permuted = torch.matmul(ktraj_permuted, rot_matrix)
                    # ktraj_rot = ktraj_rot_permuted.permute(2, 0, 1) # Back to (2, n_points, n_time)
                    
                    # # 5. Create a new, temporary physics operator for the EI loss
                    # #    Crucially, we reuse the original dcomp, as discussed.
                    # physics_rot = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, ktraj_rot, eval_dcomp)
                    
                    # # 6. Calculate the EI loss with the rotated physics
                    # y_rot = physics_rot(inv=False, data=x2_complex, smaps=val_csmap).to(val_csmap.device)
                    
                    # # For the reconstruction step, you MUST use the rotated physics again
                    # x3, *_ = model(y_rot, physics_rot, val_csmap, acceleration_encoding, epoch=None)

                    # val_ei_loss = ei_loss_fn.metric(x3, val_t_img)

                    val_running_ei_loss += val_ei_loss.item()
                    val_loader_tqdm.set_postfix(
                        val_mc_loss=val_mc_loss.item(), val_ei_loss=val_ei_loss.item()
                    )
                else:
                    val_loader_tqdm.set_postfix(val_mc_loss=val_mc_loss.item())


                ## Evaluation
                ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, _ = eval_sample(val_kspace_batch, val_csmap, val_ground_truth, val_x_recon, eval_physics, val_mask, val_grasp_img_tensor, acceleration, eval_dir, f'epoch{epoch}', device)
                epoch_eval_ssims.append(ssim)
                epoch_eval_psnrs.append(psnr)
                epoch_eval_mses.append(mse)
                epoch_eval_lpipses.append(lpips)
                epoch_eval_dc_mses.append(dc_mse)
                epoch_eval_dc_maes.append(dc_mae)

                if recon_corr is not None:
                    epoch_eval_curve_corrs.append(recon_corr)



        # Calculate and store average validation evaluation metrics
        epoch_eval_ssim = np.mean(epoch_eval_ssims)
        epoch_eval_psnr = np.mean(epoch_eval_psnrs)
        epoch_eval_mse = np.mean(epoch_eval_mses)
        epoch_eval_lpips = np.mean(epoch_eval_lpipses)
        epoch_eval_dc_mse = np.mean(epoch_eval_dc_mses)
        epoch_eval_dc_mae = np.mean(epoch_eval_dc_maes)
        epoch_eval_curve_corr = np.mean(epoch_eval_curve_corrs)

        eval_ssims.append(epoch_eval_ssim)
        eval_psnrs.append(epoch_eval_psnr)
        eval_mses.append(epoch_eval_mse)
        eval_lpipses.append(epoch_eval_lpips)
        eval_dc_mses.append(epoch_eval_dc_mse) 
        eval_dc_maes.append(epoch_eval_dc_mae)    
        eval_curve_corrs.append(epoch_eval_curve_corr)  
        
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
                eval_lpipses=eval_lpipses,
                eval_dc_mses=eval_dc_mses,
                eval_dc_maes=eval_dc_maes,
                eval_curve_corrs=eval_curve_corrs
            )
            model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
            save_checkpoint(model, optimizer, epoch + 1, train_curves, val_curves, eval_curves, model_save_path)
            print(f'Model saved to {model_save_path}')


            # Plot MC Loss
            # plt.figure()
            # plt.plot(train_mc_losses, label="Training MC Loss")
            # plt.plot(val_mc_losses, label="Validation MC Loss")
            # plt.xlabel("Epoch")
            # plt.ylabel("MC Loss")
            # plt.title("Measurement Consistency Loss")
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "mc_losses.png"))
            # plt.close()

            # # Plot Train and Val Losses Individually
            # plt.figure()
            # plt.plot(train_mc_losses)
            # plt.xlabel("Epoch")
            # plt.ylabel("MC Loss")
            # plt.title("Training Measurement Consistency Loss")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "train_mc_losses.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(val_mc_losses)
            # plt.xlabel("Epoch")
            # plt.ylabel("MC Loss")
            # plt.title("Validation Measurement Consistency Loss")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "val_mc_losses.png"))
            # plt.close()

            # if use_ei_loss:
            #     # Plot EI Loss
            #     plt.figure()
            #     plt.plot(train_ei_losses, label="Training EI Loss")
            #     plt.plot(val_ei_losses, label="Validation EI Loss")
            #     plt.xlabel("Epoch")
            #     plt.ylabel("EI Loss")
            #     plt.title("Equivariant Imaging Loss")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.savefig(os.path.join(output_dir, "ei_losses.png"))
            #     plt.close()

            #     plt.figure()
            #     plt.plot(train_ei_losses)
            #     plt.xlabel("Epoch")
            #     plt.ylabel("EI Loss")
            #     plt.title("Training Equivariant Imaging Loss")
            #     plt.grid(True)
            #     plt.savefig(os.path.join(output_dir, "train_ei_losses.png"))
            #     plt.close()

            #     plt.figure()
            #     plt.plot(val_ei_losses)
            #     plt.xlabel("Epoch")
            #     plt.ylabel("EI Loss")
            #     plt.title("Validation Equivariant Imaging Loss")
            #     plt.grid(True)
            #     plt.savefig(os.path.join(output_dir, "val_ei_losses.png"))
            #     plt.close()


            # plt.figure()
            # plt.plot(train_adj_losses, label="Training Adjoint Loss")
            # plt.plot(val_adj_losses, label="Validation Adjoint Loss")
            # plt.xlabel("Epoch")
            # plt.ylabel("Adjoint Loss")
            # plt.title("CNN Adjoint Loss")
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "adj_losses.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(train_adj_losses)
            # plt.xlabel("Epoch")
            # plt.ylabel("Adjoint Loss")
            # plt.title("Training CNN Adjoint Loss")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "train_adj_losses.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(val_adj_losses)
            # plt.xlabel("Epoch")
            # plt.ylabel("Adjoint Loss")
            # plt.title("Validation CNN Adjoint Loss")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "val_adj_losses.png"))
            # plt.close()


            # plot losses in one figure
            # Set the seaborn style
            sns.set_style("whitegrid")

            # Create a figure and a set of subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # Plot Training Adjoint Loss
            sns.lineplot(x=range(len(train_adj_losses)), y=train_adj_losses, ax=axes[0, 0])
            axes[0, 0].set_title("Training Adjoint Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Adjoint Loss")

            # Plot Training MC Loss
            sns.lineplot(x=range(len(train_mc_losses)), y=train_mc_losses, ax=axes[0, 1])
            axes[0, 1].set_title("Training MC Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("MC Loss")

            # Plot Training EI Loss
            sns.lineplot(x=range(len(train_ei_losses)), y=train_ei_losses, ax=axes[0, 2])
            axes[0, 2].set_title("Training EI Loss")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("EI Loss")

            # Plot Validation Adjoint Loss
            sns.lineplot(x=range(len(val_adj_losses)), y=val_adj_losses, ax=axes[1, 0], color='orange')
            axes[1, 0].set_title(f"Validation Adjoint Loss ({N_spokes_eval} spokes/frame)")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Adjoint Loss")

            # Plot Validation MC Loss
            sns.lineplot(x=range(len(val_mc_losses)), y=val_mc_losses, ax=axes[1, 1], color='orange')
            axes[1, 1].set_title(f"Validation MC Loss ({N_spokes_eval} spokes/frame)")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("MC Loss")

            # Plot Validation EI Loss
            sns.lineplot(x=range(len(val_ei_losses)), y=val_ei_losses, ax=axes[1, 2], color='orange')
            axes[1, 2].set_title(f"Validation EI Loss ({N_spokes_eval} spokes/frame)")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("EI Loss")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "losses.png"))
            plt.close()


            # plt.figure()
            # plt.plot(lambda_Ls)
            # plt.xlabel("Epoch")
            # plt.ylabel("Lambda_L Value")
            # plt.title("Lambda_L During Training")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "lambda_L.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(lambda_Ss)
            # plt.xlabel("Epoch")
            # plt.ylabel("Lambda_S Value")
            # plt.title("Lambda_S During Training")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "lambda_S.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(lambda_spatial_Ls)
            # plt.xlabel("Epoch")
            # plt.ylabel("Lambda_spatial_L Value")
            # plt.title("Lambda_spatial_L During Training")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "lambda_spatial_L.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(lambda_spatial_Ss)
            # plt.xlabel("Epoch")
            # plt.ylabel("Lambda_spatial_S Value")
            # plt.title("Lambda_spatial_S During Training")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "lambda_spatial_S.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(gammas)
            # plt.xlabel("Epoch")
            # plt.ylabel("Gamma Value")
            # plt.title("Step Size During Training")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "gamma.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(lambda_steps)
            # plt.xlabel("Epoch")
            # plt.ylabel("Lambda_step Value")
            # plt.title("Lambda_step During Training")
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, "lambda_step.png"))
            # plt.close()

            # plot learnable parameters in one figure
            # Set the seaborn style
            sns.set_style("whitegrid")

            # Create a figure and a set of subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            sns.lineplot(x=range(len(lambda_Ls)), y=lambda_Ls, ax=axes[0, 0])
            axes[0, 0].set_title("Lambda_L Parameter Value")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Lambda_L")

            sns.lineplot(x=range(len(lambda_Ss)), y=lambda_Ss, ax=axes[0, 1])
            axes[0, 1].set_title("Lambda_S Parameter Value")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Lambda_S")

            sns.lineplot(x=range(len(lambda_spatial_Ls)), y=lambda_spatial_Ls, ax=axes[0, 2])
            axes[0, 2].set_title("Spatial Lambda_L Parameter Value")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("Spatial Lambda_L")

            sns.lineplot(x=range(len(lambda_spatial_Ss)), y=lambda_spatial_Ss, ax=axes[1, 0])
            axes[1, 0].set_title("Spatial Lambda_S Parameter Value")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Spatial Lambda_S")

            sns.lineplot(x=range(len(gammas)), y=gammas, ax=axes[1, 1])
            axes[1, 1].set_title("Gamma Parameter Value")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Gamma")

            sns.lineplot(x=range(len(lambda_steps)), y=lambda_steps, ax=axes[1, 2])
            axes[1, 2].set_title("Lambda Step Parameter Value")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("Lambda Step")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "parameters.png"))
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
            # plt.figure()
            # plt.plot(eval_ssims)
            # plt.xlabel("Epoch")
            # plt.ylabel("SSIM")
            # plt.title("Evaluation SSIM")
            # plt.grid(True)
            # plt.savefig(os.path.join(eval_dir, "eval_ssims.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(eval_psnrs)
            # plt.xlabel("Epoch")
            # plt.ylabel("PSNR")
            # plt.title("Evaluation PSNR")
            # plt.grid(True)
            # plt.savefig(os.path.join(eval_dir, "eval_psnrs.png"))
            # plt.close()

            # plt.figure()
            # plt.plot(eval_mses)
            # plt.xlabel("Epoch")
            # plt.ylabel("MSE")
            # plt.title("Evaluation MSE")
            # plt.grid(True)
            # plt.savefig(os.path.join(eval_dir, "eval_mses.png"))
            # plt.close()


            # plt.figure()
            # plt.plot(eval_lpipses)
            # plt.xlabel("Epoch")
            # plt.ylabel("LPIPS")
            # plt.title("Evaluation LPIPS")
            # plt.grid(True)
            # plt.savefig(os.path.join(eval_dir, "eval_lpipses.png"))
            # plt.close()


            # plt.figure()
            # plt.plot(eval_dc_maes)
            # plt.xlabel("Epoch")
            # plt.ylabel("k-space MAE")
            # plt.title("Evaluation Data Consistency (MAE)")
            # plt.grid(True)
            # plt.savefig(os.path.join(eval_dir, "eval_dc_maes.png"))
            # plt.close()


            # plt.figure()
            # plt.plot(eval_curve_corrs)
            # plt.xlabel("Epoch")
            # plt.ylabel("Pearson Correlation of Tumor Enhancement Curve")
            # plt.title("Pearson Correlation")
            # plt.grid(True)
            # plt.savefig(os.path.join(eval_dir, "eval_curve_correlations.png"))
            # plt.close()

            # plot evaluation metrics in one figure
            # Set the seaborn style
            sns.set_style("whitegrid")

            # Create a figure and a set of subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Evaluation Metrics Over Epochs ({N_spokes_eval} spokes/frame)', fontsize=20)

            sns.lineplot(x=range(len(eval_ssims)), y=eval_ssims, ax=axes[0, 0])
            axes[0, 0].set_title("Evaluation SSIM")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("SSIM")

            sns.lineplot(x=range(len(eval_psnrs)), y=eval_psnrs, ax=axes[0, 1])
            axes[0, 1].set_title("Evaluation PSNR")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("PSNR")

            sns.lineplot(x=range(len(eval_mses)), y=eval_mses, ax=axes[0, 2])
            axes[0, 2].set_title("Evaluation Image MSE")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("MSE")

            sns.lineplot(x=range(len(eval_lpipses)), y=eval_lpipses, ax=axes[1, 0])
            axes[1, 0].set_title("Evaluation LPIPS")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("LPIPS")

            sns.lineplot(x=range(len(eval_dc_maes)), y=eval_dc_maes, ax=axes[1, 1])
            axes[1, 1].set_title("Evaluation k-space MAE")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("MAE")

            sns.lineplot(x=range(len(eval_curve_corrs)), y=eval_curve_corrs, ax=axes[1, 2])
            axes[1, 2].set_title("Tumor Enhancement Curve Correlation")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("Pearson Correlation Coefficient")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, "eval_metrics.png"))
            plt.close()


            plt.figure()
            plt.plot(eval_dc_mses)
            plt.xlabel("Epoch")
            plt.ylabel("k-space MSE")
            plt.title("Evaluation Data Consistency (MSE)")
            plt.grid(True)
            plt.savefig(os.path.join(eval_dir, "eval_dc_mses.png"))
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
        print(f"Recon LPIPS: {epoch_eval_lpips:.4f} ± {np.std(epoch_eval_lpipses):.4f}")
        print(f"Recon DC MSE: {epoch_eval_dc_mse:.4f} ± {np.std(epoch_eval_dc_mses):.4f}")
        print(f"Recon DC MAE: {epoch_eval_dc_mae:.4f} ± {np.std(epoch_eval_dc_maes):.4f}")
        print(f"Recon Enhancement Curve Correlation: {epoch_eval_curve_corr:.4f} ± {np.std(epoch_eval_curve_corrs):.4f}")
        print(f"GRASP SSIM: {np.mean(grasp_ssims):.4f} ± {np.std(grasp_ssims):.4f}")
        print(f"GRASP PSNR: {np.mean(grasp_psnrs):.4f} ± {np.std(grasp_psnrs):.4f}")
        print(f"GRASP MSE: {np.mean(grasp_mses):.4f} ± {np.std(grasp_mses):.4f}")
        print(f"GRASP LPIPS: {np.mean(grasp_lpipses):.4f} ± {np.std(grasp_lpipses):.4f}")
        print(f"GRASP DC MSE: {np.mean(grasp_dc_mses):.6f} ± {np.std(grasp_dc_mses):.4f}")
        print(f"GRASP DC MAE: {np.mean(grasp_dc_maes):.6f} ± {np.std(grasp_dc_maes):.4f}")
        print(f"GRASP Enhancement Curve Correlation: {np.mean(grasp_curve_corrs):.6f} ± {np.std(grasp_curve_corrs):.4f}")


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
    eval_lpipses=eval_lpipses,
    eval_dc_mses=eval_dc_mses,
    eval_dc_maes=eval_dc_maes,
    eval_curve_corrs=eval_curve_corrs,
)
model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
save_checkpoint(model, optimizer, epochs + 1, train_curves, val_curves, eval_curves, model_save_path)
print(f'Model saved to {model_save_path}')


# save final evaluation metrics
metrics_path = os.path.join(eval_dir, "eval_metrics.csv")

with open(metrics_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Recon', 'SSIM', 'PSNR', 'MSE', 'LPIPS', 'DC MSE', 'DC MAE', 'EC Correlation'])
    writer.writerow(['DL', 
                     f'{epoch_eval_ssim:.4f} ± {np.std(epoch_eval_ssims):.4f}', 
                     f'{epoch_eval_psnr:.4f} ± {np.std(epoch_eval_psnrs):.4f}', 
                     f'{epoch_eval_mse:.4f} ± {np.std(epoch_eval_mses):.4f}',
                     f'{epoch_eval_lpips:.4f} ± {np.std(epoch_eval_lpipses):.4f}',  
                     f'{epoch_eval_dc_mse:.4f} ± {np.std(epoch_eval_dc_mses):.4f}', 
                     f'{epoch_eval_dc_mae:.4f} ± {np.std(epoch_eval_dc_maes):.4f}', 
                     f'{epoch_eval_curve_corr:.4f} ± {np.std(epoch_eval_curve_corrs):.4f}'])
    writer.writerow(['GRASP', 
                     f'{np.mean(grasp_ssims):.4f} ± {np.std(grasp_ssims):.4f}', 
                     f'{np.mean(grasp_psnrs):.4f} ± {np.std(grasp_psnrs):.4f}', 
                     f'{np.mean(grasp_mses):.4f} ± {np.std(grasp_mses):.4f}', 
                     f'{np.mean(grasp_lpipses):.4f} ± {np.std(grasp_lpipses):.4f}', 
                     f'{np.mean(grasp_dc_mses):.4f} ± {np.std(grasp_dc_mses):.4f}', 
                     f'{np.mean(grasp_dc_maes):.4f} ± {np.std(grasp_dc_maes):.4f}', 
                     f'{np.mean(grasp_curve_corrs):.4f} ± {np.std(grasp_curve_corrs):.4f}'])



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
        "spokes_per_frame": 2,
        "num_frames": 22, # 4 * 22 = 88 total spokes
        "slice": slice(0, 22), # The entire 22-frame duration
        "description": "Stress test: max temporal points, 2 spokes"
    },
    {
        "spokes_per_frame": 4,
        "num_frames": 22, # 4 * 22 = 88 total spokes
        "slice": slice(0, 22), # The entire 22-frame duration
        "description": "Stress test: max temporal points, 4 spokes"
    },
    {
        "spokes_per_frame": 8,
        "num_frames": 22, # 8 * 22 = 176 total spokes
        "slice": slice(0, 22), # The entire 22-frame duration
        "description": "Stress test: max temporal points, 8 spokes"
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




with torch.no_grad():

    spf_recon_ssim = {}
    spf_recon_psnr = {}
    spf_recon_mse = {}
    spf_recon_lpips = {}
    spf_recon_dc_mse = {}
    spf_recon_dc_mae = {}
    spf_recon_corr = {}
    spf_grasp_ssim = {}
    spf_grasp_psnr = {}
    spf_grasp_mse = {}
    spf_grasp_lpips = {}
    spf_grasp_dc_mse = {}
    spf_grasp_dc_mae = {}
    spf_grasp_corr = {}


    print("--- Running Stress Test Evaluation (Budget: 176 spokes) ---")
    for eval_config in STRESS_TEST_PLAN:

        stress_test_ssims = []
        stress_test_psnrs = []
        stress_test_mses = []
        stress_test_lpipses = []
        stress_test_dc_mses = []
        stress_test_dc_maes = []
        stress_test_corrs = []
        stress_test_grasp_ssims = []
        stress_test_grasp_psnrs = []
        stress_test_grasp_mses = []
        stress_test_grasp_lpipses = []
        stress_test_grasp_dc_mses = []
        stress_test_grasp_dc_maes = []
        stress_test_grasp_corrs = []

        spokes = eval_config["spokes_per_frame"]
        time_slice = eval_config["slice"]
        num_frames = eval_config["num_frames"]

        eval_spf_dataset.spokes_per_frame = spokes
        eval_spf_dataset.window = time_slice
        eval_spf_dataset.num_frames = num_frames


        for csmap, ground_truth, grasp_img, mask, grasp_path in tqdm(eval_spf_loader, desc="Variable Spokes Per Frame Evaluation"):


            csmap = csmap.squeeze(0).to(device)   # Remove batch dim
            ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

            print("ground truth shape: ", ground_truth.shape)
            print("csmap shape: ", csmap.shape)


            # SIMULATE KSPACE
            ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, spokes, num_frames)
            physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))

            sim_kspace = physics(False, ground_truth, csmap)

            kspace = sim_kspace.squeeze(0).to(device) # Remove batch dim

            # calculate acceleration factor
            acceleration = torch.tensor([N_full / int(spokes)], dtype=torch.float, device=device)

            if config['model']['encode_acceleration']:
                acceleration_encoding = acceleration
            else: 
                acceleration_encoding = None


            # check if GRASP image exists or if we need to perform GRASP recon
            if type(grasp_img) is int or len(grasp_img.shape) == 1:
                print(f"No GRASP file found, performing reconstruction with {spokes} spokes/frame and {num_frames} frames.")

                grasp_img = GRASPRecon(csmap, sim_kspace, spokes, num_frames, grasp_path[0])

                grasp_recon_torch = torch.from_numpy(grasp_img).permute(2, 0, 1) # T, H, W
                grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

                grasp_img = torch.flip(grasp_recon_torch, dims=[-3])
                grasp_img = torch.rot90(grasp_img, k=3, dims=[-3,-1]).unsqueeze(0)

            grasp_img = grasp_img.to(device)

            
            x_recon, *_ = model(
                kspace.to(device), physics, csmap, acceleration_encoding, epoch=None, norm=config['model']['norm']
            )

            ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
            ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')


            ## Evaluation
            ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr = eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img, acceleration, eval_dir, f"{spokes}spf", device)
            stress_test_ssims.append(ssim)
            stress_test_psnrs.append(psnr)
            stress_test_mses.append(mse)
            stress_test_lpipses.append(lpips)
            stress_test_dc_mses.append(dc_mse)
            stress_test_dc_maes.append(dc_mae)

            if recon_corr is not None:
                stress_test_corrs.append(recon_corr)
                stress_test_grasp_corrs.append(grasp_corr)


            ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp = eval_grasp(kspace, csmap, ground_truth, grasp_img, physics, device, eval_dir)
            stress_test_grasp_ssims.append(ssim_grasp)
            stress_test_grasp_psnrs.append(psnr_grasp)
            stress_test_grasp_mses.append(mse_grasp)
            stress_test_grasp_lpipses.append(lpips_grasp)
            stress_test_grasp_dc_mses.append(dc_mse_grasp)
            stress_test_grasp_dc_maes.append(dc_mae_grasp)


            spf_recon_ssim[spokes] = np.mean(stress_test_ssims)
            spf_recon_psnr[spokes] = np.mean(stress_test_psnrs)
            spf_recon_mse[spokes] = np.mean(stress_test_mses)
            spf_recon_lpips[spokes] = np.mean(stress_test_lpipses)
            spf_recon_dc_mse[spokes] = np.mean(stress_test_dc_mses)
            spf_recon_dc_mae[spokes] = np.mean(stress_test_dc_maes)
            spf_recon_corr[spokes] = np.mean(stress_test_corrs)

            spf_grasp_ssim[spokes] = np.mean(stress_test_grasp_ssims)
            spf_grasp_psnr[spokes] = np.mean(stress_test_grasp_psnrs)
            spf_grasp_mse[spokes] = np.mean(stress_test_grasp_mses)
            spf_grasp_lpips[spokes] = np.mean(stress_test_grasp_lpipses)
            spf_grasp_dc_mse[spokes] = np.mean(stress_test_grasp_dc_mses)
            spf_grasp_dc_mae[spokes] = np.mean(stress_test_grasp_dc_maes)
            spf_grasp_corr[spokes] = np.mean(stress_test_grasp_corrs)


        # plot an example image comparison at the given temporal resolution
        # ground_truth = ground_truth.squeeze()
        # ground_truth = torch.abs(ground_truth[0] + 1j * ground_truth[1])
        # ground_truth = rearrange(ground_truth, 't h w -> h w t')

        # x_recon = x_recon.squeeze()
        # x_recon = torch.abs(x_recon[0] + 1j * x_recon[1])

        # grasp_img = grasp_img.squeeze()
        # grasp_img = torch.abs(grasp_img[0] + 1j * grasp_img[1])
        # grasp_img = rearrange(grasp_img, 'h t w -> h w t')

        
        # filename = os.path.join(eval_dir, f'{spokes}spf_time_series.png')
        # plot_time_series(ground_truth.cpu().numpy(), x_recon.cpu().numpy(), grasp_img.cpu().numpy(), filename)



        # Save Results
        spf_metrics_path = os.path.join(eval_dir, "eval_metrics.csv")
        with open(spf_metrics_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', "LPIPS", 'DC MSE', 'DC MAE', 'EC Correlation'])

            writer.writerow(['DL', spokes, 
            f'{np.mean(stress_test_ssims):.4f} ± {np.std(stress_test_ssims):.4f}', 
            f'{np.mean(stress_test_psnrs):.4f} ± {np.std(stress_test_psnrs):.4f}', 
            f'{np.mean(stress_test_mses):.4f} ± {np.std(stress_test_mses):.4f}', 
            f'{np.mean(stress_test_lpipses):.4f} ± {np.std(stress_test_lpipses):.4f}', 
            f'{np.mean(stress_test_dc_mses):.4f} ± {np.std(stress_test_dc_mses):.4f}',
            f'{np.mean(stress_test_dc_maes):.4f} ± {np.std(stress_test_dc_maes):.4f}',
            f'{np.mean(stress_test_corrs):.4f} ± {np.std(stress_test_corrs):.4f}'
            ])

            writer.writerow(['GRASP', spokes, 
            f'{np.mean(stress_test_grasp_ssims):.4f} ± {np.std(stress_test_grasp_ssims):.4f}', 
            f'{np.mean(stress_test_grasp_psnrs):.4f} ± {np.std(stress_test_grasp_psnrs):.4f}', 
            f'{np.mean(stress_test_grasp_mses):.4f} ± {np.std(stress_test_grasp_mses):.4f}', 
            f'{np.mean(stress_test_grasp_lpipses):.4f} ± {np.std(stress_test_grasp_lpipses):.4f}', 
            f'{np.mean(stress_test_grasp_dc_mses):.4f} ± {np.std(stress_test_grasp_dc_mses):.4f}',
            f'{np.mean(stress_test_grasp_dc_maes):.4f} ± {np.std(stress_test_grasp_dc_maes):.4f}',
            f'{np.mean(stress_test_grasp_corrs):.4f} ± {np.std(stress_test_grasp_corrs):.4f}',
            ])



    print("--- Running Main Evaluation (Budget: 320 spokes) ---")
    for eval_config in MAIN_EVALUATION_PLAN:

        spf_eval_ssims = []
        spf_eval_psnrs = []
        spf_eval_mses = []
        spf_eval_lpipses = []
        spf_eval_dc_mses = []
        spf_eval_dc_maes = []
        spf_eval_curve_corrs = []
        spf_grasp_ssims = []
        spf_grasp_psnrs = []
        spf_grasp_mses = []
        spf_grasp_lpipses = []
        spf_grasp_dc_mses = []
        spf_grasp_dc_maes = []
        spf_grasp_curve_corrs = []

        spokes = eval_config["spokes_per_frame"]
        time_slice = eval_config["slice"]
        num_frames = eval_config["num_frames"]

        eval_spf_dataset.spokes_per_frame = spokes
        eval_spf_dataset.window = time_slice
        eval_spf_dataset.num_frames = num_frames


        for csmap, ground_truth, grasp_img, mask, grasp_path in tqdm(eval_spf_loader, desc="Variable Spokes Per Frame Evaluation"):


            csmap = csmap.squeeze(0).to(device)   # Remove batch dim
            ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

            # SIMULATE KSPACE
            ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, spokes, num_frames)
            physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))

            sim_kspace = physics(False, ground_truth, csmap)

            kspace = sim_kspace.squeeze(0).to(device) # Remove batch dim
            
            # calculate acceleration factor
            acceleration = torch.tensor([N_full / int(spokes)], dtype=torch.float, device=device)

            if config['model']['encode_acceleration']:
                acceleration_encoding = acceleration
            else: 
                acceleration_encoding = None


            # check if GRASP image exists or if we need to perform GRASP recon
            if type(grasp_img) is int or len(grasp_img.shape) == 1:
                print(f"No GRASP file found, performing reconstruction with {spokes} spokes/frame and {num_frames} frames.")

                grasp_img = GRASPRecon(csmap, sim_kspace, spokes, num_frames, grasp_path[0])

                grasp_recon_torch = torch.from_numpy(grasp_img).permute(2, 0, 1) # T, H, W
                grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

                grasp_img = torch.flip(grasp_recon_torch, dims=[-3])
                grasp_img = torch.rot90(grasp_img, k=3, dims=[-3,-1]).unsqueeze(0)

            grasp_img = grasp_img.to(device)


            x_recon, *_ = model(
                kspace.to(device), physics, csmap, acceleration_encoding, epoch=None, norm=config['model']['norm']
            )

            ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
            ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')


            ## Evaluation
            ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr = eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img, acceleration, eval_dir, f'{spokes}spf', device)
            spf_eval_ssims.append(ssim)
            spf_eval_psnrs.append(psnr)
            spf_eval_mses.append(mse)
            spf_eval_lpipses.append(lpips)
            spf_eval_dc_mses.append(dc_mse)
            spf_eval_dc_maes.append(dc_mae)

            if recon_corr is not None:
                spf_eval_curve_corrs.append(recon_corr)
                spf_grasp_curve_corrs.append(grasp_corr)


            ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp = eval_grasp(kspace, csmap, ground_truth, grasp_img, physics, device, eval_dir)
            spf_grasp_ssims.append(ssim_grasp)
            spf_grasp_psnrs.append(psnr_grasp)
            spf_grasp_mses.append(mse_grasp)
            spf_grasp_lpipses.append(lpips_grasp)
            spf_grasp_dc_mses.append(dc_mse_grasp)
            spf_grasp_dc_maes.append(dc_mae_grasp)


        # plot an example image comparison at the given temporal resolution
        # ground_truth = ground_truth.squeeze()
        # ground_truth = torch.abs(ground_truth[0] + 1j * ground_truth[1])
        # ground_truth = rearrange(ground_truth, 't h w -> h w t')

        # x_recon = x_recon.squeeze()
        # x_recon = torch.abs(x_recon[0] + 1j * x_recon[1])

        # grasp_img = grasp_img.squeeze()
        # grasp_img = torch.abs(grasp_img[0] + 1j * grasp_img[1])
        # grasp_img = rearrange(grasp_img, 'h t w -> h w t')

        # filename = os.path.join(eval_dir, f'{spokes}spf_time_series.png')
        # plot_time_series(ground_truth.cpu().numpy(), x_recon.cpu().numpy(), grasp_img.cpu().numpy(), filename)


        spf_recon_ssim[spokes] = np.mean(spf_eval_ssims)
        spf_recon_psnr[spokes] = np.mean(spf_eval_psnrs)
        spf_recon_mse[spokes] = np.mean(spf_eval_mses)
        spf_recon_lpips[spokes] = np.mean(spf_eval_lpipses)
        spf_recon_dc_mse[spokes] = np.mean(spf_eval_dc_mses)
        spf_recon_dc_mae[spokes] = np.mean(spf_eval_dc_maes)
        spf_recon_corr[spokes] = np.mean(spf_eval_curve_corrs)

        spf_grasp_ssim[spokes] = np.mean(spf_grasp_ssims)
        spf_grasp_psnr[spokes] = np.mean(spf_grasp_psnrs)
        spf_grasp_mse[spokes] = np.mean(spf_grasp_mses)
        spf_grasp_lpips[spokes] = np.mean(spf_grasp_lpipses)
        spf_grasp_dc_mse[spokes] = np.mean(spf_grasp_dc_mses)
        spf_grasp_dc_mae[spokes] = np.mean(spf_grasp_dc_maes)
        spf_grasp_corr[spokes] = np.mean(spf_grasp_curve_corrs)


        # Save Results
        spf_metrics_path = os.path.join(eval_dir, "eval_metrics.csv")
        with open(spf_metrics_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', 'LPIPS', 'DC MSE', 'DC MAE', 'EC Correlation'])

            writer.writerow(['DL', spokes, 
            f'{np.mean(spf_eval_ssims):.4f} ± {np.std(spf_eval_ssims):.4f}', 
            f'{np.mean(spf_eval_psnrs):.4f} ± {np.std(spf_eval_psnrs):.4f}', 
            f'{np.mean(spf_eval_mses):.4f} ± {np.std(spf_eval_mses):.4f}', 
            f'{np.mean(spf_eval_lpipses):.4f} ± {np.std(spf_eval_lpipses):.4f}', 
            f'{np.mean(spf_eval_dc_mses):.4f} ± {np.std(spf_eval_dc_mses):.4f}',
            f'{np.mean(spf_eval_dc_maes):.4f} ± {np.std(spf_eval_dc_maes):.4f}',
            f'{np.mean(spf_eval_curve_corrs):.4f} ± {np.std(spf_eval_curve_corrs):.4f}'
            ])

            writer.writerow(['GRASP', spokes, 
            f'{np.mean(spf_grasp_ssims):.4f} ± {np.std(spf_grasp_ssims):.4f}', 
            f'{np.mean(spf_grasp_psnrs):.4f} ± {np.std(spf_grasp_psnrs):.4f}', 
            f'{np.mean(spf_grasp_mses):.4f} ± {np.std(spf_grasp_mses):.4f}', 
            f'{np.mean(spf_grasp_lpipses):.4f} ± {np.std(spf_grasp_lpipses):.4f}', 
            f'{np.mean(spf_grasp_dc_mses):.4f} ± {np.std(spf_grasp_dc_mses):.4f}',
            f'{np.mean(spf_grasp_dc_maes):.4f} ± {np.std(spf_grasp_dc_maes):.4f}',
            f'{np.mean(spf_grasp_curve_corrs):.4f} ± {np.std(spf_grasp_curve_corrs):.4f}'
            ])

    

    
# Plot Metrics vs Spokes Per Frame
# plt.figure()
# plt.plot(list(spf_recon_ssim.keys()), list(spf_recon_ssim.values()), label="DL Recon", marker='o')
# plt.plot(list(spf_grasp_ssim.keys()), list(spf_grasp_ssim.values()), label="GRASP Recon", marker='o')
# plt.xlabel("Spokes per Frame")
# plt.ylabel("SSIM")
# plt.title("Evaluation SSIM vs Spokes per Frame")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "spf_eval_ssim.png"))
# plt.close()


# plt.figure()
# plt.plot(list(spf_recon_psnr.keys()), list(spf_recon_psnr.values()), label="DL Recon", marker='o')
# plt.plot(list(spf_grasp_psnr.keys()), list(spf_grasp_psnr.values()), label="GRASP Recon", marker='o')
# plt.xlabel("Spokes per Frame")
# plt.ylabel("PSNR")
# plt.title("Evaluation PSNR vs Spokes per Frame")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "spf_eval_psnr.png"))
# plt.close()


# plt.figure()
# plt.plot(list(spf_recon_mse.keys()), list(spf_recon_mse.values()), label="DL Recon", marker='o')
# plt.plot(list(spf_grasp_mse.keys()), list(spf_grasp_mse.values()), label="GRASP Recon", marker='o')
# plt.xlabel("Spokes per Frame")
# plt.ylabel("MSE")
# plt.title("Evaluation Image MSE vs Spokes per Frame")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "spf_eval_mse.png"))
# plt.close()

# plt.figure()
# plt.plot(list(spf_recon_lpips.keys()), list(spf_recon_lpips.values()), label="DL Recon", marker='o')
# plt.plot(list(spf_grasp_lpips.keys()), list(spf_grasp_lpips.values()), label="GRASP Recon", marker='o')
# plt.xlabel("Spokes per Frame")
# plt.ylabel("LPIPS")
# plt.title("Evaluation LPIPS vs Spokes per Frame")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "spf_eval_lpips.png"))
# plt.close()


# plt.figure()
# plt.plot(list(spf_recon_dc_mse.keys()), list(spf_recon_dc_mse.values()), label="DL Recon", marker='o')
# plt.plot(list(spf_grasp_dc_mse.keys()), list(spf_grasp_dc_mse.values()), label="GRASP Recon", marker='o')
# plt.xlabel("Spokes per Frame")
# plt.ylabel("k-space MSE")
# plt.title("Data Consistency Evaluation (MSE) vs Spokes per Frame")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "spf_eval_dc_mse.png"))
# plt.close()

# plt.figure()
# plt.plot(list(spf_recon_dc_mae.keys()), list(spf_recon_dc_mae.values()), label="DL Recon", marker='o')
# plt.plot(list(spf_grasp_dc_mae.keys()), list(spf_grasp_dc_mae.values()), label="GRASP Recon", marker='o')
# plt.xlabel("Spokes per Frame")
# plt.ylabel("k-space MAE")
# plt.title("Data Consistency Evaluation (MAE) vs Spokes per Frame")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "spf_eval_dc_mae.png"))
# plt.close()


# plt.figure()
# plt.plot(list(spf_recon_corr.keys()), list(spf_recon_corr.values()), label="DL Recon", marker='o')
# plt.plot(list(spf_grasp_corr.keys()), list(spf_grasp_corr.values()), label="GRASP Recon", marker='o')
# plt.xlabel("Spokes per Frame")
# plt.ylabel("Pearson Correlation")
# plt.title("Pearson Correlation of Tumor Enhancement Curve vs Spokes per Frame")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "spf_eval_curve_correlations.png"))
# plt.close()


# plot variable spokes/frame evaluation metrics in one figure
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))


sns.lineplot(x=list(spf_recon_ssim.keys()), 
             y=list(spf_recon_ssim.values()), 
             label="DL Recon", 
             marker='o',
             ax=axes[0, 0])

sns.lineplot(x=list(spf_grasp_ssim.keys()), 
             y=list(spf_grasp_ssim.values()), 
             label="Standard Recon", 
             marker='o',
             ax=axes[0, 0])

axes[0, 0].set_title("Evaluation SSIM vs Spokes/Frame")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("SSIM")


sns.lineplot(x=list(spf_recon_psnr.keys()), 
             y=list(spf_recon_psnr.values()), 
             label="DL Recon", 
             marker='o',
             ax=axes[0, 1])

sns.lineplot(x=list(spf_grasp_psnr.keys()), 
             y=list(spf_grasp_psnr.values()), 
             label="Standard Recon", 
             marker='o',
             ax=axes[0, 1])
axes[0, 1].set_title("Evaluation PSNR vs Spokes/Frame")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("PSNR")


sns.lineplot(x=list(spf_recon_mse.keys()), 
             y=list(spf_recon_mse.values()), 
             label="DL Recon", 
             marker='o',
             ax=axes[0, 2])

sns.lineplot(x=list(spf_grasp_mse.keys()), 
             y=list(spf_grasp_mse.values()), 
             label="Standard Recon", 
             marker='o',
             ax=axes[0, 2])
axes[0, 2].set_title("Evaluation Image MSE vs Spokes/Frame")
axes[0, 2].set_xlabel("Epoch")
axes[0, 2].set_ylabel("MSE")


sns.lineplot(x=list(spf_recon_lpips.keys()), 
             y=list(spf_recon_lpips.values()), 
             label="DL Recon", 
             marker='o',
             ax=axes[1, 0])

sns.lineplot(x=list(spf_grasp_lpips.keys()), 
             y=list(spf_grasp_lpips.values()), 
             label="Standard Recon", 
             marker='o',
             ax=axes[1, 0])
axes[1, 0].set_title("Evaluation LPIPS vs Spokes/Frame")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("LPIPS")

sns.lineplot(x=list(spf_recon_dc_mae.keys()), 
             y=list(spf_recon_dc_mae.values()), 
             label="DL Recon", 
             marker='o',
             ax=axes[1, 1])

sns.lineplot(x=list(spf_grasp_dc_mae.keys()), 
             y=list(spf_grasp_dc_mae.values()), 
             label="Standard Recon", 
             marker='o',
             ax=axes[1, 1])
axes[1, 1].set_title("Evaluation k-space MAE vs Spokes/Frame")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("MAE")

sns.lineplot(x=list(spf_recon_corr.keys()), 
             y=list(spf_recon_corr.values()), 
             label="DL Recon", 
             marker='o',
             ax=axes[1, 2])

sns.lineplot(x=list(spf_grasp_corr.keys()), 
             y=list(spf_grasp_corr.values()), 
             label="Standard Recon", 
             marker='o',
             ax=axes[1, 2])
axes[1, 2].set_title("Tumor Enhancement Curve Correlation vs Spokes/Frame")
axes[1, 2].set_xlabel("Epoch")
axes[1, 2].set_ylabel("Pearson Correlation Coefficient")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spf_eval_metrics.png"))
plt.close()