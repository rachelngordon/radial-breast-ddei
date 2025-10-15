import os
import matplotlib.pyplot as plt
import torch
import yaml
from dataloader import SimulatedSPFDataset
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from radial_lsfp import MCNUFFT
from utils import prep_nufft, to_torch_complex, GRASPRecon, sliding_window_inference, load_checkpoint
from eval import eval_grasp, eval_sample, calc_dc
import csv
import seaborn as sns
import h5py
import argparse
import json
from lsfpnet_encoding import LSFPNet, ArtifactRemovalLSFPNet
import math
import sigpy as sp
from raw_kspace_eval import eval_raw_kspace


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train ReconResNet model.")
parser.add_argument(
    "--exp_name", type=str, required=True, help="Name of the experiment"
)
args = parser.parse_args()

# Load the configuration file
with open(f"output/{args.exp_name}/config.yaml", "r") as file:
    config = yaml.safe_load(file)

split_file = config["data"]["split_file"]
model_type = config["model"]["name"]
device = torch.device(config["training"]["device"])
H, W = config["data"]["height"], config["data"]["width"]
N_time, N_samples, N_coils = (
    config["data"]["timeframes"],
    config["data"]["samples"],
    config["data"]["coils"]
)
Ng = config["data"]["fpg"] 

total_spokes = config["data"]["total_spokes"]

N_spokes = int(total_spokes / N_time)
N_full = config['data']['height'] * math.pi / 2

N_slices = 83

eval_chunk_size = config["evaluation"]["chunk_size"]
eval_chunk_overlap = config["evaluation"]["chunk_overlap"]


# define model
initial_lambdas = {'lambda_L': config['model']['lambda_L'], 
                    'lambda_S': config['model']['lambda_S'], 
                    'lambda_spatial_L': config['model']['lambda_spatial_L'],
                    'lambda_spatial_S': config['model']['lambda_spatial_S'],
                    'gamma': config['model']['gamma'],
                    'lambda_step': config['model']['lambda_step']}

output_dir = os.path.join(config["experiment"]["output_dir"], args.exp_name)
block_dir = os.path.join(output_dir, "block_outputs")
eval_dir = os.path.join(output_dir, "eval_results")
data_dir = config["data"]["root_dir"]

lsfp_backbone = LSFPNet(LayerNo=config["model"]["num_layers"], 
                        lambdas=initial_lambdas, 
                        channels=config['model']['channels'],
                        style_dim=config['model']['style_dim'],
                        svd_mode=config['model']['svd_mode'],
                        use_lowk_dc=config['model']['use_lowk_dc'],
                        lowk_frac=config['model']['lowk_frac'],
                        lowk_alpha=config['model']['lowk_alpha'],
                        film_bounded=config['model']['film_bounded'],
                        film_gain=config['model']['film_gain'],
                        film_identity_init=config['model']['film_identity_init'],
                        svd_noise_std=config['model']['svd_noise_std'],
                        film_L=config['model']['film_L'],
                        )
    
if config['model']['encode_acceleration'] and config['model']['encode_time_index']:
    model = ArtifactRemovalLSFPNet(lsfp_backbone, block_dir, channels=2).to(device)
else:
    model = ArtifactRemovalLSFPNet(lsfp_backbone, block_dir, channels=1).to(device)



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["model"]["optimizer"]["lr"],
    betas=(config["model"]["optimizer"]["b1"], config["model"]["optimizer"]["b2"]),
    eps=config["model"]["optimizer"]["eps"],
    weight_decay=config["model"]["optimizer"]["weight_decay"],
)

checkpoint_file = f'output/{args.exp_name}/{args.exp_name}_model.pth'
model, optimizer, start_epoch, target_w_ei, step0_train_ei_loss, epoch_train_mc_loss, train_curves, val_curves, eval_curves = load_checkpoint(model, optimizer, checkpoint_file)




# load data
with open(split_file, "r") as fp:
    splits = json.load(fp)


val_dro_patient_ids = splits["val_dro"]
val_patient_ids = splits["val"]


num_slices_to_eval = 1



# EVALUATE WITH VARIABLE SPOKES PER FRAME

MAIN_EVALUATION_PLAN = [
    {
        "spokes_per_frame": 8,
        "num_frames": 36, # 8 * 36 = 288 total spokes
        "description": "High temporal resolution"
    },
    {
        "spokes_per_frame": 16,
        "num_frames": 18, # 16 * 18 = 288 total spokes
        "description": "High temporal resolution"
    },
    {
        "spokes_per_frame": 24,
        "num_frames": 12, # 24 * 12 = 288 total spokes
        "description": "Good temporal resolution"
    },
    {
        "spokes_per_frame": 32,
        "num_frames": 8, # 36 * 8 = 288 total spokes
        "description": "Standard temporal resolution"
    },
]



# --- Stress Test Plan ---
# Designed to push the limits with very few spokes per frame.
# This has a different (lower) total spoke budget.

STRESS_TEST_PLAN = [
    {
        "spokes_per_frame": 2,
        "num_frames": 144, # 2 * 144 = 288 total spokes
        "description": "Stress test: max temporal points, 2 spokes"
    },
    {
        "spokes_per_frame": 4,
        "num_frames": 72, # 4 * 72 = 288 total spokes
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
    shuffle=False,
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
    spf_raw_dc_mse = {}
    spf_raw_dc_mae = {}
    spf_raw_grasp_dc_mse = {}
    spf_raw_grasp_dc_mae = {}

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
        num_frames = eval_config["num_frames"]

        eval_spf_dataset.spokes_per_frame = spokes
        eval_spf_dataset.num_frames = num_frames
        eval_spf_dataset._update_sample_paths()


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

            if config['model']['encode_time_index'] == False:
                start_timepoint_index = None
            else:
                start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)


            # check if GRASP image exists or if we need to perform GRASP recon
            if type(grasp_img) is int or len(grasp_img.shape) == 1:
                print(f"No GRASP file found, performing reconstruction with {spokes} spokes/frame and {num_frames} frames.")

                grasp_img = GRASPRecon(csmap, sim_kspace, spokes, num_frames, grasp_path[0])

                grasp_recon_torch = torch.from_numpy(grasp_img).permute(2, 0, 1) # T, H, W
                grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

                grasp_img = torch.flip(grasp_recon_torch, dims=[-3])
                grasp_img = torch.rot90(grasp_img, k=3, dims=[-3,-1]).unsqueeze(0)

            grasp_img = grasp_img.to(device)

            if num_frames > eval_chunk_size:
                print("Performing sliding window eval...")
                x_recon, _ = sliding_window_inference(H, W, num_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, eval_chunk_size, eval_chunk_overlap, kspace, csmap, acceleration_encoding, start_timepoint_index, model, epoch=None, device=device)  
            else:
                x_recon, *_ = model(
                    kspace.to(device), physics, csmap, acceleration_encoding, start_timepoint_index, epoch=None, norm=config['model']['norm']
                )

            ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
            ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')


            ## Evaluation
            ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr = eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img, acceleration, int(spokes), eval_dir, f"{spokes}spf", device)
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



        # evaluate on raw k-space
        print(f"Evaluating on raw k-space with {num_slices_to_eval} slices...")
        raw_dc_mse, raw_dc_mae, raw_grasp_dc_mse, raw_grasp_dc_mae, raw_dc_std_mse, raw_dc_std_mae, raw_dc_std_grasp_mse, raw_dc_std_grasp_mae = eval_raw_kspace(num_slices_to_eval, val_patient_ids, data_dir, model, spokes, N_slices, num_frames, eval_chunk_size, eval_chunk_overlap, H, W, ktraj, dcomp, nufft_ob, adjnufft_ob, physics, acceleration_encoding, start_timepoint_index, device, output_dir, label=f"{spokes}spf")

        spf_raw_dc_mse[spokes] = raw_dc_mse
        spf_raw_dc_mae[spokes] = raw_dc_mae
        spf_raw_grasp_dc_mse[spokes] = raw_grasp_dc_mse
        spf_raw_grasp_dc_mae[spokes] = raw_grasp_dc_mae



        # # Calculate and store average validation evaluation metrics
        # if global_rank == 0 or not config['training']['multigpu']:
        #     epoch_eval_ssim = np.mean(epoch_eval_ssims)
        #     epoch_eval_psnr = np.mean(epoch_eval_psnrs)
        #     epoch_eval_mse = np.mean(epoch_eval_mses)
        #     epoch_eval_lpips = np.mean(epoch_eval_lpipses)
        #     epoch_eval_dc_mse = np.mean(epoch_eval_dc_mses)
        #     epoch_eval_dc_mae = np.mean(epoch_eval_dc_maes)
        #     epoch_eval_curve_corr = np.mean(epoch_eval_curve_corrs)

        #     eval_ssims.append(epoch_eval_ssim)
        #     eval_psnrs.append(epoch_eval_psnr)
        #     eval_mses.append(epoch_eval_mse)
        #     eval_lpipses.append(epoch_eval_lpips)
        #     eval_dc_mses.append(epoch_eval_dc_mse) 
        #     eval_dc_maes.append(epoch_eval_dc_mae) 
        #     eval_raw_dc_mses.append(raw_dc_mse) 
        #     eval_raw_dc_maes.append(raw_dc_mae)   
        #     eval_curve_corrs.append(epoch_eval_curve_corr)   
                

        # Save Results
        spf_metrics_path = os.path.join(eval_dir, "eval_metrics.csv")
        with open(spf_metrics_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', "LPIPS", 'DC MSE', 'DC MAE', 'EC Correlation'])

            csvwriter.writerow(['DL', spokes, 
            f'{np.mean(stress_test_ssims):.4f} ± {np.std(stress_test_ssims):.4f}', 
            f'{np.mean(stress_test_psnrs):.4f} ± {np.std(stress_test_psnrs):.4f}', 
            f'{np.mean(stress_test_mses):.4f} ± {np.std(stress_test_mses):.4f}', 
            f'{np.mean(stress_test_lpipses):.4f} ± {np.std(stress_test_lpipses):.4f}', 
            f'{np.mean(stress_test_dc_mses):.4f} ± {np.std(stress_test_dc_mses):.4f}',
            f'{np.mean(stress_test_dc_maes):.4f} ± {np.std(stress_test_dc_maes):.4f}',
            f'{raw_dc_mse:.4f} ± {raw_dc_std_mse:.4f}',
            f'{raw_dc_mae:.4f} ± {raw_dc_std_mae:.4f}',
            f'{np.mean(stress_test_corrs):.4f} ± {np.std(stress_test_corrs):.4f}'
            ])

            csvwriter.writerow(['GRASP', spokes, 
            f'{np.mean(stress_test_grasp_ssims):.4f} ± {np.std(stress_test_grasp_ssims):.4f}', 
            f'{np.mean(stress_test_grasp_psnrs):.4f} ± {np.std(stress_test_grasp_psnrs):.4f}', 
            f'{np.mean(stress_test_grasp_mses):.4f} ± {np.std(stress_test_grasp_mses):.4f}', 
            f'{np.mean(stress_test_grasp_lpipses):.4f} ± {np.std(stress_test_grasp_lpipses):.4f}', 
            f'{np.mean(stress_test_grasp_dc_mses):.4f} ± {np.std(stress_test_grasp_dc_mses):.4f}',
            f'{np.mean(stress_test_grasp_dc_maes):.4f} ± {np.std(stress_test_grasp_dc_maes):.4f}',
            f'{raw_grasp_dc_mse:.4f} ± {raw_dc_std_grasp_mse:.4f}',
            f'{raw_grasp_dc_mae:.4f} ± {raw_dc_std_grasp_mae:.4f}',
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
        num_frames = eval_config["num_frames"]

        eval_spf_dataset.spokes_per_frame = spokes
        eval_spf_dataset.num_frames = num_frames
        eval_spf_dataset._update_sample_paths()

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
            
            if config['model']['encode_time_index'] == False:
                start_timepoint_index = None
            else:
                start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)


            # check if GRASP image exists or if we need to perform GRASP recon
            if type(grasp_img) is int or len(grasp_img.shape) == 1:
                print(f"No GRASP file found, performing reconstruction with {spokes} spokes/frame and {num_frames} frames.")

                grasp_img = GRASPRecon(csmap, sim_kspace, spokes, num_frames, grasp_path[0])

                grasp_recon_torch = torch.from_numpy(grasp_img).permute(2, 0, 1) # T, H, W
                grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

                grasp_img = torch.flip(grasp_recon_torch, dims=[-3])
                grasp_img = torch.rot90(grasp_img, k=3, dims=[-3,-1]).unsqueeze(0)

            grasp_img = grasp_img.to(device)

            if num_frames > eval_chunk_size:
                print("Performing sliding window eval...")
                x_recon, _ = sliding_window_inference(H, W, num_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, eval_chunk_size, eval_chunk_overlap, kspace, csmap, acceleration_encoding, start_timepoint_index, model, epoch=None, device=device)  
            else:
                x_recon, *_ = model(
                kspace.to(device), physics, csmap, acceleration_encoding, start_timepoint_index, epoch=None, norm=config['model']['norm']
                )

            

            ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
            ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')


            ## Evaluation
            ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr = eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img, acceleration, int(spokes), eval_dir, f'{spokes}spf', device)
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


        # evaluate on raw k-space
        print(f"Evaluating on raw k-space with {num_slices_to_eval} slices...")
        raw_dc_mse, raw_dc_mae, raw_grasp_dc_mse, raw_grasp_dc_mae, raw_dc_std_mse, raw_dc_std_mae, raw_dc_std_grasp_mse, raw_dc_std_grasp_mae = eval_raw_kspace(num_slices_to_eval, val_patient_ids, data_dir, model, spokes, N_slices, num_frames, eval_chunk_size, eval_chunk_overlap, H, W, ktraj, dcomp, nufft_ob, adjnufft_ob, physics, acceleration_encoding, start_timepoint_index, device, output_dir, label=f"{spokes}spf")

        spf_raw_dc_mse[spokes] = raw_dc_mse
        spf_raw_dc_mae[spokes] = raw_dc_mae
        spf_raw_grasp_dc_mse[spokes] = raw_grasp_dc_mse
        spf_raw_grasp_dc_mae[spokes] = raw_grasp_dc_mae
        
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

            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', 'LPIPS', 'DC MSE', 'DC MAE', 'EC Correlation'])

            csvwriter.writerow(['DL', spokes, 
            f'{np.mean(spf_eval_ssims):.4f} ± {np.std(spf_eval_ssims):.4f}', 
            f'{np.mean(spf_eval_psnrs):.4f} ± {np.std(spf_eval_psnrs):.4f}', 
            f'{np.mean(spf_eval_mses):.4f} ± {np.std(spf_eval_mses):.4f}', 
            f'{np.mean(spf_eval_lpipses):.4f} ± {np.std(spf_eval_lpipses):.4f}', 
            f'{np.mean(spf_eval_dc_mses):.4f} ± {np.std(spf_eval_dc_mses):.4f}',
            f'{np.mean(spf_eval_dc_maes):.4f} ± {np.std(spf_eval_dc_maes):.4f}',
            f'{raw_dc_mse:.4f} ± {raw_dc_std_mse:.4f}',
            f'{raw_dc_mae:.4f} ± {raw_dc_std_mae:.4f}',
            f'{np.mean(spf_eval_curve_corrs):.4f} ± {np.std(spf_eval_curve_corrs):.4f}'
            ])

            csvwriter.writerow(['GRASP', spokes, 
            f'{np.mean(spf_grasp_ssims):.4f} ± {np.std(spf_grasp_ssims):.4f}', 
            f'{np.mean(spf_grasp_psnrs):.4f} ± {np.std(spf_grasp_psnrs):.4f}', 
            f'{np.mean(spf_grasp_mses):.4f} ± {np.std(spf_grasp_mses):.4f}', 
            f'{np.mean(spf_grasp_lpipses):.4f} ± {np.std(spf_grasp_lpipses):.4f}', 
            f'{np.mean(spf_grasp_dc_mses):.4f} ± {np.std(spf_grasp_dc_mses):.4f}',
            f'{np.mean(spf_grasp_dc_maes):.4f} ± {np.std(spf_grasp_dc_maes):.4f}',
            f'{raw_grasp_dc_mse:.4f} ± {raw_dc_std_grasp_mse:.4f}',
            f'{raw_grasp_dc_mae:.4f} ± {raw_dc_std_grasp_mae:.4f}',
            f'{np.mean(spf_grasp_curve_corrs):.4f} ± {np.std(spf_grasp_curve_corrs):.4f}'
            ])

    

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
axes[0, 0].set_xlabel("Spokes per Frame")
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
axes[0, 1].set_xlabel("Spokes per Frame")
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
axes[0, 2].set_xlabel("Spokes per Frame")
axes[0, 2].set_ylabel("MSE")


# sns.lineplot(x=list(spf_recon_lpips.keys()), 
#             y=list(spf_recon_lpips.values()), 
#             label="DL Recon", 
#             marker='o',
#             ax=axes[1, 0])

# sns.lineplot(x=list(spf_grasp_lpips.keys()), 
#             y=list(spf_grasp_lpips.values()), 
#             label="Standard Recon", 
#             marker='o',
#             ax=axes[1, 0])
# axes[1, 0].set_title("Evaluation LPIPS vs Spokes/Frame")
# axes[1, 0].set_xlabel("Spokes per Frame")
# axes[1, 0].set_ylabel("LPIPS")

sns.lineplot(x=list(spf_raw_dc_mse.keys()), 
    y=list(spf_raw_dc_mse.values()), 
    label="DL Recon", 
    marker='o',
    ax=axes[1, 0])

sns.lineplot(x=list(spf_grasp_lpips.keys()), 
            y=list(spf_grasp_lpips.values()), 
            label="Standard Recon", 
            marker='o',
            ax=axes[1, 0])
axes[1, 0].set_title("Evaluation Raw k-space MAE vs Spokes/Frame")
axes[1, 0].set_xlabel("Spokes per Frame")
axes[1, 0].set_ylabel("MAE")

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
axes[1, 1].set_title("Evaluation Simulated k-space MAE vs Spokes/Frame")
axes[1, 1].set_xlabel("Spokes per Frame")
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
axes[1, 2].set_xlabel("Spokes per Frame")
axes[1, 2].set_ylabel("Pearson Correlation Coefficient")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spf_eval_metrics.png"))
plt.close()