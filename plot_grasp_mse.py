import argparse
import json
import os
import matplotlib.pyplot as plt
import torch
import yaml
from dataloader import ZFSliceDataset, SimulatedDataset, SimulatedSPFDataset
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transform import VideoRotate, VideoDiffeo, SubsampleTime, MonophasicTimeWarp, TemporalNoise, TimeReverse
from ei import EILoss
from mc import MCLoss
from lsfpnet_encoding import LSFPNet, ArtifactRemovalLSFPNet
from radial_lsfp import MCNUFFT
from utils import prep_nufft, log_gradient_stats, plot_enhancement_curve, get_cosine_ei_weight, plot_reconstruction_sample, get_git_commit, save_checkpoint, load_checkpoint, to_torch_complex, GRASPRecon, sliding_window_inference, set_seed
from eval import eval_grasp, eval_sample
import seaborn as sns
import math


# load data
split_file = "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json"
with open(split_file, "r") as fp:
    splits = json.load(fp)

val_patient_ids = splits["val"]
val_dro_patient_ids = splits["val_dro"]


root_dir = "/ess/scratch/scratch1/rachelgordon/dro_dataset"
model_type = "LSFPNet"

eval_spf_dataset = SimulatedSPFDataset(
    root_dir=root_dir, 
    model_type=model_type, 
    patient_ids=val_dro_patient_ids,
    )


eval_spf_loader = DataLoader(
    eval_spf_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

device = torch.device("cuda")
N_samples = 640
exp_name = "plot_grasp_metrics"
output_dir = os.path.join("output", exp_name)
eval_dir = os.path.join(output_dir, "eval_results")


MAIN_EVALUATION_PLAN = [
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
            # {
            #     "spokes_per_frame": 24,
            #     "num_frames": 12, # 24 * 12 = 288 total spokes
            #     "description": "Good temporal resolution"
            # },
            # {
            #     "spokes_per_frame": 32,
            #     "num_frames": 8, # 36 * 8 = 288 total spokes
            #     "description": "Standard temporal resolution"
            # },
        ]



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

for eval_config in MAIN_EVALUATION_PLAN:
                
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
                
                
                for csmap, ground_truth, grasp_img, mask, grasp_path in eval_spf_loader:

                    print("grasp_path: ", grasp_path)


                    csmap = csmap.squeeze(0).to(device)   # Remove batch dim
                    ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

                    # SIMULATE KSPACE
                    ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, spokes, num_frames)
                    physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))

                    sim_kspace = physics(False, ground_truth, csmap)

                    kspace = sim_kspace.squeeze(0).to(device) # Remove batch dim


                    # check if GRASP image exists or if we need to perform GRASP recon
                    # if type(grasp_img) is int or len(grasp_img.shape) == 1:
                    #     print(f"No GRASP file found, performing reconstruction with {spokes} spokes/frame and {num_frames} frames.")

                    #     grasp_img = GRASPRecon(csmap, sim_kspace, spokes, num_frames, grasp_path[0])

                    #     grasp_recon_torch = torch.from_numpy(grasp_img).permute(2, 0, 1) # T, H, W
                    #     grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

                    #     grasp_img = torch.flip(grasp_recon_torch, dims=[-3])
                    #     grasp_img = torch.rot90(grasp_img, k=3, dims=[-3,-1]).unsqueeze(0)

                    grasp_img = grasp_img.to(device)

                    ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
                    ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')


                    ## Evaluation
                    # ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr = eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img, acceleration, int(spokes), eval_dir, f"{spokes}spf", device)
                    # stress_test_ssims.append(ssim)
                    # stress_test_psnrs.append(psnr)
                    # stress_test_mses.append(mse)
                    # stress_test_lpipses.append(lpips)
                    # stress_test_dc_mses.append(dc_mse)
                    # stress_test_dc_maes.append(dc_mae)

                    # if recon_corr is not None:
                    #     stress_test_corrs.append(recon_corr)
                    #     stress_test_grasp_corrs.append(grasp_corr)


                    ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp = eval_grasp(kspace, csmap, ground_truth, grasp_img, physics, device, eval_dir)

                    stress_test_grasp_ssims.append(ssim_grasp)
                    stress_test_grasp_psnrs.append(psnr_grasp)
                    stress_test_grasp_mses.append(mse_grasp)
                    stress_test_grasp_lpipses.append(lpips_grasp)
                    stress_test_grasp_dc_mses.append(dc_mse_grasp)
                    stress_test_grasp_dc_maes.append(dc_mae_grasp)

                spf_grasp_ssim[spokes] = np.mean(stress_test_grasp_ssims)
                spf_grasp_psnr[spokes] = np.mean(stress_test_grasp_psnrs)
                spf_grasp_mse[spokes] = np.mean(stress_test_grasp_mses)
                spf_grasp_lpips[spokes] = np.mean(stress_test_grasp_lpipses)
                spf_grasp_dc_mse[spokes] = np.mean(stress_test_grasp_dc_mses)
                spf_grasp_dc_mae[spokes] = np.mean(stress_test_grasp_dc_maes)
                spf_grasp_corr[spokes] = np.mean(stress_test_grasp_corrs)



                    # spf_grasp_ssim[spokes] = np.mean(stress_test_grasp_ssims)
                    # spf_grasp_psnr[spokes] = np.mean(stress_test_grasp_psnrs)
                    # spf_grasp_mse[spokes] = np.mean(stress_test_grasp_mses)
                    # spf_grasp_lpips[spokes] = np.mean(stress_test_grasp_lpipses)
                    # spf_grasp_dc_mse[spokes] = np.mean(stress_test_grasp_dc_mses)
                    # spf_grasp_dc_mae[spokes] = np.mean(stress_test_grasp_dc_maes)
                    # spf_grasp_corr[spokes] = np.mean(stress_test_grasp_corrs)


N_full = 320 * math.pi / 2

# temporal resolution = frames/second,  150 seconds / timeframes

# accelerations = {}
# for spf in spf_grasp_mse.keys():

#     acceleration = N_full / int(spf)
#     accelerations[acceleration] = spf_grasp_mse[spf]

temp_resolutions = {}
for spf in spf_grasp_mse.keys():
    num_timeframes = round(288 / int(spf), 0)
    temp_res = round(150 / num_timeframes, 0)
    temp_resolutions[temp_res] = spf_grasp_mse[spf]


# Create the line plot
sns.lineplot(x=list(temp_resolutions.keys()), y=list(temp_resolutions.values()), marker='o')

plt.title("MSE of GRASP Reconstruction vs. Temporal Resolution", fontsize=16)
plt.xlabel("Temporal Resolution (seconds/frame)", fontsize=14)
plt.ylabel("MSE", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True) # Add a grid for better readability

plt.savefig('grasp_mse_spf.png')
