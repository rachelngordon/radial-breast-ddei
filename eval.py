import os
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchmetrics
import time
from dataloader import SimulatedDataset
from lsfpnet import to_torch_complex, from_torch_complex
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm # A library for a nice progress bar
from scipy.stats import mannwhitneyu


## Evaluate Spatial Image Quality

def calc_image_metrics(input, reference, device):
    """
    Calculates image metrics for a given input and reference image.
    """

    # --- Initialize Metrics ---
    # We will compute metrics frame by frame. data_range is important for PSNR.
    # We normalize images to [0, 1], so the data_range is 1.0.
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    mse = torchmetrics.MeanSquaredError().to(device)

    ssim = ssim(input, reference)
    psnr = psnr(input, reference)
    mse = mse(input, reference)

    return ssim.item(), psnr.item(), mse.item()
    


## Evaluate Data Consistency in k-space

def calc_dc(input, reference, device):
    """
    Calculates data consistency MSE for a given input and reference k-space tensor.
    """

    mse = torchmetrics.MeanSquaredError().to(device)

    print(input.shape)
    print(reference.shape)
    input = from_torch_complex(input)
    reference = from_torch_complex(reference)

    mse = mse(input.to(device), reference.to(device))

    return mse.item()



## Estimate PK Parameters

def tofts_model(t, Ktrans, ve, aif_t, aif_c):
    """Calculates the tissue concentration curve using the standard Tofts model."""
    ve = max(ve, 1e-6)
    interp_func = PchipInterpolator(aif_t, aif_c, extrapolate=True)
    aif_interp = interp_func(t)
    impulse_response = Ktrans * np.exp(-t * (Ktrans / ve))
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    Ct = np.convolve(aif_interp, impulse_response, mode='full')[:len(t)] * dt
    return Ct

def signal_to_concentration(signal_curve, S0_pixel, T10_pixel, TR, r1, flip_angle_rad):
    """Converts an MRI signal curve S(t) to a concentration curve C(t)."""
    if S0_pixel < 1e-6:
        return np.zeros_like(signal_curve)
    norm_signal = signal_curve / S0_pixel
    sin_fa, cos_fa = np.sin(flip_angle_rad), np.cos(flip_angle_rad)
    denominator = sin_fa - norm_signal * cos_fa
    E1 = (sin_fa - norm_signal) / np.maximum(denominator, 1e-9)
    E1 = np.maximum(E1, 1e-9)
    R1_t = -np.log(E1) / TR
    R10 = 1.0 / T10_pixel
    concentration_curve = (R1_t - R10) / r1
    return np.maximum(0, concentration_curve)

def estimate_pk_parameters(
    reconstructed_images: np.ndarray,
    aif_t: np.ndarray,
    aif_c: np.ndarray,
    S0_map: np.ndarray,
    T10_map: np.ndarray,
    TR: float = 4.87e-3,
    r1: float = 4.3,
    flip_angle_deg: float = 10.0
) -> np.ndarray:
    """
    Estimates pharmacokinetic parameters (Ktrans, ve) from reconstructed DCE-MRI images.

    Args:
        reconstructed_images (np.ndarray): A (H, W, Time) array of dynamic images,
                                           THIS IS THE OUTPUT FROM YOUR DL MODEL.
        aif_t, aif_c (np.ndarray): The time points and concentrations for the AIF.
        S0_map, T10_map (np.ndarray): Baseline maps from the ground truth DRO.
        TR, r1, flip_angle_deg: Sequence parameters.

    Returns:
        np.ndarray: A (H, W, 4) array containing the estimated [ve, Ktrans, 0, 0] maps.
    """
    height, width, num_frames = reconstructed_images.shape
    flip_angle_rad = np.deg2rad(flip_angle_deg)
    time_points = aif_t
    ktrans_map = np.zeros((height, width))
    ve_map = np.zeros((height, width))

    fitting_func = lambda t, Ktrans, ve: tofts_model(t, Ktrans, ve, aif_t, aif_c)

    print("Estimating PK parameters from the reconstructed images...")
    for r in tqdm(range(height), desc="Fitting PK Model"):
        for c in range(width):
            if S0_map[r, c] < np.mean(S0_map) * 0.1:
                continue
            
            signal_curve = np.abs(reconstructed_images[r, c, :])
            concentration_curve = signal_to_concentration(
                signal_curve, S0_map[r, c], T10_map[r, c], TR, r1, flip_angle_rad
            )
            
            try:
                initial_guess = [0.1 / 60, 0.2] # Ktrans in s^-1
                bounds = ([0, 0], [2.0 / 60, 1.0])
                params, _ = curve_fit(
                    fitting_func, time_points, concentration_curve, p0=initial_guess, bounds=bounds, method='trf'
                )
                ktrans_map[r, c] = params[0] * 60 # Convert from s^-1 to min^-1
                ve_map[r, c] = params[1]
            except RuntimeError:
                pass # Fit failed, leave as 0

    zeros_map = np.zeros_like(ktrans_map)
    # The order [ve, Ktrans, 0, 0] is chosen to align with the DRO's ground truth order.
    # DRO parMap is [ve, vp, Fp, PS]. We are comparing ve vs ve, and Ktrans vs vp.
    estimated_pk_map = np.stack([ve_map, ktrans_map, zeros_map, zeros_map], axis=-1)
    return estimated_pk_map




## Evaluate Temporal Fidelity

def evaluate_reconstruction_fidelity(
    ground_truth_params: np.ndarray,
    estimated_params: np.ndarray,
    masks: dict,
    param_names: list = None,
    regions_to_evaluate: list = None,
    display_plots: bool = True
) -> dict:
    """
    Evaluates the fidelity of reconstructed pharmacokinetic (PK) parameters against ground truth.

    This function performs a quantitative and visual comparison, mimicking the evaluation
    methods described in the research paper (e.g., Figures 6 and 8).

    Args:
        ground_truth_params (np.ndarray): The ground truth PK parameter map, typically a
                                          (H, W, 4) array from the `gen_dro` output.
        estimated_params (np.ndarray): The PK parameter map estimated from your reconstructed
                                       images, with the same shape as ground_truth_params.
        masks (dict): A dictionary of boolean masks for different tissue regions, typically
                      from the `gen_dro` output (e.g., dro_results['mask']).
        param_names (list, optional): A list of names for the 4 parameters.
                                      Defaults to ['ve', 'vp', 'Fp', 'PS'].
        regions_to_evaluate (list, optional): A list of region names (keys in the `masks`
                                            dict) to analyze. Defaults to all available masks.
        display_plots (bool): If True, generates and shows summary plots.

    Returns:
        dict: A nested dictionary containing the evaluation results (median error and p-value)
              for each region and each parameter.
    """
    if param_names is None:
        param_names = ['ve', 'vp', 'Fp (F_p)', 'PS'] # As ordered in gen_dro

    if regions_to_evaluate is None:
        regions_to_evaluate = [name for name, mask in masks.items() if mask.any()]

    if ground_truth_params.shape != estimated_params.shape:
        raise ValueError("Ground truth and estimated parameter maps must have the same shape.")

    evaluation_results = {}
    print("--- Reconstruction Fidelity Evaluation ---")
    print("-" * 40)

    # --- 1. Quantitative and Statistical Evaluation ---
    for region in regions_to_evaluate:
        if region not in masks or not masks[region].any():
            continue

        print(f"Region: {region.capitalize()}")
        evaluation_results[region] = {}
        # NOTE: change to numerical value associated with specified region
        mask = masks[region]

        for i, p_name in enumerate(param_names):
            # Extract values from the specified region using the mask
            gt_values = ground_truth_params[:, :, i][mask]
            est_values = estimated_params[:, :, i][mask]

            # Avoid division by zero for relative error calculation
            gt_values_safe = gt_values.copy()
            gt_values_safe[gt_values_safe == 0] = 1e-9 # Add a small epsilon

            # Metric 1: Median Relative Error
            relative_error = (est_values - gt_values) / gt_values_safe
            median_err = np.median(relative_error)

            # Metric 2: Wilcoxon Rank-Sum Test (Mann-Whitney U)
            # This tests if the two distributions are significantly different
            try:
                stat, p_value = mannwhitneyu(gt_values, est_values, alternative='two-sided')
            except ValueError: # Happens if all values are identical
                stat, p_value = 0, 1.0

            evaluation_results[region][p_name] = {
                'median_relative_error': median_err,
                'p_value': p_value
            }

            print(f"  - {p_name:<10}: Median Error = {median_err:+.2%}, p-value = {p_value:.4f}")

    if not display_plots:
        return evaluation_results

    # --- 2. Visual Evaluation ---
    num_params = ground_truth_params.shape[2]
    fig, axes = plt.subplots(num_params, 3, figsize=(15, 4 * num_params), sharex=True, sharey=True)
    fig.suptitle("Visual Comparison of PK Parameter Maps", fontsize=16)

    for i in range(num_params):
        p_name = param_names[i]
        gt_map = ground_truth_params[:, :, i]
        est_map = estimated_params[:, :, i]
        error_map = est_map - gt_map

        # Determine shared color limits for GT and Estimated maps
        vmax = np.percentile(gt_map[gt_map > 0], 99) if (gt_map > 0).any() else 1.0
        vmin = 0

        # Plot Ground Truth
        im_gt = axes[i, 0].imshow(gt_map, vmin=vmin, vmax=vmax, cmap='viridis')
        axes[i, 0].set_title(f"Ground Truth: {p_name}")
        axes[i, 0].axis('off')
        fig.colorbar(im_gt, ax=axes[i, 0])

        # Plot Estimated
        im_est = axes[i, 1].imshow(est_map, vmin=vmin, vmax=vmax, cmap='viridis')
        axes[i, 1].set_title(f"Your Estimation: {p_name}")
        axes[i, 1].axis('off')
        fig.colorbar(im_est, ax=axes[i, 1])

        # Plot Error Map
        # Use a diverging colormap and center it at zero
        err_vmax = np.percentile(np.abs(error_map), 99)
        im_err = axes[i, 2].imshow(error_map, vmin=-err_vmax, vmax=err_vmax, cmap='coolwarm')
        axes[i, 2].set_title(f"Error Map (Est - GT)")
        axes[i, 2].axis('off')
        fig.colorbar(im_err, ax=axes[i, 2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return evaluation_results


## Evaluation Loop 

def eval_model(model, device, config, output_dir, physics_objects):
    print("\n" + "="*80)
    print("--- Starting Evaluation on Simulated Dataset ---")
    print("="*80)

    # --- 1. Setup DataLoader for Simulated Data ---
    simulated_data_path = config["evaluation"]["simulated_dataset_path"]
    model_type = config["model"]["name"]

    try:
        eval_dataset = SimulatedDataset(root_dir=simulated_data_path, model_type=model_type, num_samples=config['evaluation']['num_samples'])
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Skipping evaluation.")
        return

    # Use a batch size of 1 for evaluation to process one sample at a time
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4)

    physics = physics_objects['physics']


    # Store results
    all_ssim_scores = []
    all_psnr_scores = []
    all_mse_scores = []

    all_ssim_scores_grasp = []
    all_psnr_scores_grasp = []
    all_mse_scores_grasp = []

    # --- 3. Evaluation Loop ---
    model.eval()
    with torch.no_grad():
        for i, (kspace, csmap, ground_truth, grasp_recon, parMap, aif, S0, T10, mask) in enumerate(tqdm(eval_loader, desc="Evaluating on Simulated Data")):

            start = time.time()
            
            # kspace_complex = to_torch_complex(kspace).to(device)
            kspace_complex = kspace.to(device)
            csmap = csmap.to(device)
            ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)
            grasp_recon = grasp_recon.to(device) # Shape: (1, 2, H, T, W)
            
            # --- Perform Inference (handle different model types) ---
            kspace_complex = kspace_complex.squeeze(0) # Remove batch dim
            csmap_complex = csmap.squeeze(0)   # Remove batch dim

            with torch.no_grad():
                scale = torch.quantile(kspace_complex.abs(), 0.99) + 1e-8
            kspace_norm = kspace_complex / scale

            x_recon, _ = model(
                kspace_norm, 
                physics, 
                csmap_complex, 
                physics_objects['dcomp']
            ) # Output shape (B, C, H, W, T)


            end = time.time()

            print(f"Time for Inference: {end-start}")

            # --- 4. Prepare Tensors for Metric Calculation ---
            # Convert complex images to magnitude
            recon_mag = torch.sqrt(x_recon[:, 0, ...]**2 + x_recon[:, 1, ...]**2)
            gt_mag = torch.sqrt(ground_truth[:, 0, ...]**2 + ground_truth[:, 1, ...]**2)
            grasp_mag = torch.sqrt(grasp_recon[:, 0, ...]**2 + grasp_recon[:, 1, ...]**2)

            # Normalize each image in the time series to [0, 1] for fair comparison
            for t in range(recon_mag.shape[-1]): # Iterate over time frames

                frame_recon = recon_mag[..., t]
                frame_gt = gt_mag[:, t, :, :]
                frame_grasp = grasp_mag[:, :, t, :]

                # Normalize by max value of each frame
                # NOTE: Make sure this is the correct normalization with my training scheme
                if frame_recon.max() > 0:
                    frame_recon = frame_recon / frame_recon.max()
                if frame_gt.max() > 0:
                    frame_gt = frame_gt / frame_gt.max()
                if frame_grasp.max() > 0:
                    frame_grasp = frame_grasp / frame_grasp.max()

                # Add channel dimension for torchmetrics: (B, H, W) -> (B, 1, H, W)
                frame_recon = frame_recon.unsqueeze(1)
                frame_gt = frame_gt.unsqueeze(1)
                frame_grasp = frame_grasp.unsqueeze(1)
                
                # Calculate Spatial Image Quality Metrics
                ssim, psnr, mse = calc_image_metrics(frame_recon, frame_gt, device)
                all_ssim_scores.append(ssim)
                all_psnr_scores.append(psnr)
                all_mse_scores.append(mse)


                ssim_grasp, psnr_grasp, mse_grasp = calc_image_metrics(frame_grasp, frame_gt, device)
                all_ssim_scores_grasp.append(ssim_grasp)
                all_psnr_scores_grasp.append(psnr_grasp)
                all_mse_scores_grasp.append(mse_grasp)


            # Calculate Data Consistency Metrics

            # Forward Simulation
            x_recon_complex = to_torch_complex(x_recon).squeeze()
            grasp_recon_complex = rearrange(to_torch_complex(grasp_recon).squeeze(), 'h t w -> h w t')
            kspace = kspace.squeeze()


            recon_kspace = physics(False, x_recon_complex, csmap_complex)
            grasp_kspace = physics(False, grasp_recon_complex.to(csmap_complex.dtype), csmap_complex)


            # Compute MSE
            dc = calc_dc(recon_kspace, kspace, device)
            dc_grasp = calc_dc(grasp_kspace, kspace, device)

            print("k-space MSE: ", dc)
            print("GRASP k-space MSE: ", dc_grasp)

            
            # Calculate Temporal Fidelity Metrics

            # Estimate PK Parameters
            # Define the time vector for the AIF and dynamic images
            # mask = mask[0]
            num_frames = x_recon.shape[-1]
            aif_time_points = np.linspace(0, 150, num_frames) # Time in seconds

            ground_truth_pk_map = parMap

            # Use the function from Part 1 to perform the estimation
            estimated_pk_map_from_dl = estimate_pk_parameters(
                reconstructed_images=x_recon_complex,
                aif_t=aif_time_points,
                aif_c=aif,
                S0_map=S0.cpu().numpy(),
                T10_map=T10
            )


            # ===================================================================
            # --- STEP 5: COMPARE YOUR ESTIMATED PK MAPS TO GROUND TRUTH ---
            # ===================================================================
            print("\n--- STEP 5: Final Evaluation of Your Model's Fidelity ---")

            # Now, compare the PK map from your model's output to the original ground truth PK map
            evaluation_summary = evaluate_reconstruction_fidelity(
                ground_truth_params=ground_truth_pk_map,
                estimated_params=estimated_pk_map_from_dl,
                masks=mask,
                regions_to_evaluate=['malignant'] # Focus on a key region
            )

            print("\nEvaluation Complete.")


                

    # --- 5. Compute and Report Final Results ---
    avg_ssim = np.mean(all_ssim_scores)
    std_ssim = np.std(all_ssim_scores)
    avg_psnr = np.mean(all_psnr_scores)
    std_psnr = np.std(all_psnr_scores)
    avg_mse = np.mean(all_mse_scores)
    std_mse = np.std(all_mse_scores)

    avg_ssim_grasp = np.mean(all_ssim_scores_grasp)
    std_ssim_grasp = np.std(all_ssim_scores_grasp)
    avg_psnr_grasp = np.mean(all_psnr_scores_grasp)
    std_psnr_grasp = np.std(all_psnr_scores_grasp)
    avg_mse_grasp = np.mean(all_mse_scores_grasp)
    std_mse_grasp = np.std(all_mse_scores_grasp)

    print("\n--- Evaluation Complete ---")
    print(f"  Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
    print(f"  Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"  Average GRASP SSIM: {avg_ssim_grasp:.4f} ± {std_ssim_grasp:.4f}")
    print(f"  Average GRASP PSNR: {avg_psnr_grasp:.4f} ± {std_psnr_grasp:.4f}")
    print(f"  Average GRASP MSE: {avg_mse_grasp:.4f} ± {std_mse_grasp:.4f}")
    print("-" * 27)

    # Save results to a file
    results_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(results_path, "w") as f:
        f.write("Evaluation Metrics on Simulated Dataset\n")
        f.write("="*40 + "\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Experiment: {os.path.basename(output_dir)}\n")
        f.write(f"Number of evaluation samples: {len(eval_dataset)}\n")
        f.write(f"Number of time frames per sample: {ground_truth.shape[2]}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average SSIM: {avg_ssim:.4f} (Std: {std_ssim:.4f})\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} (Std: {std_psnr:.4f})\n")
        f.write(f"Average MSE: {avg_mse:.4f} (Std: {std_mse:.4f})\n")
        f.write(f"Average GRASP SSIM: {avg_ssim_grasp:.4f} (Std: {std_ssim_grasp:.4f})\n")
        f.write(f"Average GRASP PSNR: {avg_psnr_grasp:.4f} (Std: {std_psnr_grasp:.4f})\n")
        f.write(f"Average GRASP MSE: {avg_mse_grasp:.4f} (Std: {std_mse_grasp:.4f})\n")

    print(f"Results saved to {results_path}")
    print("="*80 + "\n")