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
from skimage.metrics import structural_similarity as ssim_map_func
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
from typing import List, Dict


# ==========================================================
# EVALUATION FUNCTIONS
# ==========================================================

def calc_image_metrics(input, reference, data_range, device):
    """
    Calculates image metrics for a given input and reference image.
    """

    # --- Initialize Metrics ---
    # We will compute metrics frame by frame. data_range is important for PSNR.
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    psnr = torchmetrics.PeakSignalNoiseRatio(data_range=data_range).to(device)
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

    input = from_torch_complex(input)
    reference = from_torch_complex(reference)

    mse = mse(input.to(device), reference.to(device))

    return mse.item()



def evaluate_reconstruction_fidelity(
    ground_truth_params: np.ndarray,
    estimated_params: np.ndarray,
    masks: dict,
    param_names: list = None,
    regions_to_evaluate: list = None,
    display_plots: bool = True,
    filename: str = 'pk_param_maps.png'
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
            print("ground_truth_params: ", type(ground_truth_params))
            # Extract values from the specified region using the mask
            gt_values = ground_truth_params[:, :, i][mask]
            est_values = estimated_params[:, :, i][mask]

            # Avoid division by zero for relative error calculation
            print("gt_values: ", type(gt_values))
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
    plt.savefig(filename)
    plt.close()

    return evaluation_results


# ==========================================================
# PLOTTING FUNCTIONS
# ==========================================================

def plot_spatial_quality(
    recon_img: np.ndarray,
    gt_img: np.ndarray,
    grasp_img: np.ndarray,
    time_frame_index: int,
    filename: str,
    data_range: float
):
    """
    Generates a comparison plot for a single time frame in a 2x4 grid.
    Each row includes: Ground Truth, Reconstruction, Error Map, and SSIM Map.

    Args:
        recon_img (np.ndarray): Your model's reconstructed image for this frame.
        gt_img (np.ndarray): The ground truth image for this frame.
        grasp_img (np.ndarray): The GRASP reconstruction image for this frame.
        time_frame_index (int): The index of the time frame for titling.
        filename (str): The path to save the output plot.
    """

    # Calculate error maps
    error_map_dl = recon_img - gt_img
    error_map_grasp = grasp_img - gt_img

    # Calculate SSIM maps
    _, ssim_map_dl = ssim_map_func(gt_img, recon_img, data_range=data_range, full=True)
    _, ssim_map_grasp = ssim_map_func(gt_img, grasp_img, data_range=data_range, full=True)

    # Create a 2x4 plot grid
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f"Spatial Quality Comparison at Time Frame {time_frame_index}", fontsize=20)

    # --- Top Row: DL Reconstruction Comparison ---
    axes[0, 0].imshow(gt_img, cmap='gray')
    axes[0, 0].set_title("Ground Truth")

    axes[0, 1].imshow(recon_img, cmap='gray')
    axes[0, 1].set_title("DL Reconstruction")

    im_err_dl = axes[0, 2].imshow(error_map_dl, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title("DL Error Map (Recon - GT)")
    fig.colorbar(im_err_dl, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_ssim_dl = axes[0, 3].imshow(ssim_map_dl, cmap='viridis', vmin=0, vmax=1)
    axes[0, 3].set_title("DL SSIM Map")
    fig.colorbar(im_ssim_dl, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # --- Bottom Row: GRASP Reconstruction Comparison ---
    axes[1, 0].imshow(gt_img, cmap='gray')
    axes[1, 0].set_title("Ground Truth")

    axes[1, 1].imshow(grasp_img, cmap='gray')
    axes[1, 1].set_title("GRASP Reconstruction")

    im_err_grasp = axes[1, 2].imshow(error_map_grasp, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title("GRASP Error Map (Recon - GT)")
    fig.colorbar(im_err_grasp, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im_ssim_grasp = axes[1, 3].imshow(ssim_map_grasp, cmap='viridis', vmin=0, vmax=1)
    axes[1, 3].set_title("GRASP SSIM Map")
    fig.colorbar(im_ssim_grasp, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    # Turn off axes for all plots
    for ax in axes.flat:
        ax.axis('off')

    # plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(filename)
    plt.close()



def plot_temporal_curves(
    gt_img_stack: np.ndarray,
    recon_img_stack: np.ndarray,
    grasp_img_stack: np.ndarray,
    masks: dict,
    time_points: np.ndarray,
    filename: str
):
    """
    Plots the mean signal intensity vs. time for different tissue regions.
    This is CRITICAL for debugging PK model fitting.

    Args:
        gt_img_stack (np.ndarray): Time series of ground truth images (H, W, T).
        recon_img_stack (np.ndarray): Time series of your model's images (H, W, T).
        grasp_img_stack (np.ndarray): Time series of GRASP images (H, W, T).
        masks (dict): Dictionary of boolean NumPy masks for different regions.
        time_points (np.ndarray): The time vector for the x-axis.
        filename (str): The path to save the output plot.
    """
    regions = [r for r in ['malignant', 'glandular', 'muscle'] if r in masks and masks[r].any()]
    if not regions:
        print("No relevant regions found in mask to plot temporal curves.")
        return

    fig, axes = plt.subplots(1, len(regions), figsize=(7 * len(regions), 5), sharey=True)
    if len(regions) == 1: axes = [axes] # Ensure axes is always a list
    fig.suptitle("Temporal Fidelity: Mean Signal vs. Time", fontsize=16)

    for i, region in enumerate(regions):
        mask = masks[region]

        # Calculate mean signal in the masked region for each time point
        gt_curve = [gt_img_stack[:, :, t][mask].mean() for t in range(gt_img_stack.shape[2])]
        recon_curve = [recon_img_stack[:, :, t][mask].mean() for t in range(recon_img_stack.shape[2])]
        grasp_curve = [grasp_img_stack[:, :, t][mask].mean() for t in range(grasp_img_stack.shape[2])]

        # Plot
        # --- CHANGES ARE HERE ---
        # Added marker='o' to show dots for each time point
        axes[i].plot(time_points, gt_curve, 'k-', label='Ground Truth', linewidth=2, marker='o')
        axes[i].plot(time_points, recon_curve, 'r--', label='DL Recon', marker='o')
        axes[i].plot(time_points, grasp_curve, 'b:', label='GRASP Recon', marker='o')
        # --- END OF CHANGES ---
        
        axes[i].set_title(f"Region: {region.capitalize()}")
        axes[i].set_xlabel("Time (s)")
        axes[i].grid(True)
        axes[i].legend()

    axes[0].set_ylabel("Mean Signal Intensity")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()


# def plot_temporal_curves(
#     gt_img_stack: np.ndarray,
#     recon_img_stack: np.ndarray,
#     grasp_img_stack: np.ndarray,
#     masks: dict,
#     time_points: np.ndarray,
#     filename: str
# ):
#     """
#     Plots the mean signal intensity vs. time for different tissue regions.
#     This is CRITICAL for debugging PK model fitting.

#     Args:
#         gt_img_stack (np.ndarray): Time series of ground truth images (H, W, T).
#         recon_img_stack (np.ndarray): Time series of your model's images (H, W, T).
#         grasp_img_stack (np.ndarray): Time series of GRASP images (H, W, T).
#         masks (dict): Dictionary of boolean NumPy masks for different regions.
#         time_points (np.ndarray): The time vector for the x-axis.
#         filename (str): The path to save the output plot.
#     """
#     regions = [r for r in ['malignant', 'glandular', 'muscle'] if r in masks and masks[r].any()]
#     if not regions:
#         print("No relevant regions found in mask to plot temporal curves.")
#         return

#     fig, axes = plt.subplots(1, len(regions), figsize=(7 * len(regions), 5), sharey=True)
#     if len(regions) == 1: axes = [axes] # Ensure axes is always a list
#     fig.suptitle("Temporal Fidelity: Mean Signal vs. Time", fontsize=16)

#     for i, region in enumerate(regions):
#         mask = masks[region]
        
#         # Calculate mean signal in the masked region for each time point
#         gt_curve = [gt_img_stack[:, :, t][mask].mean() for t in range(gt_img_stack.shape[2])]
#         recon_curve = [recon_img_stack[:, :, t][mask].mean() for t in range(recon_img_stack.shape[2])]
#         grasp_curve = [grasp_img_stack[:, :, t][mask].mean() for t in range(grasp_img_stack.shape[2])]

#         # Plot
#         axes[i].plot(time_points, gt_curve, 'k-', label='Ground Truth', linewidth=2)
#         axes[i].plot(time_points, recon_curve, 'r--', label='DL Recon')
#         axes[i].plot(time_points, grasp_curve, 'b:', label='GRASP Recon')
#         axes[i].set_title(f"Region: {region.capitalize()}")
#         axes[i].set_xlabel("Time (s)")
#         axes[i].grid(True)
#         axes[i].legend()

#     axes[0].set_ylabel("Mean Signal Intensity")
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(filename)
#     plt.close()



def plot_single_temporal_curve(
    img_stack: np.ndarray,
    masks: Dict[str, np.ndarray],
    time_points: np.ndarray,
    filename: str,
    # New arguments required for this specific plot style:
    frames_to_show: List[int] = None
):
    """
    Generates a comprehensive analysis plot for a single sample, showing the
    Tumor Contrast Enhancement Curve (CEC) and corresponding image frames with
    the tumor Region of Interest (ROI) highlighted.

    This function is modified to produce a detailed analysis plot for the
    'malignant' tissue type, using the ground truth data.

    Args:
        gt_img_stack (np.ndarray): Time series of ground truth images (H, W, T).
        recon_img_stack (np.ndarray): Unused in this plot, kept for signature compatibility.
        grasp_img_stack (np.ndarray): Unused in this plot, kept for signature compatibility.
        masks (dict): Dictionary of boolean NumPy masks. Expects a 'malignant' key.
        time_points (np.ndarray): The time vector for the x-axis (e.g., frame numbers).
        filename (str): The path to save the output plot.
        sample_name (str): The name of the sample for the main plot title.
        frames_to_show (List[int]): A list of 4 frame indices to display in the
                                    image grid and highlight on the curve.
                                    If None, defaults to [0, 6, 13, 20].
    """
    # This function now specifically targets the 'malignant' tumor region.
    region_key = 'malignant'
    if region_key not in masks or not masks[region_key].any():
        print(f"'{region_key}' mask not found or is empty. Skipping plot generation.")
        return

    tumor_mask = masks[region_key]

    if frames_to_show is None:
        # frames_to_show = [0, 6, 20, 21] # Default frames from the example
        frames_to_show = [0, 6, 13, 20] # Default frames from the example
    if len(frames_to_show) != 4:
        raise ValueError(f"This function is designed to show exactly 4 frames, but {len(frames_to_show)} were provided.")

    # --- 1. Setup Figure and Layout ---
    fig = plt.figure(figsize=(20, 8.5))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.1, wspace=0.1)

    ax_curve = fig.add_subplot(gs[:, 0:2])
    ax_imgs = [
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
    ]

    # fig.suptitle(f"Enhancement Curve", fontsize=22, y=0.98)

    # --- 2. Plot Tumor Enhancement Curve (Left Panel) ---
    mean_curve = [img_stack[:, :, t][tumor_mask].mean() for t in range(img_stack.shape[2])]

    # Plot the full curve with markers
    ax_curve.plot(time_points, mean_curve, 'o-', label='Mean Tumor Signal', linewidth=2, markersize=6)

    # Highlight specific points on the curve with red stars
    highlight_times = [time_points[i] for i in frames_to_show]
    highlight_vals = [mean_curve[i] for i in frames_to_show]
    ax_curve.plot(highlight_times, highlight_vals, 'r*', markersize=18, zorder=10) # zorder to ensure stars are on top

    # Formatting the curve plot
    ax_curve.set_title("Tumor Contrast Enhancement Curve (CEC)", fontsize=18, pad=10)
    ax_curve.set_xlabel("Time Frame", fontsize=16)
    ax_curve.set_ylabel("Mean Signal Intensity", fontsize=16)
    ax_curve.legend(fontsize=14)
    ax_curve.grid(True, linestyle='--')
    ax_curve.tick_params(axis='both', which='major', labelsize=14)

    # --- 3. Plot Image Frames with ROI (Right Panel) ---
    # Find contours of the tumor mask to draw an outline
    contours = find_contours(tumor_mask, 0.5)

    # Use consistent intensity scaling for all image frames
    # vmin, vmax = np.percentile(img_stack, [1, 99.5])

    for i, frame_idx in enumerate(frames_to_show):
        ax = ax_imgs[i]
        image = img_stack[:, :, frame_idx]

        # Display the image
        ax.imshow(image, cmap='gray')#, vmin=vmin, vmax=vmax)

        # Overlay the tumor ROI outline
        for contour in contours:
            # contour is (row, col), plot needs (x, y)
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')

        # Formatting for each image subplot
        ax.set_title(f"Frame {frame_idx}", fontsize=16)
        ax.axis('off')

    # --- 4. Finalize and Save ---
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)



def plot_time_series(
    gt_img_stack: np.ndarray,
    recon_img_stack: np.ndarray,
    grasp_img_stack: np.ndarray,
    filename: str
):
    """
    Plots the middle 5 time points for Ground Truth, DL Recon, and GRASP.

    Args:
        gt_img_stack (np.ndarray): Time series of ground truth images (H, W, T).
        recon_img_stack (np.ndarray): Time series of your model's images (H, W, T).
        grasp_img_stack (np.ndarray): Time series of GRASP images (H, W, T).
        filename (str): The path to save the output plot.
    """
    num_frames = gt_img_stack.shape[2]
    
    # Select 5 time points: start, 1/4, 1/2, 3/4, end
    indices = np.linspace(0, num_frames - 1, 5, dtype=int)
    
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle("Temporal Series Comparison", fontsize=20)

    # --- Row 1: Ground Truth ---
    for i, frame_idx in enumerate(indices):
        img = gt_img_stack[:, :, frame_idx]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"GT: Frame {frame_idx}")
        axes[0, i].axis('off')

    # --- Row 2: DL Reconstruction ---
    for i, frame_idx in enumerate(indices):
        img = recon_img_stack[:, :, frame_idx]
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f"DL: Frame {frame_idx}")
        axes[1, i].axis('off')

    # --- Row 3: GRASP Reconstruction ---
    for i, frame_idx in enumerate(indices):
        img = grasp_img_stack[:, :, frame_idx]
        axes[2, i].imshow(img, cmap='gray')
        axes[2, i].set_title(f"GRASP: Frame {frame_idx}")
        axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()



# ==========================================================
# ESTIMATION FUNCTIONS
# ==========================================================

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

    DEBUG_PIXEL_R, DEBUG_PIXEL_C = 150, 150

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


            if r == DEBUG_PIXEL_R and c == DEBUG_PIXEL_C:
                print(f"\n--- DEBUGGING PIXEL ({r}, {c}) ---")
                
                signal_curve = np.abs(reconstructed_images[r, c, :])
                concentration_curve = signal_to_concentration(
                    signal_curve, S0_map[r, c], T10_map[r, c], TR, r1, flip_angle_rad
                )
                
                plt.figure(figsize=(10, 6))
                plt.plot(time_points, concentration_curve, 'bo', label='Measured Concentration (from DL Recon)')
                
                try:
                    params, _ = curve_fit(
                        fitting_func, time_points, concentration_curve, p0=initial_guess, bounds=bounds, method='trf'
                    )
                    ktrans_fit, ve_fit = params
                    
                    # Generate the fitted curve
                    fitted_curve = tofts_model(time_points, ktrans_fit, ve_fit, aif_t, aif_c)
                    plt.plot(time_points, fitted_curve, 'r-', label=f'Tofts Fit (Ktrans={ktrans_fit*60:.3f}, ve={ve_fit:.3f})')
                    
                except RuntimeError:
                    plt.title(f"DEBUG: Curve fit FAILED for pixel ({r}, {c})")
                    
                plt.xlabel("Time (s)")
                plt.ylabel("Concentration")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"debug_pixel_fit_{r}_{c}.png")
                plt.close()
                print("--- DEBUG PLOT SAVED ---")

    zeros_map = np.zeros_like(ktrans_map)
    # The order [ve, Ktrans, 0, 0] is chosen to align with the DRO's ground truth order.
    # DRO parMap is [ve, vp, Fp, PS]. We are comparing ve vs ve, and Ktrans vs vp.
    estimated_pk_map = np.stack([ve_map, ktrans_map, zeros_map, zeros_map], axis=-1)
    return estimated_pk_map





# ==========================================================
# EVALUATION LOOP
# ==========================================================

def eval_model(model, device, config, output_dir, physics_objects, epoch, temporal_eval=False):
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
    all_dc_scores = []

    all_ssim_scores_grasp = []
    all_psnr_scores_grasp = []
    all_mse_scores_grasp = []
    all_dc_scores_grasp = []

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

            grasp_recon = torch.rot90(grasp_recon, k=-1, dims=(2, 4))
            grasp_recon = torch.flip(grasp_recon, dims=[-1])


            
            # ==========================================================
            # PERFORM INFERENCE
            # ==========================================================
            kspace_complex = kspace_complex.squeeze(0) # Remove batch dim
            csmap_complex = csmap.squeeze(0)   # Remove batch dim

            with torch.no_grad():
                scale = torch.quantile(kspace_complex.abs(), 0.99) + 1e-8
            kspace_norm = kspace_complex / scale

            x_recon, _, = model(
                kspace_norm, 
                physics, 
                csmap_complex, 
            ) # Output shape (B, C, H, W, T)


            end = time.time()

            print(f"Time for Inference: {end-start}")



            # ==========================================================
            # EVALUATE DATA CONSISTENCY
            # ==========================================================

            # Forward Simulation
            x_recon_complex = to_torch_complex(x_recon).squeeze()
            grasp_recon_complex = rearrange(to_torch_complex(grasp_recon).squeeze(), 'h t w -> h w t')
            kspace = kspace.squeeze()


            recon_kspace = physics(False, x_recon_complex, csmap_complex)
            grasp_kspace = physics(False, grasp_recon_complex.to(csmap_complex.dtype), csmap_complex)


            # Compute MSE
            dc = calc_dc(recon_kspace, kspace, device)
            dc_grasp = calc_dc(grasp_kspace, kspace, device)

            all_dc_scores.append(dc)
            all_dc_scores_grasp.append(dc_grasp)



            # ==========================================================
            # EVALUATE SPATIAL IMAGE QUALITY
            # ==========================================================

            # calculate the single optimal scaling factor 'c'
            x_recon_np = x_recon.cpu().numpy()
            ground_truth_np = ground_truth.cpu().numpy()

            c = np.dot(x_recon_np.flatten(), ground_truth_np.flatten()) / np.dot(x_recon_np.flatten(), x_recon_np.flatten())

            recon_complex_scaled = torch.tensor(c * x_recon_np, device=device)


            # Convert complex images to magnitude
            recon_mag_scaled = torch.sqrt(recon_complex_scaled[:, 0, ...]**2 + recon_complex_scaled[:, 1, ...]**2)
            gt_mag = torch.sqrt(ground_truth[:, 0, ...]**2 + ground_truth[:, 1, ...]**2)
            grasp_mag = torch.sqrt(grasp_recon[:, 0, ...]**2 + grasp_recon[:, 1, ...]**2)




            for t in range(recon_mag_scaled.shape[-1]): # Iterate over time frames


                frame_recon = recon_mag_scaled[..., t]
                frame_gt = gt_mag[:, t, :, :]
                frame_grasp = grasp_mag[:, :, t, :]

                # calculate data range from ground truth
                data_range = frame_gt.max() - frame_gt.min()


                # Add channel dimension for torchmetrics: (B, H, W) -> (B, 1, H, W)
                frame_recon = frame_recon.unsqueeze(1)
                frame_gt = frame_gt.unsqueeze(1)
                frame_grasp = frame_grasp.unsqueeze(1).contiguous()
                
                # Calculate Spatial Image Quality Metrics
                ssim, psnr, mse = calc_image_metrics(frame_recon, frame_gt, data_range, device)
                all_ssim_scores.append(ssim)
                all_psnr_scores.append(psnr)
                all_mse_scores.append(mse)


                ssim_grasp, psnr_grasp, mse_grasp = calc_image_metrics(frame_grasp, frame_gt, data_range, device)
                all_ssim_scores_grasp.append(ssim_grasp)
                all_psnr_scores_grasp.append(psnr_grasp)
                all_mse_scores_grasp.append(mse_grasp)


            
            # ==========================================================
            # EVALUATE TEMPORAL FIDELITY
            # ==========================================================
            if temporal_eval:

                # Estimate PK Parameters
                # Define the time vector for the AIF and dynamic images
                num_frames = recon_complex_scaled.shape[-1]
                aif_time_points = np.linspace(0, 150, num_frames) # Time in seconds

                ground_truth_pk_map = parMap


                # Use the function from Part 1 to perform the estimation
                estimated_pk_map_from_dl = estimate_pk_parameters(
                    reconstructed_images=x_recon_complex.cpu().numpy(),
                    aif_t=aif_time_points,
                    aif_c=aif.cpu().numpy().squeeze(),
                    S0_map=S0.cpu().numpy().squeeze(),
                    T10_map=T10.cpu().numpy().squeeze()
                )

                # Use the function from Part 1 to perform the estimation
                estimated_pk_map_from_grasp = estimate_pk_parameters(
                    reconstructed_images=grasp_recon_complex.cpu().numpy(),
                    aif_t=aif_time_points,
                    aif_c=aif.cpu().numpy().squeeze(),
                    S0_map=S0.cpu().numpy().squeeze(),
                    T10_map=T10.cpu().numpy().squeeze()
                )


                print("\n--- STEP 5: Final Evaluation of Your Model's Fidelity ---")

                masks_np = {key: val.cpu().numpy().squeeze().astype(bool) for key, val in mask.items()}


                # Now, compare the PK map from your model's output to the original ground truth PK map
                evaluation_summary = evaluate_reconstruction_fidelity(
                    ground_truth_params=ground_truth_pk_map.cpu().numpy().squeeze(),
                    estimated_params=estimated_pk_map_from_dl,
                    masks=masks_np,
                    regions_to_evaluate=['malignant'] # Focus on a key region
                )

                grasp_evaluation_summary = evaluate_reconstruction_fidelity(
                    ground_truth_params=ground_truth_pk_map.cpu().numpy().squeeze(),
                    estimated_params=estimated_pk_map_from_grasp,
                    masks=masks_np,
                    regions_to_evaluate=['malignant'], # Focus on a key region
                    filename="pk_param_maps_grasp.png"
                )

                print("\nEvaluation Complete.")

                print("DL Recon: ", evaluation_summary)
                print("GRASP Recon: ", grasp_evaluation_summary)



            # ==========================================================
            # VISUALIZATION
            # ==========================================================

            x_recon_complex_np = to_torch_complex(recon_complex_scaled).squeeze().cpu().numpy()
            grasp_recon_complex_np = rearrange(to_torch_complex(grasp_recon).squeeze(), 'h t w -> h w t').cpu().numpy()

            gt_squeezed = ground_truth.squeeze()  # Shape: (C, T, H, W) -> (2, 22, 320, 320)
            gt_rearranged = rearrange(gt_squeezed, 'c t h w -> t c h w') # Shape: (22, 320, 320, 2)
            gt_complex_tensor = to_torch_complex(gt_rearranged) # Shape: (22, 320, 320)
            gt_final_tensor = rearrange(gt_complex_tensor, 't h w -> h w t') # Shape: (320, 320, 22)
            gt_complex_np = gt_final_tensor.cpu().numpy()

            recon_mag_np = np.abs(x_recon_complex_np)
            grasp_mag_np = np.abs(grasp_recon_complex_np)
            gt_mag_np = np.abs(gt_complex_np)
            
            masks_np = {key: val.cpu().numpy().squeeze().astype(bool) for key, val in mask.items()}
            num_frames = recon_mag_np.shape[2]
            aif_time_points = np.linspace(0, 150, num_frames)

            if i == 0:
                print("\nGenerating diagnostic plots for the first evaluation sample...")
                
                # --- Plot Spatial Quality at a Peak Enhancement Frame ---
                # Find a frame around peak enhancement (e.g., 1/3 of the way through)
                peak_frame = num_frames // 3
                data_range = gt_mag_np[:, :, peak_frame].max() - gt_mag_np[:, :, peak_frame].min()
                plot_spatial_quality(
                    recon_img=recon_mag_np[:, :, peak_frame],
                    gt_img=gt_mag_np[:, :, peak_frame],
                    grasp_img=grasp_mag_np[:, :, peak_frame],
                    time_frame_index=peak_frame,
                    filename=os.path.join(output_dir, f"spatial_quality_epoch{epoch}.png"),
                    data_range=data_range
                )

                # --- Plot Temporal Curves for Key Regions ---
                # This is the most important plot for debugging your PK results!
                plot_temporal_curves(
                    gt_img_stack=gt_mag_np,
                    recon_img_stack=recon_mag_np,
                    grasp_img_stack=grasp_mag_np,
                    masks=masks_np,
                    time_points=aif_time_points,
                    filename=os.path.join(output_dir, f"temporal_curves_epoch{epoch}.png")
                )

                plot_single_temporal_curve(
                    img_stack=recon_mag_np,
                    masks=masks_np,
                    time_points=aif_time_points,
                    filename=os.path.join(output_dir, f"recon_temporal_curve_epoch{epoch}.png")
                )

                plot_time_series(
                    gt_img_stack=gt_mag_np,
                    recon_img_stack=recon_mag_np,
                    grasp_img_stack=grasp_mag_np,
                    filename=os.path.join(output_dir, f"time_points_epoch{epoch}.png")
                )

                print("Diagnostic plots saved.")


                

    # --- 5. Compute and Report Final Results ---
    avg_ssim = np.mean(all_ssim_scores)
    std_ssim = np.std(all_ssim_scores)
    avg_psnr = np.mean(all_psnr_scores)
    std_psnr = np.std(all_psnr_scores)
    avg_mse = np.mean(all_mse_scores)
    std_mse = np.std(all_mse_scores)
    avg_dc = np.mean(all_dc_scores)
    std_dc = np.std(all_dc_scores)

    avg_ssim_grasp = np.mean(all_ssim_scores_grasp)
    std_ssim_grasp = np.std(all_ssim_scores_grasp)
    avg_psnr_grasp = np.mean(all_psnr_scores_grasp)
    std_psnr_grasp = np.std(all_psnr_scores_grasp)
    avg_mse_grasp = np.mean(all_mse_scores_grasp)
    std_mse_grasp = np.std(all_mse_scores_grasp)
    avg_dc_grasp = np.mean(all_mse_scores_grasp)
    std_dc_grasp = np.std(all_mse_scores_grasp)

    print("\n--- Evaluation Complete ---")
    print(f"  Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
    print(f"  Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"  Average DC: {avg_dc:.4f} ± {std_dc:.4f}")
    print(f"  Average GRASP SSIM: {avg_ssim_grasp:.4f} ± {std_ssim_grasp:.4f}")
    print(f"  Average GRASP PSNR: {avg_psnr_grasp:.4f} ± {std_psnr_grasp:.4f}")
    print(f"  Average GRASP MSE: {avg_mse_grasp:.4f} ± {std_mse_grasp:.4f}")
    print(f"  Average GRASP DC: {avg_dc_grasp:.4f} ± {std_dc_grasp:.4f}")
    print("-" * 27)

    # Save results to a file
    results_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(results_path, "a") as f:
        f.write(f"Evaluation Metrics on Simulated Dataset: Epoch {epoch} \n")
        f.write("="*40 + "\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Experiment: {os.path.basename(output_dir)}\n")
        f.write(f"Number of evaluation samples: {len(eval_dataset)}\n")
        f.write(f"Number of time frames per sample: {ground_truth.shape[2]}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average SSIM: {avg_ssim:.4f} (Std: {std_ssim:.4f})\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} (Std: {std_psnr:.4f})\n")
        f.write(f"Average MSE: {avg_mse:.4f} (Std: {std_mse:.4f})\n")
        f.write(f"Average DC: {avg_dc:.4f} (Std: {std_dc:.4f})\n")
        f.write(f"Average GRASP SSIM: {avg_ssim_grasp:.4f} (Std: {std_ssim_grasp:.4f})\n")
        f.write(f"Average GRASP PSNR: {avg_psnr_grasp:.4f} (Std: {std_psnr_grasp:.4f})\n")
        f.write(f"Average GRASP MSE: {avg_mse_grasp:.4f} (Std: {std_mse_grasp:.4f})\n")
        f.write(f"Average GRASP DC: {avg_dc_grasp:.4f} (Std: {std_dc_grasp:.4f})\n")

    print(f"Results saved to {results_path}")
    print("="*80 + "\n")


    return avg_ssim, avg_psnr, avg_mse, avg_dc
