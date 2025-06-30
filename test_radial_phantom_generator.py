# test_and_plot_phantom.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from radial_phantom_generator import DigitalPhantomGenerator

def plot_phantom_outputs(phantom_data: dict, z_slice_idx: int, time_points: list, coils_to_plot: list, filename_prefix: str):
    """
    Visualizes the key outputs of the DigitalPhantomGenerator.
    """
    gt_coilless = phantom_data['gt_coilless_image'].cpu().numpy()
    sens_maps = phantom_data['sensitivity_maps'].cpu().numpy()
    coil_images = phantom_data['coil_images'].cpu().numpy()

    num_time = len(time_points)
    num_coils = len(coils_to_plot)
    
    # --- Plot 1: Ground Truth Coilless Image (Magnitude and Phase) over Time ---
    fig, axes = plt.subplots(2, num_time, figsize=(4 * num_time, 8), squeeze=False)
    fig.suptitle(f"Ground Truth Coilless Image (Z-slice: {z_slice_idx})", fontsize=16)

    for i, t in enumerate(time_points):
        # Magnitude
        ax_mag = axes[0, i]
        im_mag = ax_mag.imshow(np.abs(gt_coilless[t, :, :, z_slice_idx]).T, cmap='gray')
        ax_mag.set_title(f"Magnitude (t={t})")
        ax_mag.set_xlabel("X")
        ax_mag.set_ylabel("Y")
        plt.colorbar(im_mag, ax=ax_mag)
        
        # Phase
        ax_phase = axes[1, i]
        im_phase = ax_phase.imshow(np.angle(gt_coilless[t, :, :, z_slice_idx]).T, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title(f"Phase (t={t})")
        ax_phase.set_xlabel("X")
        ax_phase.set_ylabel("Y")
        plt.colorbar(im_phase, ax=ax_phase)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{filename_prefix}_coilless_image.png")
    plt.close()

    # --- Plot 2: Sensitivity Maps (Magnitude and Phase) for selected coils ---
    fig, axes = plt.subplots(2, num_coils, figsize=(4 * num_coils, 8), squeeze=False)
    fig.suptitle(f"Sensitivity Maps (Z-slice: {z_slice_idx})", fontsize=16)
    for i, c_idx in enumerate(coils_to_plot):
        # Magnitude
        ax_mag = axes[0, i]
        im_mag = ax_mag.imshow(np.abs(sens_maps[:, :, z_slice_idx, c_idx]).T, cmap='viridis')
        ax_mag.set_title(f"Magnitude (Coil {c_idx})")
        ax_mag.set_xlabel("X")
        ax_mag.set_ylabel("Y")
        plt.colorbar(im_mag, ax=ax_mag)

        # Phase
        ax_phase = axes[1, i]
        im_phase = ax_phase.imshow(np.angle(sens_maps[:, :, z_slice_idx, c_idx]).T, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title(f"Phase (Coil {c_idx})")
        ax_phase.set_xlabel("X")
        ax_phase.set_ylabel("Y")
        plt.colorbar(im_phase, ax=ax_phase)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{filename_prefix}_sensitivity_maps.png")
    plt.close()
    
    # --- Plot 3: Final Coil Image (Magnitude) over Time for one coil ---
    coil_to_show = coils_to_plot[0]
    fig, axes = plt.subplots(1, num_time, figsize=(4 * num_time, 4.5), squeeze=False)
    fig.suptitle(f"Final Coil Image (Coil {coil_to_show}, Z-slice: {z_slice_idx})", fontsize=16)
    for i, t in enumerate(time_points):
        ax = axes[0, i]
        im = ax.imshow(np.abs(coil_images[coil_to_show, t, :, :, z_slice_idx]).T, cmap='gray')
        ax.set_title(f"Magnitude (t={t})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{filename_prefix}_final_coil_image.png")
    plt.close()

    # --- Plot 4: Time-Intensity Curve (TIC) - THE MOST IMPORTANT PLOT FOR DYNAMICS ---
    nx, ny, nz, nt, nc = generator.dims
    # We will pick a voxel inside the bright central artery to see its signal change
    source_center_xyz = generator.phys_params["source_xyz_center"]
    # Convert normalized coordinates [-1, 1] to pixel indices
    voxel_x = int((source_center_xyz[0] + 1) / 2 * nx)
    voxel_y = int((source_center_xyz[1] + 1) / 2 * ny)
    voxel_z = int((source_center_xyz[2] + 1) / 2 * nz)

    # Extract the signal intensity at this voxel over all time points
    tic_curve = np.abs(gt_coilless[:, voxel_x, voxel_y, z_slice_idx])

    plt.figure(figsize=(10, 6))
    plt.plot(range(nt), tic_curve, 'bo-', label=f'Voxel at (x={voxel_x}, y={voxel_y})')
    plt.title("Time-Intensity Curve (TIC) for a Single Voxel", fontsize=16)
    plt.xlabel("Time Point (t)")
    plt.ylabel("Signal Magnitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3. Extract the signal intensity at this specific voxel for ALL time points
    # Shape of gt_coilless is [T, X, Y, Z]
    enhancement_curve = np.abs(gt_coilless[:, voxel_x, voxel_y, voxel_z])

    # 4. Plot the curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(nt), enhancement_curve, 'bo-', label=f'Signal at Source Center')
    plt.title("Enhancement Curve at Source", fontsize=16)
    plt.xlabel("Time Point (t)")
    plt.ylabel("Signal Magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_enhancement_curve.png")
    plt.close()


if __name__ == "__main__":
    sim_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator = DigitalPhantomGenerator(
        trajectory_type='radial',
        # Fully sampled would be ~ pi/2 * 64 = 100 spokes. 32 is ~3x acceleration.
        radial_num_spokes_per_frame=32, 
        phantom_dims = [64, 64, 64, 8, 16], # phantom_dims=[320, 320, 40, 8, 16],  # [nx, ny, nz, nt, nc]
        phantom_lambda_drift = 1.5,
        phantom_source_strength = 25.0, #16.0,

        phantom_source_t_end=0.3,      # Keep the source "on" for longer
        phantom_vz=2.0,                # Make the agent flow faster
        phantom_lambda_decay=0.3,      # Make the agent wash out a bit slower
        phantom_source_xyz_center=[0.5, 0.0, -0.8],

        device=sim_device
    )

    # Generate the undersampled Radial k-space data
    phantom_data = generator.generate_accelerated_kspace()

    # 3. Print the shapes of the generated tensors to verify
    print("--- Generated Phantom Data Shapes ---")
    for name, tensor in phantom_data.items():
        if tensor is not None:
            print(f"{name}: {tensor.shape}")
        else:
            print(f"{name}: None")
    print("------------------------------------")

    # 4. Visualize the results
    z_slice_to_plot = generator.dims[2] // 2  # Plot the central z-slice
    time_points_to_plot = range(0, 8) #[0, 2, 4, 7]      # Plot a few time points
    coils_to_plot = range(0, 16) #[0:15]                 # Plot a few coils

    plot_phantom_outputs(
        phantom_data=phantom_data,
        z_slice_idx=z_slice_to_plot,
        time_points=time_points_to_plot,
        coils_to_plot=coils_to_plot,
        filename_prefix='phantom_output_plots/phantom'
    )