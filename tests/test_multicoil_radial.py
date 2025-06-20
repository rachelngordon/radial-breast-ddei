import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# --- Import your project files ---
# Make sure these paths are correct relative to where you run the script
from dataloader import SliceDataset
from radial import DynamicRadialPhysics, from_torch_complex, to_torch_complex


def plot_comparison(
    img1, label1, img2, label2, plot_title, filename, output_dir="test_outputs"
):
    """Plots two dynamic images side-by-side for comparison."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to magnitude images for plotting
    img1_mag = torch.sqrt(img1[:, 0, ...] ** 2 + img1[:, 1, ...] ** 2).cpu().numpy()
    img2_mag = torch.sqrt(img2[:, 0, ...] ** 2 + img2[:, 1, ...] ** 2).cpu().numpy()

    B, T, H, W = img1_mag.shape
    fig, axes = plt.subplots(
        nrows=2, ncols=T, figsize=(T * 3, 6.5), squeeze=False
    )

    for t in range(T):
        # Plot image 1
        ax1 = axes[0, t]
        ax1.imshow(np.rot90(img1_mag[0, t], 2), cmap="gray")
        ax1.set_title(f"t = {t}")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot image 2
        ax2 = axes[1, t]
        ax2.imshow(np.rot90(img2_mag[0, t], 2), cmap="gray")
        ax2.set_title(f"t = {t}")
        ax2.set_xticks([])
        ax2.set_yticks([])

    axes[0, 0].set_ylabel(label1, fontsize=14, labelpad=10)
    axes[1, 0].set_ylabel(label2, fontsize=14, labelpad=10)

    fig.suptitle(plot_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close(fig)


def run_adjoint_test(physics, csmap, device):
    """Performs the dot-product test to verify the adjoint property."""
    print("\n--- Running Adjoint (Dot-Product) Test ---")
    B, Co, H, W = csmap.shape
    T = physics.N_time
    S = physics.N_spokes
    I = physics.N_samples

    # Create random tensors in image and k-space domains
    x_rand = torch.randn(B, 2, T, H, W, device=device)
    y_rand = torch.randn(B, 2, T, Co, S, I, device=device)

    # Move csmap to device and ensure batch dimension
    csmap = csmap.to(device)
    if csmap.dim() == 3:
        csmap = csmap.unsqueeze(0)

    # 1. Forward pass: x -> y
    y_forward = physics.A(x_rand, csmap)

    # 2. Adjoint pass: y -> x
    x_adjoint = physics.A_adjoint(y_rand, csmap)

    # Convert to complex for dot product
    x_rand_c = to_torch_complex(x_rand)
    y_rand_c = to_torch_complex(y_rand)
    y_forward_c = to_torch_complex(y_forward)
    x_adjoint_c = to_torch_complex(x_adjoint)
    
    # Calculate the two sides of the adjoint equation: <Ax, y> and <x, A*y>
    lhs = torch.sum(y_forward_c * y_rand_c.conj())
    rhs = torch.sum(x_rand_c * x_adjoint_c.conj())
    
    relative_error = torch.abs(lhs - rhs) / torch.abs(lhs)

    print(f"LHS <A(x), y>      : {lhs.item()}")
    print(f"RHS <x, A_adjoint(y)>: {rhs.item()}")
    print(f"Relative Error       : {relative_error.item():.6e}")

    if relative_error < 1e-4: # Tolerance for float32 precision
        print("✅ Adjoint test PASSED.")
    else:
        print("❌ Adjoint test FAILED. The A and A_adjoint operators are not consistent.")
        print("   This will break the MC loss during training.")


def plot_csmap_comparison(csmap_orig, csmap_flipped, output_dir="test_outputs"):
    """Plots the first coil of the original and flipped CSMaps."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the magnitude of the first coil for visualization
    csmap_orig_mag = torch.abs(csmap_orig[0, 0, ...]).cpu().numpy()
    csmap_flipped_mag = torch.abs(csmap_flipped[0, 0, ...]).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(csmap_orig_mag, cmap='viridis')
    axes[0].set_title("Original CSMap (Coil 1)")
    axes[0].axis('off')
    
    axes[1].imshow(csmap_flipped_mag, cmap='viridis')
    axes[1].set_title("Flipped CSMap (Coil 1)")
    axes[1].axis('off')
    
    plt.suptitle("CSMap Flip Verification")
    save_path = os.path.join(output_dir, "csmap_flip_verification.png")
    plt.savefig(save_path)
    print(f"Saved CSMap verification plot to {save_path}")
    plt.close(fig)


def main(config_path):
    # --- 1. Load Configuration ---
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    device = torch.device(config["training"]["device"])

    # --- 2. Load Real Data ---
    print("--- Loading a real data sample ---")
    with open(config["data"]["split_file"], "r") as fp:
        val_patient_ids = json.load(fp)["val"]

    val_dataset = SliceDataset(
        root_dir=config["data"]["root_dir"],
        patient_ids=val_patient_ids[:1],  # Just use one patient for the test
        dataset_key=config["data"]["dataset_key"],
        slice_idx=config["dataloader"]["slice_idx"],
        N_coils=config["data"]["coils"],
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    measured_kspace, csmap, grasp_img = next(iter(val_loader))

    # Move data to the target device
    measured_kspace = measured_kspace.to(device)
    csmap = csmap.to(device)
    grasp_img = grasp_img.to(device)

    grasp_img = torch.flip(grasp_img, dims=[3, 4])

    csmap_flipped = torch.flip(csmap, dims=[2])
    plot_csmap_comparison(csmap, csmap_flipped)

    # --- 3. Instantiate Physics Operator ---
    physics = DynamicRadialPhysics(
        im_size=(config["data"]["height"], config["data"]["width"]),
        N_spokes=int(config["data"]["total_spokes"] / config["data"]["timeframes"]),
        N_samples=config["data"]["spokes_per_frame"],
        N_time=config["data"]["timeframes"],
        N_coils=config["data"]["coils"],
    )

    # === TEST 1: Zero-Filled Reconstruction from Real Data ===
    print("\n--- Running Test 1: Zero-Filled Reconstruction ---")
    with torch.no_grad():
        x_zf = physics.A_adjoint(measured_kspace, csmap)

    plot_comparison(
        x_zf, "ZF Recon (A_adjoint)",
        grasp_img, "GRASP Benchmark",
        "Test 1: ZF Reconstruction vs. GRASP",
        "test1_zf_vs_grasp"
    )
    print("Observation: The ZF Recon should show anatomy but have heavy streaking artifacts.")

    # === TEST 2: Round-Trip Test (A -> A_adjoint) ===
    print("\n--- Running Test 2: Round-Trip (A -> A_adjoint) ---")
    with torch.no_grad():
        # Use the clean GRASP image as the ground truth input
        y_generated = physics.A(grasp_img, csmap)
        x_reconstructed = physics.A_adjoint(y_generated, csmap)

    plot_comparison(
        x_reconstructed, "Reconstructed from A(GRASP)",
        grasp_img, "Original GRASP Image",
        "Test 2: Round-Trip Test",
        "test2_round_trip"
    )
    print("Observation: If A and A_adjoint are consistent, the reconstructed image")
    print("should look like a streaky version of the original GRASP image.")


    # === TEST 3: Mathematical Adjoint Test ===
    run_adjoint_test(physics, csmap, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the DynamicRadialPhysics operator.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(args.config)