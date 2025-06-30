# phantom_generator.py
import logging
import math
import time
from typing import List, Dict

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# fft and ifft utility functions from the original code
def fft(x: torch.Tensor, dim: tuple = (-3, -2, -1)) -> torch.Tensor:
    """Centered 3D Fast Fourier Transform."""
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )

def ifft(x: torch.Tensor, dim: tuple = (-3, -2, -1)) -> torch.Tensor:
    """Centered 3D Inverse Fast Fourier Transform."""
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )

def normalize_sens_maps_sos1(
    sens_maps: torch.Tensor,
    coil_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalizes complex sensitivity maps so that the sum-of-squares (sos) magnitude across coils is 1."""
    with torch.no_grad():
        sos = torch.sqrt(torch.sum(sens_maps.abs().pow(2), dim=coil_dim, keepdim=True))
        sens_maps_scaled = sens_maps / (sos + eps)
    return sens_maps_scaled


class DigitalPhantomGenerator:
    """
    Generates a 4D (3D+time) DCE-MRI digital phantom based on physics simulations.

    This class encapsulates the logic for:
    1. Creating a static 3D anatomical structure (Shepp-Logan based).
    2. Simulating the dynamic flow of a contrast agent using PDEs for advection-diffusion.
    3. Simulating multi-coil sensitivity maps.
    4. Combining these to produce ground truth coilless and coil-specific images.
    """
    def __init__(
        self,
        phantom_dims: List[int] = [64, 64, 32, 16, 8],  # [nx, ny, nz, nt, nc]
        phantom_vx: float = 0.0,
        phantom_vy: float = 0.0,
        phantom_vz: float = 1.5,
        phantom_D: float = 0.005,
        phantom_lambda_decay: float = 0.5,
        phantom_source_strength: float = 8.0,
        phantom_source_t_end: float = 0.15,
        phantom_source_xyz_center: List[float] = [0.0, 0.0, -0.8],
        phantom_source_xyz_std: List[float] = [0.05, 0.05, 0.05],
        phantom_lambda_drift: float = 0.05,
        phantom_S_infty: float = 0.1,
        phantom_K_phase: float = 2.0,
        phantom_T_total: float = 2.0,
        phantom_dt_ratio: float = 0.05,
        device: str = 'cpu'
    ):
        """Initializes the phantom generator with all necessary parameters."""
        if not phantom_dims or len(phantom_dims) != 5:
            raise ValueError("`phantom_dims` must be a list of 5 integers: [nx, ny, nz, nt, nc]")

        self.dims = phantom_dims
        self.device = torch.device(device)
        self.phys_params = {
            "vx": phantom_vx, "vy": phantom_vy, "vz": phantom_vz,
            "D": phantom_D, "lambda_decay": phantom_lambda_decay,
            "lambda_drift": phantom_lambda_drift, "S_infty": phantom_S_infty,
            "source_strength": phantom_source_strength, "source_t_end": phantom_source_t_end,
            "source_xyz_center": phantom_source_xyz_center, "source_xyz_std": phantom_source_xyz_std,
            "dt_ratio": phantom_dt_ratio, "K_phase": phantom_K_phase, "T_total": phantom_T_total,
        }
        logger.info(f"DigitalPhantomGenerator initialized with parameters: {self.phys_params}")


    def generate(self) -> Dict[str, torch.Tensor]:
        """
        Generates the complete set of phantom data.

        Returns:
            A dictionary containing the generated torch tensors:
            - 'gt_coilless_image': [T, X, Y, Z] complex64 - The ground truth image without coil effects.
            - 'sensitivity_maps': [X, Y, Z, C] complex64 - The simulated coil sensitivity maps.
            - 'coil_images': [C, T, X, Y, Z] complex64 - Coil-specific images.
            - 'kspace_cartesian_dense': [C, T, kX, kY, kZ] complex64 - Fully sampled k-space.
        """
        nx, ny, nz, nt, nc = self.dims
        logger.info(f"Generating Combined Phantom: Grid={nx}x{ny}x{nz}, T={nt}, Coils={nc}")

        # 1. Create static background anatomy
        static_background_img = self._create_static_spatial_phantom([nx, ny, nz]).to(self.device)

        # 2. Solve PDEs for the dynamic component
        static_mask_for_pde = (torch.abs(static_background_img) > 1e-6).float()
        static_mask_solver_shape_cpu = static_mask_for_pde.permute(2, 1, 0)  # [nz, ny, nx] for PDE solver

        pde_dynamic_component_img = self._solve_pdes_finite_difference(
            nx, ny, nz, nt, self.phys_params, static_mask_solver_shape_cpu
        ).to(self.device)

        # 3. Combine static and dynamic parts
        gt_coilless_image_combined = pde_dynamic_component_img + static_background_img.unsqueeze(-1)
        gt_coilless_image_final = gt_coilless_image_combined.permute(3, 0, 1, 2)  # T, X, Y, Z

        logger.info(f"Combined static + dynamic coilless image shape: {gt_coilless_image_final.shape}")

        # 4. Create and normalize sensitivity maps
        raw_sens_maps = self._create_phantom_sensitivity_maps([nx, ny, nz], nc).to(self.device)
        sens_maps_norm = normalize_sens_maps_sos1(raw_sens_maps, coil_dim=3)

        # 5. Create coil-specific images
        coil_images_dense = gt_coilless_image_combined.unsqueeze(4) * sens_maps_norm.unsqueeze(3)
        coil_images_dense_CTXYZ = coil_images_dense.permute(4, 3, 0, 1, 2)  # [C, T, X, Y, Z]

        # 6. Generate dense k-space via FFT
        kspace_cartesian_dense_CTKXYZ = fft(coil_images_dense_CTXYZ, dim=(-3, -2, -1))
        logger.info(f"Simulated full dense Cartesian k-space shape: {kspace_cartesian_dense_CTKXYZ.shape}")

        return {
            "gt_coilless_image": gt_coilless_image_final,
            "sensitivity_maps": sens_maps_norm,
            "coil_images": coil_images_dense_CTXYZ,
            "kspace_cartesian_dense": kspace_cartesian_dense_CTKXYZ,
        }

    def _solve_pdes_finite_difference(self, nx, ny, nz, nt_out, params, static_mask_cpu):
        """Solves the coupled artery (C) and drift (B) PDEs using FTCS finite differences."""
        # This method is copied directly from the original class, with minor adjustments
        start_time = time.time()
        sim_device = self.device
        logger.info(f"Running PDE simulation on device: {sim_device}")
        
        phantom_mask = static_mask_cpu.to(sim_device)

        dx = 2.0 / (nx - 1) if nx > 1 else 1.0
        dy = 2.0 / (ny - 1) if ny > 1 else 1.0
        dz = 2.0 / (nz - 1) if nz > 1 else 1.0
        coords_x = torch.linspace(-1, 1, nx, device=sim_device, dtype=torch.float32)
        coords_y = torch.linspace(-1, 1, ny, device=sim_device, dtype=torch.float32)
        coords_z = torch.linspace(-1, 1, nz, device=sim_device, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(coords_z, coords_y, coords_x, indexing="ij")

        T_total = params["T_total"]
        D = params["D"]
        vx, vy, vz = params["vx"], params["vy"], params["vz"]
        max_v = max(abs(vx), abs(vy), abs(vz))
        min_dsq = min(dx**2, dy**2, dz**2) if D > 1e-9 else 1.0
        min_d = min(dx, dy, dz)
        dt_max_diff = 0.5 * min_dsq / (D + 1e-9) if D > 1e-9 else float("inf")
        dt_max_adv = min_d / (max_v + 1e-9) if max_v > 1e-9 else float("inf")
        dt_max = min(dt_max_diff, dt_max_adv)
        dt = dt_max * params["dt_ratio"]
        N_steps = int(np.ceil(T_total / dt))
        logger.info(f"PDE Simulation: T_total={T_total:.2f}, dt={dt:.2e}, N_steps={N_steps}")

        v = torch.tensor([vz, vy, vx], device=sim_device, dtype=torch.float32)
        lambda_decay = torch.tensor(params["lambda_decay"], device=sim_device, dtype=torch.float32)
        lambda_drift = torch.tensor(params["lambda_drift"], device=sim_device, dtype=torch.float32)
        S_infty = torch.tensor(params["S_infty"], device=sim_device, dtype=torch.float32)
        K_phase = torch.tensor(params["K_phase"], device=sim_device, dtype=torch.float32)
        D_tensor = torch.tensor(D, device=sim_device, dtype=torch.float32)

        src_t_end = params["source_t_end"] * T_total
        src_center = torch.tensor(params["source_xyz_center"][::-1], device=sim_device, dtype=torch.float32)
        src_std = torch.tensor(params["source_xyz_std"][::-1], device=sim_device, dtype=torch.float32)
        src_std_sq = src_std.pow(2) + 1e-9
        dist_sq = ((zz - src_center[0])**2 / src_std_sq[0] + (yy - src_center[1])**2 / src_std_sq[1] + (xx - src_center[2])**2 / src_std_sq[2])
        source_spatial = params["source_strength"] * torch.exp(-0.5 * dist_sq)
        
        def source_fn(t):
            return source_spatial if t < src_t_end else torch.zeros_like(source_spatial)

        C = torch.zeros((nz, ny, nx), device=sim_device, dtype=torch.float32)
        B = torch.zeros((nz, ny, nx), device=sim_device, dtype=torch.float32)
        results_C = torch.zeros((nt_out, nz, ny, nx), device=sim_device, dtype=torch.float32)
        results_B = torch.zeros((nt_out, nz, ny, nx), device=sim_device, dtype=torch.float32)
        output_indices = torch.linspace(0, N_steps - 1, nt_out, device=sim_device).round().long()
        output_counter = 0

        pbar_pde = tqdm(range(N_steps), desc="PDE Simulation", leave=False)
        for i in pbar_pde:
            C_prev = C
            B_prev = B
            C_padded = torch.nn.functional.pad(C_prev.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="replicate").squeeze()
            
            grad_C_z = (C_padded[2:, 1:-1, 1:-1] - C_padded[:-2, 1:-1, 1:-1]) / (2 * dz)
            grad_C_y = (C_padded[1:-1, 2:, 1:-1] - C_padded[1:-1, :-2, 1:-1]) / (2 * dy)
            grad_C_x = (C_padded[1:-1, 1:-1, 2:] - C_padded[1:-1, 1:-1, :-2]) / (2 * dx)
            
            lap_C = ((C_padded[2:, 1:-1, 1:-1] - 2 * C_prev + C_padded[:-2, 1:-1, 1:-1]) / dz**2 +
                     (C_padded[1:-1, 2:, 1:-1] - 2 * C_prev + C_padded[1:-1, :-2, 1:-1]) / dy**2 +
                     (C_padded[1:-1, 1:-1, 2:] - 2 * C_prev + C_padded[1:-1, 1:-1, :-2]) / dx**2)
            
            adv_term = v[0] * grad_C_z + v[1] * grad_C_y + v[2] * grad_C_x
            dCdt = -adv_term + D_tensor * lap_C - lambda_decay * C_prev + source_fn(i * dt)
            dBdt = -lambda_drift * (B_prev - S_infty)
            
            C = C_prev + dt * dCdt
            B = B_prev + dt * dBdt
            C.clamp_(min=0.0)
            B.clamp_(min=0.0)

            if output_counter < nt_out and i == output_indices[output_counter]:
                results_C[output_counter] = C.clone()
                results_B[output_counter] = B.clone()
                output_counter += 1
        
        pbar_pde.close()
        logger.info("PDE time stepping finished.")
        
        magnitude = results_C + results_B * phantom_mask
        phase = K_phase * results_C
        coilless_complex_dynamic = magnitude * torch.exp(1j * phase)
        coilless_complex_dynamic = rearrange(coilless_complex_dynamic, "t z y x -> x y z t")
        
        logger.info(f"PDE simulation finished in {time.time() - start_time:.1f} seconds.")
        return coilless_complex_dynamic.cpu()

    def _create_static_spatial_phantom(self, spatial_dims):
        """Creates a static 3D coilless phantom image based on Shepp-Logan ellipsoids."""
        # This method is copied directly from the original class
        nx, ny, nz = spatial_dims
        x_coords = torch.linspace(-nx / 2, nx / 2, nx, dtype=torch.float32)
        y_coords = torch.linspace(-ny / 2, ny / 2, ny, dtype=torch.float32)
        z_coords = torch.linspace(-nz / 2, nz / 2, nz, dtype=torch.float32)
        yy, xx, zz = torch.meshgrid(y_coords, x_coords, z_coords, indexing="ij")
        max_dim = max(nx, ny, nz)
        scale_x, scale_y, scale_z = nx / max_dim, ny / max_dim, nz / max_dim
        
        phantom_params = [
            [0.6, 0.69 * scale_x, 0.92 * scale_y, 0.9 * scale_z, 0, 0, 0, 0], [-0.2, 0.10 * scale_x, 0.30 * scale_y, 0.4 * scale_z, 0.22 * nx / 2, 0, 0, -18], [-0.2, 0.16 * scale_x, 0.40 * scale_y, 0.4 * scale_z, -0.22 * nx / 2, 0, 0, 18], [0.1, 0.21 * scale_x, 0.25 * scale_y, 0.5 * scale_z, 0, 0.35 * ny / 2, 0, 0], [1.0, 0.04 * scale_x, 0.10 * scale_y, 0.1 * scale_z, 0, 0.10 * ny / 2, 0, 0], [1.0, 0.04 * scale_x, 0.10 * scale_y, 0.1 * scale_z, 0.10 * nx / 2, -0.10 * ny / 2, 0.2 * nz / 2, 0], [0.1, 0.02 * scale_x, 0.06 * scale_y, 0.1 * scale_z, -0.1 * nx / 2, 0.10 * ny / 2, -0.2 * nz / 2, 0],
        ]
        
        static_img_mag = torch.zeros((ny, nx, nz), dtype=torch.float32)
        for p in phantom_params:
            A, a, b, c, x0, y0, z0, phi_deg = p
            phi = torch.tensor(phi_deg * math.pi / 180.0)
            a_pix, b_pix, c_pix = a * max_dim / 2, b * max_dim / 2, c * max_dim / 2
            x_c, y_c, z_c = xx - x0, yy - y0, zz - z0
            x_rot = x_c * torch.cos(phi) + y_c * torch.sin(phi)
            y_rot = -x_c * torch.sin(phi) + y_c * torch.cos(phi)
            mask = ((x_rot / (a_pix + 1e-9))**2 + (y_rot / (b_pix + 1e-9))**2 + (z_c / (c_pix + 1e-9))**2) < 1.0
            static_img_mag[mask] += A

        object_mask = (torch.abs(static_img_mag) > 1e-4).float()
        structured_phase_map = (xx * 0.01 + yy * 0.005 - zz * 0.008) * math.pi
        random_phase_map = torch.rand_like(static_img_mag) * (2 * math.pi) - math.pi
        combined_phase_map = torch.where(object_mask > 0.5, structured_phase_map, random_phase_map)
        
        static_img_complex = static_img_mag.to(torch.complex64) * torch.exp(1j * combined_phase_map)
        return static_img_complex.permute(1, 0, 2)

    def _create_phantom_sensitivity_maps(self, spatial_dims, num_coils):
        """Creates simple Biot-Savart-like sensitivity maps."""
        # This method is copied directly from the original class
        nx, ny, nz = spatial_dims
        x_coords = torch.linspace(-1, 1, nx)
        y_coords = torch.linspace(-1, 1, ny)
        z_coords = torch.linspace(-1, 1, nz)
        yy, xx, zz = torch.meshgrid(y_coords, x_coords, z_coords, indexing="ij")
        sensitivity_maps = torch.zeros(nx, ny, nz, num_coils, dtype=torch.complex64)
        coil_radius = 1.2
        
        for i in range(num_coils):
            coil_angle = 2 * math.pi * i / num_coils
            cx, cy = coil_radius * math.cos(coil_angle), coil_radius * math.sin(coil_angle)
            distance_sq = (xx - cx)**2 + (yy - cy)**2 + zz**2
            magnitude = torch.exp(-distance_sq / (2 * 0.8**2))
            phase = (xx * -cy + yy * cx) * 1.5
            complex_map = magnitude.permute(1, 0, 2) * torch.exp(1j * phase.permute(1, 0, 2))
            sensitivity_maps[:, :, :, i] = complex_map
            
        return sensitivity_maps