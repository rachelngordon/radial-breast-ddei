import deepinv as dinv
import numpy as np
import torch
import torch.nn as nn
from deepinv.physics.time import TimeMixin
from einops import rearrange
from torchkbnufft import KbNufft, KbNufftAdjoint


def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    return torch.view_as_complex(rearrange(x, "b c ... -> b ... c").contiguous())


def from_torch_complex(x: torch.Tensor):
    """(B, ...) complex -> (B, 2, ...) real"""
    return rearrange(torch.view_as_real(x), "b ... c -> b c ...").contiguous()


# This class only knows how to handle a batch of 2D images (4D Tensors)
class RadialPhysics(dinv.physics.Physics):
    def __init__(self, im_size, N_spokes, N_samples, **kwargs):
        super().__init__(**kwargs)
        self.im_size = im_size
        self.N_spokes = N_spokes
        self.N_samples = N_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        spokelength = im_size[1] * 2
        grid_size = (int(spokelength), int(spokelength))
        self.NUFFT = KbNufft(im_size=im_size, grid_size=grid_size).to(self.device)
        self.AdjNUFFT = KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(
            self.device
        )

        self.traj, self.dcf = self.get_traj_and_dcf()
        self.traj = self.traj.to(self.device)
        self.dcf = self.dcf.to(self.device)

    def get_traj_and_dcf(self):
        # Generates a SINGLE frame trajectory (B, D, N) and DCF
        base_res = self.N_samples // 2
        base_lin = np.arange(self.N_samples).reshape(1, -1) - base_res
        ga = np.pi * (3.0 - np.sqrt(5.0))
        base_rot = np.arange(self.N_spokes).reshape(-1, 1) * ga

        traj_flat = np.zeros((self.N_spokes, self.N_samples, 2))
        traj_flat[..., 0] = np.cos(base_rot) @ base_lin
        traj_flat[..., 1] = np.sin(base_rot) @ base_lin
        traj_flat = traj_flat / base_res

        traj = torch.from_numpy(traj_flat).float()
        # Shape: (1, Dims=2, Spokes*Samples)
        traj_nufft_ready = rearrange(traj, "s i xy -> 1 xy (s i)")

        # DCF calculation
        dcf_vals = torch.sqrt(
            traj_nufft_ready[0, 0, :] ** 2 + traj_nufft_ready[0, 1, :] ** 2
        )

        # --- LIKELY FIX: Add a small epsilon to prevent division by zero ---
        epsilon = 1e-8
        dcf_vals += epsilon

        dcf = rearrange(dcf_vals, "(s i) -> 1 1 s i", s=self.N_spokes)

        return traj_nufft_ready, dcf

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input x: (B, C=2, H, W) real-valued
        x_complex = to_torch_complex(x)  # -> (B, H, W) complex
        # Add coil dimension for NUFFT
        k_complex_nufft = self.NUFFT(x_complex.unsqueeze(1), self.traj)
        # Remove coil dim and convert back to real
        y = from_torch_complex(k_complex_nufft.squeeze(1))  # -> (B, C=2, N_points)
        # Reshape to (B, C, S, I)
        return rearrange(y, "b c (s i) -> b c s i", s=self.N_spokes)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        y_flat = rearrange(y, "b c s i -> b c (s i)")
        y_complex_nufft = to_torch_complex(y_flat).unsqueeze(1)

        dcf_flat = rearrange(self.dcf, "b c s i -> b c (s i)")
        y_dcf_complex = y_complex_nufft * dcf_flat  # This multiplication is the key

        # --- DEBUG: check if y_dcf_complex has NaNs ---
        if torch.isnan(y_dcf_complex).any():
            print("!!! ERROR: NaN detected in y_dcf_complex in A_adjoint !!!")
            # This would happen if dcf_flat contained NaNs, which the epsilon should prevent.

        x_complex = self.AdjNUFFT(y_dcf_complex, self.traj).squeeze(1)
        return from_torch_complex(x_complex)


# This class now handles 5D video tensors by inheriting from our 2D class and TimeMixin
class DynamicRadialPhysics(RadialPhysics, TimeMixin):
    def __init__(self, im_size, N_spokes, N_samples, N_time, N_coils=1, **kwargs):
        super().__init__(
            im_size=im_size[:2], N_spokes=N_spokes, N_samples=N_samples, **kwargs
        )

        # This class uses N_time and N_coils.
        self.N_time = N_time
        self.N_coils = N_coils

        # The mask for dynamic physics has a time dimension
        self.mask = torch.ones(1, 2, self.N_time, self.N_spokes, self.N_samples).to(
            self.device
        )

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input x: (B, C, T, H, W)
        # Use TimeMixin to flatten time into the batch dimension
        x_flat = self.flatten(x)  # -> (B*T, C, H, W)

        # Call the base class (RadialPhysics) A method on the batch of 2D images
        y_flat = super().A(x_flat, **kwargs)  # -> (B*T, C, S, I)

        # Unflatten to restore the time dimension
        y = self.unflatten(y_flat, batch_size=x.shape[0])  # -> (B, C, T, S, I)

        # Apply the time-varying mask
        return y * self.mask

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input y: (B, C, T, S, I)
        # Apply the time-varying mask
        y_masked = y * self.mask

        # Use TimeMixin to flatten time into the batch dimension
        y_flat = self.flatten(y_masked)  # -> (B*T, C, S, I)

        # Call the base class (RadialPhysics) A_adjoint on the batch of 2D k-spaces
        x_flat = super().A_adjoint(y_flat, **kwargs)  # -> (B*T, C, H, W)

        # Unflatten to restore the time dimension
        x = self.unflatten(x_flat, batch_size=y.shape[0])  # -> (B, C, T, H, W)

        return x


class RadialDCLayer(nn.Module):
    """
    Final Data Consistency layer.
    It takes the network's current image estimate and the original measurements,
    and returns a new image estimate that is a weighted average in k-space.
    """

    def __init__(
        self,
        physics: nn.Module,  # The DC layer now requires the physics operator
        lambda_init=np.log(np.exp(1) - 1.0) / 1.0,
        learnable=True,
    ):
        super(RadialDCLayer, self).__init__()
        self.learnable = learnable
        self.lambda_ = nn.Parameter(
            torch.ones(1) * lambda_init, requires_grad=self.learnable
        )
        self.physics = physics

    def forward(self, x_img_permuted, y_kspace_meas, mask_kspace):
        # x_img_permuted from CRNN: (b, h, w, t, c)
        # y_kspace_meas from dataloader: (b, c, t, s, i)
        # mask_kspace: same shape as y_kspace_meas
        x_img = rearrange(x_img_permuted, "b h w t c -> b c t h w")
        y = y_kspace_meas

        # --- DEBUG PRINTS ---
        print("\n--- [DEBUG RadialDCLayer.forward] ---")
        print(f"Input image (x_img) has NaNs: {torch.isnan(x_img).any().item()}")
        print(f"Input image (x_img) has Infs: {torch.isinf(x_img).any().item()}")
        print(f"Input image (x_img) max value: {x_img.max().item()}")

        A_x = self.physics.A(x_img)

        print(f"k-space of net output (A_x) has NaNs: {torch.isnan(A_x).any().item()}")
        print(f"k-space of net output (A_x) has Infs: {torch.isinf(A_x).any().item()}")
        print(f"k-space of net output (A_x) max value: {A_x.max().item()}")

        print(f"Measured k-space (y) has NaNs: {torch.isnan(y).any().item()}")
        print(f"Measured k-space (y) has Infs: {torch.isinf(y).any().item()}")
        # --- END DEBUG PRINTS ---

        mask = torch.ones_like(A_x)
        lambda_ = torch.sigmoid(self.lambda_)

        k_dc = (1 - mask) * A_x + mask * (lambda_ * A_x + (1 - lambda_) * y)

        # Step 3: Transform the corrected k-space back to image space.
        # The physics operator `A_adjoint` handles all Adjoint NUFFT logic.
        x_dc_img = self.physics.A_adjoint(k_dc)

        # Step 4: Convert back to CRNN's expected permuted format.
        x_dc_permuted = rearrange(x_dc_img, "b c t h w -> b h w t c")

        return x_dc_permuted
