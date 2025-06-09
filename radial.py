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

        # --- KEEP THIS CHANGE: Increased grid size for better precision ---
        grid_size = [int(s * 2.0) for s in im_size]

        # --- Revert to using the functional KbNufft and KbNufftAdjoint ---
        self.NUFFT = KbNufft(im_size=im_size, grid_size=grid_size).to(self.device)
        self.AdjNUFFT = KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(
            self.device
        )

        self.traj, self.sqrt_dcf = self.get_traj_and_dcf()
        self.traj = self.traj.to(self.device)
        self.sqrt_dcf = self.sqrt_dcf.to(self.device)

    def get_traj_and_dcf(self, angle_offset_rad=0.0):
        base_res = self.im_size[0]
        gind = 1

        N_samples = base_res * 2

        # Check if this matches your class's N_samples
        if N_samples != self.N_samples:
            print(
                f"Warning: Vendor logic implies N_samples should be {N_samples}, but class has {self.N_samples}. Using class value."
            )
            N_samples = self.N_samples  # Trust the class value passed during init

        base_lin = np.arange(N_samples).reshape(1, -1) - (N_samples // 2)

        tau = 0.5 * (1 + 5**0.5)

        base_rad = np.pi / (gind + tau - 1)
        base_rad = np.pi / (gind + tau - 1)

        spoke_indices = np.arange(self.N_spokes)
        base_rot = (spoke_indices * base_rad + angle_offset_rad).reshape(-1, 1)

        traj_flat = np.zeros((self.N_spokes, N_samples, 2))
        traj_flat[..., 0] = np.cos(base_rot) @ base_lin
        traj_flat[..., 1] = np.sin(base_rot) @ base_lin

        # --- Now, scale for torchkbnufft which expects [-pi, pi] ---
        # The vendor code scales by dividing by 2. It's unclear what that means.
        # The robust way is to find the max radius and scale to pi.
        # The max radius in traj_flat will be N_samples / 2.
        max_radius = N_samples / 2.0
        traj_flat = (traj_flat / max_radius) * np.pi

        traj = torch.from_numpy(traj_flat).float()
        traj_nufft_ready = rearrange(traj, "s i xy -> 1 xy (s i)")

        # The DCF calculation can remain the same, it's based on the final trajectory
        dcf_vals = torch.sqrt(
            traj_nufft_ready[0, 0, :] ** 2 + traj_nufft_ready[0, 1, :] ** 2
        )
        sqrt_dcf_vals = torch.sqrt(dcf_vals)

        sqrt_dcf = rearrange(sqrt_dcf_vals, "(s i) -> 1 1 (s i)", s=self.N_spokes)
        return traj_nufft_ready, sqrt_dcf

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # The call signature is back to the original version
        x_complex = to_torch_complex(x).unsqueeze(1)
        k_complex_nufft = self.NUFFT(x_complex, self.traj)

        y_complex_weighted = k_complex_nufft * self.sqrt_dcf

        y = from_torch_complex(y_complex_weighted.squeeze(1))
        return rearrange(y, "b c (s i) -> b c s i", s=self.N_spokes)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # The call signature is back to the original version
        y_flat = rearrange(y, "b c s i -> b c (s i)")
        y_complex = to_torch_complex(y_flat).unsqueeze(1)

        y_dcf_complex = y_complex * self.sqrt_dcf

        if torch.isnan(y_dcf_complex).any():
            print("!!! ERROR: NaN detected in y_dcf_complex in A_adjoint !!!")

        x_complex = self.AdjNUFFT(y_dcf_complex, self.traj).squeeze(1)
        return from_torch_complex(x_complex)


# This class now handles 5D video tensors by inheriting from our 2D class and TimeMixin
class DynamicRadialPhysics(RadialPhysics, TimeMixin):
    def __init__(self, im_size, N_spokes, N_samples, N_time, N_coils=1, **kwargs):
        # We call the TimeMixin's init first
        TimeMixin.__init__(self)

        # We call the base Physics init, not RadialPhysics's init directly
        dinv.physics.Physics.__init__(self, **kwargs)

        # Store all dynamic parameters
        self.im_size = im_size[:2]  # Static image size
        self.N_spokes = N_spokes  # Spokes PER FRAME
        self.N_samples = N_samples
        self.N_time = N_time
        self.N_coils = N_coils
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Instantiate NUFFT operators ---
        grid_size = [int(s * 2.0) for s in self.im_size]
        self.NUFFT = KbNufft(im_size=self.im_size, grid_size=grid_size).to(self.device)
        self.AdjNUFFT = KbNufftAdjoint(im_size=self.im_size, grid_size=grid_size).to(
            self.device
        )

        # --- GENERATE THE FULL DYNAMIC TRAJECTORY ---
        # We generate the trajectory for ALL spokes across ALL time frames at once.
        total_spokes_in_scan = self.N_spokes * self.N_time

        # Create a temporary static physics object just to call its get_traj_and_dcf
        temp_physics = RadialPhysics(
            self.im_size, N_spokes=total_spokes_in_scan, N_samples=self.N_samples
        )

        # This full_traj has shape (1, 2, N_spokes*N_time*N_samples)
        full_traj, full_sqrt_dcf = temp_physics.get_traj_and_dcf()

        # --- Reshape the trajectory and DCF to be time-aware ---
        # New shape: (T, 1, 2, S*I) for easy selection later
        self.traj_per_frame = rearrange(
            full_traj, "b c (t s i) -> t b c (s i)", t=self.N_time, s=self.N_spokes
        )
        self.sqrt_dcf_per_frame = rearrange(
            full_sqrt_dcf, "b c (t s i) -> t b c (s i)", t=self.N_time, s=self.N_spokes
        )

        self.traj_per_frame = self.traj_per_frame.to(self.device)
        self.sqrt_dcf_per_frame = self.sqrt_dcf_per_frame.to(self.device)

        self.mask = torch.ones(1, 2, self.N_time, self.N_spokes, self.N_samples).to(
            self.device
        )

    # --- We need to override A and A_adjoint to use the per-frame trajectory ---
    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x has shape (B, C, T, H, W)
        B, C, T, H, W = x.shape
        output_kspace_frames = []

        for t in range(T):
            x_frame = x[:, :, t, :, :].unsqueeze(2)  # -> (B, C, 1, H, W)
            x_flat = self.flatten(x_frame)  # -> (B, C, H, W)

            # Use the trajectory for this specific time frame 't'
            traj_t = self.traj_per_frame[t]
            sqrt_dcf_t = self.sqrt_dcf_per_frame[t]

            x_complex = to_torch_complex(x_flat).unsqueeze(1)
            k_complex_nufft = self.NUFFT(x_complex, traj_t)
            y_complex_weighted = k_complex_nufft * sqrt_dcf_t

            y_frame = from_torch_complex(y_complex_weighted.squeeze(1))
            y_frame_reshaped = rearrange(
                y_frame, "b c (s i) -> b c s i", s=self.N_spokes
            )
            output_kspace_frames.append(y_frame_reshaped)

        y = torch.stack(output_kspace_frames, dim=2)  # Stack along the time dimension
        return y * self.mask

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # y has shape (B, C, T, S, I)
        B, C, T, S, I = y.shape
        output_image_frames = []

        y_masked = y * self.mask

        for t in range(T):
            y_frame = y_masked[:, :, t, :, :]  # -> (B, C, S, I)

            # Use the trajectory for this specific time frame 't'
            traj_t = self.traj_per_frame[t]
            sqrt_dcf_t = self.sqrt_dcf_per_frame[t]

            y_flat = rearrange(y_frame, "b c s i -> b c (s i)")
            y_complex = to_torch_complex(y_flat).unsqueeze(1)
            y_dcf_complex = y_complex * sqrt_dcf_t

            x_complex_frame = self.AdjNUFFT(y_dcf_complex, traj_t).squeeze(1)
            x_frame = from_torch_complex(x_complex_frame)  # -> (B, C, H, W)
            output_image_frames.append(x_frame)

        x = torch.stack(output_image_frames, dim=2)  # -> (B, C, T, H, W)
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

        A_x = self.physics.A(x_img)

        lambda_ = torch.sigmoid(self.lambda_)

        k_dc = (1 - mask_kspace) * A_x + mask_kspace * (
            lambda_ * A_x + (1 - lambda_) * y
        )

        # Step 3: Transform the corrected k-space back to image space.
        # The physics operator `A_adjoint` handles all Adjoint NUFFT logic.
        x_dc_img = self.physics.A_adjoint(k_dc)

        # Step 4: Convert back to CRNN's expected permuted format.
        x_dc_permuted = rearrange(x_dc_img, "b c t h w -> b h w t c")

        return x_dc_permuted
