import torch
import torch.nn as nn
from torchkbnufft import KbNufft, KbNufftAdjoint
import numpy as np
from einops import rearrange

class RadialDCLayerSingleCoil(nn.Module):
    def __init__(
        self,
        im_size,
        grid_size,
        lambda_init: float = np.log(np.exp(1) - 1.0) / 1.0,
        learnable: bool = True,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.device = device
        self.lambda_ = nn.Parameter(torch.ones(1) * lambda_init, requires_grad=learnable)

        # Forward NUFFT and adjoint (no smaps)
        self.nufft_op   = KbNufft(im_size=im_size, grid_size=grid_size).to(device)
        self.adjnufft_op = KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(device)

    def forward(self, x, y_radial, ktraj):
        """
        x: (batch, 1, H, W, 2) real
        y_radial: (n_samples, n_spokes, 2) real
        ktraj: (1, 2, n_samples*n_spokes) float
        """
        # x_c = x.to(dtype=torch.complex64)

        # (1) Forward NUFFT (no smaps => single‐coil)
        A_x = self.nufft_op(x.contiguous(), ktraj.contiguous(), smaps=None, norm='ortho')  # (batch, 1, n_samples*n_spokes, 2)

        # reshape simulated k-space
        A_x = rearrange(A_x, "b c r i -> b c i r ")#.to(dtype)
        A_x = torch.reshape(A_x, (1, 1, 2, 288, 640, 1)).squeeze()
        A_x = rearrange(A_x, 'i sp sam -> sam sp i')

        # apply density compensation function to simulated k-space
        dcf = np.sqrt(ktraj[..., 0] ** 2 + ktraj[..., 1] ** 2)  # shape: (N_TIME, N_SPOKES)
        dcf_tensor = (
            torch.tensor(dcf).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        A_x = A_x * dcf_tensor

        # (2) Weighted combine
        lambda_c = torch.sigmoid(self.lambda_).type(torch.complex64)
        k_dc = lambda_c * A_x + (1 - lambda_c) * y_radial

        # (3) Adjoint NUFFT
        x_dc = self.adjnufft_op(k_dc, ktraj, smaps=None, norm='ortho')  # (batch, 1, H, W)
        return x_dc
    
    def extra_repr(self):
        return f"lambda (raw)={self.lambda_.item():.4g}, learnable={self.lambda_.requires_grad}"




class RadialDCLayer(nn.Module):
    """
    Radial Data Consistency layer using torchkbnufft.
    Mirrors the Cartesian DCLayer, but with NUFFT-based forward/adjoint.
    """

    def __init__(
        self,
        im_size,
        grid_size,
        lambda_init: float = np.log(np.exp(1) - 1.0) / 1.0,
        learnable: bool = True,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Args:
            im_size (tuple of int): (height, width) for the NUFFT image domain.
            grid_size (tuple of int): (grid_h, grid_w) for the NUFFT gridding.
            lambda_init (float): initial value for the λ‐parameter (in log‐domain).
            learnable (bool): whether λ should be trainable.
            device (torch.device): where to place NUFFT operators and parameters.
        """
        super().__init__()
        self.device = device

        # Learnable λ parameter (same usage as Cartesian DCLayer)
        self.lambda_ = nn.Parameter(torch.ones(1) * lambda_init, requires_grad=learnable)

        # NUFFT forward operator: image -> non‐uniform k-space
        # We will pass smaps (coil sensitivities) at runtime to .forward(...)
        self.nufft_op = KbNufft(im_size=im_size, grid_size=grid_size).to(device)

        # NUFFT adjoint operator: k-space -> image (with coil sensitivities)
        self.adjnufft_op = KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(device)

    def forward(self, x, y_radial, ktraj, smaps):
        """
        Perform one Data Consistency pass on radial data.

        Args:
            x (torch.Tensor):
                Current image estimate of shape (batch, 1, height, width) if single-coil,
                or (batch, 1, height, width) for multi‐coil input (the layer applies
                smaps internally). Here we assume x is a single‐channel image.

            y_radial (torch.Tensor):
                Measured radial k-space, shape (batch, coils, n_samples, n_spokes).
                This is exactly what your simulation routines output (before reshaping).

            ktraj (torch.Tensor):
                Precomputed radial trajectory, shape (n_spokes * n_samples, 2)
                or (batch‐compatible) as required by torchkbnufft. In your helper
                you had shape (n_spokes * n_samples, time, 2), but here time=1.

            smaps (torch.Tensor):
                Coil‐sensitivity maps, shape (batch, coils, height, width).

        Returns:
            x_dc (torch.Tensor):
                Image after enforcing radial data consistency, shape (batch, 1, height, width).
        """

        # — Step 1: forward NUFFT to get simulated radial k-space from x —
        #    A_x has shape (batch, coils, n_samples, n_spokes)
        #    We feed x and smaps into nufft_op; KbNufft expects x of shape
        #    (batch, coils, height, width) if we pass smaps, so we first
        #    need to broadcast x across coils and multiply by smaps ourselves.

        # Expand x to multi-coil images: (batch, coils, height, width) = smaps * x
        # Here x: (batch, 1, H, W) -> broadcast to (batch, coils, H, W)
        # then multiply by complex smaps
        # Because torchkbnufft expects complex data as two real channels,
        # we assume smaps is already complex‐valued encoded as (batch, coils, 2, H, W)
        # or (batch, coils, H, W) complex‐dtype.  For simplicity, assume smaps is
        # complex‐valued (dtype=torch.complex64) of shape (batch, coils, H, W).
        #
        # If your smaps is real+imag as two separate real channels, you must
        # stack them into a complex tensor: e.g.
        #    smaps_complex = smaps_real + 1j * smaps_imag
        #
        # Then do:
        #    coil_images = x * smaps_complex

        # (1a) Convert x to complex: (batch, 1, H, W)  -> (batch, 1, H, W) complex
        x_complex = x.to(dtype=torch.complex64)

        # (1b) Expand x_complex across coils and multiply by smaps:
        #      smaps is (batch, coils, H, W) complex => coil_images (batch, coils, H, W)
        coil_images = x_complex * smaps

        # (1c) Run NUFFT: shape out is (batch, coils, n_samples, n_spokes)
        #      Note: KbNufft.forward signature: forward(x_coil, traj, smaps=None, norm='ortho')
        #      Since we already multiplied by smaps, we pass smaps=None here.
        A_x = self.nufft_op(coil_images, ktraj, smaps=None, norm='ortho')
        # A_x: (batch, coils, n_samples, n_spokes) complex

        # — Step 2: combine simulated k-space (A_x) and measured k-space (y_radial) —
        # Both A_x and y_radial are assumed to be EXACTLY at the same sample locations.
        # The Cartesian code does:
        #    k_dc = (1 – mask)*FFT(x) + mask*( λ·FFT(x) + (1–λ)·y )
        # For radial we typically have no “background” unsampled region—every entry in
        # y_radial corresponds to a sampled spoke+sample. So we simply do:
        #    k_dc = λ·A_x + (1–λ)·y_radial
        # The automatic “mask” is effectively an all‐ones boolean over the sampled points.

        # Broadcast λ to complex dtype:
        lambda_val = torch.sigmoid(self.lambda_)  # optional: clamp λ in (0,1)
        lambda_c = lambda_val.to(x.device).type(torch.complex64)

        k_dc = lambda_c * A_x + (1.0 - lambda_c) * y_radial

        # — Step 3: inverse NUFFT (adjoint) to get updated image estimate —
        # We want to go from k_dc (batch, coils, n_samples, n_spokes) back to a
        # single‐coil image.  Use the NUFFT adjoint with smaps provided so that
        # it internally performs sum_{c} conj(smaps_c) · adjNUFFT( k_dc_c ).
        #
        # KbNufftAdj signature: adjoint(kspace, traj, smaps=..., norm='ortho')
        # If we pass smaps, it will combine coil images for us.
        x_dc_complex = self.adjnufft_op(k_dc, ktraj, smaps=smaps, norm='ortho')
        # x_dc_complex: (batch, 1, H, W) complex

        # Convert back to real image (if MRI is magnitude‐only) or keep complex if you want
        # a complex‐valued image.  In many DC‐CNN setups, you treat the image as complex,
        # so return x_dc_complex directly.  If your network expects real images, you can
        # take real part or magnitude:
        #
        #    x_dc_real = x_dc_complex.real
        #
        # Here we assume complex‐valued output:
        return x_dc_complex

    def extra_repr(self):
        return f"lambda (raw)={self.lambda_.item():.4g}, learnable={self.lambda_.requires_grad}"
