# from __future__ import annotations
import deepinv as dinv
import numpy as np
import torch
import torch.nn as nn
from deepinv.physics.time import TimeMixin
from einops import rearrange
from torchkbnufft import KbNufft, KbNufftAdjoint
from noise import ZeroNoise
import warnings
from torch import Tensor


def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    return torch.view_as_complex(rearrange(x, "b c ... -> b ... c").contiguous())


def from_torch_complex(x: torch.Tensor):
    """(B, ...) complex -> (B, 2, ...) real"""
    return rearrange(torch.view_as_real(x), "b ... c -> b c ...").contiguous()



class Physics(torch.nn.Module):  # parent class for forward models
    r"""
    Parent class for forward operators

    It describes the general forward measurement process

    .. math::

        y = \noise(\forw(x))

    where :math:`x` is an image of :math:`n` pixels, :math:`y` is the measurements of size :math:`m`,
    :math:`\forw:\xset\mapsto \yset` is a deterministic mapping capturing the physics of the acquisition
    and :math:`\noise:\yset\mapsto \yset` is a stochastic mapping which characterizes the noise affecting
    the measurements.

    :param Callable A: forward operator function which maps an image to the observed measurements :math:`x\mapsto y`.
    :param deepinv.physics.NoiseModel, Callable noise_model: function that adds noise to the measurements :math:`N(z)`.
        See the noise module for some predefined functions.
    :param Callable sensor_model: function that incorporates any sensor non-linearities to the sensing process,
        such as quantization or saturation, defined as a function :math:`\eta(z)`, such that
        :math:`y=\eta\left(N(A(x))\right)`. By default, the `sensor_model` is set to the identity :math:`\eta(z)=z`.
    :param int max_iter: If the operator does not have a closed form pseudoinverse, the gradient descent algorithm
        is used for computing it, and this parameter fixes the maximum number of gradient descent iterations.
    :param float tol: If the operator does not have a closed form pseudoinverse, the gradient descent algorithm
        is used for computing it, and this parameter fixes the absolute tolerance of the gradient descent algorithm.
    :param str solver: least squares solver to use. Only gradient descent is available for non-linear operators.
    """

    def __init__(
        self,
        A=lambda x, **kwargs: x,
        noise_model=ZeroNoise(),
        sensor_model=lambda x: x,
        solver="gradient_descent",
        max_iter=50,
        tol=1e-4,
    ):
        super().__init__()
        self.noise_model = noise_model
        self.sensor_model = sensor_model
        self.forw = A
        self.SVD = False  # flag indicating SVD available
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver

    def __mul__(self, other):
        r"""
        Concatenates two forward operators :math:`A = A_1\circ A_2` via the mul operation

        The resulting operator keeps the noise and sensor models of :math:`A_1`.

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (:class:`deepinv.physics.Physics`) concatenated operator

        """

        warnings.warn(
            "You are composing two physics objects. The resulting physics will not retain the original attributes. "
            "You may instead retrieve attributes of the original physics by indexing the resulting physics."
        )
        return compose(other, self, max_iter=self.max_iter, tol=self.tol)

    def stack(self, other):
        r"""
        Stacks two forward operators :math:`A(x) = \begin{bmatrix} A_1(x) \\ A_2(x) \end{bmatrix}`

        The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
        each entry corresponds to the measurements of the corresponding operator.

        Returns a :class:`deepinv.physics.StackedPhysics` object.

        See :ref:`physics_combining` for more information.

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (:class:`deepinv.physics.StackedPhysics`) stacked operator

        """
        return stack(self, other)

    def forward(self, x, csmaps, **kwargs):
        r"""
        Computes forward operator

        .. math::

                y = N(A(x), \sigma)


        :param torch.Tensor, list[torch.Tensor] x: signal/image
        :return: (:class:`torch.Tensor`) noisy measurements

        """
        return self.sensor(self.noise(self.A(x, csmaps, **kwargs), **kwargs))
    
    def A(self, x, **kwargs):
        r"""
        Computes forward operator :math:`y = A(x)` (without noise and/or sensor non-linearities)

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (:class:`torch.Tensor`) clean measurements

        """
        return self.forw(x, **kwargs)

    def sensor(self, x):
        r"""
        Computes sensor non-linearities :math:`y = \eta(y)`

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (:class:`torch.Tensor`) clean measurements
        """
        return self.sensor_model(x)

    def set_noise_model(self, noise_model, **kwargs):
        r"""
        Sets the noise model

        :param Callable noise_model: noise model
        """
        self.noise_model = noise_model

    def noise(self, x, **kwargs) -> Tensor:
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)`

        :param torch.Tensor x:  clean measurements
        :param None, float noise_level: optional noise level parameter
        :return: noisy measurements

        """

        return self.noise_model(x, **kwargs)

    def A_dagger(self, y, x_init=None):
        r"""
        Computes an inverse as:

        .. math::

            x^* \in \underset{x}{\arg\min} \quad \|\forw{x}-y\|^2.

        This function uses gradient descent to find the inverse. It can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist.

        :param torch.Tensor y: a measurement :math:`y` to reconstruct via the pseudoinverse.
        :param torch.Tensor x_init: initial guess for the reconstruction.
        :return: (:class:`torch.Tensor`) The reconstructed image :math:`x`.

        """

        if self.solver == "gradient_descent":
            if x_init is None:
                x_init = self.A_adjoint(y)

            x = x_init

            lr = 1e-1
            loss = torch.nn.MSELoss()
            for _ in range(self.max_iter):
                x = x - lr * self.A_vjp(x, self.A(x) - y)
                err = loss(self.A(x), y)
                if err < self.tol:
                    break
        else:
            raise NotImplementedError(
                f"Solver {self.solver} not implemented for A_dagger"
            )

        return x.clone()

    def set_ls_solver(self, solver, max_iter=None, tol=None):
        r"""
        Change default solver for computing the least squares solution:

        .. math::

            x^* \in \underset{x}{\arg\min} \quad \|\forw{x}-y\|^2.

        :param str solver: solver to use. If the physics are non-linear, the only available solver is `'gradient_descent'`.
            For linear operators, the options are `'CG'`, `'lsqr'`, `'BiCGStab'` and `'minres'` (see :func:`deepinv.optim.utils.least_squares` for more details).
        :param int max_iter: maximum number of iterations for the solver.
        :param float tol: relative tolerance for the solver, stopping when :math:`\|A(x) - y\| < \text{tol} \|y\|`.
        """

        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        self.solver = solver

    def A_vjp(self, x, v):
        r"""
        Computes the product between a vector :math:`v` and the Jacobian of the forward operator :math:`A` evaluated at :math:`x`, defined as:

        .. math::

            A_{vjp}(x, v) = \left. \frac{\partial A}{\partial x}  \right|_x^\top  v.

        By default, the Jacobian is computed using automatic differentiation.

        :param torch.Tensor x: signal/image.
        :param torch.Tensor v: vector.
        :return: (:class:`torch.Tensor`) the VJP product between :math:`v` and the Jacobian.
        """
        _, vjpfunc = torch.func.vjp(self.A, x)
        return vjpfunc(v)[0]

    def update(self, **kwargs):
        r"""
        Update the parameters of the physics: forward operator and noise model.

        :param dict kwargs: dictionary of parameters to update.
        """
        self.update_parameters(**kwargs)
        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)

    def update_parameters(self, **kwargs):
        r"""

        Update the parameters of the forward operator.

        :param dict kwargs: dictionary of parameters to update.
        """
        if kwargs:
            for key, value in kwargs.items():
                if (
                    value is not None
                    and hasattr(self, key)
                    and isinstance(value, torch.Tensor)
                ):
                    self.register_buffer(key, value)



# This class only knows how to handle a batch of 2D images (4D Tensors)
class RadialPhysics(Physics):
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

    def A(self, x: torch.Tensor, csmaps, **kwargs) -> torch.Tensor:
        # The call signature is back to the original version
        x_complex = to_torch_complex(x).unsqueeze(1)
        k_complex_nufft = self.NUFFT(x_complex, self.traj, csmaps)

        y_complex_weighted = k_complex_nufft * self.sqrt_dcf

        y = from_torch_complex(y_complex_weighted.squeeze(1))
        return rearrange(y, "b c (s i) -> b c s i", s=self.N_spokes)

    def A_adjoint(self, y: torch.Tensor, csmaps, **kwargs) -> torch.Tensor:
        # The call signature is back to the original version
        y_flat = rearrange(y, "b c s i -> b c (s i)")
        y_complex = to_torch_complex(y_flat).unsqueeze(1)

        y_dcf_complex = y_complex * self.sqrt_dcf

        if torch.isnan(y_dcf_complex).any():
            print("!!! ERROR: NaN detected in y_dcf_complex in A_adjoint !!!")

        x_complex = self.AdjNUFFT(y_dcf_complex, self.traj, csmaps).squeeze(1)
        return from_torch_complex(x_complex)


# This class now handles 5D video tensors by inheriting from our 2D class and TimeMixin
class DynamicRadialPhysics(RadialPhysics, TimeMixin):
    def __init__(self, im_size, N_spokes, N_samples, N_time, N_coils=1, **kwargs):
        # We call the TimeMixin's init first
        TimeMixin.__init__(self)

        # We call the base Physics init, not RadialPhysics's init directly
        # dinv.physics.Physics.__init__(self, **kwargs)
        Physics.__init__(self, **kwargs)

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

        if self.N_coils == 1:
            self.mask = torch.ones(1, 2, self.N_time, self.N_spokes, self.N_samples).to(
                self.device
            )
        else:
            self.mask = torch.ones(1, 2, self.N_time, self.N_coils, self.N_spokes, self.N_samples).to(
                self.device
            )


    # --- We need to override A and A_adjoint to use the per-frame trajectory ---
    def A(self, x: torch.Tensor, csmap, **kwargs) -> torch.Tensor:
        # x has shape (B, C, T, H, W)
        B, C, T, H, W = x.shape
        output_kspace_frames = []

        csmap = csmap.to(self.device)

        x_complex = to_torch_complex(x) # -> (B, T, H, W)

        for t in range(T):
            x_complex_frame = x_complex[:, t, :, :].unsqueeze(1)  # -> (B, Co, H, W)

            if self.N_coils > 1:
                x_complex_frame = x_complex_frame * csmap.to(x_complex_frame.dtype)

            # Use the trajectory for this specific time frame 't'
            traj_t = self.traj_per_frame[t]
            sqrt_dcf_t = self.sqrt_dcf_per_frame[t]


            k_complex_nufft = self.NUFFT(x_complex_frame, traj_t)
            y_complex_weighted = k_complex_nufft * sqrt_dcf_t


            if self.N_coils == 1:
                y_frame = from_torch_complex(y_complex_weighted.squeeze(1))
                y_frame_reshaped = rearrange(
                    y_frame, "b c (s i) -> b c s i", s=self.N_spokes
                )
            else: 
                y_frame = from_torch_complex(y_complex_weighted)
                y_frame_reshaped = rearrange(
                    y_frame, "b c co (s i) -> b c co s i", s=self.N_spokes
                )

            output_kspace_frames.append(y_frame_reshaped)

        y = torch.stack(output_kspace_frames, dim=2)  # Stack along the time dimension

        return y * self.mask

    def A_adjoint(self, y: torch.Tensor, csmap: torch.Tensor, **kwargs) -> torch.Tensor:

        # check if input has coil dimension
        if len(y.shape) == 5:
            B, C, T, S, I = y.shape
        elif len(y.shape) == 6:
            B, C, T, Co, S, I = y.shape

        output_image_frames = []

        y_masked = y * self.mask

        for t in range(T):
            y_frame = y_masked[:, :, t, :, :]  # -> (B, C, S, I)

            # Use the trajectory for this specific time frame 't'
            traj_t = self.traj_per_frame[t]
            sqrt_dcf_t = self.sqrt_dcf_per_frame[t]

            if self.N_coils == 1:
                y_flat = rearrange(y_frame, "b c s i -> b c (s i)")
                y_complex = to_torch_complex(y_flat).unsqueeze(1)

                y_dcf_complex = y_complex * sqrt_dcf_t
                x_complex_frame = self.AdjNUFFT(y_dcf_complex, traj_t).squeeze(1) # -> (B, H, W)

            else:
                y_flat = rearrange(y_frame, "b c co s i -> b c co (s i)")
                y_complex = to_torch_complex(y_flat)

                y_dcf_complex = y_complex * sqrt_dcf_t
                x_complex_frame = self.AdjNUFFT(y_dcf_complex, traj_t) # -> (B, Co, H, W)


            if self.N_coils > 1:

                csmap = csmap.to(x_complex_frame.dtype).to(self.device)

                # Multiply raw images by conjugate of sensitivity maps
                sens_weighted_imgs = x_complex_frame * csmap.conj()

                # Calculate the sum-of-squares of the sensitivity maps
                combined_numerator = torch.sum(sens_weighted_imgs, dim=1)
                sos_sens_maps = torch.sum(csmap.abs()**2, dim=1)
                epsilon = torch.finfo(sos_sens_maps.dtype).eps
                
                # Obtain final combined image
                x_complex_frame = combined_numerator / (sos_sens_maps + epsilon)

            
            x_frame = from_torch_complex(x_complex_frame)
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

    def forward(self, x_img_permuted, y_kspace_meas, mask_kspace, csmap):
        # x_img_permuted from CRNN: (b, h, w, t, c)
        # y_kspace_meas from dataloader: (b, c, t, s, i)
        # mask_kspace: same shape as y_kspace_meas
        x_img = rearrange(x_img_permuted, "b h w t c -> b c t h w")
        y = y_kspace_meas

        A_x = self.physics.A(x_img, csmap)

        lambda_ = torch.sigmoid(self.lambda_)

        k_dc = (1 - mask_kspace) * A_x + mask_kspace * (
            lambda_ * A_x + (1 - lambda_) * y
        )

        # Step 3: Transform the corrected k-space back to image space.
        # The physics operator `A_adjoint` handles all Adjoint NUFFT logic.
        x_dc_img = self.physics.A_adjoint(k_dc, csmap)

        # Step 4: Convert back to CRNN's expected permuted format.
        x_dc_permuted = rearrange(x_dc_img, "b c t h w -> b h w t c")

        return x_dc_permuted
