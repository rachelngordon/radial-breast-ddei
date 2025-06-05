import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from torchkbnufft import KbNufft, KbNufftAdjoint
from einops import rearrange
import deepinv as dinv


class RadialDCLayer(nn.Module):
    """
        Radial Data Consistency layer from DC-CNN, apply for single coil mainly

        Expected Input Shapes: 
            im_size: (H, W, Ntime)
            x: (1, 1, H, W, 2, Ntime)
            y: (1, 1, Nrad, 2, Ntime)
            traj: (1, 2, Nrad (spokes * samples), Ntime)
            dcf: (1, 1, 1, Nrad, Ntime)
    """
    def __init__(
        self, 
        im_size,
        lambda_init=np.log(np.exp(1)-1.)/1., 
        learnable=True
        ):
        """
        Args:
            lambda_init (float): Init value of data consistency block (DCB)
        """
        super(RadialDCLayer, self).__init__()
        self.learnable = learnable
        self.lambda_ = nn.Parameter(torch.ones(1) * lambda_init, requires_grad=self.learnable)

        self.norm = "ortho"
        self.dtype = torch.float
        self.im_size = im_size

        spokelength=im_size[1]*2
        grid_size = (int(spokelength),int(spokelength))

        self.NUFFT =  KbNufft(im_size=im_size[:2], grid_size=grid_size).to(self.dtype)
        self.AdjNUFFT = KbNufftAdjoint(im_size=im_size[:2], grid_size=grid_size).to(self.dtype)

    # forward/adjoint operator functions adapted from: https://github.com/koflera/DynamicRadCineMRI
    def apply_A(self, x_tensor, ktraj):
		
		#for each time point apply the forward model;
        # if self.csmap is not None:
        #     kdat_list = [self.NUFFT(x_tensor[...,kt].contiguous(),
        #                     self.ktraj_tensor[...,kt].contiguous(), smaps=self.csmap, norm=self.norm) for kt in range(self.im_size[2])]
        # else:

        if x_tensor.shape[-1] == 2:
            x_tensor = rearrange(x_tensor, 'b h w t i -> b h w i t')

        if len(x_tensor.shape) == 5:
            x_tensor = x_tensor.unsqueeze(1)

        kdat_list = [self.NUFFT(x_tensor[...,kt].contiguous(),
                        ktraj[...,kt].contiguous(), norm=self.norm) for kt in range(self.im_size[2])]
			
        kdat_tensor = torch.stack(kdat_list,dim=-1)
		
        return kdat_tensor
    
    def apply_AH(self, k_tensor, ktraj):

        k_tensor = k_tensor.to(self.dtype)

        #for each time point apply the adjoint NUFFT-operator;
        # if self.csmap is not None:
        #     xrec_list = [self.AdjNUFFT(k_tensor[...,kt].contiguous(),
        #                         ktraj[...,kt].to(self.dtype).contiguous(), smaps=self.csmap, norm=self.norm) for kt in range(self.im_size[2])] 
        # else:
 
        xrec_list = [self.AdjNUFFT(k_tensor[...,kt].contiguous(),
                            ktraj[...,kt].to(self.dtype).contiguous(), norm=self.norm) for kt in range(self.im_size[2])] 

        xrec_tensor = torch.stack(xrec_list,dim=-1)

        return xrec_tensor

    def apply_Adag(self, k_tensor, ktraj, dcf):
		
		# multiply k-space data with dcomp
        k_tensor = rearrange(k_tensor, 'b c Nrad i t -> b c i Nrad t')
        dcomp_k_tensor = dcf * k_tensor
        dcomp_k_tensor = dcomp_k_tensor.permute(0, 1, 3, 2, 4) # shape: (1, 12, 11520, 2, 20)

		# apply adjoint
        xrec_tensor = self.apply_AH(dcomp_k_tensor.to(torch.complex128), ktraj)
		
        return xrec_tensor

    def forward(self, x, y, ktraj, dcf):

        # need to implement for all timeframes still (either loop over them or pass all to NUFFT)
        A_x = self.apply_A(x, ktraj)

        # NOTE: temporary fix, only works when Nrad is shifted forward one in shape
        if y.shape[2] != 2:
            y = rearrange(y, 'b t s i c -> b c s i t')
        else:
            y = rearrange(y, 'b t i s c -> b c s i t')


        k_dc = self.lambda_.to(x.device) * A_x + (1 - self.lambda_.to(x.device)) * y

        x_dc = self.apply_Adag(k_dc, ktraj, dcf)

        return x_dc

    def extra_repr(self):
        return f"lambda={self.lambda_.item():.4g}, learnable={self.learnable}"
    


class RadialPhysics(dinv.physics.Physics):
    """
    Physics operator that obtains radial trajectory and performs NUFFT and Adjoint NUFFT on radial data for unrolled network.
    """
    def __init__(
        self, 
        im_size,
        N_spokes, 
        N_samples,
        N_time,
        N_coils=1
        ):
        """
        Args:
            lambda_init (float): Init value of data consistency block (DCB)
        """
        super(RadialPhysics, self).__init__()

        self.norm = "ortho"
        self.dtype = torch.float
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.im_size = im_size
        self.N_spokes = N_spokes
        self.N_samples = N_samples
        self.N_time = N_time
        self.N_coils = N_coils

        spokelength=im_size[1]*2
        grid_size = (int(spokelength),int(spokelength))

        self.NUFFT =  KbNufft(im_size=im_size[:2], grid_size=grid_size).to(self.dtype).to(self.device)
        self.AdjNUFFT = KbNufftAdjoint(im_size=im_size[:2], grid_size=grid_size).to(self.dtype).to(self.device)

        self.base_res = N_samples // 2
        self.traj, self.dcf = self.get_traj_and_dcf()

        self.traj = self.traj.to(self.device)
        self.dcf = self.dcf.to(self.device)


    def get_traj_and_dcf(self, gind=1):

        N_tot_spokes = self.N_spokes * self.N_time

        N_samples = self.base_res * 2

        base_lin = np.arange(N_samples).reshape(1, -1) - self.base_res

        tau = 0.5 * (1 + 5**0.5)
        base_rad = np.pi / (gind + tau - 1)

        base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

        traj = np.zeros((N_tot_spokes, N_samples, 2))

        traj[..., 0] = np.cos(base_rot) @ base_lin
        traj[..., 1] = np.sin(base_rot) @ base_lin

        traj = traj / 2

        traj = traj.reshape(self.N_time, self.N_spokes, N_samples, 2)

        traj = rearrange(traj, 't sp sam i -> (sp sam) t i')

        traj = traj / np.mean(np.abs(traj))
        traj = torch.tensor(traj)*torch.tensor([1, -1])

        # compute the density compensation function from trajectory
        dcf = np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2)
        dcf = dcf.clone().detach().unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # combine real and imaginary components in k-space trajectory
        traj = rearrange(traj, "s t i -> i s t").unsqueeze(0)


        return traj, dcf
    
    
    def apply_dcf(self, y):

        # NOTE: temporary fix, only works when Nrad is shifted forward one in shape
        if y.shape[-2] != self.dcf.shape[-2]:
            y = rearrange(y, 'b c s i t -> b c i s t')


        y = self.dcf * y
        y = y.permute(0, 1, 3, 2, 4) # shape: (1, 12, 11520, 2, 20)

        return y


    
    def A(self, x: Tensor, **kwargs) -> torch.Tensor:

        #for each time point apply the forward model;
        # if self.csmap is not None:
        #     kdat_list = [self.NUFFT(x_tensor[...,kt].contiguous(),
        #                     self.ktraj_tensor[...,kt].contiguous(), smaps=self.csmap, norm=self.norm) for kt in range(self.im_size[2])]
        # else:
        
        # remove coil dimension if necessary for now
        if len(x.shape) == 5:
            x = x.unsqueeze(1)

        y = [self.NUFFT(x[...,kt].to(self.device).contiguous(),
                        self.traj[...,kt].to(self.device).contiguous(), norm=self.norm) for kt in range(self.im_size[2])]

        y = torch.stack(y, dim=-1).to(self.dtype)
		
        return y

    
    def A_adjoint(
        self, y: Tensor, mag: bool = False, **kwargs
    ) -> Tensor:
        """Adjoint operator.

        Optionally perform magnitude to reduce channel dimension.

        :param torch.Tensor y: input kspace of shape `(B,2,T,H,W)`
        :param torch.Tensor mask: optionally set mask on-the-fly, see class docs for shapes allowed.
        :param bool mag: perform complex magnitude.
        """

        # apply dcf
        y = self.apply_dcf(y).to(self.dtype)

        # apply adjoint

        #for each time point apply the adjoint NUFFT-operator;
        # if self.csmap is not None:
        #     xrec_list = [self.AdjNUFFT(k_tensor[...,kt].contiguous(),
        #                         ktraj[...,kt].to(self.dtype).contiguous(), smaps=self.csmap, norm=self.norm) for kt in range(self.im_size[2])] 
        # else:

        x = [self.AdjNUFFT(y[...,kt].to(self.device).contiguous(),
                            self.traj[...,kt].to(self.dtype).to(self.device).contiguous(), norm=self.norm) for kt in range(self.im_size[2])] 

        x = torch.stack(x,dim=-1)

        x = rearrange(x, 'b c h w i t -> b w h t i c').squeeze(-1) # note: squeezing coil dimension for current single coil implementation

        if mag:
            return torch.abs(x[..., 0] + 1j * x[..., 1])
        else: 
            return x

		
    def A_dagger(self, y: Tensor, **kwargs) -> torch.Tensor:
        return self.A_adjoint(y, **kwargs)
