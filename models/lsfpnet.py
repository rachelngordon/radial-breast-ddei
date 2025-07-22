import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from lsp import Project_inf, Wxs, Wtxs
from einops import rearrange


def print_grad_hook(name):
    """A simple hook function to print gradient stats for complex or real tensors."""
    def hook(grad):
        if grad is not None:
            print(f"--- Grad flowing back to '{name}' ---")
            print(f"    grad.shape: {grad.shape}")
            print(f"    grad.dtype: {grad.dtype}") # Good to add for debugging

            # --- The Fix ---
            # If the gradient is complex, calculate stats on its magnitude.
            if torch.is_complex(grad):
                grad_abs = grad.abs()
                # norm is fine on complex, but we can be consistent
                print(f"    grad.norm(): {grad.norm().item():.6e}")
                print(f"    grad_abs.mean(): {grad_abs.mean().item():.6e}")
                print(f"    grad_abs.max():  {grad_abs.max().item():.6e}")
                print(f"    grad_abs.min():  {grad_abs.min().item():.6e}")
            else: # If it's a real tensor, proceed as before
                print(f"    grad.norm(): {grad.norm().item():.6e}")
                print(f"    grad.mean(): {grad.mean().item():.6e}")
                print(f"    grad.max():  {grad.max().item():.6e}")
                print(f"    grad.min():  {grad.min().item():.6e}")

            # isnan check also needs to be on real and imag parts for complex
            if torch.is_complex(grad):
                num_nans = torch.isnan(grad.real).sum() + torch.isnan(grad.imag).sum()
            else:
                num_nans = torch.isnan(grad).sum()

            print(f"    Number of NaNs: {num_nans}")
            print("-" * 30)
        else:
            print(f"--- Grad flowing back to '{name}' is None ---")
            print("-" * 30)
    return hook

dtype = torch.complex64

def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    return torch.view_as_complex(rearrange(x, "b c ... -> b ... c").contiguous())


def from_torch_complex(x: torch.Tensor):
    """(B, ...) complex -> (B, 2, ...) real"""
    return rearrange(torch.view_as_real(x), "b ... c -> b c ...").contiguous()


# define LSFP-Net Block
class BasicBlock(nn.Module):
    def __init__(self, channels=32):
        super(BasicBlock, self).__init__()

        self.channels = channels

        self.lambda_L = nn.Parameter(torch.tensor([0.0025]))
        self.lambda_S = nn.Parameter(torch.tensor([0.05]))
        self.lambda_spatial_L = nn.Parameter(torch.tensor([5e-2]))
        self.lambda_spatial_S = nn.Parameter(torch.tensor([5e-2]))

        # --- MODIFIED VALUES ---
        # # Was 0.0025. Increase dramatically to match the scale of the singular values.
        # self.lambda_L = nn.Parameter(torch.tensor([5.0])) 

        # # Was 0.05. Decrease to be closer to the input signal magnitude.
        # self.lambda_S = nn.Parameter(torch.tensor([0.1]))
        
        # # Was 0.05. Decrease to be closer to the input signal magnitude.
        # self.lambda_spatial_L = nn.Parameter(torch.tensor([0.1]))
        
        # # Was 0.05. Decrease to be closer to the input signal magnitude.
        # self.lambda_spatial_S = nn.Parameter(torch.tensor([0.1]))
        # --- END MODIFIED VALUES ---

        self.gamma = nn.Parameter(torch.tensor([0.5]))
        self.lambda_step = nn.Parameter(torch.tensor([1/10]))

        # self.gamma = 0.5
        # self.lambda_step = 1/10

        self.conv1_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 1, 3, 3, 3)))
        self.conv2_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
        self.conv3_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))

        self.conv1_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
        self.conv2_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
        self.conv3_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, 3, 3, 3)))

        self.conv1_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 1, 3, 3, 3)))
        self.conv2_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
        self.conv3_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))

        self.conv1_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
        self.conv2_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
        self.conv3_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, 3, 3, 3)))

    def forward(self, M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S, csmaps):

        handles = []

        c = self.lambda_step / self.gamma
        nx, ny, nt = M0.size()

        # gradient
        temp_data = torch.reshape(L + S, [nx, ny, nt])
        temp_data = param_E(inv=False, data=temp_data, smaps=csmaps).to(param_d.device)
        gradient = param_E(inv=True, data=temp_data - param_d, smaps=csmaps)
        gradient = torch.reshape(gradient, [nx * ny, nt]).to(param_d.device)

        # HOOK 1: Check the main data consistency gradient
        if self.training and gradient.requires_grad:
            handles.append(gradient.register_hook(print_grad_hook("Data Consistency Gradient")))

        # pb_L
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        # pb_L = F.leaky_relu(pb_L, negative_slope=0.1)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        # pb_L = F.leaky_relu(pb_L, negative_slope=0.1)
        pb_L = F.conv3d(pb_L, self.conv3_backward_l, padding=1)

        pb_L = torch.reshape(torch.squeeze(pb_L), [2, nx * ny, nt])
        pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

        # y_L
        y_L = L - self.gamma * gradient - self.gamma * pt_L - self.gamma * pb_L

        # pt_L

        # --- DEBUG PRINT ---
        print(f"SVD Input Values (c * y_L + pt_L) mag: {torch.mean(torch.abs(c * y_L + pt_L)):.4f}")

        svd_input = c * y_L + pt_L

        # HOOK 2: Check the gradient right before the SVD
        if self.training and gradient.requires_grad:
            handles.append(svd_input.register_hook(print_grad_hook("SVD Input")))

        Ut, St, Vt = torch.linalg.svd((svd_input), full_matrices=False)

        # HOOK 3: Check the gradient flowing out of the singular values
        if self.training and gradient.requires_grad:
            handles.append(St.register_hook(print_grad_hook("Singular Values (St)")))

        # --- DEBUG PRINT ---
        print(f"SVD Singular Values (St) mag: {torch.mean(torch.abs(St)):.4f}, Lambda_L: {self.lambda_L.item():.4f}")

        temp_St = torch.diag(Project_inf(St, self.lambda_L))
        pt_L = Ut.mm(temp_St).mm(Vt)

        # HOOK 4: Check the gradient of the low-rank component after SVD thresholding
        if self.training and gradient.requires_grad:
            handles.append(pt_L.register_hook(print_grad_hook("pt_L after SVD")))

        # update p_L
        temp_y_L_input = torch.cat((torch.real(y_L), torch.imag(y_L)), 0).to(torch.float32)
        temp_y_L_input = torch.reshape(temp_y_L_input, [2, nx, ny, nt]).unsqueeze(1)
        temp_y_L = F.conv3d(temp_y_L_input, self.conv1_forward_l, padding=1)
        temp_y_L = F.relu(temp_y_L)
        # temp_y_L = F.leaky_relu(temp_y_L, negative_slope=0.1)
        temp_y_L = F.conv3d(temp_y_L, self.conv2_forward_l, padding=1)
        temp_y_L = F.relu(temp_y_L)
        # temp_y_L = F.leaky_relu(temp_y_L, negative_slope=0.1)
        temp_y_L_output = F.conv3d(temp_y_L, self.conv3_forward_l, padding=1)

        temp_y_L = temp_y_L_output + p_L
        temp_y_L = temp_y_L[0, :, :, :, :] + 1j * temp_y_L[1, :, :, :, :]

        # --- DEBUG PRINT ---
        print(f"Spatial input mag: {torch.mean(torch.abs(c * temp_y_L)):.4f}, Lambda_Spatial_L: {self.lambda_spatial_L.item():.4f}")

        p_L = Project_inf(c * temp_y_L, self.lambda_spatial_L)

        # new pb_L
        p_L = torch.cat((torch.real(p_L), torch.imag(p_L)), 0).to(torch.float32)
        p_L = torch.reshape(p_L, [2, self.channels, nx, ny, nt])
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        # pb_L = F.leaky_relu(pb_L, negative_slope=0.1)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        # pb_L = F.leaky_relu(pb_L, negative_slope=0.1)
        pb_L_output = F.conv3d(pb_L, self.conv3_backward_l, padding=1)

        pb_L = torch.reshape(pb_L_output, [2, nx * ny, nt])
        pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

        # L
        L = L - self.gamma * gradient - self.gamma * pt_L - self.gamma * pb_L

        # HOOK 5: Check the final gradient for the L component
        if self.training and gradient.requires_grad:
            handles.append(L.register_hook(print_grad_hook("Final L for this block")))

        # adjoint loss: adjloss_L = psi * x * y - psi_t * y * x
        adjloss_L = temp_y_L_output * p_L - pb_L_output * temp_y_L_input

        # pb_S
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        # pb_S = F.leaky_relu(pb_S, negative_slope=0.1)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        # pb_S = F.leaky_relu(pb_S, negative_slope=0.1)
        pb_S = F.conv3d(pb_S, self.conv3_backward_s, padding=1)

        pb_S = torch.reshape(pb_S, [2, nx * ny, nt])
        pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

        # y_S
        y_S = S - self.gamma * gradient - self.gamma * Wtxs(pt_S) - self.gamma * pb_S

        # pt_S
        # --- DEBUG PRINT ---
        print(f"Spatial input mag: {torch.mean(torch.abs(c * Wxs(y_S) + pt_S)):.4f}, Lambda_S: {self.lambda_S.item():.4f}")

        pt_S = Project_inf(c * Wxs(y_S) + pt_S, self.lambda_S)

        if self.training and gradient.requires_grad:
            handles.append(pt_S.register_hook(print_grad_hook("pt_S")))

        # update p_S
        temp_y_S_input = torch.cat((torch.real(y_S), torch.imag(y_S)), 0).to(torch.float32)
        temp_y_S_input = torch.reshape(temp_y_S_input, [2, nx, ny, nt]).unsqueeze(1)
        temp_y_S = F.conv3d(temp_y_S_input, self.conv1_forward_s, padding=1)
        temp_y_S = F.relu(temp_y_S)
        # temp_y_S = F.leaky_relu(temp_y_S, negative_slope=0.1)
        temp_y_S = F.conv3d(temp_y_S, self.conv2_forward_s, padding=1)
        temp_y_S = F.relu(temp_y_S)
        # temp_y_S = F.leaky_relu(temp_y_S, negative_slope=0.1)
        temp_y_S_output = F.conv3d(temp_y_S, self.conv3_forward_s, padding=1)

        temp_y_Sp = temp_y_S_output + p_S
        temp_y_Sp = temp_y_Sp[0, :, :, :, :] + 1j * temp_y_Sp[1, :, :, :, :]

        # --- DEBUG PRINT ---
        print(f"Spatial input mag: {torch.mean(torch.abs(c * temp_y_Sp)):.4f}, Lambda_Spatial_S: {self.lambda_spatial_S.item():.4f}")
        p_S = Project_inf(c * temp_y_Sp, self.lambda_spatial_S)

        if self.training and gradient.requires_grad:
            handles.append(p_S.register_hook(print_grad_hook("p_S")))

        # new pb_S
        p_S = torch.cat((torch.real(p_S), torch.imag(p_S)), 0).to(torch.float32)
        p_S = torch.reshape(p_S, [2, self.channels, nx, ny, nt])
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        # pb_S = F.leaky_relu(pb_S, negative_slope=0.1)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        # pb_S = F.leaky_relu(pb_S, negative_slope=0.1)
        pb_S_output = F.conv3d(pb_S, self.conv3_backward_s, padding=1)

        pb_S = torch.reshape(pb_S_output, [2, nx * ny, nt])
        pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

        # S
        S = S - self.gamma * gradient - self.gamma * Wtxs(pt_S) - self.gamma * pb_S

        if self.training and gradient.requires_grad:
            handles.append(S.register_hook(print_grad_hook("S")))

        # adjoint loss: adjloss_S = psi * x * y - psi_t * y * x
        adjloss_S = temp_y_S_output * p_S - pb_S_output * temp_y_S_input

        return [L, S, adjloss_L, adjloss_S, pt_L, pt_S, p_L, p_S, handles]


# define LSFP-Net
class LSFPNet(nn.Module):
    def __init__(self, LayerNo, channels=32):
        super(LSFPNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.channels = channels

        for ii in range(LayerNo):
            onelayer.append(BasicBlock(channels=self.channels))

        self.fcs = nn.ModuleList(onelayer)


    def forward(self, M0, param_E, param_d, csmap):

        # M0 = M0[..., 0] + 1j * M0[..., 1]
        # param_d = param_d[..., 0] + 1j * param_d[..., 1]

        nx, ny, nt = M0.size()
        L = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        S = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        pt_L = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        pt_S = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        p_L = torch.zeros([2, self.channels, nx, ny, nt], dtype=torch.float32).to(param_d.device)
        p_S = torch.zeros([2, self.channels, nx, ny, nt], dtype=torch.float32).to(param_d.device)

        layers_adj_L = []
        layers_adj_S = []

        for ii in range(self.LayerNo):
            [L, S, layer_adj_L, layer_adj_S, pt_L, pt_S, p_L, p_S, handles] = self.fcs[ii](M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S, csmap)
            layers_adj_L.append(layer_adj_L)
            layers_adj_S.append(layer_adj_S)

        L = torch.reshape(L, [nx, ny, nt])
        S = torch.reshape(S, [nx, ny, nt])

        return [L, S, layers_adj_L, layers_adj_S, handles]
    

class EWrapper:
    """
    Make `physics` look like the callable interface expected by LSFP-Net:
        E(inv=False, data=image)  -> k-space
        E(inv=True , data=data )  -> image
    Scaling is handled *outside*, so we just delegate.
    """
    def __init__(self, physics, csmap):
        self.physics, self.csmap = physics, csmap

    def __call__(self, *, inv: bool, data: torch.Tensor) -> torch.Tensor:
        if inv:   # adjoint
            data = from_torch_complex(data.unsqueeze(0))
            data = self.physics.A_adjoint(data, self.csmap)
            data = to_torch_complex(data).squeeze(0)
            return rearrange(data, 't h w -> h w t')
        
        else:     # forward
            data = from_torch_complex(data.unsqueeze(0))
            data = self.physics.A(data, self.csmap)
            return to_torch_complex(data).squeeze(0)
        

class ArtifactRemovalLSFPNet(nn.Module):
    def __init__(self, backbone_net, **kwargs):
        super(ArtifactRemovalLSFPNet, self).__init__()
        self.backbone_net = backbone_net

    @staticmethod
    def _normalise(zf: torch.Tensor, data: torch.Tensor):
        """
        Per-dynamic-series max-magnitude scaling (paper default).
        Both `zf` (image) and `data` (k-space) share the SAME scalar.
        """
        with torch.no_grad():
            scale = zf.abs().max()                       # scalar, grads OK

        # Add a small epsilon to prevent division by zero
        scale = scale + 1e-8

        return zf / scale.detach(), data / scale.detach(), scale.detach()

    def forward(self, y, E, csmap, dcf, norm=True, **kwargs):

        # 1. Get the initial ZF recon. This defines our target energy/scale.
        x_init = E(inv=True, data=y, smaps=csmap)

        # 2. Permute and normalize the input for the network
        # x_init_permuted = rearrange(x_init, "b c t h w -> b h w t c")
        if norm:
            print(x_init.shape)
            print(x_init.dtype)
            print("initial mag values: ")
            print("min: ", torch.abs(x_init).min())
            print("max: ", torch.abs(x_init).max())
            print("mean: ", torch.abs(x_init).mean())
            x_init_norm, y_norm, scale = self._normalise(x_init, y)
            print("normalized values: ")
            print("min: ", torch.abs(x_init_norm).min())
            print("max: ", torch.abs(x_init_norm).max())
            print("mean: ", torch.abs(x_init_norm).mean())
        else:
            x_init_norm = x_init
            y_norm = y

        L, S, adjoint_L, adjoint_S, handles = self.backbone_net(x_init_norm, E, y_norm, csmap)

        total_adjoint_loss_scalar = 0
        if adjoint_L: # Check if the list is not empty
            for loss_tensor in adjoint_L:
                # Use torch.mean(torch.abs(...)) to get a scalar value for each tensor
                total_adjoint_loss_scalar += torch.mean(torch.abs(loss_tensor))
                # total_adjoint_loss_scalar += torch.mean(loss_tensor**2)
        if adjoint_S:
            for loss_tensor in adjoint_S:
                total_adjoint_loss_scalar += torch.mean(torch.abs(loss_tensor))
                # total_adjoint_loss_scalar += torch.mean(loss_tensor**2)
                
        if norm:
            recon = (L + S) * scale                  # rescale to original units
        else:
            recon = (L + S)


        # 4) stack & convert back to (B,2,T,H,W) float32
        x_hat = torch.stack((recon.real, recon.imag), dim=0).unsqueeze(0)  # (B,2,H,W,T)

        return x_hat, total_adjoint_loss_scalar, handles