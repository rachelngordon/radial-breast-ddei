import torch
import torch.nn as nn
from einops import rearrange


# --- Helper function for normalization ---
def _normalize_batch(x):
    """
    Normalizes each item in the batch to the range [-1, 1] based on its absolute maximum value.
    This preserves the internal structure and relative enhancement curve shape.
    Shape of x: (B, H, W, T, C)
    """
    b, h, w, t, c = x.shape

    # FIX: Use .reshape() instead of .view() to handle non-contiguous tensors from einops/permute.
    x_flat = x.reshape(b, -1)

    # Get the max absolute value for each item in the batch
    # Add a small epsilon to prevent division by zero for black images
    max_vals, _ = torch.max(torch.abs(x_flat), dim=1, keepdim=True)
    max_vals = max_vals + 1e-8

    # FIX: Use .reshape() here as well for consistency and safety.
    max_vals_reshaped = max_vals.reshape(b, 1, 1, 1, 1)

    return x / max_vals_reshaped


class ConvRNNCell(nn.Module):
    """
    Convolutional RNN cell.

    ### FIDELITY ENHANCEMENT:
    - Removed spectral_norm wrapper from convolutional layers. This reduces regularization,
      allowing the network to potentially learn finer details of the contrast enhancement,
      at the cost of relying more on other stabilization techniques.
    """

    def __init__(self, in_chans, out_chans, bias=True):
        super(ConvRNNCell, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        # Removed spectral_norm for higher fidelity, as requested.
        self.i2h = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=bias)
        self.h2h = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        # The logic here is sound. The error came from malformed inputs.
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.i2h(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        return hidden


class BCRNN(nn.Module):
    """
    Bidirectional Convolutional RNN.
    Note: The original DDEI implementation calls this BCRNN but it is a standard
    unidirectional CRNN operating over the time dimension. We keep the naming for consistency.
    """

    def __init__(self, in_chans, chans, n_layers, bias=True, device="cuda"):
        super(BCRNN, self).__init__()
        self.in_chans = in_chans
        self.chans = chans
        self.n_layers = n_layers
        self.device = device
        self.CRNN_model = ConvRNNCell(in_chans, chans, bias=bias)

    def forward(self, input, hidden_iteration):
        B, H, W, T, C = input.shape

        # Permute from (B, H, W, T, C) to (T, B, C, H, W) for temporal processing
        input = input.permute(3, 0, 4, 1, 2).contiguous()
        hidden_iteration = hidden_iteration.permute(3, 0, 4, 1, 2).contiguous()

        # Initialize hidden state for the RNN
        hidden = torch.zeros(B, self.chans, H, W, device=self.device)

        hidden_list = []
        for i in range(T):
            hidden = self.CRNN_model(input[i], hidden_iteration[i], hidden)
            hidden_list.append(hidden)

        # Stack hidden states and permute back to (B, H, W, T, Chans)
        out = torch.stack(hidden_list, dim=0).permute(1, 3, 4, 0, 2).contiguous()
        return out


class CRNN(nn.Module):
    """
    Main Unrolled CRNN architecture.

    ### STABILIZATION & BUG FIX:
    1.  **Inter-Cascade Normalization**: The output of the data consistency (DC) layer from each
        cascade is normalized before being fed into the next cascade. This is the primary fix
        for the exploding activation problem. It breaks the amplification feedback loop.
    2.  **Residual Block Fix**: The code block for processing the BCRNN output and adding the
        residual connection was reshaped incorrectly, which was the likely cause of the
        `size mismatch` error. This has been completely rewritten to be correct.
    3.  **Cleaned up Logic**: The variable naming and flow within the forward pass are clarified
        to distinguish between different states of the image estimate (e.g., `x_pre_dc`, `x_post_dc`).
    """

    def __init__(self, num_cascades, chans, in_chans=2, datalayer=None, **kwargs):
        super(CRNN, self).__init__()
        self.num_cascades = num_cascades
        self.chans = chans
        self.in_chans = in_chans
        self.datalayer = datalayer

        self.bcrnn = BCRNN(in_chans, chans, n_layers=1, **kwargs)
        self.res_conv = nn.Conv2d(chans, in_chans, kernel_size=1, stride=1, padding=0)

    def forward(self, x_init_permuted, y, mask):
        B, H, W, T, C = x_init_permuted.shape
        net = {}

        # Initial image estimate (output of A_adjoint)
        net["t0_x0"] = x_init_permuted

        # The input to the first cascade is the initial estimate
        x_cascade_in = net["t0_x0"]

        for i in range(self.num_cascades):
            # --- Recurrent Denoiser ---
            # The BCRNN denoises the input from the previous step.
            # `net["t0_x0"]` provides the initial context at each time step.
            bcrnn_out = self.bcrnn(x_cascade_in, net["t0_x0"])

            # --- BUG FIX & Reshaping for Residual Connection ---
            # Original code had incorrect permute/flatten operations.
            # Correct reshaping:
            # Input: (B, H, W, T, chans) -> (B*T, chans, H, W)
            bcrnn_out_reshaped = bcrnn_out.permute(
                0, 3, 4, 1, 2
            ).contiguous()  # -> (B, T, chans, H, W)
            bcrnn_out_reshaped = bcrnn_out_reshaped.reshape(B * T, self.chans, H, W)

            # Apply 1x1 conv to project back to input channels
            res_out_reshaped = self.res_conv(
                bcrnn_out_reshaped
            )  # -> (B*T, in_chans, H, W)

            # Reshape back to original tensor layout
            # -> (B, T, in_chans, H, W) -> (B, H, W, T, in_chans)
            res_out = res_out_reshaped.reshape(B, T, self.in_chans, H, W)
            res_out = res_out.permute(0, 3, 4, 1, 2).contiguous()

            # Add residual connection
            x_pre_dc = x_cascade_in + res_out

            # --- Data Consistency ---
            x_post_dc = self.datalayer(x_pre_dc, y, mask)

            # --- STABILIZATION: Inter-Cascade Normalization ---
            # Normalize the output of the DC layer before it becomes the
            # input to the next cascade. This prevents value explosion.
            # We skip normalization on the very last cascade's output.
            if i < self.num_cascades - 1:
                x_cascade_in = _normalize_batch(x_post_dc)
            else:
                # This is the final output of the entire CRNN
                x_cascade_in = x_post_dc

        return x_cascade_in  # This is the final reconstructed image


# This wrapper class remains the same as in your original file.
class ArtifactRemovalCRNN(nn.Module):
    def __init__(self, backbone_net, **kwargs):
        super(ArtifactRemovalCRNN, self).__init__()
        self.backbone_net = backbone_net

    def forward(self, y, physics, **kwargs):
        # Initial reconstruction x0 = A_H(y)
        x_init = physics.A_adjoint(y)

        # --- Input Normalization ---
        # Normalize the initial estimate before feeding it to the network.
        # This is a good practice for stability.

        # NOTE: I simplified the logic here. We rearrange once, normalize, then rearrange back.
        # This avoids the double rearrange which was a bit confusing and prone to error.
        x_init_permuted = rearrange(x_init, "b c t h w -> b h w t c")
        x_init_norm_permuted = _normalize_batch(x_init_permuted)

        mask = self.backbone_net.datalayer.physics.mask

        x_hat_permuted = self.backbone_net(x_init_norm_permuted, y, mask)

        # Permute back to standard (B, C, T, H, W) for loss calculation
        x_hat = rearrange(x_hat_permuted, "b h w t c -> b c t h w")

        return x_hat
