import torch
import torch.nn as nn
from einops import rearrange


# --- Helper function for normalization ---
def _normalize_batch(x):
    b, h, w, t, c = x.shape
    x_flat = x.reshape(b, -1)
    max_vals, _ = torch.max(torch.abs(x_flat), dim=1, keepdim=True)
    max_vals = max_vals + 1e-8
    max_vals_reshaped = max_vals.reshape(b, 1, 1, 1, 1)
    return x / max_vals_reshaped


def _renormalize_by_input(x_after_dc, x_before_dc):
    norm_before = torch.linalg.vector_norm(x_before_dc.flatten(1), dim=1, keepdim=True)
    norm_after = torch.linalg.vector_norm(x_after_dc.flatten(1), dim=1, keepdim=True)
    scaling_factor = norm_before / (norm_after + 1e-8)
    return x_after_dc * scaling_factor.view(-1, 1, 1, 1, 1)


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

    def forward(self, x_init_permuted, y, mask):  # 'mask' is the unused dummy mask
        B, H, W, T, C = x_init_permuted.shape
        x_cascade_in = x_init_permuted

        for i in range(self.num_cascades):
            # --- 1. Regularization Block (Network's guess) ---
            bcrnn_out = self.bcrnn(x_cascade_in, x_init_permuted)
            bcrnn_out_reshaped = (
                bcrnn_out.permute(0, 3, 4, 1, 2)
                .contiguous()
                .reshape(B * T, self.chans, H, W)
            )
            res_out_reshaped = self.res_conv(bcrnn_out_reshaped)
            res_out = (
                res_out_reshaped.reshape(B, T, self.in_chans, H, W)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )
            x_pre_dc = x_cascade_in + res_out

            # --- 2. Data Consistency Block (The Explosion) ---
            # Use the correct physics mask from the datalayer itself
            x_post_dc = self.datalayer(x_pre_dc, y, self.datalayer.physics.mask)

            # --- 3. Normalization Block (The Fix) ---
            # This is the crucial step to prevent the feedback loop explosion.
            if i < self.num_cascades - 1:
                # Choose your normalization strategy. Renormalizing is often better.
                x_cascade_in = _renormalize_by_input(x_post_dc, x_pre_dc)
            else:
                # On the last cascade, don't normalize the final output
                x_cascade_in = x_post_dc

        return x_cascade_in


class ArtifactRemovalCRNN(nn.Module):
    def __init__(self, backbone_net, **kwargs):
        super(ArtifactRemovalCRNN, self).__init__()
        self.backbone_net = backbone_net

    def forward(self, y, physics, **kwargs):
        # 1. Get the initial ZF recon. This defines our target energy/scale.
        x_init = physics.A_adjoint(y)

        # We need its norm before any permutations.
        norm_of_zf_recon = torch.linalg.vector_norm(x_init)

        # 2. Permute and normalize the input for the network
        x_init_permuted = rearrange(x_init, "b c t h w -> b h w t c")
        x_init_norm_permuted = _normalize_batch(x_init_permuted)

        mask = torch.ones_like(y)

        # 3. Get the raw, high-magnitude output from the backbone
        x_hat_permuted_raw = self.backbone_net(x_init_norm_permuted, y, mask)

        # 4. Convert it back to standard image tensor format
        x_hat_raw = rearrange(x_hat_permuted_raw, "b h w t c -> b c t h w")

        # --- 5. Rescale the final output ---
        # Rescale the network's huge output to match the norm of the initial ZF recon.
        # This gives the loss function two tensors of a similar magnitude to compare.
        norm_of_raw_output = torch.linalg.vector_norm(x_hat_raw)

        scaling_factor = norm_of_zf_recon / (norm_of_raw_output + 1e-8)

        x_hat_final = x_hat_raw * scaling_factor

        return x_hat_final
