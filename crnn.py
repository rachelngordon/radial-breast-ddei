import torch
import torch.nn as nn
from einops import rearrange
import wandb


def log(key, value, step):
    # In a real scenario, this would be wandb.log({key: value, "step": step})
    print(f"Step {step} - {key}: {value:.4f}")


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


def normalize_batch_percentile(x, percentile=99.5):
    """
    Normalizes each item in the batch based on a high percentile of its absolute values.

    This method is more robust to extreme outliers (e.g., from NUFFT artifacts)
    than normalizing by the absolute maximum. It ensures that the scaling factor
    is determined by the bulk of the image signal, not a single rogue pixel.

    Shape of x: (B, H, W, T, C) or any shape starting with Batch.
    """
    # Ensure a percentile value is valid
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    b = x.shape[0]

    # Use .reshape() to handle potentially non-contiguous tensors.
    # We flatten all dimensions except the batch dimension.
    x_flat = x.reshape(b, -1)

    # Calculate the specified percentile of the absolute values for each item in the batch.
    # We work with absolute values to find a suitable scaling factor for the signal magnitude.
    # Note: torch.quantile expects q to be in the range [0, 1].
    q = percentile / 100.0
    # The [0] at the end is to extract the values tensor from the (values, indices) tuple that some versions of PyTorch might return
    # or to simplify the output dimension.
    percentile_vals = torch.quantile(torch.abs(x_flat), q, dim=1, keepdim=True)

    # Add a small epsilon to prevent division by zero for blank/zero images.
    norm_factors = percentile_vals + 1e-8

    # Reshape the normalization factors to be broadcastable with the original tensor x.
    # The shape will be (B, 1, 1, 1, 1) to match the input dimensions.
    dims_to_unsqueeze = [1] * (x.dim() - 1)
    norm_factors_reshaped = norm_factors.view(b, *dims_to_unsqueeze)

    return x / norm_factors_reshaped


def normalize_batch_percentile_and_clip(x, percentile=99.5, clip_value=10.0):
    """
    Combines robust percentile normalization with hard clipping to tame extreme outliers.
    """
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    b = x.shape[0]
    x_flat = x.reshape(b, -1)
    q = percentile / 100.0

    percentile_vals = torch.quantile(torch.abs(x_flat), q, dim=1, keepdim=True)
    norm_factors = percentile_vals + 1e-8

    dims_to_unsqueeze = [1] * (x.dim() - 1)
    norm_factors_reshaped = norm_factors.view(b, *dims_to_unsqueeze)
    
    normalized_x = x / norm_factors_reshaped
    
    # Clip the result to a reasonable range
    clipped_x = torch.clamp(normalized_x, -clip_value, clip_value)
    
    return clipped_x


def normalize_batch_standardize(x, epsilon=1e-8):
    """
    Standardizes each item in the batch to have a mean of 0 and a standard deviation of 1.

    This is a very common and robust normalization technique in deep learning. It centers
    the data and scales it, which can significantly improve network stability and
    convergence, especially when dealing with inputs that have varying offsets or scales.

    Shape of x: (B, H, W, T, C) or any shape starting with the Batch dimension.
    """
    # Get the batch size
    b = x.shape[0]

    # Use .reshape() to handle potentially non-contiguous tensors from operations like permute.
    # We flatten all dimensions except the batch dimension to compute stats over each item.
    x_flat = x.reshape(b, -1)

    # Calculate the mean for each item in the batch.
    # `keepdim=True` is crucial to maintain a shape of (B, 1) for broadcasting.
    mean = torch.mean(x_flat, dim=1, keepdim=True)

    # Calculate the standard deviation for each item in the batch.
    std = torch.std(x_flat, dim=1, keepdim=True)

    # Add the small epsilon to the standard deviation to prevent division by zero
    # for any blank or constant-valued images in the batch.
    safe_std = std + epsilon

    # Reshape the mean and std tensors to be broadcastable with the original tensor x.
    # This dynamically creates the correct number of singleton dimensions.
    # e.g., (B, 1) -> (B, 1, 1, 1, 1)
    dims_to_unsqueeze = [1] * (x.dim() - 1)
    mean_reshaped = mean.view(b, *dims_to_unsqueeze)
    std_reshaped = safe_std.view(b, *dims_to_unsqueeze)

    # Apply the standardization formula: (x - mean) / std
    return (x - mean_reshaped) / std_reshaped


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

    def forward(self, x_init_permuted, y, mask, csmap):  # 'mask' is the unused dummy mask
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
            x_post_dc = self.datalayer(x_pre_dc, y, self.datalayer.physics.mask, csmap)

            # --- 3. Normalization Block (The Fix) ---
            # This is the crucial step to prevent the feedback loop explosion.
            global_step = 0
            if i < self.num_cascades - 1:
                # Log metrics BEFORE normalization
                # log(f"cascade_{i}/pre_norm_mean", x_post_dc.mean().item(), global_step)
                # log(f"cascade_{i}/pre_norm_std", x_post_dc.std().item(), global_step)
                # log(f"cascade_{i}/pre_norm_min", x_post_dc.min().item(), global_step)
                # log(f"cascade_{i}/pre_norm_max", x_post_dc.max().item(), global_step)

                # Choose your normalization strategy. Renormalizing is often better.
                # x_cascade_in = _renormalize_by_input(x_post_dc, x_pre_dc)

                # z score normalization
                #x_cascade_in = normalize_batch_standardize(x_post_dc)
                # x_cascade_in = normalize_batch_percentile_and_clip(x_post_dc)

                # per-frame normalization
                # Reshape for per-frame normalization. Current shape: (B, H, W, T, C)
                B, H, W, T, C = x_post_dc.shape
                x_reshaped = x_post_dc.permute(0, 3, 1, 2, 4).contiguous().view(B * T, H, W, C) # Reshape to (B*T, H, W, C)

                # Standardize each frame independently
                x_cascade_in = normalize_batch_standardize(x_reshaped)

                # Reshape back to the original format for the next BCRNN pass
                x_cascade_in = x_cascade_in.view(B, T, H, W, C).permute(0, 2, 3, 1, 4).contiguous()


                # Log metrics AFTER normalization
                # log(f"cascade_{i}/post_norm_mean", x_cascade_in.mean().item(), global_step)
                # log(f"cascade_{i}/post_norm_std", x_cascade_in.std().item(), global_step)
                # log(f"cascade_{i}/post_norm_min", x_cascade_in.min().item(), global_step)
                # log(f"cascade_{i}/post_norm_max", x_cascade_in.max().item(), global_step)
            else:
                # On the last cascade, don't normalize the final output
                x_cascade_in = x_post_dc

        return x_cascade_in


class ArtifactRemovalCRNN(nn.Module):
    def __init__(self, backbone_net, **kwargs):
        super(ArtifactRemovalCRNN, self).__init__()
        self.backbone_net = backbone_net

    def forward(self, y, physics, csmap, **kwargs):
        # 1. Get the initial ZF recon. This defines our target energy/scale.
        x_init = physics.A_adjoint(y, csmap)

        # We need its norm before any permutations.
        norm_of_zf_recon = torch.linalg.vector_norm(x_init)

        # 2. Permute and normalize the input for the network
        x_init_permuted = rearrange(x_init, "b c t h w -> b h w t c")

        # x_init_norm_permuted = _normalize_batch(x_init_permuted)
        # x_init_norm_permuted = normalize_batch_percentile(x_init_permuted)
        # x_init_norm_permuted = normalize_batch_standardize(x_init_permuted)
        # x_init_norm_permuted = normalize_batch_percentile_and_clip(x_init_permuted)

        # per-frame normalization
        B, H, W, T, C = x_init_permuted.shape
        x_init_permuted = x_init_permuted.permute(0, 3, 1, 2, 4).contiguous().view(B * T, H, W, C) # Reshape to (B*T, H, W, C)

        # Standardize each frame independently
        x_init_norm_permuted = normalize_batch_standardize(x_init_permuted)

        # Reshape back to the original format for the next BCRNN pass
        x_init_norm_permuted = x_init_norm_permuted.view(B, T, H, W, C).permute(0, 2, 3, 1, 4).contiguous()


        mask = physics.mask #torch.ones_like(y)

        # 3. Get the raw, high-magnitude output from the backbone
        x_hat_permuted_raw = self.backbone_net(x_init_norm_permuted, y, mask, csmap)

        # 4. Convert it back to standard image tensor format
        x_hat_raw = rearrange(x_hat_permuted_raw, "b h w t c -> b c t h w")

        # --- 5. Rescale the final output ---
        # Rescale the network's huge output to match the norm of the initial ZF recon.
        # This gives the loss function two tensors of a similar magnitude to compare.
        norm_of_raw_output = torch.linalg.vector_norm(x_hat_raw)

        scaling_factor = norm_of_zf_recon / (norm_of_raw_output + 1e-8)

        x_hat_final = x_hat_raw * scaling_factor

        return x_hat_final
