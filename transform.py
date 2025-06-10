
import deepinv as dinv
import torch
from deepinv.transform import Transform
from einops import rearrange
from torchvision.transforms import InterpolationMode



class VideoRotate(dinv.transform.Rotate):
    """A Rotate transform that correctly handles 5D video tensors by flattening time into the batch dimension."""

    def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
        # First, check if we even need to flatten. If it's already 4D, just rotate.
        if not self._check_x_5D(x):
            return super()._transform(x, **params)

        # It's a 5D video tensor. Flatten time into the batch dimension.
        B = x.shape[0]
        x_flat = dinv.physics.TimeMixin.flatten(x)  # (B, C, T, H, W) -> (B*T, C, H, W)

        # The parent's _transform method can now work correctly on the 4D tensor (batch of 2D images).
        # We need to get the right parameters for this new batch size.
        # The `get_params` is usually called before `_transform`, so we should be okay.
        # However, to be safe, let's pass a modified params dictionary.
        flat_params = self.get_params(x_flat)

        transformed_flat = super()._transform(x_flat, **flat_params)

        # Unflatten to restore the original 5D video shape.
        return dinv.physics.TimeMixin.unflatten(transformed_flat, batch_size=B)


class VideoDiffeo(dinv.transform.CPABDiffeomorphism):
    """A Diffeomorphism transform that correctly handles 5D video tensors."""

    def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
        if not self._check_x_5D(x):
            return super()._transform(x, **params)

        B = x.shape[0]
        x_flat = dinv.physics.TimeMixin.flatten(x)
        flat_params = self.get_params(x_flat)
        transformed_flat = super()._transform(x_flat, **flat_params)
        return dinv.physics.TimeMixin.unflatten(transformed_flat, batch_size=B)


# class SubsampleTime(Transform):
#     r"""
#     Augments a video by taking a random contiguous temporal sub-sequence.
#     This is suitable for non-cyclical data like contrast enhancement curves,
#     as it preserves the local arrow of time.

#     :param int n_trans: Number of transformed versions to generate per input image.
#     :param float subsample_ratio: The ratio of the total time frames to keep (e.g., 0.8 for 80%).
#     :param torch.Generator rng: Random number generator.
#     """

#     def __init__(self, *args, subsample_ratio: float = 0.8, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.flatten_video_input = False  # We operate directly on the 5D tensor
#         assert 0.0 < subsample_ratio <= 1.0, "subsample_ratio must be between 0 and 1."
#         self.subsample_ratio = subsample_ratio

#     def _get_params(self, x: torch.Tensor) -> dict:
#         """Generates a random start index for the temporal crop."""
#         total_time_frames = x.shape[2]  # Shape is (B, C, T, H, W)
#         subsample_length = int(total_time_frames * self.subsample_ratio)
#         if subsample_length >= total_time_frames:
#             # Handle edge case where ratio is 1.0 or rounds up
#             return {"start_indices": torch.zeros(self.n_trans, dtype=torch.long)}

#         max_start_index = total_time_frames - subsample_length
#         start_indices = torch.randint(
#             low=0, high=max_start_index + 1, size=(self.n_trans,), generator=self.rng
#         )
#         return {"start_indices": start_indices}

#     def _transform(
#         self, x: torch.Tensor, start_indices: torch.Tensor, **kwargs
#     ) -> torch.Tensor:
#         """Performs the temporal subsampling and resizes back to the original length."""
#         B, C, total_time_frames, H, W = x.shape
#         subsample_length = int(total_time_frames * self.subsample_ratio)

#         if subsample_length >= total_time_frames:
#             return x.repeat(self.n_trans, 1, 1, 1, 1)

#         output_list = []
#         for start_idx in start_indices:
#             # 1. Take the temporal subsequence
#             sub_sequence = x[:, :, start_idx : start_idx + subsample_length, :, :]

#             # 2. Flatten all non-time dimensions into one giant "channel" dimension for interpolation.
#             # Pattern: (Batch, Channels, Time, Height, Width) -> (Batch, (Channels*Height*Width), Time)
#             flat_for_interp = rearrange(sub_sequence, "b c t h w -> b (c h w) t")

#             # 3. Interpolate along the time dimension (the last dimension).
#             # This is a 1D interpolation.
#             resized_flat = torch.nn.functional.interpolate(
#                 flat_for_interp,
#                 size=total_time_frames,
#                 mode="linear",
#                 align_corners=False,
#             )

#             # 4. Un-flatten the dimensions back to the original video format.
#             # Einops can do this because it knows how (c h w) was constructed.
#             # Pattern: (Batch, (Channels*Height*Width), Time) -> (Batch, Channels, Time, Height, Width)
#             resized_sequence = rearrange(
#                 resized_flat, "b (c h w) t -> b c t h w", c=C, h=H, w=W
#             )

#             output_list.append(resized_sequence)

#         return torch.cat(output_list, dim=0)
    

class SubsampleTime(Transform):
    r"""
    Augments a video by taking a random contiguous temporal sub-sequence of a
    RANDOM length, and then interpolating it back to the original length.

    :param int n_trans: Number of transformed versions to generate per input image.
    :param tuple[float, float] subsample_ratio_range: The min and max ratio of the
                                                     total time frames to keep (e.g., (0.7, 0.95)).
    :param torch.Generator rng: Random number generator.
    """

    def __init__(self, *args, subsample_ratio_range: tuple[float, float] = (0.7, 0.95), **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False  # Operate on 5D tensor
        
        min_r, max_r = subsample_ratio_range
        assert 0.0 < min_r <= max_r <= 1.0, "subsample_ratio_range must be a valid range (min, max) between 0 and 1."
        self.subsample_ratio_range = subsample_ratio_range

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        Generates a random ratio and a random start index for each transform.
        """
        total_time_frames = x.shape[2]  # Shape is (B, C, T, H, W)
        min_r, max_r = self.subsample_ratio_range
        
        # 1. Sample a random ratio for each of the `n_trans` transforms
        #    torch.rand generates in [0, 1), so we scale and shift to get [min_r, max_r)
        ratios = min_r + (max_r - min_r) * torch.rand(self.n_trans, generator=self.rng)

        start_indices = []
        for ratio in ratios:
            # 2. Calculate length and max start index based on THIS transform's ratio
            subsample_length = int(total_time_frames * ratio.item())
            if subsample_length >= total_time_frames:
                start_indices.append(0)
                continue

            max_start_index = total_time_frames - subsample_length
            start_idx = torch.randint(
                low=0, high=max_start_index + 1, size=(1,), generator=self.rng
            ).item()
            start_indices.append(start_idx)
            
        return {"ratios": ratios, "start_indices": torch.tensor(start_indices, dtype=torch.long)}

    def _transform(
        self, x: torch.Tensor, ratios: torch.Tensor, start_indices: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Performs the temporal subsampling and resizing for each requested transform."""
        B, C, total_time_frames, H, W = x.shape
        assert B == 1, "This transform implementation assumes a batch size of 1 for simplicity."

        output_list = []
        # We generate `n_trans` augmented versions from the single input `x`
        for i in range(self.n_trans):
            ratio = ratios[i]
            start_idx = start_indices[i]
            subsample_length = int(total_time_frames * ratio.item())

            if subsample_length >= total_time_frames:
                output_list.append(x.clone()) # Use clone to avoid issues
                continue

            # 1. Take the subsequence
            sub_sequence = x[:, :, start_idx : start_idx + subsample_length, :, :]

            # 2. Flatten for interpolation
            flat_for_interp = rearrange(sub_sequence, "b c t h w -> b (c h w) t")

            # 3. Interpolate back to original time dimension length
            resized_flat = torch.nn.functional.interpolate(
                flat_for_interp,
                size=total_time_frames,
                mode="linear",
                align_corners=False,
            )

            # 4. Un-flatten back to video format
            resized_sequence = rearrange(
                resized_flat, "b (c h w) t -> b c t h w", c=C, h=H, w=W
            )
            output_list.append(resized_sequence)

        return torch.cat(output_list, dim=0)
    


class PeakAwareBiPhasicWarp(Transform):
    r"""
    An enhancement-peak-aware temporal augmentation that warps BOTH the wash-in
    and wash-out phases independently with different random ratios.

    It finds the time of peak enhancement, splits the video, and then
    time-warps (compresses/stretches) both phases before reassembling them.

    :param tuple[float, float] warp_ratio_range: The min/max ratio for
                                                 compressing a phase. e.g., (0.6, 0.95).
    """

    def __init__(self, *args, warp_ratio_range: tuple[float, float] = (0.6, 0.95), **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False
        min_r, max_r = warp_ratio_range
        assert 0.0 < min_r <= max_r <= 1.0, "warp_ratio_range must be valid."
        self.warp_ratio_range = warp_ratio_range

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        Generate two independent warp ratios for each requested transform:
        one for wash-in and one for wash-out.
        """
        params_list = []
        min_r, max_r = self.warp_ratio_range
        
        for _ in range(self.n_trans):
            # Generate TWO independent random ratios from the specified range
            washin_ratio = min_r + (max_r - min_r) * torch.rand(1, generator=self.rng).item()
            washout_ratio = min_r + (max_r - min_r) * torch.rand(1, generator=self.rng).item()
            
            params_list.append({
                "washin_ratio": washin_ratio,
                "washout_ratio": washout_ratio
            })
        
        return {"params_list": params_list}

    def _transform(self, x: torch.Tensor, params_list: list[dict], **kwargs) -> torch.Tensor:
        """Applies the independent, bi-phasic warp."""
        assert x.shape[0] == 1, "This transform assumes a batch size of 1 for the input."

        # Option 1: Find a single, global peak index for simplicity
        mean_signal_curve = torch.mean(x, dim=(0, 1, 3, 4)) # Avg over B, C, H, W
        peak_idx = torch.argmax(mean_signal_curve)

        # Edge case: If peak is at the start/end, there's no wash-in/wash-out to warp.
        if peak_idx <= 0 or peak_idx >= x.shape[2] - 1:
            return x.repeat(self.n_trans, 1, 1, 1, 1)

        output_list = []
        for params in params_list:
            # 1. Split the video into three parts
            wash_in_phase = x[:, :, :peak_idx, :, :]
            peak_frame = x[:, :, peak_idx:peak_idx+1, :, :]
            wash_out_phase = x[:, :, peak_idx+1:, :, :]

            # 2. Warp BOTH phases using their respective independent ratios
            warped_wash_in = self._warp_phase(wash_in_phase, params["washin_ratio"])
            warped_wash_out = self._warp_phase(wash_out_phase, params["washout_ratio"])

            # 3. Reassemble the video
            new_x = torch.cat([warped_wash_in, peak_frame, warped_wash_out], dim=2)
            output_list.append(new_x)

        return torch.cat(output_list, dim=0)

    def _warp_phase(self, phase_tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """Helper function to subsample and interpolate a video phase."""
        B, C, T_phase, H, W = phase_tensor.shape
        if T_phase <= 1: # Cannot warp a single frame or empty tensor
            return phase_tensor

        subsample_len = max(1, int(T_phase * ratio)) # Ensure at least 1 frame

        # Reshape for interpolation
        flat_for_interp = rearrange(phase_tensor, "b c t h w -> b (c h w) t")
        
        # Subsample by interpolating down, then interpolate back up.
        # This creates a smooth warp effect.
        subsampled_flat = F.interpolate(flat_for_interp, size=subsample_len, mode='linear', align_corners=False)
        resized_flat = F.interpolate(subsampled_flat, size=T_phase, mode='linear', align_corners=False)

        # Reshape back to video format
        warped_phase = rearrange(resized_flat, "b (c h w) t -> b c t h w", c=C, h=H, w=W)
        return warped_phase