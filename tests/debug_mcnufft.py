import torch
import torchkbnufft as tkbn
import numpy as np

# Paste the inspect_tensor function here
def inspect_tensor(tensor, name="Tensor"):
    # ... (code from above) ...
    if not isinstance(tensor, torch.Tensor):
        print(f"--- {name} is not a Tensor (type: {type(tensor)}) ---")
        return

    print(f"--- Inspection of: {name} ---")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    
    # Check for non-finite values
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count > 0:
        print(f"  WARNING: Contains {nan_count} NaN values!")
    if inf_count > 0:
        print(f"  WARNING: Contains {inf_count} Inf values!")

    # For complex tensors, inspect real and imaginary parts
    if tensor.is_complex():
        tensor_real = tensor.real
        tensor_imag = tensor.imag
        print("  --- Real Part Stats ---")
        print(f"    Min: {tensor_real.min().item():.6f}, Max: {tensor_real.max().item():.6f}")
        print(f"    Mean: {tensor_real.mean().item():.6f}, Std: {tensor_real.std().item():.6f}")
        print("  --- Imaginary Part Stats ---")
        print(f"    Min: {tensor_imag.min().item():.6f}, Max: {tensor_imag.max().item():.6f}")
        print(f"    Mean: {tensor_imag.mean().item():.6f}, Std: {tensor_imag.std().item():.6f}")
    # For real tensors
    else:
        print(f"  Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
        print(f"  Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
        
    print("-" * (25 + len(name)))


# Paste your MCNUFFT class here
class MCNUFFT(torch.nn.Module):
    # ... (your exact MCNUFFT class code) ...
    def __init__(self, nufft_ob, adjnufft_ob, ktraj):
        super(MCNUFFT, self).__init__()
        self.nufft_ob = nufft_ob
        self.adjnufft_ob = adjnufft_ob
        self.ktraj = torch.squeeze(ktraj)
        # self.dcomp = torch.squeeze(dcomp)

    def forward(self, inv, data, smaps):
        # We will add inspection calls inside this method
        print(f"\n{'='*20} MCNUFFT {'FORWARD' if not inv else 'ADJOINT'} PASS {'='*20}")
        inspect_tensor(data, "Input 'data'")
        inspect_tensor(smaps, "Input 'smaps'")
        
        data = torch.squeeze(data)  # delete redundant dimension
        Nx = smaps.shape[2]
        Ny = smaps.shape[3]

        if inv:  # adjoint nufft
            smaps = smaps.to(data.dtype)
            if len(data.shape) > 2:  # multi-frame
                x = torch.zeros([Nx, Ny, data.shape[2]], dtype=data.dtype, device=data.device)
                for ii in range(0, data.shape[2]):
                    kd = data[:, :, ii]
                    k = self.ktraj[:, :, ii]
                    # d = self.dcomp[:, ii]
                    # inspect_tensor(d, f"dcomp for frame {ii}")
                    
                    kd_weighted = kd #* d
                    inspect_tensor(kd_weighted, f"Weighted k-space (kd * d) for frame {ii}")

                    x_temp = self.adjnufft_ob(kd_weighted.unsqueeze(0), k, smaps=smaps)
                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)
                inspect_tensor(x, "Final Output 'x' (Adjoint)")

            # ... (single frame logic can be debugged similarly if needed)
        else:  # forward nufft
            if len(data.shape) > 2:  # multi-frame
                x = torch.zeros([smaps.shape[1], self.ktraj.shape[1], data.shape[-1]], dtype=data.dtype, device=data.device)
                for ii in range(0, data.shape[-1]):
                    image = data[:, :, ii]
                    k = self.ktraj[:, :, ii]
                    
                    inspect_tensor(image, f"Input image for frame {ii}")
                    
                    x_temp = self.nufft_ob(image.unsqueeze(0).unsqueeze(0), k, smaps=smaps)
                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)
                inspect_tensor(x, "Final Output 'x' (Forward k-space)")

            # ... (single frame logic can be debugged similarly if needed)
        return x

# --- Main Debugging Script ---
if __name__ == "__main__":
    # --- 1. Define Realistic Parameters ---
    im_size = (128, 128)
    N_coils = 8
    N_time = 6
    N_spokes = 32
    N_samples_per_spoke = im_size[0] * 2  # Oversampling factor of 2
    
    # --- 2. Create Dummy NUFFT Objects & Trajectory ---
    # This mimics the `prep_nufft` logic
    nufft_ob = tkbn.KbNufft(im_size=im_size)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)
    
    # Create a dummy trajectory and density compensation
    # Shape: (2, N_time * N_spokes * N_samples_per_spoke)
    total_samples = N_time * N_spokes * N_samples_per_spoke
    ktraj_flat = (torch.rand(2, total_samples) - 0.5) * 2 * np.pi
    
    # IMPORTANT: Calculate a realistic dcomp
    # dcomp_flat = tkbn.calc_density_compensation_function(ktraj=ktraj_flat, im_size=im_size)

    # Reshape them to match MCNUFFT's expected input
    # ktraj shape: (2, N_spokes * N_samples, N_time)
    ktraj_3d = ktraj_flat.reshape(2, N_spokes * N_samples_per_spoke, N_time)
    # dcomp shape: (N_spokes * N_samples, N_time)
    # dcomp_3d = dcomp_flat.reshape(N_spokes * N_samples_per_spoke, N_time)

    # --- 3. Instantiate MCNUFFT ---
    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_3d)

    print("\n" + "="*50)
    print("INSPECTING INITIAL PHYSICS MODEL PARAMETERS")
    print("="*50)
    inspect_tensor(physics.ktraj, "physics.ktraj")
    # inspect_tensor(physics.dcomp, "physics.dcomp")
    print("="*50 + "\n")

    # --- 4. Create Dummy Data for the Adjoint Test ---
    # Create a random image space tensor `x`
    x_true = torch.randn(*im_size, N_time, dtype=torch.complex64)
    inspect_tensor(x_true, "Initial image-space tensor 'x_true'")

    # Create random coil sensitivity maps `smaps`
    # Shape: (1, N_coils, H, W)
    smaps = torch.randn(1, N_coils, *im_size, dtype=torch.complex64)
    # Normalize smaps as is often done in practice (sum-of-squares = 1)
    smaps = smaps / torch.sqrt(torch.sum(smaps.abs()**2, dim=1, keepdim=True))
    inspect_tensor(smaps, "Initial 'smaps'")

    # --- 5. Perform the Adjoint Test ---
    print("\n" + "="*50)
    print("           RUNNING ADJOINTNESS TEST")
    print("="*50)
    # a) Compute A(x) to get k-space data y
    # Note: data needs to be in shape (H, W, T) for your forward pass
    y = physics(inv=False, data=x_true, smaps=smaps)
    inspect_tensor(y, "Result of Forward NUFFT, 'y'")

    # b) Create a random k-space vector `z` to compute dot products with
    z = torch.randn_like(y)
    inspect_tensor(z, "Random k-space vector 'z' for dot product")

    # c) Compute A_H(z)
    x_adj_z = physics(inv=True, data=z, smaps=smaps)
    inspect_tensor(x_adj_z, "Result of Adjoint NUFFT on 'z', 'x_adj_z'")
    
    # d) Calculate the two sides of the adjoint equation: <Ax, z> and <x, A_H z>
    # <Ax, z> = sum(conj(Ax) * z)
    inner_product_1 = torch.sum(y.conj() * z)
    
    # <x, A_H z> = sum(conj(x) * A_H z)
    inner_product_2 = torch.sum(x_true.conj() * x_adj_z)

    print("\n" + "="*50)
    print("           ADJOINTNESS TEST RESULTS")
    print("="*50)
    print(f"<Ax, z>  = {inner_product_1.item()}")
    print(f"<x, A_H z> = {inner_product_2.item()}")
    
    # Check how close they are
    relative_error = torch.abs(inner_product_1 - inner_product_2) / torch.abs(inner_product_1)
    print(f"Relative Error: {relative_error.item():.8f}")
    
    if relative_error < 1e-4: # A generous tolerance for NUFFT
        print("\nSUCCESS: The relative error is small. The operator likely satisfies the adjoint property.")
    else:
        print("\nFAILURE: The relative error is large. The operator is NOT a proper adjoint.")
        print("This is a likely source of your vanishing gradients.")
    print("="*50)