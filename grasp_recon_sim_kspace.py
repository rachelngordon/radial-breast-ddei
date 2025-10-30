import argparse
import h5py
import os
import pathlib

import numpy as np
import sigpy as sp
import torch
import cupy as cp

from sigpy.mri import app
from einops import rearrange
import matplotlib.pyplot as plt
import argparse
from utils import to_torch_complex, prep_nufft
from radial_lsfp import MCNUFFT

def get_traj(spokes_per_frame, csmaps=False, N_spokes=13, N_time=1, base_res=320, gind=1):

    N_tot_spokes = N_spokes * N_time

    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    if spokes_per_frame == 288 and csmaps == False:
        return traj
    else:
        return np.squeeze(traj)

# %% compute coil sensitivity maps
def get_coil(ksp, spokes_per_frame, device=sp.Device(-1)):

    N_coils, N_spokes, N_samples = ksp.shape

    base_res = N_samples // 2

    ishape = [N_coils] + [base_res] * 2

    traj = get_traj(spokes_per_frame, csmaps=True, N_spokes=N_spokes, N_time=1,
                    base_res=base_res, gind=1)

    dcf = (traj[..., 0]**2 + traj[..., 1]**2)**0.5

    F = sp.linop.NUFFT(ishape, traj)

    cim = F.H(ksp * dcf)
    cim = sp.fft(cim, axes=(-2, -1))

    mps = app.EspiritCalib(cim, device=device).run()
    # print(type(mps))

    mps = sp.to_device(mps)
    # print("After to device: ", type(mps))

    return mps

parser = argparse.ArgumentParser()
parser.add_argument("--spokes_per_frame", type=int, required=True)
args = parser.parse_args()

# load DRO
dro_id = 30
sample_dir = f'/ess/scratch/scratch1/rachelgordon/dro_dataset/dro_144frames/sample_{dro_id:03d}_sub{dro_id}'

csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))

ground_truth_complex = dro['ground_truth_images']
ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) # T, H, W
ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)

# CSMaps: (H, W, C) -> (1, C, H, W) [batch, coils, h, w]
csmaps_torch = torch.from_numpy(csmaps).permute(2, 0, 1).unsqueeze(0)

# simulate k-space
device = torch.device("cuda")
csmap = csmaps_torch.to(device)
ground_truth = ground_truth_torch.to(device).unsqueeze(0) # Shape: (1, 2, T, H, W)

ground_truth_for_physics = rearrange(to_torch_complex(ground_truth), 'b t h w -> b h w t')

# SIMULATE KSPACE
# define physics object for evaluation
N_samples = 640
spokes_per_frame = args.spokes_per_frame
N_time_eval = int(288 / args.spokes_per_frame)
eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, spokes_per_frame, N_time_eval)
eval_ktraj = eval_ktraj.to(device)
eval_dcomp = eval_dcomp.to(device)
eval_nufft_ob = eval_nufft_ob.to(device)
eval_adjnufft_ob = eval_adjnufft_ob.to(device)

eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

ksp_f = eval_physics(False, ground_truth_for_physics, csmap)
ksp_f = rearrange(ksp_f, 'c (sp sam) t -> (sp t) sam c', sp=spokes_per_frame).unsqueeze(-1).cpu().numpy()
print("kspace shape: ", ksp_f.shape)



device = sp.Device(0 if torch.cuda.is_available() else -1)



ksp = np.transpose(ksp_f, (3, 2, 0, 1))

N_slices, N_coils, N_spokes, N_samples = ksp.shape

base_res = N_samples // 2

N_time = N_spokes // spokes_per_frame

N_spokes_prep = N_time * spokes_per_frame

ksp_redu = ksp[:, :, :N_spokes_prep, :]
print('  ksp_redu shape: ', ksp_redu.shape)

# %% retrospecitvely split spokes
ksp_prep = np.swapaxes(ksp_redu, 0, 2)
ksp_prep_shape = ksp_prep.shape
ksp_prep = np.reshape(ksp_prep, [N_time, spokes_per_frame] + list(ksp_prep_shape[1:]))
ksp_prep = np.transpose(ksp_prep, (3, 0, 2, 1, 4))


ksp_prep = ksp_prep[:, :, None, :, None, :, :]
print('  ksp_prep shape: ', ksp_prep.shape)
print('  ksp_prep dtype: ', ksp_prep.dtype)


# %% trajectories
traj = get_traj(spokes_per_frame, N_spokes=spokes_per_frame,
                N_time=N_time, base_res=base_res,
                gind=1)
print('  traj shape: ', traj.shape)



# %% slice-by-slice recon

# coil sensitivity maps
print('> compute coil sensitivity maps')
C = get_coil(ksp[0], spokes_per_frame, device=device)
C = C[:, None, :, :]
print('  coil shape: ', C.shape)


# recon
k1 = ksp_prep[0]
print("k1 shape: ", k1.shape)
print("k1 dtype: ", k1.dtype)

print("---- k-space input shape: ", k1.shape) # ---- k-space input shape:  (8, 1, 16, 1, 36, 640)
print("---- csmaps input shape: ", C.shape) # ---- csmaps input shape:  (16, 1, 320, 320)
print("---- traj input shape: ", traj.shape) # ---- traj input shape:  (8, 36, 640, 2)
R1 = app.HighDimensionalRecon(k1, C,
                combine_echo=False,
                lamda=0.001,
                coord=traj,
                regu='TV', regu_axes=[0],
                max_iter=10,
                solver='ADMM', rho=0.1,
                device=device,
                show_pbar=False,
                verbose=False).run()
print("R1 dtype: ", R1.dtype)

    # slice_count += 1

    # if slice_count >= num_slices:
    #     break

acq_slices = sp.to_device(R1)

acq_slices = cp.array(acq_slices)
acq_slices = cp.asnumpy(acq_slices)
print("acq_slices dtype: ", acq_slices.dtype)
# acq_slices = np.squeeze(abs(acq_slices))

# acq_slices = acq_slices.squeeze(axis=(2, 3, 4))
acq_slices = acq_slices.squeeze()
 
# acq_slices = acq_slices.squeeze(axis=(2, 3, 4))
print("acq_slices dtype: ", acq_slices.dtype)

print("acq_slices shape: ", acq_slices.shape)

# slice_idx_to_plot = 95
if spokes_per_frame < 288:
    # select first timeframe
    plt.imshow(np.abs(acq_slices[0]), cmap='gray')
else:
    plt.imshow(np.abs(acq_slices), cmap='gray')

plt.axis("off")
plt.savefig(f'/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/sim_grasp_recon_{spokes_per_frame}spf_dro_{dro_id}.png')