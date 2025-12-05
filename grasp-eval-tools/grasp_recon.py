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

# %% read in k-space data
patient_id = '001'
IN_DIR = f'/ess/scratch/scratch1/rachelgordon/fastMRI_breast_data/full_kspace/fastMRI_breast_{patient_id}_2.h5'
spokes_per_frame = args.spokes_per_frame
center_partition = 31
images_per_slab = 192

device = sp.Device(0 if torch.cuda.is_available() else -1)

f = h5py.File(IN_DIR, 'r')
ksp_f = f['kspace'][:].T
ksp_f = np.transpose(ksp_f, (4, 3, 2, 1, 0))
print('> kspace shape ', ksp_f.shape)
f.close()


print("original k-space type: ", ksp_f.dtype)
ksp = ksp_f[0] + 1j * ksp_f[1]
print("k-space type after real/imag combined: ", ksp_f.dtype)
ksp = np.transpose(ksp, (3, 2, 0, 1))


# zero-fill the slice dimension if necessaary
partitions = ksp.shape[0]


if images_per_slab > partitions + 1:
    shift = int(images_per_slab / 2 - center_partition)
else:
    shift = 0
    print("slices less than or equal to partitions + 1.")

ksp_zf = np.zeros_like(ksp, shape=[images_per_slab] + list(ksp.shape[1:]))
ksp_zf[shift : shift + partitions, ...] = ksp


ksp_zf = sp.fft(ksp_zf, axes=(0,))


N_slices, N_coils, N_spokes, N_samples = ksp_zf.shape

base_res = N_samples // 2

N_time = N_spokes // spokes_per_frame

N_spokes_prep = N_time * spokes_per_frame

ksp_redu = ksp_zf[:, :, :N_spokes_prep, :]
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

slice_loop = range(N_slices)

acq_slices = []
# num_slices = 97
# slice_count = 0
# for s in slice_loop:
s = 95
print('>>> slice ', str(s).zfill(3))

# coil sensitivity maps
print('> compute coil sensitivity maps')
C = get_coil(ksp_zf[s], spokes_per_frame, device=device)
C = C[:, None, :, :]
print('  coil shape: ', C.shape)


# recon
k1 = ksp_prep[s]
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
acq_slices.append(R1)

    # slice_count += 1

    # if slice_count >= num_slices:
    #     break

acq_slices = sp.to_device(acq_slices)

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
plt.savefig(f'/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/grasp_recon_{spokes_per_frame}spf_fastMRI_{patient_id}_slice{s}.png')