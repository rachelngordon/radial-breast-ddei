import torch
import torch.nn as nn
import numpy as np
from time import time

dtype = torch.complex64

# class MCNUFFT(nn.Module):
#     def __init__(self, nufft_ob, adjnufft_ob, ktraj, dcomp):
#         super(MCNUFFT, self).__init__()
#         self.nufft_ob = nufft_ob
#         self.adjnufft_ob = adjnufft_ob
#         self.ktraj = torch.squeeze(ktraj)
#         self.dcomp = torch.squeeze(dcomp)

#     def forward(self, inv, data, smaps):

#         data = torch.squeeze(data)  # delete redundant dimension
#         Nx = smaps.shape[2]
#         Ny = smaps.shape[3]

#         if inv:  # adjoint nufft

#             smaps = smaps.to(dtype)

#             if len(data.shape) > 2:  # multi-frame

#                 x = torch.zeros([Nx, Ny, data.shape[2]], dtype=dtype)

#                 for ii in range(0, data.shape[2]):
#                     kd = data[:, :, ii]
#                     k = self.ktraj[:, :, ii]
#                     d = self.dcomp[:, ii]

#                     kd = kd.unsqueeze(0)
#                     d = d.unsqueeze(0).unsqueeze(0)

#                     tt1 = time()
#                     x_temp = self.adjnufft_ob(kd * d, k, smaps=smaps)

#                     x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)
#                     tt2 = time()
#                     # print('adjnufft time is %.6f' % (tt2 - tt1))

#                     # plt.figure()
#                     # plt.imshow(np.abs(x_temp.numpy()), 'gray')
#                     # plt.show()

#             else:  # single frame

#                 kd = data.unsqueeze(0)
#                 d = self.dcomp.unsqueeze(0).unsqueeze(0)
#                 x = self.adjnufft_ob(kd * d, self.ktraj, smaps=smaps)
#                 x = torch.squeeze(x) / np.sqrt(Nx * Ny)

#         else:  # forward nufft

#             if len(data.shape) > 2:  # multi-frame

#                 x = torch.zeros([smaps.shape[1], self.ktraj.shape[1], data.shape[-1]], dtype=dtype)

#                 for ii in range(0, data.shape[-1]):

#                     image = data[:, :, ii]
#                     k = self.ktraj[:, :, ii]

#                     image = image.unsqueeze(0).unsqueeze(0)
#                     x_temp = self.nufft_ob(image, k, smaps=smaps)
#                     x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)

#             else:  # single frame

#                 image = data.unsqueeze(0).unsqueeze(0)
#                 x = self.nufft_ob(image, self.ktraj, smaps=smaps)
#                 x = torch.squeeze(x) / np.sqrt(Nx * Ny)

#         return x


class MCNUFFT(nn.Module):
    def __init__(self, nufft_ob, adjnufft_ob, ktraj, dcomp):
        super(MCNUFFT, self).__init__()
        self.nufft_ob = nufft_ob
        self.adjnufft_ob = adjnufft_ob
        self.ktraj = torch.squeeze(ktraj)
        self.dcomp = torch.squeeze(dcomp)

    def forward(self, inv, data, smaps):
        # Squeeze is fine, but let's get original shapes
        data = torch.squeeze(data)
        Nx, Ny = smaps.shape[2], smaps.shape[3]

        if len(data.shape) > 2:  # multi-frame
            is_complex_data = torch.is_complex(data)
            
            # --- Vectorized approach ---
            if inv: # Adjoint NUFFT (k-space -> image)
                # Original shape: [coils, samples, time]
                # We need [batch, coils, samples] for nufft, so permute time to batch
                kd = data.permute(2, 0, 1) # -> [time, coils, samples]
                d = self.dcomp.permute(1, 0) # -> [time, samples]
                k = self.ktraj.permute(2, 0, 1) # -> [time, samples, 2]

                # Unsqueeze for coils/smaps dim
                d = d.unsqueeze(1) 
                
                # Perform one batched operation
                x_temp = self.adjnufft_ob(kd * d, k, smaps=smaps.to(dtype))
                # Output shape: [time, 1, Nx, Ny]
                
                # Reshape back to [Nx, Ny, time]
                x = x_temp.squeeze(1).permute(1, 2, 0) / np.sqrt(Nx * Ny)

            else: # Forward NUFFT (image -> k-space)
                # Original shape: [Nx, Ny, time]
                # We need [batch, 1, Nx, Ny] for nufft, so permute
                image = data.permute(2, 0, 1).unsqueeze(1) # -> [time, 1, Nx, Ny]
                k = self.ktraj.permute(2, 0, 1) # -> [time, samples, 2]
                
                # Perform one batched operation
                x_temp = self.nufft_ob(image, k, smaps=smaps)
                # Output shape: [time, coils, samples]
                
                # Reshape back to [coils, samples, time]
                x = x_temp.permute(1, 2, 0) / np.sqrt(Nx * Ny)
        else:  # single frame (original logic is fine)
            if inv:
                kd = data.unsqueeze(0)
                d = self.dcomp.unsqueeze(0).unsqueeze(0)
                x = self.adjnufft_ob(kd * d, self.ktraj, smaps=smaps.to(dtype))
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)
            else:
                image = data.unsqueeze(0).unsqueeze(0)
                x = self.nufft_ob(image, self.ktraj, smaps=smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)
        return x