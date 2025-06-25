import torch
import torch.nn as nn
import numpy as np
from time import time

dtype = torch.complex64

class MCNUFFT(nn.Module):
    def __init__(self, nufft_ob, adjnufft_ob, ktraj, dcomp, smaps):
        super(MCNUFFT, self).__init__()
        self.nufft_ob = nufft_ob
        self.adjnufft_ob = adjnufft_ob
        self.ktraj = torch.squeeze(ktraj)
        self.dcomp = torch.squeeze(dcomp)
        self.smaps = smaps

    def forward(self, inv, data):
        data = torch.squeeze(data)  # delete redundant dimension
        Nx = self.smaps.shape[2]
        Ny = self.smaps.shape[3]

        if inv:  # adjoint nufft

            self.smaps = self.smaps.to(dtype)

            if len(data.shape) > 2:  # multi-frame

                x = torch.zeros([Nx, Ny, data.shape[2]], dtype=dtype)

                for ii in range(0, data.shape[2]):
                    kd = data[:, :, ii]
                    k = self.ktraj[:, :, ii]
                    d = self.dcomp[:, ii]

                    kd = kd.unsqueeze(0)
                    d = d.unsqueeze(0).unsqueeze(0)

                    tt1 = time()

                    x_temp = self.adjnufft_ob(kd * d, k, smaps=self.smaps)

                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)
                    tt2 = time()
                    # print('adjnufft time is %.6f' % (tt2 - tt1))

                    # plt.figure()
                    # plt.imshow(np.abs(x_temp.numpy()), 'gray')
                    # plt.show()

            else:  # single frame

                kd = data.unsqueeze(0)
                d = self.dcomp.unsqueeze(0).unsqueeze(0)
                x = self.adjnufft_ob(kd * d, self.ktraj, smaps=self.smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)

        else:  # forward nufft

            if len(data.shape) > 2:  # multi-frame

                x = torch.zeros([self.smaps.shape[1], self.ktraj.shape[1], data.shape[-1]], dtype=dtype)

                for ii in range(0, data.shape[-1]):
                    image = data[:, :, ii]
                    k = self.ktraj[:, :, ii]

                    image = image.unsqueeze(0).unsqueeze(0)
                    x_temp = self.nufft_ob(image, k, smaps=self.smaps)
                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)

            else:  # single frame

                image = data.unsqueeze(0).unsqueeze(0)
                x = self.nufft_ob(image, self.ktraj, smaps=self.smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)

        return x