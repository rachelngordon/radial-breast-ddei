import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import h5py
from time import time
import torchkbnufft as tkbn
from torch.utils.data import TensorDataset, DataLoader
from utils import trajGR, normabs, MCNUFFT
import matplotlib.pyplot as plt
import argparse
from lsfpnet import LSFPNet
import yaml
import json
from dataloader import SliceDataset
from mc import MCLoss
from einops import rearrange
conv_num = 3
dtype = torch.complex64

################# hyper parameters #################
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train ReconResNet model.")
parser.add_argument(
    "--config",
    type=str,
    required=False,
    default="config.yaml",
    help="Path to the configuration file",
)
parser.add_argument(
    "--exp_name", type=str, required=True, help="Name of the experiment"
)
parser.add_argument(
    "--from_checkpoint",
    type=bool,
    required=False,
    default=False,
    help="Whether to load from a checkpoint",
)
args = parser.parse_args()

# print experiment name and git commit
# commit_hash = get_git_commit()
# print(f"Running experiment on Git commit: {commit_hash}")

exp_name = args.exp_name
print(f"Experiment: {exp_name}")

# Load the configuration file
if args.from_checkpoint == True:
    with open(f"output/{exp_name}/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    with open(args.config, "r") as file:
        new_config = yaml.safe_load(file)
    
    end_epoch = new_config['training']["epochs"]
else:
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    end_epoch = config['training']["epochs"]


output_dir = os.path.join(config["experiment"]["output_dir"], exp_name)
os.makedirs(output_dir, exist_ok=True)

eval_dir = os.path.join(output_dir, "eval_results")
os.makedirs(eval_dir, exist_ok=True)


# Save the configuration file
if args.from_checkpoint == False:
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

# load params
split_file = config["data"]["split_file"]

batch_size = config["dataloader"]["batch_size"]
max_subjects = config["dataloader"]["max_subjects"]

layer_num = config["model"]["num_layers"]
learning_rate = config["model"]["optimizer"]["lr"]

mc_loss_weight = config["model"]["losses"]["mc_loss"]["weight"]
adj_loss_weight = config["model"]["losses"]["adj_loss"]["weight"]

use_ei_loss = config["model"]["losses"]["use_ei_loss"]
target_weight = config["model"]["losses"]["ei_loss"]["weight"]
warmup = config["model"]["losses"]["ei_loss"]["warmup"]
duration = config["model"]["losses"]["ei_loss"]["duration"]

save_interval = config["training"]["save_interval"]
plot_interval = config["training"]["plot_interval"]
# device = torch.device(config["training"]["device"])
start_epoch = 0

model_type = config["model"]["name"]

H, W = config["data"]["height"], config["data"]["width"]
Nt, Nsample, N_coils, N_time_eval = (
    config["data"]["timeframes"],
    config["data"]["spokes_per_frame"],
    config["data"]["coils"],
    config["data"]["eval_timeframes"]
)
Nspokes = int(config["data"]["total_spokes"] / Nt)
Ng = Nt  # frames per group
Mg = Nt // Ng  # total number of groups
indG = np.arange(0, Nt, 1, dtype=int)
indG = np.reshape(indG, [Mg, Ng])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mc_loss_fn = MCLoss(model_type="LSFPNet")

################# Load training data ##############


# load data
with open(split_file, "r") as fp:
    splits = json.load(fp)


# NOTE: need to look into why I am only loading 88 training samples and not 192
if max_subjects < 300:
    max_train = int(max_subjects * (1 - config["data"]["val_split_ratio"]))

    train_patient_ids = splits["train"][:max_train]
    

else:
    train_patient_ids = splits["train"]



train_dataset = SliceDataset(
    root_dir=config["data"]["root_dir"],
    patient_ids=train_patient_ids,
    dataset_key=config["data"]["dataset_key"],
    file_pattern="*.h5",
    slice_idx=config["dataloader"]["slice_idx"],
    N_coils=N_coils
)


train_loader = DataLoader(
    train_dataset,
    batch_size=config["dataloader"]["batch_size"],
    shuffle=config["dataloader"]["shuffle"],
    num_workers=config["dataloader"]["num_workers"],
)



# filename = "./Datasets/IMRI_Simu6_test.mat"
# iMRI_data = h5py.File(filename, 'r')
# brainI_ref = np.transpose(iMRI_data['brainI_ref_test'])  # reference
# c = np.transpose(iMRI_data['b1_test'])   # coil sensitivity maps

# brainI_ref = np.transpose(brainI_ref, [3, 0, 1, 2])
# c = np.transpose(c, [3, 2, 0, 1])
# c.dtype = 'complex128'

# overSample = 2
# Nitem = brainI_ref.shape[0]
# Nsample = brainI_ref.shape[1] * overSample
# Nt = brainI_ref.shape[3]
# Ncoil = c.shape[1]

# Nspokes = spf  # number of spokes per frame
# Ng = fpg  # frames per group
# Mg = Nt // Ng  # total number of groups
# indG = np.arange(0, Nt, 1, dtype=int)
# indG = np.reshape(indG, [Mg, Ng])

# # prepare training dataset
# brainI_ref = torch.from_numpy(brainI_ref)
# c = torch.tensor(c, dtype=dtype)

# train_dataset = TensorDataset(brainI_ref, c)
# train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

################### prepare NUFFT ################
def prep_nufft(Nsample, Nspokes, Ng):

    overSmaple = 2
    im_size = (int(Nsample/overSmaple), int(Nsample/overSmaple))
    grid_size = (Nsample, Nsample)

    ktraj = trajGR(Nsample, Nspokes * Ng)
    ktraj = torch.tensor(ktraj, dtype=torch.float)
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)
    dcomp = dcomp.squeeze()

    ktraju = np.zeros([2, Nspokes * Nsample, Ng], dtype=float)
    dcompu = np.zeros([Nspokes * Nsample, Ng], dtype=complex)

    for ii in range(0, Ng):
        ktraju[:, :, ii] = ktraj[:, (ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]
        dcompu[:, ii] = dcomp[(ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]

    ktraju = torch.tensor(ktraju, dtype=torch.float)
    dcompu = torch.tensor(dcompu, dtype=dtype)

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)  # forward nufft
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)  # backward nufft

    return ktraju, dcompu, nufft_ob, adjnufft_ob

def radial_down_sampling(images, param_E):

    k_und = param_E(inv=False, data=images)
    k_und = k_und.to(images.device)
    im_und = param_E(inv=True, data=k_und)

    return k_und, im_und


ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(Nsample, Nspokes, Ng)
ktraj = ktraj.to(device)
dcomp = dcomp.to(device)
nufft_ob = nufft_ob.to(device)
adjnufft_ob = adjnufft_ob.to(device)

########### Network and Data preparation #############

# new network
model = LSFPNet(layer_num)
model = nn.DataParallel(model, [0])
model = model.to(device)

# print parameter number
print_flag = 0
if print_flag:
    print(model)
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Mloss = nn.MSELoss()

# dir of model and log
# model_dir = "./%s/LSFP_Net_C%d_B%d_SPF%d_FPG%d_adj2" % (args.model_dir, conv_num, layer_num, spf, fpg)
# log_name = "./%s/Log_LSFP_Net_C%d_B%d_SPF%d_FPG%d_adj2.txt" % (args.log_dir, conv_num, layer_num, spf, fpg)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# if not os.path.exists(args.log_dir):
#     os.makedirs(args.log_dir)

################### training loop #####################
# if start_epoch > 0:
#     pre_model_dir = output_dir
#     model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

for epoch_i in range(start_epoch + 1, end_epoch + 1):


    for idx, (x_train, c_train, _) in enumerate(train_loader):

        # convert to complex
        x_train = (x_train[:, 0, ...] + 1j*x_train[:, 1, ...]).to(device)
        x_train = rearrange(x_train, 'b t c sp sam -> b c (sp sam) t').to(device)
        c_train = c_train.to(x_train.dtype)

        print("x: ", x_train.shape, x_train.dtype)
        print("csmaps: ", c_train.shape, c_train.dtype)

        random_group = torch.randperm(Mg)
        x_train_it = x_train.squeeze()
        smap1 = c_train.squeeze().to(device)
        param_E = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp, smap1)

        for p in random_group[0:1]:
            M0 = param_E(inv=True, data=x_train)
            param_d = x_train_it
            print("param_d input: ", param_d.shape)
            # ground_truth = x_train_it[:, :, indG[p, :]]
            # gt = ground_truth.type(dtype).to(device)
            # param_d, M0 = radial_down_sampling(gt, param_E)

            time1 = time()
            [L, S, loss_layers_adj_L, loss_layers_adj_S] = model(M0, param_E, param_d)
            M = torch.abs(L + S)
            M_recon = M.cpu().data.numpy()
            time2 = time()

            # temp_ground_truth = ground_truth.type(torch.float32).to(device)
            loss_constraint_L = torch.square(torch.mean(loss_layers_adj_L[0])) / layer_num
            loss_constraint_S = torch.square(torch.mean(loss_layers_adj_S[0])) / layer_num

            for k in range(layer_num - 1):
                loss_constraint_S += torch.square(torch.mean(loss_layers_adj_S[k + 1])) / layer_num
                loss_constraint_L += torch.square(torch.mean(loss_layers_adj_L[k + 1])) / layer_num

            gamma = torch.Tensor([0.01]).to(device)
            # loss_ref = Mloss(M, temp_ground_truth)
            print("param_d: ", param_d.shape, param_d.dtype)
            print("M: ", M.shape, M.dtype)
            print("c_train: ", c_train.shape, c_train.dtype)
            loss_ref = mc_loss_fn(param_d.to(device), M, param_E, c_train)
            loss = loss_ref + torch.mul(gamma, loss_constraint_L + loss_constraint_S)
            # loss = torch.mul(gamma, loss_constraint_L + loss_constraint_S)

            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(abs(ground_truth[:, :, Ng-1]), 'gray')
            # plt.subplot(1, 3, 2)
            # plt.imshow(abs(M0.numpy()[:, :, Ng-1]), 'gray')
            # plt.subplot(1, 3, 3)
            # plt.imshow(abs(M_recon[:, :, Ng-1]), 'gray')
            # plt.show()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_data = "[%02d/%02d] Loss_ref: %.5f, Loss_adj_L: %.5f, Loss_adj_S: %.5f, Loss: %.5f\n" % \
                          (epoch_i, end_epoch, loss_ref, loss_constraint_L, loss_constraint_S, loss)
            output_data2 = "[%02d/%02d] Loss_ref: %.5f, Loss_adj_L: %.5f, Loss_adj_S: %.5f,Loss: %.5f，time: %.4f\n" % \
                           (epoch_i, end_epoch, loss_ref, loss_constraint_L, loss_constraint_S, loss, (time2-time1))
            print(output_data2)

    # output_file = open(log_name, 'a')
    # output_file.write(output_data)
    # output_file.close()
    print(output_data)

    if epoch_i % 5 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (output_dir, epoch_i))  # save only the parameters