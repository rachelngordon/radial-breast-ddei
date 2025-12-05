# breastMRI-recon
Breast MRI reconstruction with DCE-MRI dataset from NYU

## Environment Set Up
```bash
micromamba create -n recon_mri python=3.11 pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
micromamba activate recon_mri

python -m pip install tqdm
python -m pip install pydicom
python -m pip install numba
python -m pip install scipy
python -m pip install pywavelets
python -m pip install h5py
python -m pip install matplotlib

micromamba install -c conda-forge cupy cudnn cutensor nccl  # if you have GPU
micromamba install -c conda-forge numpy=1.24

git clone https://github.com/ZhengguoTan/sigpy.git
cd sigpy
python -m pip install -e . 

micromamba install -c conda-forge nibabel

python -m pip install pandas
python -m pip install torchio
python -m pip install tensorboard

git clone https://github.com/soumickmj/Tricorder.git
cd Tricorder
python -m pip install -e . 

python -m pip install black
python -m pip install pytorch_msssim

git clone https://github.com/soumickmj/pytorch-complex.git
cd pytorch-complex
python -m pip install -e . 

micromamba install -c conda-forge ipykernel

python -m pip install scikit-image 
python -m pip install torchkbnufft

python -m pip install git-filter-repo
python -m pip install wandb

python -m pip install mirtorch
python -m pip install wget

pip install openpyxl
python -m pip install torchmetrics

python -m pip install opencv-python

```

## References
Preprocessing code is adapted from code provided with the fastMRI breast dataset: https://github.com/eddysolo/demo_dce_recon
ReconResNet code is adapted from: https://github.com/soumickmj/NCC1701/tree/main
Data Consistency code is adapted from: https://github.com/koflera/DynamicRadCineMRI/tree/main
