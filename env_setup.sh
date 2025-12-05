#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="recon_mri"

# 0) Create env from YAML
micromamba create -n "${ENV_NAME}" -f env_min.yaml -y

# 1) Add the extra conda-forge bits (if not already listed in env_min.yaml)
micromamba install -n "${ENV_NAME}" -c conda-forge \
  numba scipy pywavelets tqdm "numpy<2.0" -y

# 2) Install normal pip requirements into that env
micromamba run -n "${ENV_NAME}" python -m pip install --no-build-isolation -r requirements.txt

# 3) deepinv (fixed commit, non-editable)
micromamba run -n "${ENV_NAME}" python -m pip install --no-build-isolation \
  "git+https://github.com/deepinv/deepinv.git@6cc66b6b85ace803fae4cf3be1c25b54eb1ace91"

# 4) editable git deps
micromamba run -n "${ENV_NAME}" python -m pip install --no-build-isolation -e \
  "git+https://github.com/soumickmj/pytorch-complex.git@f5d0c3511a3f3c7cb7f138ee7f42a699d3f8209a#egg=pytorch_complex"

micromamba run -n "${ENV_NAME}" python -m pip install --no-build-isolation -e \
  "git+https://github.com/jinh0park/pytorch-ssim-3D.git@ada88564a754cd857730d649c511384dd41f9b4e#egg=pytorch_ssim"

# sigpy from your local src checkout
micromamba run -n "${ENV_NAME}" python -m pip install --no-build-isolation -e src/sigpy

# 5) (optional) final pass over requirements, if you really want
micromamba run -n "${ENV_NAME}" python -m pip install --no-build-isolation -r requirements.txt

echo
echo "Env ${ENV_NAME} setup complete."
echo "To use it interactively:"
echo "  micromamba activate ${ENV_NAME}"
