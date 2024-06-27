#!/bin/bash

# module load
module load turing/cuda/12.1
module load turing/cudnn/8.9.7
module load turing/nccl/2.20.5
module load turing/hpcx/2.17.1

# python virtualenv
source .env/bin/activate

# pip install
pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

# install pytorch
pip install -r requirements.txt

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine install
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.6

# flash-attention install
pip uninstall flash-attn

git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.4.2
pip install -e .

# huggingface install
pip install transformers accelerate

# storage install
pip install zarr tensorstore
