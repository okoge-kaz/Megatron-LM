#!/bin/bash
#SBATCH --job-name=install
#SBATCH --partition=h100
#SBATCH --time=0-01:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/install/%x-%j.out
#SBATCH --error=outputs/install/%x-%j.out

set -e

module load gc1/cuda/12.1
module load gc1/cudnn/9.2.0
module load gc1/nccl/2.20.5
module load gc1/hpcx/2.18.1

source .env_transformer_engine/bin/activate

# pip install
pip install --upgrade pip
pip install --upgrade wheel cmake ninja

pip install -r requirements.txt

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine
git submodule update --init --recursive
export NVTE_FRAMEWORK=pytorch
pip install -e .

# flash-attention install
pip install git+https://github.com/Dao-AILab/flash-attention@v2.5.6 --no-build-isolation