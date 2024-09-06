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

source .env_pytorch/bin/activate

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.3.1

git submodule sync
git submodule update --init --recursive

export USE_CUDA=1
export USE_NCCL=1
export USE_MPI=1
export USE_DISTRIBUTED=1
export CMAKE_PREFIX_PATH=$(dirname $(which mpicc))/../

pip install -r requirements.txt
pip install numpy

CC=mpicc CXX=mpicxx python setup.py install
CC=mpicc CXX=mpicxx python setup.py develop

# check
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.distributed.is_mpi_available())"
python -c "import torch; print(torch.distributed.is_nccl_available())"

# after install pytorch, please reinstall apex, flash-attention, and transformer-engine
# if not, you will get an error related to ImportError: /bb/1/llm/gaf51275/2024/Megatron-LM-latest/.env/lib/python3.10/site-packages/amp_C.cpython-310-x86_64-linux-gnu.so: undefined symbol

pip install --upgrade pip
pip install --upgrade wheel cmake ninja

pip install -r requirements_pytorch.txt

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.9

# flash-attention install
pip install git+https://github.com/Dao-AILab/flash-attention@v2.5.8 --no-build-isolation
