#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:01:00:00
#$ -j y
#$ -o outputs/install/
#$ -cwd

# Load modules
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

set -e

# swich virtual env
source .env/bin/activate

cd ..

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
