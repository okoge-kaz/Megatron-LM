#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/install/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles
module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.17/2.17.1-1
module load hpcx/2.12
module load gcc/11.4.0

# python virtualenv
source .env/bin/activate

# pip install
pip install --upgrade pip
pip install --upgrade setuptools wheel cmake ninja packaging

pip install -r requirements.txt

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine
# A100 1枚だと CPU memory 不足でエラーになる
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git

cd TransformerEngine
git checkout v1.6

git submodule update --init --recursive
export NVTE_FRAMEWORK=pytorch
pip install .

# flash-attention install
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.4.2
python setup.py install

# huggingface install
pip install transformers accelerate zarr tensorstore
