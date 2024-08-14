#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/install/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
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
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

# flash-attention install
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.8
pip install -e .

# huggingface install
pip install transformers accelerate zarr tensorstore
