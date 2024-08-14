#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -p -5

# priotiry: -5: normal, -4: high, -3: highest

# Load modules
module use ~/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.1/2.18.3
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# Set environment variables
source .env/bin/activate

pip install --upgrade pip

# Install packages
pip install -r requirements.txt

# nvidia apex
cd ..

git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git

cd TransformerEngine
export NVTE_FRAMEWORK=pytorch
pip install .

# flash-atten
cd ..
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention

git checkout v2.4.2

pip install -e .
