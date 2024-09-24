#!/bin/bash
#SBATCH --job-name=install
#SBATCH --partition=p5.48xlarge
#SBATCH --time=0-01:00:00
#SBATCH --nodes 1
#SBATCH --output=outputs/install/%x-%j.out
#SBATCH --error=outputs/install/%x-%j.out

set -e

module load cuda/12.1
module load cudnn/8.9.3

source .env/bin/activate

# pip install
pip install --upgrade pip
pip install --upgrade wheel cmake ninja

pip install -r requirements.txt

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.10

# flash-attention install
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.8
pip install -e .
