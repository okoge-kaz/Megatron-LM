#!/bin/sh
#PBS -q rt_HF
#PBS -N install
#PBS -l select=2:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -o outputs/install/
#PBS -P gag51395

cd $PBS_O_WORKDIR
mkdir -p outputs/install

source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles

module load cuda/12.4
module load cudnn/9.1.1
module load nccl/2.21.5
module load hpcx/2.18.1

source .env/bin/activate

pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

# install pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install packages
pip install -r requirements.txt

# nvidia apex
cd ..
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine (v1.11 support flash-atten-v3)
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.12
