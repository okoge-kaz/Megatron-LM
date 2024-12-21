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

source .env_pytorch/bin/activate

pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

# install pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.5.1
git submodule sync
git submodule update --init --recursive

# pytorch requirements
pip install -r requirements.txt

# nccl, mpi, backend pytorch
# build options: https://github.com/pytorch/pytorch/blob/main/setup.py
# if you reinstall pytorch from source with different settings, you need to remove the previous installation: python setup.py clean
export USE_DISTRIBUTED=1
export TORCH_USE_CUDA_DSA=1

export CUDA_VERSION=12.4
export CUDNN_VERSION=9.1.1
export CXX_COMPILER="/usr/bin/c++"
export CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow"
export LAPACK_INFO=mkl
export PERF_WITH_AVX=1
export PERF_WITH_AVX2=1
export USE_CUDA=ON
export USE_CUDNN=ON
export USE_CUSPARSELT=1
export USE_EXCEPTION_PTR=1
export USE_GFLAGS=OFF
export USE_GLOG=OFF
export USE_GLOO=ON
export USE_MKL=ON
export USE_MKLDNN=ON
export USE_MPI=ON
export USE_NCCL=1
export USE_NNPACK=ON
export USE_OPENMP=ON
export USE_ROCM=OFF
export USE_ROCM_KERNEL_ASSERT=OFF
export TORCH_VERSION=2.5.1

export MAX_JOBS=16

# set tmp dir
export TMPDIR="/groups/gag51395/tmp"
export TMP="/groups/gag51395/tmp"

python setup.py develop

pip install triton

# Install packages
pip install -r requirements.txt

# nvidia apex
cd ..
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine (v1.11 support flash-atten-v3)
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.12
