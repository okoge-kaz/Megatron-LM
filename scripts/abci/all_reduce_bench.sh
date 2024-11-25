#!/bin/sh
#PBS -q rt_HF
#PBS -N all_reduce_bench
#PBS -l select=16:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/all_reduce_bench
#PBS -P gcg51558

cd $PBS_O_WORKDIR
mkdir -p outputs/all_reduce_bench

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles

module load cuda/12.4
module load cudnn/9.1.1
module load nccl/2.21.5
module load hpcx/2.18.1

source .env/bin/activate

# distributed settings
JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="h200"

NODEFILE=$PBS_NODEFILE
NODE_COUNT=$(sort -u $NODEFILE | wc -l)
NUM_NODES=$NODE_COUNT
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x LIBRARY_PATH \
  -x PATH \
  -x INCLUDE \
  -x CUDA_HOME \
  -x CUDA_PATH \
  -x CUDA_NVCC_EXECUTABLE \
  -x CPATH \
  -x CUDNN_PATH \
  -x CUDNN_INCLUDE_DIR \
  -x CUDNN_LIBRARY_DIR \
  -x CUDNN_ROOT_DIR \
  -x NCCL_HOME \
  -x NCCL_INCLUDE_DIR \
  -x NCCL_LIBRARY_DIR \
  -x OMPI_HOME \
  -x MPI_HOME \
  -bind-to none \
  python scripts/gcp/all_reduce_bench.py
