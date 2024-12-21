#!/bin/sh
#PBS -q rt_HF
#PBS -N megatron-to-hf
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/convert/megatron-to-hf/
#PBS -P gcg51558

set -e

cd $PBS_O_WORKDIR
mkdir -p outputs/convert/megatron-to-hf

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

export TMPDIR="/groups/gag51395/tmp"
export TMP="/groups/gag51395/tmp"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="h200"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# distributed settings
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=4
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

ITERATION=200
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_DIST_CHECKPOINT_DIR=/groups/gag51395/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/dist-ckpt/LR2.5E-5-MINLR2.5E-6-WD0.1
MEGATRON_CHECKPOINT_DIR=/groups/gag51395/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/torch-ckpt/LR2.5E-5-MINLR2.5E-6-WD0.1
HF_CHECKPOINT_DIR=/groups/gag51395/checkpoints/megatron-to-hf/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}
mkdir -p ${MEGATRON_CHECKPOINT_DIR}

CURRENT_ITERATION=$(cat "${MEGATRON_DIST_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

echo $ITERATION > "${MEGATRON_DIST_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B
TOKENIZER_MODEL=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B/tokenizer.json

# dist checkpoint -> megatron checkpoint
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -x LD_LIBRARY_PATH \
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
  python pretrain_gpt.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --use-checkpoint-args \
  --ckpt-convert-format torch \
  --load ${MEGATRON_DIST_CHECKPOINT_DIR} \
  --ckpt-convert-save ${MEGATRON_CHECKPOINT_DIR} \
  --use-mpi \
  --micro-batch-size 1 \
  --global-batch-size 1024 \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --train-iters 25000 \
  --mock-data \
  --distributed-backend nccl \
  --lr 2.5E-5 \
  --min-lr 2.5E-6 \
  --lr-decay-style cosine \
  --lr-decay-iters 25000 \
  --lr-warmup-iters 1000 \
  --bf16 \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --weight-decay 0.1 \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rotary-base 500000.0 \
  --rope-factor 8.0 \
  --rope-low-freq-factor 1.0 \
  --rope-high-freq-factor 4.0 \
  --rope-original-max-positional-embeddings 8192 \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --attention-softmax-in-fp32 \
  --accumulate-allreduce-grads-in-fp32 \
  --transformer-impl "transformer_engine"

echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR}/torch \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --true-vocab-size 128256 \
  --llama-3-1 \
  --megatron-path /groups/gag51395/src/fujii/Megatron-LM-v0.9

# change checkpoint iteration
echo $CURRENT_ITERATION > "${MEGATRON_DIST_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
echo $CURRENT_ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
