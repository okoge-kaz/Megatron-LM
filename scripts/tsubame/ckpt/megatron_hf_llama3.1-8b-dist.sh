#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -o outputs/convert/megatron_hf/$JOB_ID.log
#$ -e outputs/convert/megatron_hf/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

START_ITERATION=100
END_ITERATION=100
INCREMENT=500

for ITERATION in $(seq $START_ITERATION $INCREMENT $END_ITERATION); do
  FORMATTED_ITERATION=$(printf "%07d" $ITERATION)
  echo -e "Converting iteration ${ITERATION}\n"

  # model config
  MEGATRON_DIST_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR2.5E-5-MINLR2.5E-6-WD0.1-v0.9.0
  MEGATRON_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR2.5E-5-MINLR2.5E-6-WD0.1-v0.9.0-no-dist
  HF_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/megatron-to-hf/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/iter_${FORMATTED_ITERATION}

  mkdir -p ${HF_CHECKPOINT_DIR}
  mkdir -p ${MEGATRON_CHECKPOINT_DIR}

  CURRENT_ITERATION=$(cat "${MEGATRON_DIST_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

  echo $ITERATION > "${MEGATRON_DIST_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

  # tokenizer config
  TOKENIZER_MODEL_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B
  TOKENIZER_MODEL=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json

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
    -x PATH \
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
    --megatron-path /gs/bs/tga-NII-LLM/src/Megatron-LM-v0.9

  # change checkpoint iteration
  echo $CURRENT_ITERATION > "${MEGATRON_DIST_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
  echo $CURRENT_ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
done
