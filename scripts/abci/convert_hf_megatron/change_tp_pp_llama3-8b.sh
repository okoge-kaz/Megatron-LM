#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:00:30:00
#$ -j y
#$ -o outputs/megatron-to-hf/
#$ -cwd

# Load modules
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

# swich virtual env
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=4

TARGET_TENSOR_PARALLEL_SIZE=2
TARGET_PIPELINE_PARALLEL_SIZE=2

ITERATION=2500
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

EXPERIMENT=fineweb-edu

# model config
MEGATRON_CHECKPOINT_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b/${EXPERIMENT}/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct1/LR2.5E-5-MINLR2.5E-6-WD0.1
TARGET_CHECKPOINT_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b/${EXPERIMENT}/tp2-pp2-ct1/LR2.5E-5-MINLR2.5E-6-WD0.1/

mkdir -p ${TARGET_CHECKPOINT_DIR}

CURRENT_ITERATION=$(cat "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3-70B

export CUDA_DEVICE_MAX_CONNECTIONS=1
# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver mcore \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${TARGET_CHECKPOINT_DIR} \
  --target-tensor-parallel-size ${TARGET_TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${TARGET_PIPELINE_PARALLEL_SIZE} \
  --position-embedding-type rope \
  --loader-transformer-impl transformer_engine \
  --saver-transformer-impl transformer_engine \
  --megatron-path /bb/llm/gaf51275/2024/Megatron-LM-latest

# change checkpoint iteration
echo $CURRENT_ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
