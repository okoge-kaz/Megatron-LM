#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/change_tp_pp/
#$ -cwd

set -e

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

# distributed settings
TARGET_TENSOR_PARALLEL_SIZE=2
TARGET_PIPELINE_PARALLEL_SIZE=4

BASE_TENSOR_PARALLEL_SIZE=2
BASE_PIPELINE_PARALLEL_SIZE=2

# model config
BASE_CHECKPOINT_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b/exp6/tp${BASE_TENSOR_PARALLEL_SIZE}-pp${BASE_PIPELINE_PARALLEL_SIZE}-ct1-LR2.5E-5-MINLR2.5E-6-WD0.1-WARMUP1000
TARGET_CHECKPOINT_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b/exp6/tp${TARGET_TENSOR_PARALLEL_SIZE}-pp${TARGET_PIPELINE_PARALLEL_SIZE}-ct1-LR2.5E-5-MINLR2.5E-6-WD0.1-WARMUP1000

mkdir -p ${TARGET_CHECKPOINT_DIR}

export CUDA_DEVICE_MAX_CONNECTIONS=1

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver mcore \
  --megatron-path /bb/llm/gaf51275/2024/Megatron-LM \
  --target-tensor-parallel-size ${TARGET_TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${TARGET_PIPELINE_PARALLEL_SIZE} \
  --load-dir ${BASE_CHECKPOINT_DIR} \
  --save-dir ${TARGET_CHECKPOINT_DIR}
