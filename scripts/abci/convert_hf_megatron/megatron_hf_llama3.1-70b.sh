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
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=8
ITERATION=1
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/bb/llm/gaf51275/checkpoints/hf-to-megatron/Llama-3.1-70b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
HF_CHECKPOINT_DIR=/bb/llm/gaf51275/2024/checkpoints/megatron-to-hf/Llama-3.1-70b/checkpoint-convert-test/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

CURRENT_ITERATION=$(cat "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3.1-70B

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --llama-3-1 \
  --megatron-path /bb/llm/gaf51275/2024/Megatron-LM-latest

# change checkpoint iteration
echo $CURRENT_ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
