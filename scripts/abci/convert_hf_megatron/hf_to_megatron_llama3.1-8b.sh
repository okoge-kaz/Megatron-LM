#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:01:00:00
#$ -j y
#$ -o outputs/hf-to-megatron/
#$ -cwd

# Load modules
source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.17/2.17.1-1
module load hpcx/2.12
module load gcc/11.4.0


# swich virtual env
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=1

# model config
HF_CHECKPOINT_DIR=/bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3.1-8B
MEGATRON_CHECKPOINT_DIR=/bb/llm/gaf51275/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json

export CUDA_DEVICE_MAX_CONNECTIONS=1

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama3_hf \
  --saver mcore \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --bf16 \
  --saver-transformer-impl "transformer_engine"
