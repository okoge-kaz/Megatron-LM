#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --time=1:00:00
#SBATCH --partition=p5.48xlarge
#SBATCH --nodes 1
#SBATCH --output=outputs/convert/hf-megatron/%x-%j.out
#SBATCH --error=outputs/convert/hf-megatron/%x-%j.out

# aws hyperpod's auto-resume functionality donesn't support Gres

# module load
module load gc1/cuda/12.1
module load gc1/cudnn/9.2.0
module load gc1/nccl/2.20.5
module load gc1/hpcx/2.18.1

# swich virtual env
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=2

# model config
HF_CHECKPOINT_DIR=/fsx/ubuntu/hf-checkpoints/Meta-Llama-3.1-8B
MEGATRON_CHECKPOINT_DIR=/fsx/ubuntu/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/fsx/ubuntu/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json

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
