#!/bin/bash
#SBATCH --job-name=ckpt-convert
#SBATCH --time=5:00:00
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/convert/megatron-to-hf/%x-%j.out
#SBATCH --error=outputs/convert/megatron-to-hf/%x-%j.out

set -e

# module load
module load turing/cuda/12.1
module load turing/cudnn/8.9.7
module load turing/nccl/2.20.5
module load turing/hpcx/2.17.1

# open file limit
ulimit -n 65536 1048576

# python virtualenv
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=8

ITERATION=1
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/home/ext_kazuki_fujii_turing_motors_c/checkpoints/hf-to-megatron/Llama-3-70b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
HF_CHECKPOINT_DIR=/home/ext_kazuki_fujii_turing_motors_c/checkpoints/megatron-to-hf/Llama-3-70b

mkdir -p ${HF_CHECKPOINT_DIR}

echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/home/ext_kazuki_fujii_turing_motors_c/hf-checkpoints/Meta-Llama-3-70B

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl "transformer_engine" \
  --megatron-path /home/ext_kazuki_fujii_turing_motors_c/src/Megatron-LM
