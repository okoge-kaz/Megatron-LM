#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:03:00:00
#$ -j y
#$ -o outputs/megatron-to-hf/
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
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=8

ITERATION=2500
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/groups/gag51395/checkpoints/Llama-3-70b/
HF_CHECKPOINT_DIR=/groups/gag51395/checkpoints/megatron-to-hf/Llama-3-70b-hf/LR1.0e-5-MINLR1.0E-6-WD0.1/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

echo $ITERATION >"${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/groups/gag51395/hf-checkpoints/Meta-Llama-3-8B

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
  --megatron-path /groups/gag51395/src/Megatron-LM