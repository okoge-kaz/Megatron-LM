#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -o outputs/convert/hf_megatron/$JOB_ID.log
#$ -e outputs/convert/hf_megatron/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=2

ITERATION=2500
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

EXPERIMENT=datacom-lm

# model config
MEGATRON_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/Llama-3.1-8b/${EXPERIMENT}/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct1/LR2.5E-5-MINLR2.5E-6-WD0.1
HF_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/megatron-to-hf/Llama-3.1-8b/${EXPERIMENT}/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct1-LR2.5E-5-MINLR2.5E-6-WD0.1/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

CURRENT_ITERATION=$(cat "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B

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
  --megatron-path /gs/bs/tga-bayes-crest/fujii/Megatron-LM

# change checkpoint iteration
echo $CURRENT_ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
