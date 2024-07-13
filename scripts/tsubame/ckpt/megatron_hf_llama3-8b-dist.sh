#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -o outputs/convert/megatron_hf/$JOB_ID.log
#$ -e outputs/convert/megatron_hf/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

set -e

# swich virtual env
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=2

ITERATION=10
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/Llama-3-8b/dist-ckpt/tp2-pp2-ct1-LR2.5E-5-MINLR2.5E-6-WD0.1-WARMUP1000
HF_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/megatron-to-hf/Llama-3-8b-dist-ckpt/LR2.5E-5-MINLR2.5E-6-WD0.1-WARMUP1000/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/gs/bs/tga-bayes-crest/fujii/hf-checkpoints/Meta-Llama-3-8B

# dist-ckpt
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=1

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
  --megatron-path /gs/bs/tga-bayes-crest/fujii/Megatron-LM \
  --use-torch-dist-ckpt \
  --original-tp-size ${TENSOR_PARALLEL_SIZE} \
  --original-pp-size ${PIPELINE_PARALLEL_SIZE} \
