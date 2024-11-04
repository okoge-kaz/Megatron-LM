#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -o outputs/convert/hf_megatron/$JOB_ID.log
#$ -e outputs/convert/hf_megatron/$JOB_ID.log
#$ -p -3

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=4

# model config
HF_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Mixtral-8x7B-v0.1
MEGATRON_CHECKPOINT_DIR=/gs/bs/tgh-NII-LLM/checkpoints/hf-to-megatron/mixtral-8x7b-v0.1/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/gs/bs/tga-NII-LLM/hf-checkpoints/Mixtral-8x7B-v0.1/tokenizer.json

export CUDA_DEVICE_MAX_CONNECTIONS=1

# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mixtral_hf \
  --saver mcore \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --saver-transformer-impl "transformer_engine"
