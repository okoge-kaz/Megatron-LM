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

# switch virtual env
source .env_pytorch_mpi/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1

# model config
HF_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B
MEGATRON_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json

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
