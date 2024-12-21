#!/bin/sh
#PBS -q rt_HF
#PBS -N hf-to-megatron
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/convert/hf-to-megatron/
#PBS -P gag51395

cd $PBS_O_WORKDIR
mkdir -p outputs/convert/hf-to-megatron

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles

module load cuda/12.4
module load cudnn/9.1.1
module load nccl/2.21.5
module load hpcx/2.18.1

source .env/bin/activate

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1

# model config
HF_CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B
MEGATRON_CHECKPOINT_DIR=/groups/gag51395/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B/tokenizer.json

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
