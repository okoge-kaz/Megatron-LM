#!/bin/sh
#PBS -q rt_HF
#PBS -N tp-pp-change
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/tp-pp/
#PBS -P gag51395

cd $PBS_O_WORKDIR
mkdir -p outputs/tp-pp

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
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=4

TARGET_TENSOR_PARALLEL_SIZE=2
TARGET_PIPELINE_PARALLEL_SIZE=2

ITERATION=2500
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/groups/gag51395/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct1/LR2.5E-5-MINLR2.5E-6-WD0.1
TARGET_CHECKPOINT_DIR=/groups/gag51395/checkpoints/Llama-3.1-8b/tp${TARGET_TENSOR_PARALLEL_SIZE}-pp${TARGET_PIPELINE_PARALLEL_SIZE}/LR2.5E-5-MINLR2.5E-6-WD0.1

mkdir -p ${TARGET_CHECKPOINT_DIR}

CURRENT_ITERATION=$(cat "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B

export CUDA_DEVICE_MAX_CONNECTIONS=1
# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver mcore \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${TARGET_CHECKPOINT_DIR} \
  --target-tensor-parallel-size ${TARGET_TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${TARGET_PIPELINE_PARALLEL_SIZE} \
  --position-embedding-type rope \
  --loader-transformer-impl transformer_engine \
  --saver-transformer-impl transformer_engine \
  --megatron-path /groups/gag51395/src/fujii/Megatron-LM-v0.9

# change checkpoint iteration
echo $CURRENT_ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
