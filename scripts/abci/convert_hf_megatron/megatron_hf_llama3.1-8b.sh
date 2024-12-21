#!/bin/sh
#PBS -q rt_HF
#PBS -N megatron-to-hf
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/convert/megatron-to-hf/
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
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1

# iteration settings
START_ITERATION=15000
END_ITERATION=30000
STEP=2500

# model config
MEGATRON_CHECKPOINT_DIR=/groups/gag51395/checkpoints/Llama-3.1-swallow-8b-v0.2/tp2-pp1-ct1/LR2.5E-5-MINLR2.5E-6-WD0.1

# tokenizer config
TOKENIZER_MODEL_DIR=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B

# hf checkpoint dir base
HF_CHECKPOINT_DIR_BASE=/groups/gag51395/checkpoints/megatron-to-hf/Llama-3.1-swallow-8b-v0.2/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${HF_CHECKPOINT_DIR_BASE}

export CUDA_DEVICE_MAX_CONNECTIONS=1

# iterate through specified iterations
for ITERATION in $(seq $START_ITERATION $STEP $END_ITERATION); do
  FORMATTED_ITERATION=$(printf "%07d" $ITERATION)
  HF_CHECKPOINT_DIR=${HF_CHECKPOINT_DIR_BASE}/iter_${FORMATTED_ITERATION}

  mkdir -p ${HF_CHECKPOINT_DIR}

  CURRENT_ITERATION=$(cat "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

  echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

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
    --true-vocab-size 128256 \
    --megatron-path /groups/gag51395/src/fujii/Megatron-LM-v0.9 \
    --llama-3-1

  # reset checkpoint iteration
  echo $CURRENT_ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
done

echo "checkpoint conversion completed"
