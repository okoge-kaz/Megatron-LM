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
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=4
ITERATION=1
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/groups/gag51395/checkpoints/hf-to-megatron/Llama-3.1-70b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
HF_CHECKPOINT_DIR=/groups/gag51395/checkpoints/megatron-to-hf/Llama-3.1-70b/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

CURRENT_ITERATION=$(cat "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-70B

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
  --megatron-path /groups/gag51395/src/fujii/Megatron-LM-v0.9

# change checkpoint iteration
echo $CURRENT_ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
