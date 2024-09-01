#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=0:1:00:00
#$ -j y
#$ -o outputs/Llama-3.1-8b-ablation/
#$ -cwd

# Load modules
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

set -e

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="a100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# model config
# llama-3.1-8b: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # intermediate size (HuggingFace)
NUM_LAYERS=32
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=8
SEQ_LENGTH=8192

# distributed settings
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

PIPLINE_MODEL_CHUNKS=1
LAYERS_PER_VIRTUAL_PIPELINE_STAGE=$((${NUM_LAYERS} / ${PIPELINE_PARALLEL_SIZE} / ${PIPLINE_MODEL_CHUNKS}))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512
TRAIN_STEPS=12500
LR_DECAY_ITERS=12500

LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json
CHECKPOINT_DIR=/bb/llm/gaf51275/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3.1-8b-ablation/exp4/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}

echo ${CHECKPOINT_SAVE_DIR}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 845605789 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# ja ml filtered
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2205983556 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2327676963 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3050722077 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2646246510 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2374899743 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2284874515 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2126411962 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2068687102 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1950139686 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1956797591 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2179866718 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1227693577 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-50_text_document"

# ja-en parallel corpus
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 220668524.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/default_plain_text_format_text_document"

# en wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1106738082 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/en_wiki_merged_train_text_document"

# en cosmopedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 631789032 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_automathtext_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9971123.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_khanacademy_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 47349782.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_openstax_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 456129185.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stanford_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1308907664 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stories_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 661481292.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v1_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 538518707.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v2_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 78188425.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_wikihow_train_text_document"

# en datacomp-lm baseline
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1002501435 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_01_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1003339926 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_02_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 969491425.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_03_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1004117604 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_04_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1004668606 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_05_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1003214575 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_06_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1004445999 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_07_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1003994433 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_08_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1002342179 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_09_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1001883818 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_10_of_10_text_document"

# code stack v2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3233055633 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/bigcode/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-00_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4766944367 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/bigcode/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-01_text_document"

# job name
JOB_NAME="Llama-3.1-8b-ABCI-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

# interleaved pipeline
PIPELINE_ARGS="--pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}"

if [[ ${PIPLINE_MODEL_CHUNKS} -gt 1 ]]; then
  echo "Interleaved pipeline is enabled: layers per virtual pipeline stage = ${LAYERS_PER_VIRTUAL_PIPELINE_STAGE}"

  PIPELINE_ARGS="${PIPELINE_ARGS} --num-layers-per-virtual-pipeline-stage ${LAYERS_PER_VIRTUAL_PIPELINE_STAGE}"
fi

# timer (profiling)
LOG_TIMER=False

TIMER_ARGS="--log-throughput"

if [[ ${LOG_TIMER} == "True" ]]; then
  TIMER_ARGS="${TIMER_ARGS} --log-timers-to-tensorboard"
  TIMER_ARGS="${TIMER_ARGS} --timing-log-level 2"
fi

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -bind-to none \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  ${PIPELINE_ARGS} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --group-query-attention \
  --num-query-groups ${NUM_KEY_VALUE_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --reset-position-ids \
  --reset-attention-mask \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 990,10,0 \
  --distributed-backend nccl \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-decay-iters ${LR_DECAY_ITERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --save-interval 500 \
  --no-initialization \
  --exit-on-missing-checkpoint \
  --eval-interval 500 \
  --eval-iters 10 \
  --bf16 \
  --no-initialization \
  --exit-on-missing-checkpoint \
  --use-checkpoint-args \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rope-theta 500000.0 \
  --rope-factor 8.0 \
  --rope-low-freq-factor 1.0 \
  --rope-high-freq-factor 4.0 \
  --rope-original-max-positional-embeddings 8192 \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --attention-softmax-in-fp32 \
  --accumulate-allreduce-grads-in-fp32 \
  --transformer-impl "transformer_engine" \
  --use-mpi \
  --use-z-loss \
  ${TIMER_ARGS} \
  --log-straggler \
  --disable-straggler-on-startup \
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-3.1-8B-ablation" \
  --wandb-entity "prj-jalm"
