#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=00:02:00:00
#$ -o outputs/Llama-3.1-8b/$JOB_ID.log
#$ -e outputs/Llama-3.1-8b/$JOB_ID.log
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

export TMPDIR="/gs/bs/tge-gc24sp03/cache"
export TMP="/gs/bs/tge-gc24sp03/cache"

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# model config
# llama-3.1-8b: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # intermediate size (HuggingFace)
NUM_LAYERS=32
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=8
SEQ_LENGTH=8192

# distributed settings
TENSOR_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

PIPELINE_MODEL_CHUNKS=1
LAYERS_PER_VIRTUAL_PIPELINE_STAGE=$((${NUM_LAYERS} / ${PIPELINE_PARALLEL_SIZE} / ${PIPELINE_MODEL_CHUNKS}))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000
LR_DECAY_ITERS=25000

LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json
CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/gs/bs/tga-NII-LLM/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# Japanese Wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3382423156 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# Japanese LLM filter top10
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 24667669179 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-llm-top10/dump_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18460481240 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-llm-top10/dump_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 34341682865 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-llm-top10/dump_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 30462184620 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-llm-top10/dump_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10299249644 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-llm-top10/dump_4_text_document"

# Japanese Wikipedia filter top10
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11467145217 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-wiki-top10/dump_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6757409720 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-wiki-top10/dump_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16204663356 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-wiki-top10/dump_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14840947061 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-wiki-top10/dump_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4116143943 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/filter-v2-wiki-top10/dump_4_text_document"

# English - Japanese Parallel Corpus
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 882674099 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/default_plain_text_format_text_document"

# English Wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4426952329 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/en_wiki_merged_train_text_document"

# English Cosmopedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2527156128 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_automathtext_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 39884494 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_khanacademy_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 189399130 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_openstax_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1824516742 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stanford_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5235630656 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stories_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2645925170 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v1_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2154074830 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v2_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 312753702 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_wikihow_train_text_document"

# English DCML
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1730421007 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_01_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1731868329 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_02_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1673442322 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_03_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1733210681 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_04_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1734161769 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_05_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1731651960 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_06_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1733777525 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_07_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1732998076 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_08_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1730146115 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_09_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1729354936 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_10_of_10_text_document"

# Code Stack v2 filtered
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 37500000000 /gs/fs/tgh-24IDU/datasets/Llama-3.1-transformers/stack-v2-filtered/stack_v2_python_compile_pylint_ja_or_en_text_document"
# job name
JOB_NAME="Llama-3.1-8b-TSUBAME-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

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

if [[ ${PIPELINE_MODEL_CHUNKS} -gt 1 ]]; then
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

# pytorch profiler
TENSORBOARD_DIR="/gs/fs/tgh-24IDU/tensorboard/bf16-te-v1.9/${NUM_NODES}-nodes"
mkdir -p ${TENSORBOARD_DIR}

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -x UB_SKIPMC=1 \
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
  --tp-comm-overlap \
  --distributed-timeout-minutes 30 \
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
  --ckpt-format "torch_dist" \
  --async-save \
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
  --rotary-base 500000.0 \
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
  --wandb-project "Megatron-LM-TransformerEngine-v1.10" \
  --wandb-entity "okoge"
