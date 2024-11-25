#!/bin/sh
#PBS -q rt_HF
#PBS -N llama-3.1-8b
#PBS -l select=2:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/Llama-3.1-8B/
#PBS -P gcg51558

cd $PBS_O_WORKDIR
mkdir -p outputs/Llama-3.1-8B

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
JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="h200"

NODEFILE=$PBS_NODEFILE
NODE_COUNT=$(sort -u $NODEFILE | wc -l)
NUM_NODES=$NODE_COUNT
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

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
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

PIPELINE_MODEL_CHUNKS=1
LAYERS_PER_VIRTUAL_PIPELINE_STAGE=$((${NUM_LAYERS} / ${PIPELINE_PARALLEL_SIZE} / ${PIPELINE_MODEL_CHUNKS}))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=27500
LR_DECAY_ITERS=27500

LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B/tokenizer.json
CHECKPOINT_DIR=/groups/gag51395/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/groups/gag51395/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}

echo ${CHECKPOINT_SAVE_DIR}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3382423156 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"
# ja lm top10
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19423361558 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-llm-top10/dump_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14535812000 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-llm-top10/dump_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 27040695169 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-llm-top10/dump_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 23985972142 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-llm-top10/dump_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8109645389 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-llm-top10/dump_4_text_document"
# ja wiki top10
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8729168518 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-wiki-top10/dump_0_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5143962780 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-wiki-top10/dump_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12335523317 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-wiki-top10/dump_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11297417571 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-wiki-top10/dump_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3133344302 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/quality_filtering_ablation/filter-v2-wiki-top10/dump_4_text_document"

# ja en parallel corpus
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 882674099 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/default_plain_text_format_text_document"
# en wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4426952329 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/en_wiki_merged_train_text_document"
# en cosmopedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2527156128 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_automathtext_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 39884494 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_khanacademy_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 189399130 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_openstax_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1824516742 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stanford_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5235630656 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stories_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2645925170 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v1_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2154074830 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v2_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 312753702 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_wikihow_train_text_document"
# en dcml
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1067033136 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_01_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1067925601 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_02_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1031898250 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_03_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1068753339 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_04_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1069339810 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_05_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1067792181 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_06_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1069102872 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_07_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1068622240 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_08_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1066863628 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_09_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1066375762 /groups/gag51395/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_10_of_10_text_document"

# code stack v2 filter
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 32000000000 /groups/gag51395/datasets/Meta-Llama-3.1_original_transformers-4.45.2/stack_v2_python_compile_pylint_ja_or_en_text_document"

# job name
JOB_NAME="Llama-3.1-8b-ABCI-3.0-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

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
TENSORBOARD_DIR="${CHECKPOINT_SAVE_DIR}/tensorboard/${NUM_NODES}-nodes"
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
  -x NCCL_DEBUG=INFO \
  -x LD_LIBRARY_PATH \
  -x LIBRARY_PATH \
  -x PATH \
  -x INCLUDE \
  -x CUDA_HOME \
  -x CUDA_PATH \
  -x CUDA_NVCC_EXECUTABLE \
  -x CPATH \
  -x CUDNN_PATH \
  -x CUDNN_INCLUDE_DIR \
  -x CUDNN_LIBRARY_DIR \
  -x CUDNN_ROOT_DIR \
  -x NCCL_HOME \
  -x NCCL_INCLUDE_DIR \
  -x NCCL_LIBRARY_DIR \
  -x OMPI_HOME \
  -x MPI_HOME \
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
  --wandb-project "ABCI3.0" \
  --wandb-entity "okoge"
