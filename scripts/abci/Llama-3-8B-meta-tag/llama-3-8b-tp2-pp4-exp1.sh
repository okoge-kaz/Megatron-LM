#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=5:00:00:00
#$ -j y
#$ -o outputs/Llama-3-8b-meta-tag/
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
# llama-3-8b: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # intermediate size (HuggingFace)
NUM_LAYERS=32
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=8
SEQ_LENGTH=8192

# distributed settings
TENSOR_PARALLEL_SIZE=2   # fixed
PIPELINE_PARALLEL_SIZE=4 # num layers 32: Llama-2 8B
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=12500
LR_DECAY_ITERS=12500

LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/bb/llm/gaf51275/hf-checkpoints/meta-tag-Llama-3-8B/tokenizer.json
CHECKPOINT_DIR=/groups/gag51395/checkpoints/hf-to-megatron/Llama-3-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b-meta-tag/exp1/tp2-pp4-ct1-LR2.5E-5-MINLR2.5E-6-WD0.1-WARMUP1000

echo ${CHECKPOINT_SAVE_DIR}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wiki/base
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1691211578 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/wiki-base_text_document"

# swallow:annotate-base
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 58346236.14 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 23827873.13 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 28854530.45 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16460654.8 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 21305921.64 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20691877.89 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7436330.28 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13642589.48 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5786884.247 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2654899.248 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18364706.49 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3665052.668 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5055174.058 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11740166.16 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2528239.491 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3965270.647 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4279495.975 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14338988.54 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11003467.79 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8050779.789 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10232642.88 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9695352.299 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 191498312.9 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 48424852.48 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 72166881.1 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 148290629.3 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16733206.92 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 169194658 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 84853011.4 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 159353591.5 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 175595192.8 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 579952653.6 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 593449909.6 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4986023032 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4055655887 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1841842964 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2448553024 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1872256023 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1705544484 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1629832243 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1317115245 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1916444886 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1801132404 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1534304344 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1374117985 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1433298205 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1405303929 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1376818827 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1186524984 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1175895394 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1347082342 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1275878623 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1476232984 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1465606179 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1162247949 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 936149004 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1038897331 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1198863159 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1242966222 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1245830480 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1439410870 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1285607098 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1473579077 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1268741984 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1163208960 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1315018658 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1338610882 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1377542344 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1425152899 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1528765670 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1322436426 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1803446900 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1631233430 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1418895541 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1577189415 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1398836449 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1454788187 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1166923524 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1190546877 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1218297043 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1355822886 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1492851211 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1612274645 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1874139233 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2104992045 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1750818442 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1517233818 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1512521994 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1462752238 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1407387790 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1408365922 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1525856751 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1669224498 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1824456545 /groups/gag51395/datasets/binarized/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-base/CC-MAIN-2023-50_text_document"


# job name
JOB_NAME="Llama-3-8b-meta-tag-exp1-ABCI-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -bind-to none \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
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
  --begin-of-special-token-id 128002 \
  --end-of-special-token-id 128003 \
  --reset-position-ids \
  --reset-attention-mask \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 995,5,0 \
  --distributed-backend nccl \
  --init-method-std 0.02 \
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
  --save-interval 100 \
  --no-initialization \
  --exit-on-missing-checkpoint \
  --eval-interval 500 \
  --eval-iters 10 \
  --bf16 \
  --use-checkpoint-args \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rope-theta 500000.0 \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --attention-softmax-in-fp32 \
  --transformer-impl "transformer_engine" \
  --use-mpi \
  --use-z-loss \
  --log-throughput \
  --log-straggler \
  --disable-straggler-on-startup \
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-3-8B-meta-tag" \
  --wandb-entity "prj-jalm"
