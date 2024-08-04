#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=5:00:00:00
#$ -j y
#$ -o outputs/Llama-3-8b-wiki-like/
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
GLOBAL_BATCH_SIZE=512
TRAIN_STEPS=12500  # 50B Tokens
LR_DECAY_ITERS=12500

LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/groups/gag51395/hf-checkpoints/Meta-Llama-3-8B/tokenizer.json
CHECKPOINT_DIR=/groups/gag51395/checkpoints/hf-to-megatron/Llama-3-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b/wiki-like/exp11/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}

echo ${CHECKPOINT_SAVE_DIR}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1691211578 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# swallow ml heuristics + swllow v2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 40722400.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16477740.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17277208.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7178840.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8309216.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6279422 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1939791.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2951734.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1114610.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 590639.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3789196 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 734741.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 939744.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2150529.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 430575.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 708473.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1114908 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4132753.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2541638 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1795867.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2191383.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2028350.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 76228334 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17939406.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 27365518.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 54094237.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5699522.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 45286000.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22246667.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 49310499.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 50457747.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 188955412.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 175730726 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1296443022 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 922292916.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 440646136.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 499805041.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 434568522.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 438630002 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 327813746.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 238234890.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 413520365.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 344782139.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 275974728.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 227449219.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 239949176.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 251123682 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 273984388.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 199148006 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 221655924 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 231773897.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 250384208.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 274280795.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 338731940 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 316510279.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 250860474.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 290690660 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 343507661.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 351776782 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 350062295.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 402662440 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 369487624.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 382021779.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 327804492 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 294060938.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 339452746.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 387771476.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 380436943.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 411784616 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 422799614 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 381774894.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 470729269.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 459623782.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 410892130 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 473194084.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 476752858.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 466166776.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 386852944.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 425234284.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 372704763.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 432881698 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 463610244 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 458836979.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 484148788.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 634539681.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 550410157.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 493970964.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 475246027.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 442286362 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 430279790 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 405622336 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 407007157.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 453404767.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 255356034.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18142734.67 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6331165.462 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8735773.487 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4628954.897 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6406892.669 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5612544.153 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1931240.426 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3587397.618 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1511248.971 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 734833.0486 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4744659.305 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 860242.6748 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1300770.917 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3329086.468 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 633379.3725 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1032039.99 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1104898.14 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4143787.04 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3162615.672 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2423453.245 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3318007.756 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2554770.655 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 35928276.72 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8890610.394 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14609787.81 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 36882209.12 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3659498.339 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 33875378.47 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16569367.42 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 33107981.22 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 38708080.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 155707389.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 158359265.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1235432588 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1040831582 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 421141500.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 530396868.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 693025218.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 413753499 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 319149994.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 315523460.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 477798107.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 419006663.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 323511446.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 299557576.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 332725679.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 300997242.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 312360358.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 261692208 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 242367913.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 303171380.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 400251060.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 446910787.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 365333715.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 314737532.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 223667719.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 263492657.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 296781839.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 277861297.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 274722755.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 376924745.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 292104008.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 336439075.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 300517988.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 249936745.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 272128739.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 308790917.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 327523194.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 327246417.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 350535447.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 286959894.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 440703987.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 365528899.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 293947562.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 345697381.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 323694925.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 330953936.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 250842607.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 331087685.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 298811070.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 329605129.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 324021636.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 286620914.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 295288670.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 474039901.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 379412721.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 334018365.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 346087040.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 329269932.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 316353432.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 314396822.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 315514891.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 392609128.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 419319424 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-base/CC-MAIN-2023-50_text_document"


# job name
JOB_NAME="Llama-3-8b-wiki-like-exp11-ABCI-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

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
  --wandb-project "Llama-3-8B-wiki-like-ablation" \
  --wandb-entity "prj-jalm"
