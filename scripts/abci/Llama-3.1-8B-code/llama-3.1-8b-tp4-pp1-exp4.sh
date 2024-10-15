#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=5:00:00:00
#$ -j y
#$ -o outputs/Llama-3.1-8b-code/
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
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=2
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

PIPLINE_MODEL_CHUNKS=1
LAYERS_PER_VIRTUAL_PIPELINE_STAGE=$((${NUM_LAYERS} / ${PIPELINE_PARALLEL_SIZE} / ${PIPLINE_MODEL_CHUNKS}))

# training config
MICRO_BATCH_SIZE=2
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
CHECKPOINT_SAVE_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3.1-8b-code/exp4/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}

echo ${CHECKPOINT_SAVE_DIR}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# japanese wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 845605789 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# japanese swallow v2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7511751.598 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2651338.278 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3714122.608 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2036580.211 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2871990.994 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2553386.583 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 878224.4745 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1655009.802 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 700903.1105 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 339761.3537 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2205054.528 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 402267.2395 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 605794.6275 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1567788.014 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 303092.0128 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 492916.2168 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 527497.474 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1932224.614 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1497931.786 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1141391.613 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1562568.814 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1185522.906 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15295070.77 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3816763.685 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6268900.726 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16308301.12 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1567811.081 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14853587.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7279005.48 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14558442.89 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17282883.75 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 69292447.78 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 71124180.46 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 557142312.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 475875517.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 189899995.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 240087011 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 322723532.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 186276874.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 143895209 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 145150232.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 218846630.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 192471018.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 147917868.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 138344887.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 154151155 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 137822502.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 142551296.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 120608433.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 109904127 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 139634595.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 189031719.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 208775497.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 166488254.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 142752606.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100010358.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 118205037.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 132616134.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 123326949.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 121792495.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 170148572.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 129563408 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 150781030.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 134979809.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 111718166.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 121270268.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 137703868.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 147055187.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 146268825.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 156895619 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 127015326.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 198772107.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 163470408.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 130416710.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 153932293.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 143469365.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 146747627.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 110279818.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 148301778.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 133836630.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 146393040.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 143719623.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 125949593.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 130182280.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 211846180.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 167793519.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 147924738.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 155157649.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 146697872.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 141045301.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 139799955.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 140472662 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 176575348.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 95478419.71 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-50_text_document"

# japanese swallow v2 top10
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 25451500.25 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10298588 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10798255.25 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4486775.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5193260.25 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3924638.75 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1212369.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1844834.25 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 696631.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 369149.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2368247.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 459213.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 587340.25 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1344081 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 269109.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 442796 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 696817.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2582970.75 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1588523.75 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1122417 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1369614.75 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1267719 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 47642708.75 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11212129 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17103449 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 33808898.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3562201.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 28303750.25 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13904167 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 30819062 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 31536092 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 118097132.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 109831703.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 810276888.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 576433072.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 275403835.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 312378151 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 271605326.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 274143751.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 204883591.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 148896806.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 258450228.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 215488837.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 172484205.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 142155762 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 149968235.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 156952301.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 171240242.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 124467503.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 138534952.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 144858685.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 156490130.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 171425497.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 211707462.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 197818924.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 156787796.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 181681662.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 214692288.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 219860488.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 218788934.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 251664025 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 230929765.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 238763612.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 204877807.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 183788086.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 212157966.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 242357172.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 237773089.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 257365385 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 264249758.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 238609309.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 294205793.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 287264864.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 256807581.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 295746302.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 297970536.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 291354235.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 241783090.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 265771427.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 232940477 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 270551061.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 289756402.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 286773112.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 302592993 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 396587301 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 344006348.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 308731852.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 297028767 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 276428976.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 268924868.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 253513960 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 254379473.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 283377979.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 159597521.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-50_text_document"

# japanese english parallel corpus
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 220668524.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/default_plain_text_format_text_document"

# english wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1106738082 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/en_wiki_merged_train_text_document"

# english cosmopedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 631789032 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_automathtext_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9971123.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_khanacademy_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 47349782.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_openstax_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 456129185.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stanford_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1308907664 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stories_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 661481292.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v1_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 538518707.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v2_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 78188425.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_wikihow_train_text_document"

# english datacomplm baseline
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

# code stack v2 python
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8000000000 /bb/llm/gaf51275/datasets/Meta-Llama-3.1_original_transformers-4.45.2/stack_v2_python_compile_pylint_ja_or_en_text_document"

# job name
JOB_NAME="Llama-3.1-8b-exp4-ABCI-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

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
  --wandb-project "Llama-3.1-8B-code" \
  --wandb-entity "prj-jalm"
