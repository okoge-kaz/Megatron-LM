#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=4:10:00:00
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
CHECKPOINT_SAVE_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b-meta-tag/exp3/tp2-pp4-ct1-LR2.5E-5-MINLR2.5E-6-WD0.1-WARMUP1000

echo ${CHECKPOINT_SAVE_DIR}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wiki base
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1691211578 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/wiki-base_text_document"

# swallow:annotate quality
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 58366175.14 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 23822673.79 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 28851556.96 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16460159.6 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 21305393.18 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20685435.7 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7433447.401 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13639059.1 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5785238.474 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2654549.755 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 18363698.79 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3663749.584 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5054198.96 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11738730.14 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2527491.387 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3964771.873 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4279110.643 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14339973.63 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11001900.08 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8047075.55 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10228288.84 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9693115.396 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 191207452.7 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 48352602.93 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 72068993.19 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 148109921.2 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16717683.41 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 169283662 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 84884111.12 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 159396826 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 175544220 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 579731699.9 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 593314131.4 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4983980356 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4054262895 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1840643489 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2447501152 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1871088635 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1705224384 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1628705804 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1316306259 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1915459258 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1800253778 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1533363649 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1373434009 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1432516615 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1404785601 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1376566129 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1186125456 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1175465269 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1346616172 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1275685088 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1475712133 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1465332642 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1162223833 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 936231974.5 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1038982011 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1199095029 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1243092691 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1245996683 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1439889139 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1285943589 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1473758090 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1269048924 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1163343400 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1315099290 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1339015416 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1377835191 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1425579541 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1529067780 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1322738409 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1804486141 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1631802803 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1419364531 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1577807794 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1399361999 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1455553932 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1167437332 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1191281099 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1218990003 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1356677361 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1493366560 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1611921763 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1873245499 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2105300274 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1751174746 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1517628007 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1512986078 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1463484438 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1407961401 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1409091599 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1526685798 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1670691475 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1826000928 /bb/llm/gaf51275/datasets/Meta-tag-Meta-Llama-3_original_transformers-4.40.1/annotate-quality/CC-MAIN-2023-50_text_document"

# job name
JOB_NAME="Llama-3-8b-meta-tag-exp3-ABCI-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

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
  --split 990,5,5 \
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
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-3-8B-meta-tag" \
  --wandb-entity "prj-jalm"
