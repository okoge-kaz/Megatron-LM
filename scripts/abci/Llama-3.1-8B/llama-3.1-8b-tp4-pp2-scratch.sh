#!/bin/bash
#$ -l rt_AF=8
#$ -l h_rt=0:0:30:00
#$ -j y
#$ -o outputs/Llama-3.1-8b-tflops/
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
CONTEXT_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

PIPLINE_MODEL_CHUNKS=1
LAYERS_PER_VIRTUAL_PIPELINE_STAGE=$((${NUM_LAYERS} / ${PIPELINE_PARALLEL_SIZE} / ${PIPLINE_MODEL_CHUNKS}))

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
TOKENIZER_MODEL=/bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json
CHECKPOINT_DIR=/bb/llm/gaf51275/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/bb/llm/gaf51275/2024/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}

echo ${CHECKPOINT_SAVE_DIR}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3382423156 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# ja swallow v2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 30221726.23 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10667022 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14942879.19 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8193690.751 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11554765.15 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10272936.91 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3533324.99 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6658533.963 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2819915.121 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1366948.105 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8871506.651 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1618425.506 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2437269.01 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6307618.074 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1219417.83 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1983129.868 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2122259.241 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7773841.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6026568.34 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4592114.689 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6286619.878 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4769666.334 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 61536039.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15355830.88 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 25221414.61 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 65612527.94 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6307710.878 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 59759838.72 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 29285328.19 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 58572394.14 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 69533526.81 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 278781501.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 286151037.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2241528120 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1914570713 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 764016966.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 965932356.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1298400530 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 749440210.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 578927771.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 583977055.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 880476793.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 774360859.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 595111975.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 556597386.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 620190104.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 554495698.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 573520863.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 485239031.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 442172826.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 561786219 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 760523668.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 839958010.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 669825458 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 574330784.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 402367627.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 475569542.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 533549127.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 496176324.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 490002819.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 684551867.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 521267216.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 606631219.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 543058804.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 449471177.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 487901763.3 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 554018402.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 591641185.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 588477446.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 631231795.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 511015623.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 799711775.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 657683878.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 524700274.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 619309565.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 577214493.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 590403790.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 443684332.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 596656544.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 538459498.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 588977194.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 578221344.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 506727901.9 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 523757100.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 852312164.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 675076877.8 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 595139614.2 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 624239491.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 590203613.7 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 567461853.6 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 562451502.1 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 565157976.5 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 710408456 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 384134462.4 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-50_text_document"

# ja swallow ML filtered
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101806001 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 41194352 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 43193021 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17947102 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20773041 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15698555 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4849478 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7379337 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2786526 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1476598 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9472990 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1836854 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2349361 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5376324 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1076438 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1771184 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2787270 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10331883 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6354095 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4489668 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5478459 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5070876 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 190570835 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 44848516 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 68413796 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 135235594 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14248806 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 113215001 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 55616668 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 123276248 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 126144368 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 472388531 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 439326815 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3241107555 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2305732291 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1101615341 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1249512604 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1086421307 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1096575005 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 819534366 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 595587226 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1033800913 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 861955349 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 689936822 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 568623048 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 599872942 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 627809205 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 684960971 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 497870015 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 554139810 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 579434743 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 625960522 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 685701989 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 846829850 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 791275698 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 627151187 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 726726650 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 858769153 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 879441955 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 875155738 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1006656100 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 923719061 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 955054449 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 819511230 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 735152347 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 848631866 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 969428691 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 951092358 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1029461540 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1056999035 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 954437237 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1176823173 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1149059457 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1027230325 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1182985211 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1191882147 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1165416941 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 967132362 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1063085711 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 931761908 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1082204245 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1159025610 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1147092449 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1210371972 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1586349204 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1376025393 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1234927411 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1188115068 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1105715905 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1075699475 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1014055840 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1017517893 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1133511918 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 638390086 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-50_text_document"

# ja-en parallel corpus
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 882674099 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/default_plain_text_format_text_document"

# en wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4426952329 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/en_wiki_merged_train_text_document"

# en cosmopedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2527156128 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_automathtext_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 39884494 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_khanacademy_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 189399130 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_openstax_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1824516742 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stanford_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5235630656 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stories_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2684874924 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v1_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2185784224 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v2_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 312753702 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_wikihow_train_text_document"

# en datacomp-lm baseline
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3978965646 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_01_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3982293647 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_02_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3847947687 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_03_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3985380280 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_04_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3987567231 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_05_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3981796125 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_06_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3986683692 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_07_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3984891412 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_08_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3978333553 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_09_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3976514300 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_10_of_10_text_document"

# code stack v2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12932222533 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/bigcode/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-00_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19067777467 /bb/llm/gaf51275/datasets/Meta-Llama-3_original_transformers-4.40.1/bigcode/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-01_text_document"

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
  --wandb-project "GTC25" \
  --wandb-entity "okoge"
