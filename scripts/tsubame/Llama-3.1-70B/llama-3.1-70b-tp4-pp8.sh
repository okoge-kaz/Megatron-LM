#!/bin/sh
#$ -cwd
#$ -l node_f=32
#$ -l h_rt=06:23:50:00
#$ -o outputs/Llama-3.1-70b/$JOB_ID.log
#$ -e outputs/Llama-3.1-70b/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

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
# llama-3.1-70b: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B/blob/main/config.json
HIDDEN_SIZE=8192
FFN_HIDDEN_SIZE=28672 # intermediate size (HuggingFace)
NUM_LAYERS=80
NUM_HEADS=64
NUM_KEY_VALUE_HEADS=8
SEQ_LENGTH=8192

# distributed settings
TENSOR_PARALLEL_SIZE=4 # fixed (tsubame has 4 GPUs per node)
PIPELINE_PARALLEL_SIZE=8
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

PIPLINE_MODEL_CHUNKS=1
LAYERS_PER_VIRTUAL_PIPELINE_STAGE=$((${NUM_LAYERS} / ${PIPELINE_PARALLEL_SIZE} / ${PIPLINE_MODEL_CHUNKS}))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000
LR_DECAY_ITERS=25000

LR=1.0E-5
MIN_LR=1.0E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-70B/tokenizer.json
CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-megatron/Llama-3.1-70b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/gs/bs/tga-NII-LLM/Llama-3.1-70B/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3382423156 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# ja swallow v2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 30221726.23 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10667022 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14942879.19 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8193690.751 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11554765.15 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10272936.91 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3533324.99 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6658533.963 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2819915.121 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1366948.105 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8871506.651 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1618425.506 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2437269.01 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6307618.074 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1219417.83 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1983129.868 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2122259.241 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7773841.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6026568.34 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4592114.689 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6286619.878 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4769666.334 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 61536039.3 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15355830.88 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 25221414.61 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 65612527.94 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6307710.878 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 59759838.72 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 29285328.19 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 58572394.14 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 69533526.81 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 278781501.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 286151037.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2241528120 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1914570713 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 764016966.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 965932356.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1298400530 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 749440210.3 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 578927771.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 583977055.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 880476793.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 774360859.8 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 595111975.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 556597386.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 620190104.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 554495698.5 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 573520863.9 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 485239031.6 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 442172826.9 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 561786219 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 760523668.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 839958010.6 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 669825458 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 574330784.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 402367627.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 475569542.8 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 533549127.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 496176324.9 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 490002819.7 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 684551867.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 521267216.3 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 606631219.6 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 543058804.8 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 449471177.5 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 487901763.3 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 554018402.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 591641185.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 588477446.9 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 631231795.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 511015623.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 799711775.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 657683878.6 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 524700274.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 619309565.8 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 577214493.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 590403790.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 443684332.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 596656544.8 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 538459498.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 588977194.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 578221344.7 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 506727901.9 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 523757100.7 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 852312164.5 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 675076877.8 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 595139614.2 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 624239491.5 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 590203613.7 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 567461853.6 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 562451502.1 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 565157976.5 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 710408456 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 384134462.4 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-others/CC-MAIN-2023-50_text_document"

# ja swallow ML filtered
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101806001 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 41194352 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 43193021 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17947102 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 20773041 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15698555 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4849478 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7379337 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2786526 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1476598 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9472990 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1836854 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2349361 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5376324 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1076438 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1771184 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2787270 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10331883 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6354095 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4489668 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5478459 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5070876 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 190570835 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 44848516 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 68413796 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 135235594 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 14248806 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 113215001 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 55616668 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 123276248 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 126144368 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 472388531 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 439326815 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3241107555 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2305732291 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1101615341 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1249512604 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1086421307 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1096575005 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 819534366 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 595587226 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1033800913 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 861955349 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 689936822 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 568623048 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 599872942 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 627809205 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 684960971 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 497870015 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 554139810 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 579434743 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 625960522 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 685701989 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 846829850 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 791275698 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 627151187 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 726726650 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 858769153 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 879441955 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 875155738 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1006656100 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 923719061 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 955054449 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 819511230 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 735152347 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 848631866 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 969428691 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 951092358 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1029461540 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1056999035 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 954437237 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1176823173 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1149059457 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1027230325 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1182985211 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1191882147 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1165416941 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 967132362 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1063085711 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 931761908 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1082204245 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1159025610 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1147092449 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1210371972 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1586349204 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1376025393 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1234927411 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1188115068 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1105715905 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1075699475 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1014055840 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1017517893 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1133511918 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 638390086 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/Swallow_v2/edu/filter-v2-wiki-top10/CC-MAIN-2023-50_text_document"

# ja en parallel corpus
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 882674099 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/default_plain_text_format_text_document"

# en wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4426952329 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/en_wiki_merged_train_text_document"

# en cosmopedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2527156128 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_automathtext_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 39884494 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_khanacademy_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 189399130 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_openstax_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1824516742 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stanford_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5235630656 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_stories_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2684874924 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v1_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2185784224 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_web_samples_v2_train_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 312753702 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/cosmopedia_wikihow_train_text_document"

# en datacomp-lm baseline
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3978965646 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_01_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3982293647 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_02_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3847947687 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_03_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3985380280 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_04_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3987567231 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_05_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3981796125 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_06_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3986683692 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_07_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3984891412 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_08_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3978333553 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_09_of_10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3976514300 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/global-shard_10_of_10_text_document"

# code stack v2
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 12932222533 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-00_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19067777467 /gs/fs/tgh-24IDN/datasets/binarized/Meta-Llama-3_original_transformers-4.40.1/the-stack-v2-train-smol-ids/random_sample0.1_merge/the-stack-v2-train-smol-ids-01_text_document"


# job name
JOB_NAME="Llama-3.1-70b-TSUBAME-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

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
  --distributed-timeout-minutes 15 \
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
  --save-interval 250 \
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
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-3.1-70B" \
  --wandb-entity "prj-jalm"
