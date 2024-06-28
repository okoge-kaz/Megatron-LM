#!/bin/bash

# module load
module load turing/cuda/12.1
module load turing/cudnn/8.9.7
module load turing/nccl/2.20.5
module load turing/hpcx/2.17.1

# python virtualenv
source .env/bin/activate

DATASET_DIR=/home/ext_kazuki_fujii_turing_motors_c/datasets/raw
OUTPUT_DIR=/home/ext_kazuki_fujii_turing_motors_c/datasets/binarized

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki_merged.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model /home/ext_kazuki_fujii_turing_motors_c/hf-checkpoints/Meta-Llama-3-8B/tokenizer.jsonl \
  --append-eod \
  --workers 16
