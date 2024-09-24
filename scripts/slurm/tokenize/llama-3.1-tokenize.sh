#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --time=3:00:00
#SBATCH --partition=h100
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/tokenize/%x-%j.out
#SBATCH --error=outputs/tokenize/%x-%j.out

# module load
module load gc1/cuda/12.1
module load gc1/cudnn/9.2.0
module load gc1/nccl/2.20.5
module load gc1/hpcx/2.18.1

set -e

# python virtualenv
source .env/bin/activate

DATASET_DIR=/home/kazuki_fujii/datasets/pretrain/raw
OUTPUT_DIR=/home/kazuki_fujii/datasets/pretrain/binarized/llama3

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki_merged.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model /home/kazuki_fujii/hf-checkpoints/Meta-Llama-3-8B/tokenizer.jsonl \
  --append-eod \
  --workers 128
