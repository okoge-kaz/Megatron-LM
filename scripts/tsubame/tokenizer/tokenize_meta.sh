#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -o outputs/tokenizer/$JOB_ID
#$ -e outputs/tokenizer/$JOB_ID
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

DATASET_DIR=/gs/bs/tga-NII-LLM/datasets/swallow-meta-tag
OUTPUT_DIR=/gs/bs/tga-NII-LLM/binarized/swallow-meta-tag

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python tools/preprocess_data.py \
  --input ${DATASET_DIR}/wiki-category.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja-wiki-category \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model /gs/bs/tga-NII-LLM/hf-checkpoints/meta-tag-Llama-3-8B/tokenizer.json \
  --begin-of-special-token-id 128002 \
  --end-of-special-token-id 128003 \
  --append-eod \
  --workers 64
