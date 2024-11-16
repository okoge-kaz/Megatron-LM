#!/bin/sh
#$ -cwd
#$ -l cpu_80=1
#$ -l h_rt=24:00:00
#$ -o outputs/tokenizer/$JOB_ID.log
#$ -e outputs/tokenizer/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

DATASET_DIR=/gs/fs/tga-NII-LLM/datasets/Llama2Tokenizer
OUTPUT_DIR=/gs/fs/tga-NII-LLM/datasets/Llama2Tokenizer

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python tools/preprocess_data.py \
  --input ${DATASET_DIR}/ja_wiki_merged.jsonl \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /gs/bs/tga-NII-LLM/hf-checkpoints/Mixtral-8x7B-v0.1/tokenizer.model \
  --append-eod \
  --workers 64
