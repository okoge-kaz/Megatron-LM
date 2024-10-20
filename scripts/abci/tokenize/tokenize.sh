#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/tokenize/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.20.5
module load hpcx/2.12
module load gcc/11.4.0

source .env/bin/activate

DATASET_DIR=/bb/llm/gaf51275/datasets/raw/pretrain/stack_v2_python
OUTPUT_DIR=/bb/llm/gaf51275/datasets/Meta-Llama-3.1_original_transformers-4.45.2

mkdir -p ${OUTPUT_DIR}

python tools/preprocess_data.py \
  --input ${DATASET_DIR}/merged_0.jsonl \
  --output-prefix ${OUTPUT_DIR}/stack_v2_python_all \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model /bb/llm/gaf51275/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json \
  --append-eod \
  --workers 72
