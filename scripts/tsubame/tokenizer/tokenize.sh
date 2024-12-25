#!/bin/sh
#$ -cwd
#$ -l cpu_80=1
#$ -l h_rt=24:00:00
#$ -o outputs/tokenize/$JOB_ID.log
#$ -e outputs/tokenize/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# switch virtual env
source .env/bin/activate

DATASET_DIR=/gs/bs/tgh-24IDU/datasets/raw/pretrain/finemath/finemath-4plus-jsonl
OUTPUT_DIR=/gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python tools/preprocess_data.py \
  --input ${DATASET_DIR}/finemath-4plus-merged.jsonl \
  --output-prefix ${OUTPUT_DIR}/finemath-4plus \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model /gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B/tokenizer.json \
  --append-eod \
  --workers 64
