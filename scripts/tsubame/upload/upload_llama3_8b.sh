#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=5:00:00
#$ -o outputs/upload/llama-3-8b/$JOB_ID
#$ -e outputs/upload/llama-3-8b/$JOB_ID
#$ -p -5

set -e

source .env/bin/activate

start=2500
end=7500
increment=2500

base_dirs=(
  "/gs/bs/tga-NII-LLM/checkpoints/megatron-to-hf/Llama-3-8b-block-expanison-36/exp4/tp4-pp2-ct1/LR2.5E-5-MINLR2.5E-6-WD0.1-WARMUP1000"
)

for base_dir in "${base_dirs[@]}"; do
  for ((i = start; i <= end; i += increment)); do
    upload_dir=$base_dir/iter_$(printf "%07d" $i)

    python scripts/abci/upload/upload.py \
      --ckpt-path $upload_dir \
      --repo-name tokyotech-llm/Llama-3-8b-block-expanison-36-exp4-LR2.5E-5-MINLR2.5E-6-WD0.1-iter$(printf "%07d" $i)
  done
done
