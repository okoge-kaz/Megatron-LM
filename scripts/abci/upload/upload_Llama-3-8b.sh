#!/bin/bash

set -e

start=2500
end=2500
increment=2500

upload_base_dir=/bb/llm/gaf51275/2024/checkpoints/megatron-to-hf/Llama-3-8b-hf/exp9/tp2-pp4-ct1/LR2.5e-5-MINLR2.5E-6-WD0.1

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Llama-3-8b-exp9-LR2.5e-5-MINLR2.5E-6-WD0.1-iter$(printf "%07d" $i)
done
