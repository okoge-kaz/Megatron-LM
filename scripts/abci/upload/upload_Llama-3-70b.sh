#!/bin/bash

set -e

start=10000
end=10000
increment=2500

upload_base_dir=/bb/llm/gaf51275/2024/checkpoints/megatron-to-hf/Llama-3-70b/exp6/tp8-pp16-ct1/LR1.0e-5-MINLR1.0E-6-WD0.1

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Llama-3-70b-exp6-LR1.0e-5-MINLR1.0E-6-WD0.1-iter$(printf "%07d" $i)
done
