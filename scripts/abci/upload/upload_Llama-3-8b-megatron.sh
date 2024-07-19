#!/bin/bash

set -e

start=500
end=12500
increment=500

upload_base_dir=/bb/llm/gaf51275/2024/checkpoints/Llama-3-8b/exp7/tp2-pp4-ct1/LR2.5E-5-MINLR2.5E-6-WD0.1

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name RioYokotaLab/Llama-3-8b-ablation-exp7-LR2.5e-5-MINLR2.5E-6-WD0.1-iter$(printf "%07d" $i)
done
