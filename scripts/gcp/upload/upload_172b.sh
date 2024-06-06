#!/bin/bash

source .env/bin/activate

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <start> <end>"
  exit 1
fi

start=$1
end=$2
increment=500

base_dirs=(
  "/home/ext_kazuki_fujii_rio_gsic_titech/checkpoints/megatron-to-hf/Llama-2-172b-hf"
)

for base_dir in "${base_dirs[@]}"; do
  for ((i = start; i <= end; i += increment)); do
    upload_dir=$base_dir/iter_$(printf "%07d" $i)

    python scripts/abci/upload/upload.py \
      --ckpt-path $upload_dir \
      --repo-name llm-jp/GENIAC-Llama2-172b-iter$(printf "%07d" $i)
  done
done
