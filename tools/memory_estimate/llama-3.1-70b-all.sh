#!/bin/bash

source .env/bin/activate

SEQ_LENGTH=8192

python tools/memory_estimate/optimal_setting_search.py \
  --global-batch-size 1024 \
  --hidden-size 8192 \
  --ffn-hidden-size 28672 \
  --seq-length $SEQ_LENGTH \
  --num-layers 80 \
  --num-attention-heads 64 \
  --group-query-attention \
  --swiglu \
  --num-query-groups 8 \
  --padded-vocab-size 128256 \
  --untie-embeddings-and-output-weights \
  --sequence-parallel \
  --use-distributed-optimizer \
  --hidden-dropout 0.0 \
  --attention-dropout 0.0 \
  --use-flash-attn \
  --accumulate-allreduce-grads-in-fp32 \
  --start-world-size 4 \
  --end-world-size 128 \
  --num-per-node-gpus 4 \
  --gpu-memory-size 94
