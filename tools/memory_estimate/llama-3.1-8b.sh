#!/bin/bash

source .env/bin/activate

python tools/memory_estimate/memory_consumption_estimator.py \
  --hidden-size 4096 \
  --ffn-hidden-size 14336 \
  --seq-length 8192 \
  --num-layers 32 \
  --num-attention-heads 32 \
  --group-query-attention \
  --swiglu \
  --num-key-value-heads 8 \
  --vocab-size 128256 \
  --untie-embeddings-and-output-weights \
  --tensor-parallel-size 2 \
  --context-parallel-size 2 \
  --data-parallel-size 1 \
  --use-distributed-optimizer \
  --pipeline-parallel-size 4
