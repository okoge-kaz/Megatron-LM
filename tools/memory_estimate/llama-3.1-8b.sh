#!/bin/bash

source .env/bin/activate

WORLD_SIZE=8
TENSOR_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1
MICRO_BATCH_SIZE=2
DATA_PARALLEL_SIZE=$(($WORLD_SIZE / ($TENSOR_PARALLEL_SIZE * $CONTEXT_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE)))

echo "DATA_PARALLEL_SIZE: $DATA_PARALLEL_SIZE"

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
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --sequence-parallel \
  --context-parallel-size $CONTEXT_PARALLEL_SIZE \
  --pipeline-parallel-size $PIPELINE_PARALLEL_SIZE \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --data-parallel-size $DATA_PARALLEL_SIZE \
  --use-distributed-optimizer \
  --no-dropout \
  --use-flash-attention
