#!/bin/bash

source .env/bin/activate

WORLD_SIZE=32
TENSOR_PARALLEL_SIZE=4
CONTEXT_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=4
MICRO_BATCH_SIZE=2
DATA_PARALLEL_SIZE=$(($WORLD_SIZE / ($TENSOR_PARALLEL_SIZE * $CONTEXT_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE)))
SEQ_LENGTH=8192

echo "DATA_PARALLEL_SIZE: $DATA_PARALLEL_SIZE"

python tools/memory_estimate/memory_consumption_estimator.py \
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
  --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
  --sequence-parallel \
  --context-parallel-size $CONTEXT_PARALLEL_SIZE \
  --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --data-parallel-size $DATA_PARALLEL_SIZE \
  --use-distributed-optimizer \
  --hidden-dropout 0.0 \
  --attention-dropout 0.0 \
  --use-flash-attn \
  --accumulate-allreduce-grads-in-fp32
