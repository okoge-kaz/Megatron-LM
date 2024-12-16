import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from megatron.training.theoretical_memory_usage import (
    compute_weight_and_optimizer_memory,
    compute_activation_memory,
    NUM_BYTES_IN_GIGA_BYTE,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate memory consumption of a model')

    parser.add_argument('--hidden-size', type=int, default=4096, help='Hidden size of the model')
    parser.add_argument('--ffn-hidden-size', type=int, default=14336, help='Hidden size of the feedforward network')
    parser.add_argument('--seq-length', type=int, default=8192, help='Sequence length of the model')
    parser.add_argument('--micro-batch-size', type=int, default=1, help='Micro batch size')
    parser.add_argument('--num-attention-heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--group-query-attention', action='store_true', help='Group query attention')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of layers')
    parser.add_argument('--swiglu', action='store_true', help='Use SwiGLU activation function')
    parser.add_argument('--num-experts', type=int, default=None, help='Number of experts in MoE')
    parser.add_argument('--num-query-groups', type=int, default=8, help='Number of key value heads')
    parser.add_argument('--padded-vocab-size', type=int, default=128256, help='Vocabulary size')
    parser.add_argument('--untie-embeddings-and-output-weights', action='store_true', help='Untie embeddings and output weights')
    parser.add_argument('--selective-activation-recomputation', action='store_true', help='Use selective activation recomputation')
    parser.add_argument('--hidden-dropout', type=float, default=0.0, help='Hidden dropout')
    parser.add_argument('--attention-dropout', type=float, default=0.0, help='Attention dropout')
    parser.add_argument('--use-flash-attn', action='store_true', help='Use Flash Attention')

    parser.add_argument('--tensor-model-parallel-size', type=int, default=1, help='Size of the tensor parallelism')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, help='Size of the pipeline parallelism')
    parser.add_argument('--sequence-parallel', action='store_true', help='Use sequence parallelism')
    parser.add_argument('--data-parallel-size', type=int, default=1, help='Size of the data parallelism')
    parser.add_argument('--use-distributed-optimizer', action='store_true', help='Use distributed optimizer')
    parser.add_argument('--context-parallel-size', type=int, default=1, help='Size of the context parallelism')
    parser.add_argument('--gpu-memory-limit-mb', type=int, default=40, help='GPU memory limit in GB')
    parser.add_argument('--kv-channels', type=int, help='Number of key value channels')
    parser.add_argument('--accumulate-allreduce-grads-in-fp32', action='store_true', help='Accumulate allreduce grads in FP32')
    parser.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full', 'selective'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed.')
    args = parser.parse_args()

    assert args.hidden_size % args.num_attention_heads == 0
    args.kv_channels = args.hidden_size // args.num_attention_heads
    args.virtual_pipeline_model_parallel_size = None

    weight_and_optimizer_memory = compute_weight_and_optimizer_memory(args=args) / NUM_BYTES_IN_GIGA_BYTE
    activation_memory = compute_activation_memory(args=args, num_microbatches=None) / NUM_BYTES_IN_GIGA_BYTE

    print("\n\n========== weights and optimizer memory consumption ==========")
    print("memory used (per GPU) in the first stage: ", weight_and_optimizer_memory, "GB")
    print("=============================================================\n\n")

    print("========== activation memory consumption ==========")
    print("memory used (per GPU) in the first stage: ", activation_memory, "GB")
    print("===================================================\n\n")

    print("========== total memory consumption ==========")
    print("memory used (per GPU) in the first stage: ", weight_and_optimizer_memory + activation_memory ,"GB")
    print("==============================================\n\n")
