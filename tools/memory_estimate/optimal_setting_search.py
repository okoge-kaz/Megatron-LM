import argparse
import sys
import os
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from megatron.training.theoretical_memory_usage import (
    compute_weight_and_optimizer_memory,
    compute_activation_memory,
    NUM_BYTES_IN_GIGA_BYTE,
)


def is_trainable(total_memory: float, gpu_memory_size: int) -> bool:
    """Check if the model may be trainable on the given GPU.

    Args:
        total_memory (float): weight and optimizer memory + activation memory
        gpu_memory_size (int): H100(80GB, 94GB), A100(40GB, 80GB), V100(32GB, 16GB)

    Returns:
        bool: True if the model may be trainable, False otherwise
    """
    if total_memory <= gpu_memory_size * 0.8:
        return True
    else:
        return False


def is_within_the_num_gpu_per_node(num_per_node_gpus: int, tp_size: int, cp_size: int) -> bool:
    if tp_size * cp_size <= num_per_node_gpus:
        return True
    else:
        return False


def generate_pipeline_parallel_sizes(num_layers: int) -> list[int]:
    """
    Generate pipeline parallel sizes based on the number of layers.
    """
    def all_factors(n: int) -> list[int]:
        factors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        return sorted(factors)

    # Get all factors of num_layers that can be used as pp_size
    return all_factors(num_layers)


def generate_micro_batch_sizes(
    global_batch_size: int,
    data_parallel_size: int,
) -> list[int]:
    """
    Generate micro batch sizes based on the global batch size and data parallel size.
    """
    micro_batch_sizes: list[int] = []
    micro_batch_size = 1
    while micro_batch_size * data_parallel_size <= global_batch_size:
        micro_batch_sizes.append(micro_batch_size)
        micro_batch_size *= 2
    return micro_batch_sizes


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Memory estimate for optimal setting search")
    # optimal setting search
    parser.add_argument("--start-world-size", type=int, default=1)
    parser.add_argument("--end-world-size", type=int, default=256)
    parser.add_argument("--num-per-node-gpus", type=int, default=8)
    parser.add_argument('--gpu-memory-size', type=int, default=80)

    # model configuration
    parser.add_argument('--global-batch-size', type=int, default=1024, help='Global batch size')
    parser.add_argument('--hidden-size', type=int, default=4096, help='Hidden size of the model')
    parser.add_argument('--ffn-hidden-size', type=int, default=14336, help='Hidden size of the feedforward network')
    parser.add_argument('--seq-length', type=int, default=8192, help='Sequence length of the model')
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
    parser.add_argument('--sequence-parallel', action='store_true', help='Use sequence parallelism')
    parser.add_argument('--use-distributed-optimizer', action='store_true', help='Use distributed optimizer')
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

    return args


def visualize_memory_matrix(memory_consumption_matrix, gpu_memory_size: int) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from matplotlib.colors import ListedColormap

    # Convert dictionary to DataFrame
    data = []
    for world_size, configs in memory_consumption_matrix.items():
        for (tp, cp, pp, mbs), memory in configs.items():
            data.append({
                'World Size': world_size,
                'Config': f'({tp},{cp},{pp},{mbs})',
                'Memory (GB)': memory,
                'tp': tp,
                'cp': cp,
                'pp': pp,
                'mbs': mbs
            })

    df = pd.DataFrame(data)
    # Pivot the data for heatmap
    pivot_df = df.pivot(index='Config', columns='World Size', values='Memory (GB)')
    # Sort index by tuple values
    pivot_df = pivot_df.reindex(
        sorted(pivot_df.index, key=lambda x: tuple(map(int, x.strip('()').split(','))))
    )

    # Create discrete color mapping
    def get_color_value(x):
        if pd.isna(x):
            return np.nan
        if x <= gpu_memory_size * 0.8:
            return 0  # Light green
        elif x <= gpu_memory_size:
            return 1  # Light yellow
        else:
            return 2  # Light red

    color_matrix = pivot_df.map(get_color_value)

    # Create custom colormap for discrete values
    colors = ['lightgreen', 'khaki', 'lightcoral']
    cmap = ListedColormap(colors)

    # Create figure with single plot (removed the top subplot)
    fig, ax = plt.subplots(figsize=(15, max(12, len(pivot_df) * 0.4)))

    # Create main heatmap using color_matrix
    sns.heatmap(color_matrix,
                ax=ax,
                annot=pivot_df,
                fmt='.2f',
                cmap=cmap,
                cbar=False,
                mask=pivot_df.isna(),
                annot_kws={'size': 15},
            )

    # Add horizontal lines for tp/cp/pp changes
    prev_values = None
    for idx, config in enumerate(pivot_df.index):
        current_values = tuple(map(int, config.strip('()').split(',')[:3]))
        if prev_values and current_values != prev_values:
            ax.axhline(y=idx, color='black', linewidth=2)
        prev_values = current_values

    # Customize the main plot
    ax.set_xlabel('world size', fontsize=20)
    ax.set_ylabel('(TP, CP, PP, MBS)', fontsize=20)

    # Increase y-axis tick labels font size
    ax.tick_params(axis='y', labelsize=15)  # Increased from default
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=15)

    # Add legend for color meanings
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightgreen', label=f'≤ {gpu_memory_size * 0.8:.1f} GB'),  # type: ignore
        plt.Rectangle((0,0),1,1, facecolor='khaki', label=f'≤ {gpu_memory_size:.1f} GB'),  # type: ignore
        plt.Rectangle((0,0),1,1, facecolor='lightcoral', label=f'> {gpu_memory_size:.1f} GB')  # type: ignore
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), prop={'size': 15})
    # Adjust layout
    plt.tight_layout()

    plt.savefig('memory_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    args = arg_parse()

    TENSOR_PARALLEL_SIZES = [1, 2, 4, 8]
    CONTEXT_PARALLEL_SIZES = [1, 2, 4, 8]
    PIPELINE_PARALLEL_SIZES = generate_pipeline_parallel_sizes(args.num_layers)
    # print(f"PIPELINE_PARALLEL_SIZES: {PIPELINE_PARALLEL_SIZES}")

    memory_consumption_matrix = {}
    MIN_TRAINABLE_MODEL_PARALLEL_SIZE = 1 << 20

    world_size = args.start_world_size
    while world_size <= args.end_world_size:
        memory_consumption_matrix[world_size] = {}

        for tp_size, cp_size, pp_size in itertools.product(
            TENSOR_PARALLEL_SIZES, CONTEXT_PARALLEL_SIZES, PIPELINE_PARALLEL_SIZES
        ):
            if not is_within_the_num_gpu_per_node(
                args.num_per_node_gpus, tp_size, cp_size
            ):
                continue
            if pp_size * tp_size * cp_size > world_size:
                continue
            if pp_size * tp_size * cp_size > MIN_TRAINABLE_MODEL_PARALLEL_SIZE * 2:
                continue

            args.tensor_model_parallel_size = tp_size
            args.context_parallel_size = cp_size
            args.pipeline_model_parallel_size = pp_size
            args.data_parallel_size = world_size // (tp_size * cp_size * pp_size)

            MICRO_BATCH_SIZES = generate_micro_batch_sizes(
                args.global_batch_size, args.data_parallel_size
            )

            for micro_batch_size in MICRO_BATCH_SIZES:
                args.micro_batch_size = micro_batch_size
                weight_and_optimizer_memory = compute_weight_and_optimizer_memory(args=args, verbose=False) / NUM_BYTES_IN_GIGA_BYTE
                activation_memory = compute_activation_memory(args, None, False) / NUM_BYTES_IN_GIGA_BYTE
                total_memory = weight_and_optimizer_memory + activation_memory

                memory_consumption_matrix[world_size][(tp_size, cp_size, pp_size, micro_batch_size)] = total_memory
                if is_trainable(total_memory, args.gpu_memory_size):
                    if tp_size * cp_size * pp_size < MIN_TRAINABLE_MODEL_PARALLEL_SIZE:
                        MIN_TRAINABLE_MODEL_PARALLEL_SIZE = tp_size * cp_size * pp_size
                else:
                    # break micro_batch_size increment
                    break

        world_size *= 2
    # print(memory_consumption_matrix)

    # Visualize the memory consumption matrix
    visualize_memory_matrix(
        memory_consumption_matrix=memory_consumption_matrix,
        gpu_memory_size=args.gpu_memory_size
    )
