import argparse


def compute_parameter_size(args: argparse.Namespace):
    if not args.group_query_attention:
        args.num_key_value_heads = args.num_attention_heads
    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts

    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    num_parameters_in_transformer_layers = (
        2
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (1 + (args.num_key_value_heads / args.num_attention_heads))
            )
            # MLP.
            + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier)
            # Transformer layernorms.
            + (1 / args.hidden_size)
        )
    ) + (
        # final layernorm
        args.hidden_size
    )

    embedding_size = args.hidden_size * args.vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    print(
        f"Number of parameters in the model: {num_total_parameters / 1e9:.2f}B"
    )


def compute_per_gpu_memory_consumption_weight_and_optimizer(args: argparse.Namespace):
    if not args.group_query_attention:
        args.num_key_value_heads = args.num_attention_heads
    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts

    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    embedding_size = args.hidden_size * args.vocab_size / args.tensor_parallel_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )

    if args.pipeline_parallel_size == 1:
        num_parameters_in_transformer_layers = (
            2
            * args.num_layers
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                (
                    (1 + (args.num_key_value_heads / args.num_attention_heads)) / args.tensor_parallel_size
                )
                # MLP.
                + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier) / args.tensor_parallel_size
                # Transformer layernorms.
                + (1 / args.hidden_size)  # TODO: Sequence Parallelismでは分割されるが、memoryも減る?
            )
        ) + (
            # final layernorm
            args.hidden_size  # TODO: Sequence Parallelismでは分割される?
        )
        num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
        print(
            f"Number of parameters (per GPU): {num_total_parameters / 1e9:.5f}B"
        )
        wight_and_optimizer_memory = num_total_parameters * num_bytes_per_parameter
        print(
            f"Memory consumption of weights and optimizer (per GPU): {wight_and_optimizer_memory / 1024 / 1024 / 1024:.5f}GB"
        )

    elif args.pipeline_parallel_size == 2:
        first_stage_num_parameters_in_transformer_layers = (
            2
            * (args.num_layers / args.pipeline_parallel_size)
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                (
                    (1 + (args.num_key_value_heads / args.num_attention_heads)) / args.tensor_parallel_size
                )
                # MLP.
                + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier) / args.tensor_parallel_size
                # Transformer layernorms.
                + (1 / args.hidden_size)  # TODO: Sequence Parallelismでは分割されるが、memoryも減る?
            )
        )
        first_stage_num_total_parameters = first_stage_num_parameters_in_transformer_layers + embedding_size

        second_stage_num_parameters_in_transformer_layers = (
            2
            * (args.num_layers / args.pipeline_parallel_size)
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                (
                    (1 + (args.num_key_value_heads / args.num_attention_heads)) / args.tensor_parallel_size
                )
                # MLP.
                + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier) / args.tensor_parallel_size
                # Transformer layernorms.
                + (1 / args.hidden_size)  # TODO: Sequence Parallelismでは分割されるが、memoryも減る?
            )
        ) + (
            # final layernorm
            args.hidden_size  # TODO: Sequence Parallelismでは分割される?
        )
        second_stage_num_total_parameters = second_stage_num_parameters_in_transformer_layers + embedding_size

        first_stage_weight_and_optimizer_memory = first_stage_num_total_parameters * num_bytes_per_parameter
        second_stage_weight_and_optimizer_memory = second_stage_num_total_parameters * num_bytes_per_parameter

        print(
            f"Number of parameters (per GPU) in the first stage: {first_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024:.5f}GB"
        )
        print(
            f"Number of parameters (per GPU) in the second stage: {second_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024:.5f}GB"
        )
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate memory consumption of a model')

    parser.add_argument('--hidden-size', type=int, default=4096, help='Hidden size of the model')
    parser.add_argument('--ffn-hidden-size', type=int, default=14336, help='Hidden size of the feedforward network')
    parser.add_argument('--seq-length', type=int, default=8192, help='Sequence length of the model')
    parser.add_argument('--num-attention-heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--group-query-attention', action='store_true', help='Group query attention')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of layers')
    parser.add_argument('--swiglu', action='store_true', help='Use SwiGLU activation function')
    parser.add_argument('--num-experts', type=int, default=None, help='Number of experts in MoE')
    parser.add_argument('--num-key-value-heads', type=int, default=8, help='Number of key value heads')
    parser.add_argument('--vocab-size', type=int, default=128256, help='Vocabulary size')
    parser.add_argument('--untie-embeddings-and-output-weights', action='store_true', help='Untie embeddings and output weights')

    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Size of the tensor parallelism')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='Size of the pipeline parallelism')
    parser.add_argument('--data-parallel-size', type=int, default=1, help='Size of the data parallelism')
    parser.add_argument('--use-distributed-optimizer', action='store_true', help='Use distributed optimizer')
    parser.add_argument('--context-parallel-size', type=int, default=1, help='Size of the context parallelism')
    args = parser.parse_args()

    compute_parameter_size(args=args)
    compute_per_gpu_memory_consumption_weight_and_optimizer(args=args)


