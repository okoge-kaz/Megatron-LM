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
        f"Number of parameters in the model: {num_total_parameters / 1e9:.2f}B\n"
    )


def compute_per_gpu_memory_consumption_weight_and_optimizer(args: argparse.Namespace) -> tuple[float, float, float]:
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
        18 if not args.use_distributed_optimizer else 6 + (
            12 / args.data_parallel_size // args.context_parallel_size
        )
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
                + (1 / args.hidden_size)
            )
        ) + (
            # final layernorm
            args.hidden_size
        )
        num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
        print(
            f"Number of parameters (per GPU): {num_total_parameters / 1e9:.5f}B"
        )
        wight_and_optimizer_memory = num_total_parameters * num_bytes_per_parameter
        print(
            f"Memory consumption of weights and optimizer (per GPU): {wight_and_optimizer_memory / 1024 / 1024 / 1024:.5f}GB"
        )

        return (wight_and_optimizer_memory / 1024 / 1024 / 1024, 0, 0)

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
                + (1 / args.hidden_size)
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
                + (1 / args.hidden_size)
            )
        ) + (
            # final layernorm
            args.hidden_size
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

        return (first_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024, 0,second_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024)
    else:
        assert args.num_layers % args.pipeline_parallel_size == 0

        first_stage_num_parameters_in_transformer_layers = (
            2
            * (args.num_layers // args.pipeline_parallel_size)
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
                + (1 / args.hidden_size)  # sequence parallelism で分割する方向はsequence方向なので、hidden size方向にはメモリを分割しないので減らない
            )
        )
        first_stage_num_total_parameters = first_stage_num_parameters_in_transformer_layers + embedding_size

        mid_stage_num_parameters_in_transformer_layers = (
            2
            * (args.num_layers // args.pipeline_parallel_size)
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
                + (1 / args.hidden_size)
            )
        )
        mid_stage_num_total_parameters = mid_stage_num_parameters_in_transformer_layers

        last_stage_num_parameters_in_transformer_layers = (
            2
            * (args.num_layers // args.pipeline_parallel_size)
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
                + (1 / args.hidden_size)
            )
        ) + (
            # final layernorm
            args.hidden_size
        )
        last_stage_num_total_parameters = last_stage_num_parameters_in_transformer_layers + embedding_size

        first_stage_weight_and_optimizer_memory = first_stage_num_total_parameters * num_bytes_per_parameter
        mid_stage_weight_and_optimizer_memory = mid_stage_num_total_parameters * num_bytes_per_parameter
        last_stage_weight_and_optimizer_memory = last_stage_num_total_parameters * num_bytes_per_parameter
        print(
            f"memory used (per GPU) in the first stage: {first_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024:.5f}GB"
        )
        print(
            f"memory used (per GPU) in the middle stage: {mid_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024:.5f}GB"
        )
        print(
            f"memory used (per GPU) in the last stage: {last_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024:.5f}GB"
        )

        return (
            first_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024, mid_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024, last_stage_weight_and_optimizer_memory / 1024 / 1024 / 1024
        )


def compute_per_gpu_memory_consumption_activation(args: argparse.Namespace):
    s = args.seq_length
    h = args.hidden_size
    b = args.micro_batch_size
    a = args.num_attention_heads
    k = args.num_key_value_heads

    s = s // args.context_parallel_size  # self attentionでは、s // context_parallel_size ではないが、selective activation recomputationしているので無視

    # TODO: sequence parallelを有効にするかの場合分け追加
    activation_memory = (
        # transformer layer
        2 * s * b * h # LayerNorm
        + (
            # attention
            2 * s * b * h  # x -> Q, K, V
            + (2 * s * b * h)  # Q * K^T -> attention scores (Q)
            + (2 * s * b * h) * (k / a)  # Q * K^T -> attention scores (K) (grouped query attention)
            + (  # Q*K^T -> softmax(Q*K^T)
                0 if (
                    args.selective_activation_recomputation or args.use_flash_attention
                ) else (2 * b * s * s * a)
            )
            + (  # softmax(Q*K^T) -> Dropout
                0 if args.no_dropout else 0 if (
                    args.selective_activation_recomputation or args.use_flash_attention
                ) else (1 * b * s * s * a)
            )
            + ((2 * b * s * h) * (k / a))  # attention scores * V -> attention output (V) (grouped query attention)
            + (  # Dropout(softmax(Q*K^T)) * V -> attention output (Dropout)
                0 if (
                    args.selective_activation_recomputation or args.use_flash_attention
                ) else (2 * b * a * s * s)
            )
            + (2 * b * s * h)  # attention output -> attention output (Linear)
        )
        + (0 if args.no_dropout else (b * s * h))  # attention output -> Dropout
        + (2 * b * s * h)  # Dropout(attention output -> LayerNorm
        + (
            # MLP
            # SwiGLU
            # self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            2 * b * s * h  # up_proj & gate_proj
            + 2 * b * s * args.ffn_hidden_size  # act_fn (up) -> GLU(act_fn)
            + 2 * b * s * args.ffn_hidden_size  # act_fn act_fn -> x
            + 2 * b * s * args.ffn_hidden_size  # act_fn (gate) -> x
            + 2 * b * s * args.ffn_hidden_size  # act_fn (down)
        )
        + (0 if args.no_dropout else (b * s * h)) # MLP -> Dropout
    ) * args.num_layers / args.pipeline_parallel_size / args.tensor_parallel_size

    first_stage_activation_memory = activation_memory * args.pipeline_parallel_size # 1F1B

    # first_stage_activation_memory += (  # input ot embedding (pp size microbatch in flight)
    #     8 * s * b * h * args.pipeline_parallel_size  # これの出どころ謎
    # )  # TODO: tensor parallel なし?

    # Llamaはhidden dropoutなし
    first_stage_activation_memory += 0 if args.no_dropout else ((
        # dropout in embedding layer (pp size microbatehes in flight)
        s * b * h * args.pipeline_parallel_size
    ) / args.tensor_parallel_size)

    if args.pipeline_parallel_size == 1:
        # if pp_size == 1 (lm-head cross entropy)
        first_stage_activation_memory += (
            # lm-head cross entropy (FP32)
            # output layer (layer norm) + output layer (linear)
            4 * s * b * h * (1 + args.vocab_size / h)
        ) / args.tensor_parallel_size

    print("memory used (per GPU):", first_stage_activation_memory / 1024 / 1024 / 1024, "GB")

    return first_stage_activation_memory / 1024 / 1024 / 1024, 0, 0


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
    parser.add_argument('--num-key-value-heads', type=int, default=8, help='Number of key value heads')
    parser.add_argument('--vocab-size', type=int, default=128256, help='Vocabulary size')
    parser.add_argument('--untie-embeddings-and-output-weights', action='store_true', help='Untie embeddings and output weights')
    parser.add_argument('--selective-activation-recomputation', action='store_true', help='Use selective activation recomputation')
    parser.add_argument('--no-dropout', action='store_true', help='No dropout')
    parser.add_argument('--use-flash-attention', action='store_true', help='Use Flash Attention')

    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Size of the tensor parallelism')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='Size of the pipeline parallelism')
    parser.add_argument('--sequence-parallel', action='store_true', help='Use sequence parallelism')
    parser.add_argument('--data-parallel-size', type=int, default=1, help='Size of the data parallelism')
    parser.add_argument('--use-distributed-optimizer', action='store_true', help='Use distributed optimizer')
    parser.add_argument('--context-parallel-size', type=int, default=1, help='Size of the context parallelism')
    parser.add_argument('--gpu-memory-limit-mb', type=int, default=40, help='GPU memory limit in GB')
    args = parser.parse_args()

    compute_parameter_size(args=args)

    print("========== weights and optimizer memory consumption ==========")
    first_gb, mid_gb, last_gb = compute_per_gpu_memory_consumption_weight_and_optimizer(args=args)
    print("=============================================================\n\n")

    print("========== activation memory consumption ==========")
    first_ac_gb, mid_ac_gb, last_ac_gb = compute_per_gpu_memory_consumption_activation(args=args)
    print("===================================================\n\n")

    print("========== total memory consumption ==========")
    print("memory used (per GPU) in the first stage: ", first_gb + first_ac_gb, "GB")
    print("==============================================\n\n")


