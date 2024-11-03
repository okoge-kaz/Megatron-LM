# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training."""


import math

NUM_BYTES_IN_GIGA_BYTE = 1024 * 1024 * 1024


def compute_weight_and_optimizer_memory(args, verbose=True):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
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
                (1 + (args.num_query_groups / args.num_attention_heads))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier)
            # Transformer layernorms.
            + (1 / args.hidden_size)
        )
    ) + (
        # final layer norm
        args.hidden_size
    )

    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(
            f"Number of parameters in transformer layers in billions: "
            f"{num_parameters_in_transformer_layers / 10**9}"
        )
        print(
            f"Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (    # - last layer norm
            (num_parameters_in_transformer_layers - args.hidden_size) / args.pipeline_model_parallel_size
        ) + embedding_size  # (embedding layer) for first stage
    ) / args.tensor_model_parallel_size
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
        num_parameters_on_most_loaded_model_shard += args.hidden_size  # last layer norm
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
            args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: "
                f"{num_parameters_on_other_model_shards / 10**9}"
            )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size / args.context_parallel_size)
    )
    weight_and_optimizer_memory = (
        num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.

    # Memory footprint from transformer layer (self-attention and MLP).
    s = args.seq_length
    h = args.hidden_size
    b = args.micro_batch_size
    a = args.num_attention_heads
    k = args.num_query_groups

    s = s / args.context_parallel_size
    args.no_dropout = (abs(args.attention_dropout) <= 1e-8) and (abs(args.hidden_dropout) <= 1e-8)
    args.selective_activation_recomputation = (
        args.recompute_granularity == 'selective'
    )
    print(f"INFO: No Dropout: {args.no_dropout}", flush=True)

    activation_memory = (
        # transformer layer
        2 * s * b * h  # LayerNorm
        + (
            # attention
            2 * s * b * h  # x -> Q, K, V
            + (2 * s * b * h)  # Q * K^T -> attention scores (Q)
            + (2 * s * b * h) * (k / a)  # Q * K^T -> attention scores (K) (grouped query attention)
            + (  # Q*K^T -> softmax(Q*K^T)
                0 if (
                    args.selective_activation_recomputation or args.use_flash_attn
                ) else (2 * b * s * s * a)
            )
            + (  # softmax(Q*K^T) -> Dropout
                0 if args.no_dropout else 0 if (
                    args.selective_activation_recomputation or args.use_flash_attn
                ) else (1 * b * s * s * a)
            )
            + ((2 * b * s * h) * (k / a))  # attention scores * V -> attention output (V) (grouped query attention)
            + (  # Dropout(softmax(Q*K^T)) * V -> attention output (Dropout)
                0 if (
                    args.selective_activation_recomputation or args.use_flash_attn
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
        + (0 if args.no_dropout else (b * s * h))  # MLP -> Dropout
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_GIGA_BYTE / args.tensor_model_parallel_size} GB"
        )
    activation_memory = activation_memory * args.num_layers / args.tensor_model_parallel_size

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        # 8 bytes (int64)
        8 * s * b * h * args.pipeline_model_parallel_size
    ) / args.tensor_model_parallel_size

    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += 0 if args.no_dropout else (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            # lm-head cross entropy (FP32)
            # output layer (layer norm) + output layer (linear)
            4 * s * b * h * (1 + args.padded_vocab_size / h)
        ) / args.tensor_model_parallel_size

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory


def report_theoretical_memory(args, num_microbatches=None, verbose=True):
    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_GIGA_BYTE
    )

    # Formulae here assume sequence parallelism and selective activation recomputation.
    if not (
        args.recompute_granularity == 'selective' or args.use_flash_attn is True
    ):
        print(
            f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory} GB"
        )
        return

    activation_memory = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_GIGA_BYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory} GB, "
        f"activation={activation_memory} GB, total={total_memory} GB\n"
    )
