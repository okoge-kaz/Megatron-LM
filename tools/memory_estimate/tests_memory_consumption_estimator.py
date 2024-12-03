import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from megatron.training.theoretical_memory_usage import (
    compute_weight_and_optimizer_memory,
    compute_activation_memory,
    NUM_BYTES_IN_GIGA_BYTE,
)


import pytest

# Llama-3.1-8B
@pytest.mark.parametrize(
    "args, expected_weight_and_optimizer_memory, expected_activation_memory",
    [
        # (TP, CP, PP, MBS, WorldSize) = (2, 1, 1, 1, 4) (seq_length=8192)
        (
            argparse.Namespace(
                hidden_size=4096,
                ffn_hidden_size=14336,
                seq_length=8192,
                num_layers=32,
                num_attention_heads=32,
                group_query_attention=True,
                swiglu=True,
                num_query_groups=8,
                padded_vocab_size=128256,
                untie_embeddings_and_output_weights=True,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                use_flash_attn=True,
                tensor_model_parallel_size=2,
                context_parallel_size=1,
                pipeline_model_parallel_size=1,
                micro_batch_size=1,
                data_parallel_size=2,
                use_distributed_optimizer=True,
                accumulate_allreduce_grads_in_fp32=True,
                kv_channels=4096//32,
                recompute_granularity=None,
                virtual_pipeline_model_parallel_size=None,
                num_experts=None,
            ),
            44.87260437011719,
            22.64453125,
        ),
        # (TP, CP, PP, MBS, WorldSize) = (1, 2, 1, 1, 4) (seq_length=8192)
        (
            argparse.Namespace(
                hidden_size=4096,
                ffn_hidden_size=14336,
                seq_length=8192,
                num_layers=32,
                num_attention_heads=32,
                group_query_attention=True,
                swiglu=True,
                num_query_groups=8,
                padded_vocab_size=128256,
                untie_embeddings_and_output_weights=True,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                use_flash_attn=True,
                tensor_model_parallel_size=1,
                context_parallel_size=2,
                pipeline_model_parallel_size=1,
                micro_batch_size=1,
                data_parallel_size=2,
                use_distributed_optimizer=True,
                accumulate_allreduce_grads_in_fp32=True,
                kv_channels=4096//32,
                recompute_granularity=None,
                virtual_pipeline_model_parallel_size=None,
                num_experts=None,
            ),
            67.30887222290039,
            22.64453125,
        ),
        # (TP, CP, PP, MBS, WorldSize) = (2, 1, 2, 1, 128) (seq_length=8192)
        (
            argparse.Namespace(
                hidden_size=4096,
                ffn_hidden_size=14336,
                seq_length=8192,
                num_layers=32,
                num_attention_heads=32,
                group_query_attention=True,
                swiglu=True,
                num_query_groups=8,
                padded_vocab_size=128256,
                untie_embeddings_and_output_weights=True,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                use_flash_attn=True,
                tensor_model_parallel_size=2,
                context_parallel_size=1,
                pipeline_model_parallel_size=2,
                micro_batch_size=1,
                data_parallel_size=32,
                use_distributed_optimizer=True,
                accumulate_allreduce_grads_in_fp32=True,
                kv_channels=4096//32,
                recompute_granularity=None,
                virtual_pipeline_model_parallel_size=None,
                num_experts=None,
            ),
            11.919273376464844,
            20.75,
        ),
        # (TP, CP, PP, MBS, WorldSize) = (2, 2, 1, 2, 128) (seq_length=8192)
        (
            argparse.Namespace(
                hidden_size=4096,
                ffn_hidden_size=14336,
                seq_length=8192,
                num_layers=32,
                num_attention_heads=32,
                group_query_attention=True,
                swiglu=True,
                num_query_groups=8,
                padded_vocab_size=128256,
                untie_embeddings_and_output_weights=True,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                use_flash_attn=True,
                tensor_model_parallel_size=2,
                context_parallel_size=2,
                pipeline_model_parallel_size=1,
                micro_batch_size=2,
                data_parallel_size=32,
                use_distributed_optimizer=True,
                accumulate_allreduce_grads_in_fp32=True,
                kv_channels=4096//32,
                recompute_granularity=None,
                virtual_pipeline_model_parallel_size=None,
                num_experts=None,
            ),
            23.137436628341675,
            22.64453125,
        ),
    ]
)
def test_theoretical_memory_usage(
    args, expected_weight_and_optimizer_memory, expected_activation_memory
):
    weight_and_optimizer_memory = compute_weight_and_optimizer_memory(
        args=args,
    ) / NUM_BYTES_IN_GIGA_BYTE
    assert weight_and_optimizer_memory == expected_weight_and_optimizer_memory

    activation_memory = compute_activation_memory(
        args=args,
        num_microbatches=None,
    ) / NUM_BYTES_IN_GIGA_BYTE
    assert activation_memory == expected_activation_memory
