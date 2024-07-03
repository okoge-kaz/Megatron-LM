import pytest
import torch
import torch.nn.functional as F

from transformers import LlamaForCausalLM

HF_LLAMA_7B_PATH = "/home/ext_kazuki_fujii_rio_gsic_titech/hf_checkpoints/Llama-2-7b-hf"
MEGATRON_LLAMA_7B_PATH = "/home/ext_kazuki_fujii_rio_gsic_titech/checkpoints/hf-to-megatron/Llama-2-7b/tp1-pp1"
CONVERTED_HF_LLAMA_7B_PATH = "/home/ext_kazuki_fujii_rio_gsic_titech/checkpoints/megatron-to-hf/Llama-2-7b-hf"


# Generate next n tokens function
def generate_next_tokens(model, tokenizer, inputs, num_tokens):
    input_ids = inputs['input_ids']
    for _ in range(num_tokens):
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=1)
        new_token = outputs[0, -1].unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, new_token], dim=-1)
    return input_ids


@pytest.mark.parametrize("hf_path, megatron_path, converted_hf_path", [(HF_LLAMA_7B_PATH, MEGATRON_LLAMA_7B_PATH, CONVERTED_HF_LLAMA_7B_PATH)])
def test_llama_2_hf_megatron_decoding(hf_path, megatron_path, converted_hf_path):
    model1 = LlamaForCausalLM.from_pretrained(hf_path)
    model2 = LlamaForCausalLM.from_pretrained(converted_hf_path)
    print('\n\nGreedy decoding test start\n', flush=True)

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    import random
    import numpy as np
    import torch

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    from megatron.core.models.gpt import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    config = TransformerConfig(
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_layers=model1.config.num_hidden_layers,  # type: ignore
        hidden_size=model1.config.hidden_size,  # type: ignore
        ffn_hidden_size=model1.config.intermediate_size,  # type: ignore
        num_attention_heads=model1.config.num_attention_heads,  # type: ignore
        num_query_groups=model1.config.num_key_value_heads,  # type: ignore
        layernorm_epsilon=model1.config.rms_norm_eps,  # type: ignore
        add_bias_linear=False,
        add_qkv_bias=False,
        activation_func=F.silu,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        normalization="RMSNorm",  # type: ignore
    )
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    model_mcore = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=model1.config.vocab_size,  # 32,000: Llama-2 default  # type: ignore
        max_sequence_length=model1.config.max_position_embeddings,  # 4096: Llama-2 default  # type: ignore
        pre_process=True,
        post_process=True,
        share_embeddings_and_output_weights=model1.config.tie_word_embeddings,  # False: Llama-2 default  # type: ignore
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,  # Llama-2 default
    )
    mcore_path = megatron_path + "/iter_0000001" + "/mp_rank_00" + "/model_optim_rng.pt"
    state_dict = torch.load(mcore_path)
    model_mcore.load_state_dict(state_dict["model"], strict=True)
    print("Loaded Megatron model", flush=True)

    torch.set_printoptions(threshold=10_000)

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(hf_path)
    input_text = "Tokyo is the capital of"
    inputs = tokenizer(input_text, return_tensors='pt')
    N_TOKENS = 20

    # Predict next n tokens for HF model
    hf_generated_ids = generate_next_tokens(
        model=model1,
        tokenizer=tokenizer,
        inputs=inputs,
        num_tokens=N_TOKENS,
    )
    hf_output_text = tokenizer.decode(hf_generated_ids[0], skip_special_tokens=True)

    # Predict next n tokens for Converted HF model
    converted_hf_generated_ids = generate_next_tokens(
        model=model2,
        tokenizer=tokenizer,
        inputs=inputs,
        num_tokens=N_TOKENS,
    )
    converted_hf_output_text = tokenizer.decode(converted_hf_generated_ids[0], skip_special_tokens=True)

    # Prepare input for Megatron model
    data = tokenizer.encode(input_text, return_tensors="pt")
    sequence_length = data.size(1)
    micro_batch_size = 1
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
    attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()  # type: ignore

    # Generate next n tokens for Megatron model
    mcore_generated_ids = input_ids.clone()
    for _ in range(N_TOKENS + sequence_length):
        with torch.no_grad():
            mcore_logits = model_mcore.forward(  # type: ignore
                input_ids=mcore_generated_ids,
                position_ids=position_ids,
                attention_mask=attention_mask
            )
        next_token = torch.argmax(torch.softmax(mcore_logits[:, -1, :], dim=-1), dim=-1).unsqueeze(1)
        mcore_generated_ids = torch.cat([mcore_generated_ids, next_token], dim=-1)
        position_ids = torch.arange(0, mcore_generated_ids.size(1)).unsqueeze(0).cuda()
        attention_mask = torch.ones(  # type: ignore
            (1, 1, mcore_generated_ids.size(1), mcore_generated_ids.size(1)), dtype=bool  # type: ignore
        ).cuda()

    mcore_output_texts = tokenizer.decode(mcore_generated_ids[0], skip_special_tokens=True)

    # Compare outputs
    print(f"HF Output:           {hf_output_text}", flush=True)
    print(f"Converted HF Output: {converted_hf_output_text}", flush=True)
    print(f"Megatron Output:     {mcore_output_texts}", flush=True)

    # Check if the outputs are the same
    assert hf_output_text == converted_hf_output_text == mcore_output_texts, "Outputs are not the same"
    print("All outputs are the same.")
