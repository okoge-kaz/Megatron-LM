import sys
sys.path.append("/home/ext_kazuki_fujii_rio_gsic_titech/src/Megatron-LM")

from megatron.training.tokenizer.tokenizer import _Llama2Tokenizer

LLM_JP_TOKENIZER = "/home/ext_kazuki_fujii_rio_gsic_titech/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model"

tokenizer = _Llama2Tokenizer(model_file=LLM_JP_TOKENIZER)
encoded_tokens = tokenizer.tokenize(s="サンプルテキスト")
print(f"Encoded tokens: {encoded_tokens}", flush=True)

decoded_tokens: list = []
for token in encoded_tokens:
    decoded_token = tokenizer.detokenize(token)
    decoded_tokens.append(decoded_token)

print(f"Decoded tokens: {decoded_tokens}", flush=True)

print(f"bos_id={tokenizer.bos_id}, eos_id={tokenizer.eos_id}, bos_token={tokenizer.detokenize(tokenizer.bos_id)}, eos_token={tokenizer.detokenize(tokenizer.eos_id)}", flush=True)
