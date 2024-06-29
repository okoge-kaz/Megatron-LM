from transformers import LlamaForCausalLM
import torch
import argparse


def compare_state_dicts(original_state_dict, converted_state_dict):
    differences = {}

    for key in original_state_dict.keys():
        if key not in converted_state_dict:
            differences[key] = ("Only in original state_dict", )
            continue

        original_tensor = original_state_dict[key]
        converted_tensor = converted_state_dict[key]

        if not torch.equal(original_tensor, converted_tensor):
            # それぞれのテンソルの違いを計算
            diff = original_tensor - converted_tensor
            differences[key] = (original_tensor.numpy(), converted_tensor.numpy(), diff.numpy())

    for key in converted_state_dict.keys():
        if key not in original_state_dict:
            differences[key] = ("Only in converted state_dict", )

    return differences


parser = argparse.ArgumentParser()
parser.add_argument("--base-hf-model-path", type=str, required=True)
parser.add_argument("--converted-hf-model-path", type=str, required=True)
args = parser.parse_args()

# モデルをロード
original_model = LlamaForCausalLM.from_pretrained(
    args.base_hf_model_path,
    device_map="cpu"
)
converted_model = LlamaForCausalLM.from_pretrained(
    args.converted_hf_model_path,
    device_map="cpu"
)

# state_dictを取得
original_state_dict = original_model.state_dict()  # type: ignore
converted_state_dict = converted_model.state_dict()  # type: ignore

# state_dictの差分を比較
diffs = compare_state_dicts(
    original_state_dict=original_state_dict,
    converted_state_dict=converted_state_dict
)

torch.set_printoptions(threshold=10000, precision=10)

for key, values in diffs.items():
    print(f"Key: {key}")
    if len(values) == 3:
        original_value, converted_value, diff = values
        print(f"  original Value (Shape {original_value.shape}):\n{original_value}")
        print(f"  converted Value (Shape {converted_value.shape}):\n{converted_value}")
        print(f"  Difference (Shape {diff.shape}):\n{diff}")
    else:
        print(f"  {values[0]}")
