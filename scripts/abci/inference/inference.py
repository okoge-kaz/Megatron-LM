from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoModel Inference")

    parser.add_argument("--hf-model-path", type=str, default=None, help="huggingface checkpoint path")
    parser.add_argument("--hf-tokenizer-path", type=str, default=None, help="huggingface tokenizer path")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--limit-inference-case", type=int, default=None)

    args = parser.parse_args()
    return args


def load_jsonl(file_path: str) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main() -> None:
    # argument parse
    args = parse_args()

    # load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.hf_model_path,
    )
    if torch.cuda.is_available():
        model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.hf_tokenizer_path or args.hf_model_path,
    )
    input_datasets: list[str] = ["United States of America", "東京は"]

    # inference
    with torch.no_grad():
        for index, text in enumerate(input_datasets):

            input_ids: torch.Tensor = tokenizer.encode(
                text, return_tensors="pt"  # type: ignore
            )

            output_ids = []
            for _ in range(args.num_samples):
                ids = model.generate(
                    input_ids.to(model.device),
                    max_length=256,
                    temperature=0.99,
                    top_p=0.95,
                )
                output_ids.append(ids)

            decoded_outputs = [tokenizer.decode(ids[0][len(input_ids[0]):], skip_special_tokens=True) for ids in output_ids]

            print(f"{index}: {decoded_outputs}")


if __name__ == "__main__":
    main()
