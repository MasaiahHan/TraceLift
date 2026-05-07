import argparse
import json
import os

from transformers import AutoConfig, AutoTokenizer

from .modeling_reward import DEFAULT_DIMENSION_NAMES, DEFAULT_DIMENSION_WEIGHTS, Qwen2ForReasonRewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize a Qwen2 reward-model checkpoint")
    parser.add_argument("--base_model", required=True, help="Local Qwen2.5 base/instruct model path")
    parser.add_argument("--output_dir", required=True, help="Output RM checkpoint directory")
    parser.add_argument(
        "--dimension_names",
        default=",".join(DEFAULT_DIMENSION_NAMES),
        help="Comma-separated rubric dimension names",
    )
    parser.add_argument(
        "--dimension_weights",
        default=",".join(str(x) for x in DEFAULT_DIMENSION_WEIGHTS),
        help="Comma-separated rubric dimension weights",
    )
    parser.add_argument("--disable_total_head", action="store_true")
    return parser.parse_args()


def split_csv(raw, cast=str):
    values = [cast(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("csv argument cannot be empty")
    return values


def main():
    args = parse_args()

    dimension_names = split_csv(args.dimension_names, cast=str)
    dimension_weights = split_csv(args.dimension_weights, cast=float)
    if len(dimension_names) != len(dimension_weights):
        raise ValueError("dimension_names and dimension_weights length mismatch")

    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=False)
    config.rm_dimension_names = dimension_names
    config.rm_dimension_weights = dimension_weights
    config.rm_num_labels = 5
    config.rm_use_total_head = not args.disable_total_head

    model = Qwen2ForReasonRewardModel.from_pretrained(
        args.base_model,
        config=config,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.base_model,
        "dimension_names": dimension_names,
        "dimension_weights": dimension_weights,
        "rm_num_labels": 5,
        "rm_use_total_head": not args.disable_total_head,
    }
    with open(os.path.join(args.output_dir, "rm_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
