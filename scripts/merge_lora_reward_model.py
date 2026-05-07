#!/usr/bin/env python
"""Merge a PEFT LoRA ReasonReward adapter into its Qwen2 base model.

The reward heads are saved as PEFT ``modules_to_save``. This script verifies
those heads are present in the adapter, merges LoRA weights, confirms the
unwrapped reward heads match the adapter copy, and saves a regular
Qwen2ForReasonRewardModel checkpoint.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import safe_open
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasonrm.modeling_reward import (  # noqa: E402
    DEFAULT_DIMENSION_NAMES,
    DEFAULT_DIMENSION_WEIGHTS,
    Qwen2ForReasonRewardModel,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a ReasonReward LoRA adapter into a base Qwen2 checkpoint."
    )
    parser.add_argument("--base-model", required=True, help="Base Qwen2 model path.")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory.")
    parser.add_argument("--output", required=True, help="Merged checkpoint output directory.")
    parser.add_argument(
        "--dimension-names",
        default=",".join(DEFAULT_DIMENSION_NAMES),
        help="Comma-separated reward dimension names.",
    )
    parser.add_argument(
        "--dimension-weights",
        default=",".join(str(x) for x in DEFAULT_DIMENSION_WEIGHTS),
        help="Comma-separated reward dimension weights.",
    )
    parser.add_argument("--rm-num-labels", type=int, default=5)
    parser.add_argument("--no-total-head", action="store_true", help="Disable total_head.")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
        help="dtype used to load and save the merged model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device_map. Use "auto", "cpu", or a device like "cuda:0".',
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory.")
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def split_float_csv(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_dtype(value):
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def parse_device_map(value):
    if value == "auto":
        return "auto"
    if value == "cpu":
        return {"": "cpu"}
    return {"": value}


def disable_peft_bitsandbytes_dispatch():
    """Avoid importing broken bitsandbytes paths for plain LoRA checkpoints."""
    try:
        import peft.tuners.lora.model as lora_model
    except ImportError:
        return

    lora_model.is_bnb_available = lambda: False
    lora_model.is_bnb_4bit_available = lambda: False


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists():
        if not overwrite:
            if any(output_dir.iterdir()):
                raise FileExistsError(
                    f"output directory already exists and is not empty: {output_dir}"
                )
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def adapter_reward_tensors(adapter_file):
    reward_tensors = {}
    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if "dimension_heads" in key or "total_head" in key:
                reward_tensors[key] = handle.get_tensor(key)
    return reward_tensors


def model_reward_tensors(model, num_dimensions, use_total_head):
    tensors = {}
    for idx in range(num_dimensions):
        head = model.dimension_heads[idx]
        tensors[f"base_model.model.dimension_heads.{idx}.weight"] = head.weight.detach().cpu()
        tensors[f"base_model.model.dimension_heads.{idx}.bias"] = head.bias.detach().cpu()
    if use_total_head:
        tensors["base_model.model.total_head.weight"] = model.total_head.weight.detach().cpu()
        tensors["base_model.model.total_head.bias"] = model.total_head.bias.detach().cpu()
    return tensors


def verify_reward_heads(adapter_tensors, merged_tensors):
    missing = sorted(set(merged_tensors) - set(adapter_tensors))
    if missing:
        raise RuntimeError(f"adapter is missing reward head tensors: {missing}")

    mismatched = []
    for key, merged_value in merged_tensors.items():
        adapter_value = adapter_tensors[key].to(dtype=merged_value.dtype)
        if not torch.allclose(merged_value, adapter_value, atol=0.0, rtol=0.0):
            mismatched.append(key)
    if mismatched:
        raise RuntimeError(f"merged reward heads differ from adapter tensors: {mismatched}")


def verify_no_lora_modules(model):
    lora_keys = [name for name, _ in model.named_parameters() if "lora_" in name]
    if lora_keys:
        raise RuntimeError(f"LoRA parameters still present after merge: {lora_keys[:10]}")


def write_merge_info(output_dir, args, dimension_names, dimension_weights):
    info = {
        "base_model": str(Path(args.base_model).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "output": str(output_dir.resolve()),
        "dimension_names": dimension_names,
        "dimension_weights": dimension_weights,
        "rm_num_labels": args.rm_num_labels,
        "rm_use_total_head": not args.no_total_head,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
    }
    (output_dir / "merge_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    base_model = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    output_dir = Path(args.output)
    adapter_file = adapter_dir / "adapter_model.safetensors"

    if not base_model.exists():
        raise FileNotFoundError(f"base model does not exist: {base_model}")
    if not adapter_file.exists():
        raise FileNotFoundError(f"adapter safetensors does not exist: {adapter_file}")

    prepare_output_dir(output_dir, args.overwrite)

    dimension_names = split_csv(args.dimension_names)
    dimension_weights = split_float_csv(args.dimension_weights)
    if len(dimension_names) != len(dimension_weights):
        raise ValueError("--dimension-names and --dimension-weights must have the same length")

    adapter_tensors = adapter_reward_tensors(adapter_file)
    expected_head_count = len(dimension_names) * 2 + (2 if not args.no_total_head else 0)
    if len(adapter_tensors) != expected_head_count:
        raise RuntimeError(
            f"expected {expected_head_count} reward head tensors, found {len(adapter_tensors)}"
        )

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=False)
    config.rm_dimension_names = dimension_names
    config.rm_dimension_weights = dimension_weights
    config.rm_num_labels = args.rm_num_labels
    config.rm_use_total_head = not args.no_total_head
    config.architectures = ["Qwen2ForReasonRewardModel"]
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id

    disable_peft_bitsandbytes_dispatch()
    from peft import PeftModel

    print(f"Loading base model: {base_model}", flush=True)
    model = Qwen2ForReasonRewardModel.from_pretrained(
        base_model,
        config=config,
        torch_dtype=parse_dtype(args.torch_dtype),
        device_map=parse_device_map(args.device_map),
        low_cpu_mem_usage=True,
    )

    print(f"Loading adapter: {adapter_dir}", flush=True)
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)

    print("Merging LoRA weights", flush=True)
    merged_model = peft_model.merge_and_unload()
    merged_model.config.architectures = ["Qwen2ForReasonRewardModel"]
    merged_model.config.use_cache = False

    merged_tensors = model_reward_tensors(
        merged_model,
        num_dimensions=len(dimension_names),
        use_total_head=not args.no_total_head,
    )
    verify_reward_heads(adapter_tensors, merged_tensors)
    verify_no_lora_modules(merged_model)

    print(f"Saving merged model: {output_dir}", flush=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    write_merge_info(output_dir, args, dimension_names, dimension_weights)
    print("Merge complete", flush=True)


if __name__ == "__main__":
    main()
