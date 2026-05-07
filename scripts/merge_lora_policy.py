#!/usr/bin/env python
"""Merge a PEFT LoRA causal-LM adapter into its base model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_dtype(value: str):
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def parse_device_map(value: str):
    if value == "auto":
        return "auto"
    if value == "cpu":
        return {"": "cpu"}
    return {"": value}


def disable_broken_bitsandbytes_dispatch() -> None:
    try:
        import peft.import_utils as import_utils
        import peft.tuners.lora.model as lora_model

        import_utils.is_bnb_available = lambda: False
        import_utils.is_bnb_4bit_available = lambda: False
        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
    except Exception:
        pass


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    base_model = Path(args.base_model)
    adapter = Path(args.adapter)
    output = Path(args.output)
    adapter_file = adapter / "adapter_model.safetensors"
    if not base_model.exists():
        raise FileNotFoundError(f"base model does not exist: {base_model}")
    if not adapter_file.exists():
        raise FileNotFoundError(f"adapter safetensors does not exist: {adapter_file}")

    prepare_output(output, args.overwrite)
    disable_broken_bitsandbytes_dispatch()

    from peft import PeftModel

    print(f"Loading base model: {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=parse_dtype(args.torch_dtype),
        device_map=parse_device_map(args.device_map),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Loading adapter: {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    print("Merging LoRA weights", flush=True)
    model = model.merge_and_unload()
    model.config.use_cache = True

    lora_keys = [name for name, _ in model.named_parameters() if "lora_" in name]
    if lora_keys:
        raise RuntimeError(f"LoRA parameters still present after merge: {lora_keys[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Saving merged model: {output}", flush=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output)
    (output / "merge_info.json").write_text(
        json.dumps(
            {
                "base_model": str(base_model.resolve()),
                "adapter": str(adapter.resolve()),
                "output": str(output.resolve()),
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Merge complete", flush=True)


if __name__ == "__main__":
    main()
