#!/usr/bin/env python
"""Lightweight GRPO-style training for reason generation on code tasks.

This is intentionally small and explicit because the current RRM environment
ships TRL without GRPOTrainer. The policy generates reasoning; an external
executor turns problem+reasoning into code; executable tests provide reward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasonrm.code_oracle import evaluate_completion, iter_jsonl, load_oracle_jsonl, select_oracle_tests
from tracelift.reward_utils import exec_reward as shaped_exec_reward


def disable_broken_bitsandbytes_dispatch() -> None:
    """Force PEFT LoRA to use regular torch Linear dispatch.

    The cluster image has a bitsandbytes package without the CUDA 12.4 binary.
    PEFT only needs bnb dispatch for quantized models; importing it here breaks
    ordinary bf16 LoRA unless we override the availability probe.
    """
    try:
        import peft.import_utils as import_utils
        import peft.tuners.lora.model as lora_model

        import_utils.is_bnb_available = lambda: False
        import_utils.is_bnb_4bit_available = lambda: False
        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(getattr(args, "local_rank", 0))))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return distributed, rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    if model.__class__.__name__ == "DeepSpeedEngine" and hasattr(model, "module"):
        return model.module
    return model


def reduce_mean(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(dist.get_world_size())
    return float(tensor.item())


def distributed_barrier(distributed: bool, local_rank: int) -> None:
    if not distributed:
        return
    if torch.cuda.is_available():
        try:
            dist.barrier(device_ids=[local_rank])
            return
        except TypeError:
            pass
    dist.barrier()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def load_groups(path: str, oracle_map: Dict[str, Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    groups = []
    for row in iter_jsonl(path):
        if str(row.get("problem_id")) not in oracle_map:
            continue
        if not select_oracle_tests(oracle_map[str(row.get("problem_id"))], ["reward"], 1):
            continue
        groups.append(row)
        if max_items > 0 and len(groups) >= max_items:
            break
    if not groups:
        raise ValueError("no train groups with oracle tests found")
    return groups


def truncate_chars(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0].rstrip() + "\n...[truncated]"


def build_reason_prompt(problem: str, max_problem_chars: int) -> str:
    return (
        "You are solving a programming contest problem.\n"
        "Write concise reasoning that helps derive a correct Python 3 solution.\n"
        "Do not write final code. Do not include Markdown.\n\n"
        "Problem:\n{}\n\nReasoning:\n".format(truncate_chars(problem, max_problem_chars))
    )


def build_executor_messages(problem: str, reasoning: str, max_problem_chars: int, max_reason_chars: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You solve programming contest problems. Output a complete Python 3 program only.",
        },
        {
            "role": "user",
            "content": (
                "Use the provided reasoning as guidance, then write a complete Python 3 program "
                "that solves the problem. Output code only. Do not use Markdown, do not explain.\n\n"
                "Problem:\n{}\n\nProvided reasoning:\n{}"
            ).format(truncate_chars(problem, max_problem_chars), truncate_chars(reasoning, max_reason_chars)),
        },
    ]


def build_no_reason_executor_messages(problem: str, max_problem_chars: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You solve programming contest problems. Output a complete Python 3 program only.",
        },
        {
            "role": "user",
            "content": (
                "Write a complete Python 3 program that solves the problem. Output code only. "
                "Do not use Markdown, do not explain.\n\nProblem:\n{}"
            ).format(truncate_chars(problem, max_problem_chars)),
        },
    ]


def render_completion_prompt(messages: Sequence[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role == "system":
            parts.append("Instruction:\n{}".format(content))
        else:
            parts.append(content)
    return "\n\n".join(parts).strip() + "\n\nFinal Python 3 solution:\n"


def make_opener(no_proxy: bool) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({})) if no_proxy else urllib.request.build_opener()


def post_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    timeout: float,
) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", "Bearer {}".format(api_key))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} from {}: {}".format(exc.code, url, detail[:1000])) from exc
    return json.loads(raw)


def normalize_executor_url(base_url: str, api_type: str) -> str:
    base = base_url.rstrip("/")
    if api_type == "chat":
        return base if base.endswith("/chat/completions") else base + "/chat/completions"
    if api_type == "completions":
        return base if base.endswith("/completions") else base + "/completions"
    raise ValueError("unknown api type: {}".format(api_type))


def extract_completion(response: Dict[str, Any]) -> str:
    choice = response["choices"][0]
    if isinstance(choice.get("message"), dict):
        return str(choice["message"].get("content") or "")
    return str(choice.get("text") or "")


def call_executor(
    opener: urllib.request.OpenerDirector,
    executor_url: str,
    api_type: str,
    model: str,
    api_key: Optional[str],
    messages: Sequence[Dict[str, str]],
    extra_payload: Dict[str, Any],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    if api_type == "chat":
        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    else:
        payload = {
            "model": model,
            "prompt": render_completion_prompt(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    payload.update(extra_payload)
    return extract_completion(post_json(opener, executor_url, payload, api_key, timeout))


def parse_rm_scores(response: Any, expected: int) -> List[float]:
    if isinstance(response, dict):
        for key in ("scores", "results", "outputs", "data"):
            if key in response:
                response = response[key]
                break
    if isinstance(response, (int, float)):
        response = [response]
    if not isinstance(response, list):
        raise RuntimeError("unexpected RM response: {}".format(response))
    scores = []
    for item in response:
        if isinstance(item, (int, float)):
            scores.append(float(item))
        elif isinstance(item, dict):
            for key in ("total_score", "score", "reward", "total", "value"):
                if key in item:
                    scores.append(float(item[key]))
                    break
            else:
                raise RuntimeError("cannot find score in RM item: {}".format(item))
    if len(scores) != expected:
        raise RuntimeError("RM returned {} scores for {} inputs".format(len(scores), expected))
    return scores


def call_rm(
    opener: urllib.request.OpenerDirector,
    rm_score_url: str,
    problem: str,
    reasoning: str,
    rm_api_key: Optional[str],
    rm_max_length: int,
    timeout: float,
) -> float:
    payload = {
        "items": [{"problem": problem, "reasoning": reasoning}],
        "max_length": rm_max_length,
        "batch_size": 1,
    }
    return parse_rm_scores(post_json(opener, rm_score_url, payload, rm_api_key, timeout), 1)[0]


def build_code_judge_prompt(
    problem: str,
    reasoning: str,
    max_problem_chars: int,
    max_reason_chars: int,
    final_code: Optional[str] = None,
    max_code_chars: int = 2500,
) -> str:
    if final_code is None:
        final_section = (
            "Final code/result:\n"
            "Not provided. For action_support, judge whether the reasoning gives enough actionable support "
            "for a correct implementation, instead of comparing against a specific final program."
        )
    else:
        final_section = "Final code/result:\n{}".format(truncate_chars(final_code, max_code_chars))

    return """You are judging the quality of reasoning for a code task.

You will be given:
- a coding problem
- an answer-free reasoning text
- optionally, the final code / result

Score the reasoning itself instead of the final answer alone.

Rubric dimensions (0-10, where 10 is best):
1. task_understanding: Does the reasoning correctly understand the task and constraints?
2. plan_quality: Is the proposed approach sensible and appropriate?
3. step_coherence: Are the reasoning steps logically connected and internally consistent?
4. action_support: Does the reasoning actually support a correct final code/result?
5. non_leakage: Does the reasoning avoid simply dumping the final implementation or over-leaking the final action?

Scoring rules:
- Each dimension must be scored from 0 to 10.
- Decimal scores are allowed.
- Use at most one decimal place when needed.
- 10 means excellent reasoning quality on that dimension.
- 0 means the reasoning completely fails on that dimension.

Then produce:
- rubric_score: overall score from 0 to 10 (decimal allowed)
- rubric_label: one of [strong, acceptable, weak, bad]
- rubric_reason: short explanation in 1-3 sentences

Return STRICT JSON only, with this exact schema:
{{
  "task_understanding": 0.0,
  "plan_quality": 0.0,
  "step_coherence": 0.0,
  "action_support": 0.0,
  "non_leakage": 0.0,
  "rubric_score": 0.0,
  "rubric_label": "bad",
  "rubric_reason": "..."
}}

Problem:
{problem}

Reasoning:
{reasoning}

{final_section}
""".format(
        problem=truncate_chars(problem, max_problem_chars),
        reasoning=truncate_chars(reasoning, max_reason_chars),
        final_section=final_section,
    )


def parse_judge_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start >= 0:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        break
    raise RuntimeError("judge did not return a JSON object: {}".format(text[:1000]))


def parse_judge_score(text: str, score_scale: float = 10.0) -> float:
    parsed = parse_judge_json_object(text)
    raw_score = None
    for key in ("rubric_score", "total_score", "score", "reward", "total", "value"):
        if key in parsed:
            raw_score = parsed[key]
            break
    if raw_score is None:
        raise RuntimeError("cannot find judge score in response: {}".format(parsed))
    score = float(raw_score)
    if score_scale > 0:
        score /= float(score_scale)
    return clamp(score, 0.0, 1.0)


def render_judge_completion_prompt(system_prompt: str, user_prompt: str) -> str:
    return "System:\n{}\n\nUser:\n{}\n\nReturn JSON:\n".format(system_prompt.strip(), user_prompt.strip())


def call_llm_judge_rm(
    opener: urllib.request.OpenerDirector,
    judge_url: str,
    api_type: str,
    model: str,
    api_key: Optional[str],
    problem: str,
    reasoning: str,
    extra_payload: Dict[str, Any],
    max_problem_chars: int,
    max_reason_chars: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    final_code: Optional[str] = None,
    max_code_chars: int = 2500,
    score_scale: float = 10.0,
) -> float:
    system_prompt = "You are a strict code-reasoning rubric judge. Return valid JSON only."
    user_prompt = build_code_judge_prompt(
        problem,
        reasoning,
        max_problem_chars=max_problem_chars,
        max_reason_chars=max_reason_chars,
        final_code=final_code,
        max_code_chars=max_code_chars,
    )
    if api_type == "chat":
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
    else:
        payload = {
            "model": model,
            "prompt": render_judge_completion_prompt(system_prompt, user_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    payload.update(extra_payload)
    return parse_judge_score(
        extract_completion(post_json(opener, judge_url, payload, api_key, timeout)),
        score_scale=score_scale,
    )


def warn_api_failure(args: argparse.Namespace, kind: str, exc: BaseException) -> None:
    key = "_{}_failures".format(kind)
    count = int(getattr(args, key, 0)) + 1
    setattr(args, key, count)
    if count <= getattr(args, "max_api_failure_warnings", 20):
        rank = os.environ.get("RANK", "0")
        print(
            "[rank {}] {} call failed #{}; using fallback reward: {}: {}".format(
                rank, kind, count, type(exc).__name__, str(exc)[:500]
            ),
            file=sys.stderr,
            flush=True,
        )


def binary_execution_success(execution: Dict[str, Any], fallback_reward: float = 0.0) -> float:
    if "execution_ok" in execution:
        return 1.0 if execution.get("execution_ok") else 0.0
    return 1.0 if fallback_reward >= 1.0 else 0.0


def parse_equal_success_coeffs(spec: str) -> Dict[int, float]:
    spec = (spec or "").strip()
    if not spec:
        return {}
    if spec.startswith("{"):
        raw = json.loads(spec)
        if not isinstance(raw, dict):
            raise ValueError("--equal_success_coeffs JSON must be an object")
        return {int(k): float(v) for k, v in raw.items()}
    coeffs: Dict[int, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("--equal_success_coeffs entries must look like count:coeff")
        key, value = item.split(":", 1)
        coeffs[int(key.strip())] = float(value.strip())
    return coeffs


def equal_success_coeff(success_count: int, repeats: int, coeffs: Optional[Dict[int, float]] = None) -> float:
    if success_count <= 0:
        return 0.0
    if coeffs and success_count in coeffs:
        return clamp(coeffs[success_count], 0.0, 1.0)
    if success_count >= repeats:
        return 1.0
    if repeats == 3 and success_count == 1:
        return 0.5
    return float(success_count) / float(repeats)


def reason_utility_coeff(
    with_reason_success_count: int,
    baseline_success_count: int,
    repeats: int,
    equal_coeffs: Optional[Dict[int, float]] = None,
) -> float:
    if with_reason_success_count > baseline_success_count:
        return 1.0
    if with_reason_success_count < baseline_success_count:
        return -1.0
    return equal_success_coeff(with_reason_success_count, repeats, equal_coeffs)


def combine_reason_utility_reward(
    primary_exec_reward: float,
    with_reason_success_count: int,
    baseline_success_count: int,
    repeats: int,
    rm_score: float,
    reason_weight: float,
    equal_coeffs: Optional[Dict[int, float]] = None,
) -> Tuple[float, float, float, float]:
    strict_uplift = clamp(
        (float(with_reason_success_count) - float(baseline_success_count)) / float(repeats),
        -1.0,
        1.0,
    )
    coeff = reason_utility_coeff(with_reason_success_count, baseline_success_count, repeats, equal_coeffs)
    reason_reward = clamp(rm_score, 0.0, 1.0) * coeff
    reward = clamp(primary_exec_reward + reason_weight * reason_reward, 0.0, 1.0)
    return reward, coeff, strict_uplift, reason_reward


def evaluate_reasoning_reward(
    group: Dict[str, Any],
    reasoning: str,
    oracle_map: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    opener: urllib.request.OpenerDirector,
) -> Dict[str, Any]:
    problem = str(group.get("problem") or "")
    oracle = oracle_map[str(group["problem_id"])]
    tests = select_oracle_tests(oracle, args.oracle_source_list, args.max_oracle_tests)
    if not tests:
        return {"reward": 0.0, "exec_reward": 0.0, "execution_ok": False, "error": "no_tests"}

    executor_url = normalize_executor_url(args.executor_base_url, args.executor_api_type)

    def run_executor(messages: Sequence[Dict[str, str]]) -> Tuple[Dict[str, Any], float, float]:
        try:
            code = call_executor(
                opener,
                executor_url,
                args.executor_api_type,
                args.executor_model,
                args.executor_api_key,
                messages,
                args.executor_extra_payload,
                args.executor_max_tokens,
                args.executor_temperature,
                args.api_timeout,
            )
            execution = evaluate_completion(code, tests, timeout_sec=args.exec_timeout, memory_mb=args.exec_memory_mb)
            return execution, shaped_exec_reward(execution, args.exec_reward_style), binary_execution_success(execution)
        except Exception as exc:
            warn_api_failure(args, "executor", exc)
            execution = {"execution_ok": False, "error": "{}: {}".format(type(exc).__name__, exc)}
            return execution, args.api_failure_reward, binary_execution_success({}, args.api_failure_reward)

    with_reason_messages = build_executor_messages(problem, reasoning, args.max_problem_chars, args.max_reason_chars)
    baseline_messages = build_no_reason_executor_messages(problem, args.max_problem_chars)

    primary_execution, primary_exec_reward, primary_success = run_executor(with_reason_messages)
    execution_results = [primary_execution]
    exec_reward = primary_exec_reward
    with_reason_reward = primary_exec_reward
    with_reason_success_rate = primary_success
    with_reason_success_count = None
    baseline_exec_reward = None
    baseline_success_rate = None
    baseline_success_count = None
    utility_coeff = None
    strict_uplift = None
    reward = exec_reward
    rm_score = None
    reason_reward = None
    equal_success_count = None
    if args.reward_mode == "reason_exec":
        coeff_with_rewards = []
        coeff_with_successes = []
        for _ in range(args.uplift_repeats):
            coeff_execution, coeff_reward, coeff_success = run_executor(with_reason_messages)
            execution_results.append(coeff_execution)
            coeff_with_rewards.append(coeff_reward)
            coeff_with_successes.append(coeff_success)

        baseline_rewards = []
        baseline_successes = []
        for _ in range(args.uplift_repeats):
            baseline_execution, baseline_reward, baseline_success = run_executor(baseline_messages)
            baseline_rewards.append(baseline_reward)
            baseline_successes.append(baseline_success)

        with_reason_reward = sum(coeff_with_rewards) / len(coeff_with_rewards)
        with_reason_success_count = int(sum(coeff_with_successes))
        with_reason_success_rate = sum(coeff_with_successes) / len(coeff_with_successes)
        baseline_exec_reward = sum(baseline_rewards) / len(baseline_rewards)
        baseline_success_count = int(sum(baseline_successes))
        baseline_success_rate = sum(baseline_successes) / len(baseline_successes)
        equal_success_count = with_reason_success_count if with_reason_success_count == baseline_success_count else None
        try:
            rm_score = call_rm(
                opener,
                args.rm_score_url,
                problem,
                reasoning,
                args.rm_api_key,
                args.rm_max_length,
                args.api_timeout,
            )
        except Exception as exc:
            warn_api_failure(args, "rm", exc)
            rm_score = args.rm_failure_score
        reward, utility_coeff, strict_uplift, reason_reward = combine_reason_utility_reward(
            primary_exec_reward,
            with_reason_success_count,
            baseline_success_count,
            args.uplift_repeats,
            float(rm_score),
            args.reason_weight,
            args.equal_success_coeff_map,
        )

    return {
        "reward": float(reward),
        "exec_reward": float(exec_reward),
        "primary_exec_reward": float(primary_exec_reward),
        "primary_success": float(primary_success),
        "with_reason_exec_reward": float(with_reason_reward),
        "with_reason_success_rate": float(with_reason_success_rate),
        "with_reason_success_count": with_reason_success_count,
        "baseline_exec_reward": baseline_exec_reward,
        "baseline_success_rate": baseline_success_rate,
        "baseline_success_count": baseline_success_count,
        "utility_coeff": utility_coeff,
        "strict_uplift": strict_uplift,
        "uplift": strict_uplift,
        "rm_score": rm_score,
        "reason_reward": reason_reward,
        "equal_success_count": equal_success_count,
        "execution_ok": bool(primary_execution.get("execution_ok")),
        "execution_passed": int(primary_execution.get("execution_passed") or int(primary_execution.get("execution_ok", False))),
        "execution_total": int(primary_execution.get("execution_total") or 1),
    }


def generate_reasonings(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tuple[List[str], List[List[int]], List[int]]:
    generation_model = unwrap_model(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        outputs = generation_model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = []
    token_ids = []
    for output in outputs:
        completion_ids = output[prompt_len:].tolist()
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
        token_ids.append(output.tolist())
    return completions, token_ids, [prompt_len for _ in token_ids]


def sequence_logprob(
    model: torch.nn.Module,
    input_ids: Sequence[int],
    prompt_len: int,
    device: torch.device,
) -> torch.Tensor:
    completion_len = len(input_ids) - prompt_len
    if completion_len <= 0:
        return torch.tensor(0.0, device=device)

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    try:
        outputs = model(ids, logits_to_keep=completion_len + 1)
        logits = outputs.logits
    except TypeError:
        logits = model(ids).logits

    labels = ids[:, prompt_len:]
    if logits.shape[1] == completion_len + 1:
        logits = logits[:, :-1, :]
    else:
        logits = logits[:, max(0, prompt_len - 1) : -1, :]
    if logits.shape[1] != labels.shape[1]:
        logits = logits[:, -labels.shape[1] :, :]

    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return log_probs.mean()


def save_checkpoint(model: torch.nn.Module, tokenizer: AutoTokenizer, output_dir: str, step: int) -> None:
    path = Path(output_dir) / "checkpoint-{}".format(step)
    path.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(path)
    tokenizer.save_pretrained(path)


def train(args: argparse.Namespace) -> None:
    if args.no_proxy:
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
    distributed, rank, local_rank, world_size, device = init_distributed(args)
    set_seed(args.seed + rank)
    args.oracle_source_list = [item.strip() for item in args.oracle_sources.split(",") if item.strip()]
    args.executor_extra_payload = json.loads(args.executor_extra_payload_json) if args.executor_extra_payload_json else {}
    if args.reward_mode == "reason_exec" and not args.rm_score_url:
        raise ValueError("--rm_score_url is required for reason_exec")
    if args.reward_mode == "reason_exec":
        args.uplift_exec_reward = True
        args.executor_repeats = 1
        args.baseline_repeats = int(args.uplift_repeats)
        if args.uplift_repeats <= 0:
            raise ValueError("--uplift_repeats must be positive")
        args.equal_success_coeff_map = parse_equal_success_coeffs(args.equal_success_coeffs)
        invalid_equal_counts = [
            count for count in args.equal_success_coeff_map if count < 0 or count > args.uplift_repeats
        ]
        if invalid_equal_counts:
            raise ValueError(
                "--equal_success_coeffs count(s) out of range for --uplift_repeats {}: {}".format(
                    args.uplift_repeats, sorted(invalid_equal_counts)
                )
            )
    else:
        args.equal_success_coeff_map = {}

    oracle_map = load_oracle_jsonl(args.oracle_file)
    groups = load_groups(args.train_file, oracle_map, args.max_train_items)
    rng = random.Random(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        device_map={"": local_rank if torch.cuda.is_available() else "cpu"},
    )
    model.config.use_cache = False
    if args.use_lora:
        disable_broken_bitsandbytes_dispatch()
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        if is_main_process(rank):
            model.print_trainable_parameters()

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, foreach=args.adamw_foreach)
    using_deepspeed = bool(args.deepspeed_config)
    if using_deepspeed:
        try:
            import deepspeed
        except Exception as exc:
            raise RuntimeError("--deepspeed_config requires deepspeed to be importable") from exc
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=trainable_params,
            config=args.deepspeed_config,
        )
    elif distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    opener = make_opener(args.no_proxy)
    if is_main_process(rank):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed, local_rank)
    metrics_path = Path(args.output_dir) / "metrics.jsonl"

    progress_bar = None
    steps: Iterable[int] = range(1, args.max_steps + 1)
    if is_main_process(rank) and not args.disable_tqdm and tqdm is not None:
        progress_bar = tqdm(
            steps,
            total=args.max_steps,
            dynamic_ncols=True,
            desc=f"{args.reward_mode} train",
        )
        steps = progress_bar

    for step in steps:
        model.eval()
        if args.sample_mode == "sequential":
            group = groups[((step - 1) * world_size + rank) % len(groups)]
        else:
            group = rng.choice(groups)
        prompt = build_reason_prompt(str(group.get("problem") or ""), args.max_problem_chars)
        completions, token_ids, prompt_lens = generate_reasonings(
            model,
            tokenizer,
            prompt,
            args.num_generations,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            device,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reward_infos = [
            evaluate_reasoning_reward(group, completion, oracle_map, args, opener)
            for completion in completions
        ]
        rewards = torch.tensor([info["reward"] for info in reward_infos], dtype=torch.float32, device=device)
        advantages = rewards - rewards.mean()
        std = rewards.std(unbiased=False)
        if float(std.item()) > 1e-6:
            advantages = advantages / (std + 1e-6)

        model.train()
        if using_deepspeed:
            model.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        loss_values = []
        num_sequences = max(1, len(token_ids))
        for index, (ids, prompt_len, advantage) in enumerate(zip(token_ids, prompt_lens, advantages)):
            sync_context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and index < num_sequences - 1
                else nullcontext()
            )
            with sync_context:
                logprob = sequence_logprob(model, ids, prompt_len, device)
                seq_loss = -advantage.detach() * logprob
                if using_deepspeed:
                    model.backward(seq_loss / num_sequences)
                else:
                    (seq_loss / num_sequences).backward()
                loss_values.append(seq_loss.detach().float())
        loss = torch.stack(loss_values).mean()
        if using_deepspeed:
            model.step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        metrics = {
            "step": step,
            "reward_mode": args.reward_mode,
            "uplift_repeats": args.uplift_repeats if args.reward_mode == "reason_exec" else None,
            "equal_success_coeffs": args.equal_success_coeff_map if args.reward_mode == "reason_exec" else None,
            "problem_id": group.get("problem_id"),
            "rank": rank,
            "world_size": world_size,
            "loss": float(loss.detach().float().cpu()),
            "reward_mean": float(rewards.mean().detach().cpu()),
            "reward_std": float(std.detach().cpu()),
            "exec_reward_mean": sum(info["exec_reward"] for info in reward_infos) / len(reward_infos),
            "with_reason_success_rate_mean": sum(info["with_reason_success_rate"] for info in reward_infos) / len(reward_infos),
            "baseline_success_rate_mean": None,
            "uplift_mean": None,
            "strict_uplift_mean": None,
            "utility_coeff_mean": None,
            "reason_reward_mean": None,
            "equal_success_rate": sum(1 for info in reward_infos if info.get("equal_success_count") is not None) / len(reward_infos),
            "equal_nonzero_success_rate": sum(
                1 for info in reward_infos if (info.get("equal_success_count") or 0) > 0
            )
            / len(reward_infos),
            "pass_rate": sum(1 for info in reward_infos if info["execution_ok"]) / len(reward_infos),
            "rm_score_mean": None,
            "time": time.time(),
        }
        baseline_rates = [info["baseline_success_rate"] for info in reward_infos if info.get("baseline_success_rate") is not None]
        if baseline_rates:
            metrics["baseline_success_rate_mean"] = sum(baseline_rates) / len(baseline_rates)
        uplifts = [info["uplift"] for info in reward_infos if info.get("uplift") is not None]
        if uplifts:
            metrics["uplift_mean"] = sum(uplifts) / len(uplifts)
            metrics["strict_uplift_mean"] = metrics["uplift_mean"]
        utility_coeffs = [info["utility_coeff"] for info in reward_infos if info.get("utility_coeff") is not None]
        if utility_coeffs:
            metrics["utility_coeff_mean"] = sum(utility_coeffs) / len(utility_coeffs)
        reason_rewards = [info["reason_reward"] for info in reward_infos if info.get("reason_reward") is not None]
        if reason_rewards:
            metrics["reason_reward_mean"] = sum(reason_rewards) / len(reason_rewards)
        rm_scores = [info["rm_score"] for info in reward_infos if info.get("rm_score") is not None]
        if rm_scores:
            metrics["rm_score_mean"] = sum(rm_scores) / len(rm_scores)
        for key in (
            "loss",
            "reward_mean",
            "reward_std",
            "exec_reward_mean",
            "with_reason_success_rate_mean",
            "pass_rate",
        ):
            metrics["global_" + key] = reduce_mean(float(metrics[key]), device, distributed)
        for key in (
            "baseline_success_rate_mean",
            "uplift_mean",
            "strict_uplift_mean",
            "utility_coeff_mean",
            "reason_reward_mean",
            "rm_score_mean",
        ):
            if metrics.get(key) is not None:
                metrics["global_" + key] = reduce_mean(float(metrics[key]), device, distributed)
            else:
                metrics["global_" + key] = None
        if is_main_process(rank):
            serialized_metrics = json.dumps(metrics, ensure_ascii=False)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(serialized_metrics + "\n")
            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{metrics['global_loss']:.3f}",
                    reward=f"{metrics['global_reward_mean']:.3f}",
                    pass_rate=f"{metrics['global_pass_rate']:.3f}",
                    refresh=False,
                )
                if args.print_metrics:
                    progress_bar.write(serialized_metrics)
            elif args.print_metrics:
                print(serialized_metrics, flush=True)

        if args.save_steps > 0 and step % args.save_steps == 0 and is_main_process(rank):
            save_checkpoint(model, tokenizer, args.output_dir, step)

    if is_main_process(rank) and not args.skip_final_save:
        save_checkpoint(model, tokenizer, args.output_dir, args.max_steps)
    if distributed:
        distributed_barrier(distributed, local_rank)
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", default="data/code_rm/prepared/train_groups_seed18_new_full.jsonl")
    parser.add_argument("--oracle_file", default="data/code_rm/code_test_oracles.full.statement.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reward_mode", choices=["exec_only", "reason_exec"], default="exec_only")
    parser.add_argument("--exec_reward_style", choices=["binary", "pass_rate", "coderl", "hybrid"], default="hybrid")
    parser.add_argument("--reason_weight", type=float, default=0.2)
    parser.add_argument("--reason_gate", choices=["none", "execution"], default="execution")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--sample_mode", choices=["sequential", "random"], default="random")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adamw_foreach", type=str2bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_problem_chars", type=int, default=3500)
    parser.add_argument("--max_reason_chars", type=int, default=2500)
    parser.add_argument("--oracle_sources", default="reward")
    parser.add_argument("--max_oracle_tests", type=int, default=8)
    parser.add_argument("--exec_timeout", type=float, default=2.0)
    parser.add_argument("--exec_memory_mb", type=int, default=1024)
    parser.add_argument("--executor_base_url", required=True)
    parser.add_argument("--executor_model", required=True)
    parser.add_argument("--executor_api_type", choices=["completions", "chat"], default="completions")
    parser.add_argument("--executor_api_key", default=os.getenv("EXECUTOR_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--executor_extra_payload_json", default=os.getenv("EXECUTOR_EXTRA_PAYLOAD_JSON"))
    parser.add_argument("--executor_repeats", type=int, default=1)
    parser.add_argument("--baseline_repeats", type=int, default=0)
    parser.add_argument("--uplift_repeats", type=int, default=3)
    parser.add_argument(
        "--equal_success_coeffs",
        default="",
        help="Tie rule for reason_exec when with_reason_success_count == baseline_success_count, "
        "as count:coeff pairs or JSON object. Example: 1:0.25,2:0.5,3:0.7,4:0.8,5:1",
    )
    parser.add_argument("--uplift_exec_reward", action="store_true")
    parser.add_argument("--executor_temperature", type=float, default=0.0)
    parser.add_argument("--executor_max_tokens", type=int, default=768)
    parser.add_argument("--rm_score_url", default=os.getenv("RM_SCORE_URL"))
    parser.add_argument("--rm_api_key", default=os.getenv("RM_API_KEY"))
    parser.add_argument("--rm_max_length", type=int, default=3072)
    parser.add_argument("--api_timeout", type=float, default=120.0)
    parser.add_argument("--api_failure_reward", type=float, default=0.0)
    parser.add_argument("--rm_failure_score", type=float, default=0.5)
    parser.add_argument("--max_api_failure_warnings", type=int, default=20)
    parser.add_argument("--deepspeed_config", default="")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--skip_final_save", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--no_proxy", "--no-proxy", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--print_metrics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
