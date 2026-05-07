#!/usr/bin/env python
"""TRL GRPOTrainer entrypoint for math-reason RL."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasonrm.code_oracle import iter_jsonl
from scripts.train_code_grpo import disable_broken_bitsandbytes_dispatch, make_opener, truncate_chars


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def normalize_api_url(base_url: str, api_type: str) -> str:
    base = base_url.rstrip("/")
    if api_type == "chat":
        return base if base.endswith("/chat/completions") else base + "/chat/completions"
    if api_type == "completions":
        return base if base.endswith("/completions") else base + "/completions"
    raise ValueError("unknown api type: {}".format(api_type))


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


def extract_completion(response: Dict[str, Any]) -> str:
    choice = response["choices"][0]
    if isinstance(choice.get("message"), dict):
        return str(choice["message"].get("content") or "")
    return str(choice.get("text") or "")


def render_messages(messages: Sequence[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "").strip()
        if role == "system":
            parts.append("Instruction:\n{}".format(content))
        else:
            parts.append(content)
    return "\n\n".join(parts).strip() + "\n\nFinal answer:\n"


def build_policy_prompt(problem: str, max_problem_chars: int) -> str:
    return (
        "You are solving a grade-school math problem.\n"
        "Write concise reasoning that helps another solver derive the correct final answer.\n"
        "Do not give the final answer.\n"
        "Use exactly this format:\n"
        "<reasoning>\n"
        "your reasoning\n"
        "</reasoning>\n\n"
        "Problem:\n{}\n\n"
        "Reasoning:\n"
    ).format(truncate_chars(problem, max_problem_chars))


def build_solver_messages(problem: str, reasoning: Optional[str], max_problem_chars: int, max_reason_chars: int) -> List[Dict[str, str]]:
    content = "Problem:\n{}".format(truncate_chars(problem, max_problem_chars))
    if reasoning:
        content += (
            "\n\nProvided reasoning:\n{}\n\n"
            "Use the provided reasoning as guidance if it is helpful. Return only the final answer."
        ).format(truncate_chars(reasoning, max_reason_chars))
    else:
        content += "\n\nReturn only the final answer."
    return [
        {
            "role": "system",
            "content": "You solve grade-school math problems. Return only the final numeric answer, no explanation.",
        },
        {"role": "user", "content": content},
    ]


def call_solver(
    opener: urllib.request.OpenerDirector,
    solver_url: str,
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
            "prompt": render_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    payload.update(extra_payload)
    return extract_completion(post_json(opener, solver_url, payload, api_key, timeout))


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


def call_math_rm(
    opener: urllib.request.OpenerDirector,
    rm_score_url: str,
    problem: str,
    reasoning: str,
    task_type: str,
    rm_api_key: Optional[str],
    rm_max_length: int,
    timeout: float,
) -> float:
    payload = {
        "items": [
            {
                "problem": problem,
                "reasoning": reasoning,
                "task_name": "math",
                "task_type": task_type or "math",
            }
        ],
        "max_length": rm_max_length,
        "batch_size": 1,
    }
    return parse_rm_scores(post_json(opener, rm_score_url, payload, rm_api_key, timeout), 1)[0]


def split_policy_completion(completion: str) -> str:
    text = completion or ""
    reason_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.S | re.I)
    if reason_match:
        return reason_match.group(1).strip()
    return text.strip()


def strip_latex_wrappers(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    return text


def extract_answer_text(text: str) -> str:
    text = strip_latex_wrappers(str(text or ""))
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    answer_patterns = [
        r"(?:the\s+)?answer\s+is\s*[:=]?\s*([^\n.]+)",
        r"final\s+answer\s*[:=]?\s*([^\n.]+)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].strip()
    number_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?", text)
    if number_matches:
        return number_matches[-1].strip()
    return text.strip()


def normalize_answer_string(text: str) -> str:
    text = strip_latex_wrappers(str(text or ""))
    text = text.strip().lower()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:answer|final answer)\s*[:=]\s*", "", text)
    text = text.strip(" .。")
    return text


def parse_number(text: str) -> Optional[Fraction]:
    text = normalize_answer_string(extract_answer_text(text))
    number_match = re.search(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", text)
    if not number_match:
        return None
    raw = number_match.group(0).replace(" ", "")
    try:
        return Fraction(raw)
    except Exception:
        try:
            return Fraction(float(raw)).limit_denominator(10**6)
        except Exception:
            return None


def answers_equivalent(prediction: str, reference: str) -> bool:
    pred_num = parse_number(prediction)
    ref_num = parse_number(reference)
    if pred_num is not None and ref_num is not None:
        return abs(float(pred_num - ref_num)) <= 1e-6
    return normalize_answer_string(extract_answer_text(prediction)) == normalize_answer_string(extract_answer_text(reference))


def answer_reward(prediction: str, reference: str) -> float:
    return 1.0 if answers_equivalent(prediction, reference) else 0.0


def load_train_rows(
    train_file: str,
    max_items: int,
    max_problem_chars: int,
    tokenizer: Optional[AutoTokenizer] = None,
    policy_use_chat_template: bool = False,
    policy_chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for group in iter_jsonl(train_file):
        problem = str(group.get("problem") or "")
        reference = str(group.get("reference_solution") or group.get("answer") or "").strip()
        if not problem or not reference:
            continue
        prompt = build_policy_prompt(problem, max_problem_chars)
        if policy_use_chat_template:
            if tokenizer is None:
                raise ValueError("tokenizer is required when policy_use_chat_template=True")
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **(policy_chat_template_kwargs or {}),
            )
        rows.append(
            {
                "prompt": prompt,
                "problem_id": str(group.get("problem_id") or len(rows)),
                "problem": problem,
                "reference_answer": reference,
                "task_type": str(group.get("task_type") or group.get("source") or "math"),
            }
        )
        if max_items > 0 and len(rows) >= max_items:
            break
    if not rows:
        raise ValueError("no math train rows with reference answers found")
    return rows


class MathReward:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.opener = make_opener(args.no_proxy)
        self.solver_url = normalize_api_url(args.executor_base_url, args.executor_api_type) if args.executor_base_url else ""
        self.executor_failures = 0
        self.rm_failures = 0
        self.reward_calls = 0
        self.__name__ = "math_answer_reason_reward"

    def _warn_api_failure(self, kind: str, exc: BaseException) -> None:
        if kind == "executor":
            self.executor_failures += 1
            count = self.executor_failures
        else:
            self.rm_failures += 1
            count = self.rm_failures
        if count <= self.args.max_api_failure_warnings:
            rank = os.environ.get("RANK", "0")
            print(
                "[rank {}] {} call failed #{}; using fallback reward: {}: {}".format(
                    rank, kind, count, type(exc).__name__, str(exc)[:500]
                ),
                file=sys.stderr,
                flush=True,
            )

    def _should_log_completions(self) -> bool:
        if not self.args.log_completions:
            return False
        if int(os.environ.get("RANK", "0")) != 0:
            return False
        every = max(1, int(self.args.log_completion_every))
        return self.reward_calls % every == 0

    def _log_completion(self, record: Dict[str, Any]) -> None:
        if not self._should_log_completions():
            return
        max_chars = max(0, int(self.args.log_completion_chars))
        printable = dict(record)
        for key in ("completion", "reasoning"):
            if key in printable:
                value = str(printable[key])
                printable[key] = value[:max_chars] + ("...[truncated]" if max_chars and len(value) > max_chars else "")
        if self.args.print_completions:
            print("[completion_sample] {}".format(json.dumps(printable, ensure_ascii=False)), file=sys.stderr, flush=True)

        log_file = self.args.completion_log_file
        if not log_file:
            return
        path = Path(log_file)
        if not path.is_absolute():
            path = Path(self.args.output_dir) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(printable, ensure_ascii=False) + "\n")

    def _solver_reward(self, problem: str, reasoning: Optional[str], reference: str) -> float:
        try:
            answer = call_solver(
                self.opener,
                self.solver_url,
                self.args.executor_api_type,
                self.args.executor_model,
                self.args.executor_api_key,
                build_solver_messages(problem, reasoning, self.args.max_problem_chars, self.args.max_reason_chars),
                self.args.executor_extra_payload,
                self.args.executor_max_tokens,
                self.args.executor_temperature,
                self.args.api_timeout,
            )
            return answer_reward(answer, reference)
        except Exception as exc:
            self._warn_api_failure("executor", exc)
            return self.args.api_failure_reward

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        problem_id: Sequence[str],
        problem: Sequence[str],
        reference_answer: Sequence[str],
        task_type: Sequence[str],
        **_: Any,
    ) -> List[float]:
        self.reward_calls += 1
        rewards = []
        should_log = self._should_log_completions()
        log_remaining = int(self.args.log_completion_samples) if should_log else 0
        for completion, pid, prob, ref, task in zip(completions, problem_id, problem, reference_answer, task_type):
            reasoning = split_policy_completion(str(completion))
            e2e_reward = self._solver_reward(str(prob), reasoning, str(ref))

            with_reason_reward = None
            baseline_exec_reward = None
            uplift_reward = None
            rm_score = None
            reason_reward = None
            reward = e2e_reward

            if self.args.reward_mode == "reason_exec":
                with_rewards = [
                    self._solver_reward(str(prob), reasoning, str(ref))
                    for _repeat in range(self.args.executor_repeats)
                ]
                baseline_rewards = [
                    self._solver_reward(str(prob), None, str(ref))
                    for _repeat in range(self.args.baseline_repeats)
                ]
                with_reason_reward = clamp(sum(with_rewards) / len(with_rewards), 0.0, 1.0)
                baseline_exec_reward = clamp(sum(baseline_rewards) / len(baseline_rewards), 0.0, 1.0)
                uplift_reward = clamp(with_reason_reward - baseline_exec_reward, -1.0, 1.0)
                try:
                    rm_score = call_math_rm(
                        self.opener,
                        self.args.rm_score_url,
                        str(prob),
                        reasoning,
                        str(task),
                        self.args.rm_api_key,
                        self.args.rm_max_length,
                        self.args.api_timeout,
                    )
                except Exception as exc:
                    self._warn_api_failure("rm", exc)
                    rm_score = self.args.rm_failure_score
                reason_reward = clamp(rm_score, 0.0, 1.0) * uplift_reward
                reward = self.args.e2e_reward_weight * e2e_reward + self.args.reason_reward_weight * reason_reward

            if log_remaining > 0:
                self._log_completion(
                    {
                        "call": self.reward_calls,
                        "problem_id": str(pid),
                        "reward_mode": self.args.reward_mode,
                        "reward": float(reward),
                        "e2e_reward": float(e2e_reward),
                        "rm_score": rm_score,
                        "executor_repeats": self.args.executor_repeats if self.args.reward_mode == "reason_exec" else None,
                        "baseline_repeats": self.args.baseline_repeats if self.args.reward_mode == "reason_exec" else None,
                        "with_reason_acc": with_reason_reward,
                        "no_reason_acc": baseline_exec_reward,
                        "uplift": uplift_reward,
                        "reason_reward": reason_reward,
                        "prediction": None,
                        "reference": str(ref),
                        "completion": str(completion),
                        "reasoning": reasoning,
                    }
                )
                log_remaining -= 1
            rewards.append(float(reward))
        return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", default="data/math_rm/gsm8k_even_3000/train_groups_min4.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reward_mode", choices=["exec_only", "reason_exec"], default="exec_only")
    parser.add_argument("--e2e_reward_weight", type=float, default=1.0)
    parser.add_argument("--reason_reward_weight", type=float, default=0.5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_problem_chars", type=int, default=2500)
    parser.add_argument("--max_reason_chars", type=int, default=2000)
    parser.add_argument("--policy_use_chat_template", action="store_true")
    parser.add_argument("--policy_chat_template_kwargs_json")
    parser.add_argument("--executor_base_url", default="")
    parser.add_argument("--executor_model", default="")
    parser.add_argument("--executor_api_type", choices=["completions", "chat"], default="completions")
    parser.add_argument("--executor_api_key", default=os.getenv("EXECUTOR_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--executor_extra_payload_json", default=os.getenv("EXECUTOR_EXTRA_PAYLOAD_JSON"))
    parser.add_argument("--executor_repeats", type=int, default=1)
    parser.add_argument("--baseline_repeats", type=int, default=0)
    parser.add_argument("--uplift_exec_reward", action="store_true")
    parser.add_argument("--executor_temperature", type=float, default=0.5)
    parser.add_argument("--executor_max_tokens", type=int, default=256)
    parser.add_argument("--rm_score_url", default=os.getenv("RM_SCORE_URL"))
    parser.add_argument("--rm_api_key", default=os.getenv("RM_API_KEY"))
    parser.add_argument("--rm_max_length", type=int, default=2048)
    parser.add_argument("--api_timeout", type=float, default=120.0)
    parser.add_argument("--api_failure_reward", type=float, default=0.0)
    parser.add_argument("--rm_failure_score", type=float, default=0.5)
    parser.add_argument("--max_api_failure_warnings", type=int, default=20)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--deepspeed")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    parser.add_argument("--gradient_checkpointing_kwargs_json")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_strategy", default="steps")
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--save_only_model", type=str2bool, default=False)
    parser.add_argument("--log_completions", type=str2bool, default=True)
    parser.add_argument("--log_completion_every", type=int, default=1)
    parser.add_argument("--log_completion_samples", type=int, default=1)
    parser.add_argument("--log_completion_chars", type=int, default=2000)
    parser.add_argument("--completion_log_file", default="completion_samples.jsonl")
    parser.add_argument("--print_completions", type=str2bool, default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_proxy", "--no-proxy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_proxy:
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.executor_extra_payload = json.loads(args.executor_extra_payload_json) if args.executor_extra_payload_json else {}
    args.policy_chat_template_kwargs = (
        json.loads(args.policy_chat_template_kwargs_json) if args.policy_chat_template_kwargs_json else {}
    )
    args.gradient_checkpointing_kwargs = (
        json.loads(args.gradient_checkpointing_kwargs_json) if args.gradient_checkpointing_kwargs_json else None
    )
    if not args.executor_base_url or not args.executor_model:
        raise ValueError("math GRPO requires executor_base_url and executor_model because the policy emits reasoning only")
    if args.reward_mode == "reason_exec":
        if not args.rm_score_url:
            raise ValueError("--rm_score_url is required for reason_exec")
        if not args.uplift_exec_reward:
            raise ValueError("reason_exec requires --uplift_exec_reward")
        if args.executor_repeats != 3 or args.baseline_repeats != 3:
            raise ValueError("reason_exec requires --executor_repeats 3 and --baseline_repeats 3")
        if args.e2e_reward_weight != 0.5:
            raise ValueError("--e2e_reward_weight must be exactly 0.5 for reason_exec")
        if args.reason_reward_weight != 0.5:
            raise ValueError("--reason_reward_weight must be exactly 0.5 for reason_exec")
    if args.use_lora and args.gradient_checkpointing:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(
                "LoRA math GRPO disables gradient_checkpointing to keep Qwen loss attached to LoRA parameters.",
                file=sys.stderr,
                flush=True,
            )
        args.gradient_checkpointing = False
        args.gradient_checkpointing_kwargs = None

    disable_broken_bitsandbytes_dispatch()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = load_train_rows(
        args.train_file,
        args.max_train_items,
        args.max_problem_chars,
        tokenizer=tokenizer,
        policy_use_chat_template=args.policy_use_chat_template,
        policy_chat_template_kwargs=args.policy_chat_template_kwargs,
    )
    train_dataset = Dataset.from_list(train_rows)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = args.per_device_train_batch_size * world_size
    if global_batch_size % args.num_generations != 0:
        raise ValueError(
            "global batch size (per_device_train_batch_size * world_size = {} * {}) "
            "must be divisible by num_generations={} for TRL GRPO.".format(
                args.per_device_train_batch_size, world_size, args.num_generations
            )
        )

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        report_to=[],
        use_vllm=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
        model_init_kwargs={"torch_dtype": "bfloat16", "trust_remote_code": True},
    )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=MathReward(args),
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
