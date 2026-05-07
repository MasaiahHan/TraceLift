#!/usr/bin/env python3
"""Build grouped GSM8K math Reason RM data from judged synthetic negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "outputs/math_negative_rubric_gsm8k_even_3000/filtered_bad_weak_dedup.jsonl"
DEFAULT_OUTPUT_DIR = "data/math_rm/gsm8k_even_3000"
MATH_DIMENSIONS = (
    "problem_understanding",
    "solution_strategy",
    "step_coherence",
    "calculation_correctness",
    "answer_support",
)

GSM8K_CALC_RE = re.compile(r"<<([^<>]+)>>\s*[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
GSM8K_ANSWER_RE = re.compile(r"\n?\s*####\s*.*$", re.DOTALL)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def stable_hash_int(*parts: Any) -> int:
    payload = "\0".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def clean_positive_reasoning(raw_reason: str) -> str:
    text = (raw_reason or "").strip()
    text = GSM8K_ANSWER_RE.sub("", text).strip()
    text = GSM8K_CALC_RE.sub(lambda match: match.group(1), text)
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def clean_negative_reasoning(row: dict[str, Any]) -> str:
    text = row.get("raw_negative_clean") or row.get("reason_clean") or ""
    return str(text).strip()


def bucketize_score(raw_score: Any) -> int | None:
    if raw_score is None:
        return None
    score = float(raw_score)
    bucket = int(score // 2.0)
    return min(4, max(0, bucket))


def negative_rubric(row: dict[str, Any]) -> dict[str, Any] | None:
    scores = row.get("rubric_scores") or {}
    rubric: dict[str, Any] = {}
    for name in MATH_DIMENSIONS:
        label = bucketize_score(scores.get(name))
        if label is None:
            return None
        rubric[name] = label
    total = row.get("rubric_score")
    if total is None:
        return None
    rubric["total"] = round(max(0.0, min(1.0, float(total) / 10.0)), 6)
    return rubric


def positive_rubric() -> dict[str, Any]:
    rubric = {name: 4 for name in MATH_DIMENSIONS}
    rubric["total"] = 1.0
    return rubric


def parent_id(row: dict[str, Any]) -> str:
    value = row.get("parent_source_row_index", row.get("source_row_index"))
    return str(value)


def sort_negatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("negative_kind") or ""),
            int(row.get("negative_index") or 0),
            str(row.get("reason_clean") or ""),
        ),
    )


def split_train_dev(groups: list[dict[str, Any]], dev_ratio: float, dev_size: int | None, seed: int):
    if len(groups) <= 1:
        return groups, []
    if dev_size is None:
        dev_size = int(round(len(groups) * dev_ratio))
        if dev_ratio > 0 and dev_size == 0:
            dev_size = 1
    dev_size = min(max(dev_size, 0), len(groups) - 1)
    ordered = sorted(groups, key=lambda group: stable_hash_int(seed, group["problem_id"]))
    dev_ids = {group["problem_id"] for group in ordered[:dev_size]}
    train = [group for group in groups if group["problem_id"] not in dev_ids]
    dev = [group for group in groups if group["problem_id"] in dev_ids]
    return train, dev


def build_group(parent: str, rows: list[dict[str, Any]], min_negatives: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    first = rows[0]
    positive_reason = clean_positive_reasoning(first.get("raw_reason") or "")
    stats = {
        "parent_source_row_index": parent,
        "problem": first.get("problem", ""),
        "negative_count": 0,
        "drop_reason": None,
    }
    if not positive_reason:
        stats["drop_reason"] = "empty_positive_reasoning"
        return None, stats
    if not str(first.get("problem", "")).strip():
        stats["drop_reason"] = "empty_problem"
        return None, stats
    if not str(first.get("action_gt", "")).strip():
        stats["drop_reason"] = "empty_action_gt"
        return None, stats

    negatives = []
    seen_negative_reasons = set()
    for row in sort_negatives(rows):
        reasoning = clean_negative_reasoning(row)
        if not reasoning:
            continue
        if reasoning in seen_negative_reasons:
            continue
        rubric = negative_rubric(row)
        if rubric is None:
            continue
        seen_negative_reasons.add(reasoning)
        negatives.append(
            {
                "reasoning": reasoning,
                "negative_kind": row.get("negative_kind"),
                "negative_index": row.get("negative_index"),
                "label": row.get("label", "negative"),
                "rubric": rubric,
                "rubric_label": row.get("rubric_label"),
                "rubric_score_raw": row.get("rubric_score"),
                "rubric_reason": row.get("rubric_reason"),
                "raw_reason_length": len(row.get("raw_negative_clean") or row.get("reason_clean") or ""),
                "clean_reason_length": len(reasoning),
                "metadata": {
                    "judge_meta": row.get("judge_meta") or {},
                    "source_row_index": row.get("source_row_index"),
                    "parent_source_row_index": row.get("parent_source_row_index"),
                },
            }
        )

    stats["negative_count"] = len(negatives)
    if len(negatives) < min_negatives:
        stats["drop_reason"] = "not_enough_negatives"
        return None, stats

    problem_id = "math_gsm8k_{}".format(parent.zfill(6))
    group = {
        "problem_id": problem_id,
        "source": "gsm8k",
        "task_type": "math",
        "problem": first.get("problem", ""),
        "reference_solution": first.get("action_gt", ""),
        "positive_pool": [
            {
                "reasoning": positive_reason,
                "label": "positive",
                "rubric": positive_rubric(),
                "rubric_label": "positive",
                "rubric_score_raw": 10.0,
                "raw_reason_length": len(first.get("raw_reason") or ""),
                "clean_reason_length": len(positive_reason),
            }
        ],
        "negative_bank": negatives,
        "metadata": {
            "source_dataset": first.get("source", "gsm8k"),
            "source_row_index": int(parent) if parent.isdigit() else parent,
            "answer": first.get("action_gt"),
            "negative_count": len(negatives),
            "math_dimension_names": list(MATH_DIMENSIONS),
        },
    }
    return group, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grouped math RM train/dev data from filtered GSM8K negatives.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_negatives", type=int, default=4)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--dev_size", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=18)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[parent_id(row)].append(row)

    groups = []
    dropped = []
    neg_count_hist = Counter()
    negative_kind_counts = Counter()
    for parent, parent_rows in sorted(grouped.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        group, stats = build_group(parent, parent_rows, args.min_negatives)
        neg_count_hist[stats["negative_count"]] += 1
        if group is None:
            dropped.append(stats)
            continue
        groups.append(group)
        for negative in group["negative_bank"]:
            negative_kind_counts[negative["negative_kind"]] += 1

    train_groups, dev_groups = split_train_dev(groups, args.dev_ratio, args.dev_size, args.split_seed)

    output_dir = Path(args.output_dir)
    all_output = output_dir / "all_groups_min4.jsonl"
    train_output = output_dir / "train_groups_min4.jsonl"
    dev_output = output_dir / "dev_groups_min4.jsonl"
    dropped_output = output_dir / "dropped_groups_min4.jsonl"
    summary_output = output_dir / "prepare_summary_min4.json"

    write_jsonl(all_output, groups)
    write_jsonl(train_output, train_groups)
    write_jsonl(dev_output, dev_groups)
    write_jsonl(dropped_output, dropped)

    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "dimension_names": list(MATH_DIMENSIONS),
        "num_negatives_used_by_training": args.min_negatives,
        "dev_ratio": args.dev_ratio,
        "dev_size": args.dev_size,
        "split_seed": args.split_seed,
        "input_rows": len(rows),
        "parents_seen": len(grouped),
        "groups_kept": len(groups),
        "groups_dropped": len(dropped),
        "train_groups": len(train_groups),
        "dev_groups": len(dev_groups),
        "total_negatives_kept": sum(len(group["negative_bank"]) for group in groups),
        "negative_count_histogram_per_parent": {str(k): v for k, v in sorted(neg_count_hist.items())},
        "negative_kind_counts": dict(sorted(negative_kind_counts.items())),
        "drop_reasons": dict(Counter(item["drop_reason"] for item in dropped)),
        "all_output": str(all_output),
        "train_output": str(train_output),
        "dev_output": str(dev_output),
        "dropped_output": str(dropped_output),
    }
    write_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
