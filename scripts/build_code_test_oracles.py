#!/usr/bin/env python
"""Build executable test oracles for ReasonReward code groups.

The preferred oracle is recovered from the original DeepMind CodeContests
dataset. When a problem cannot be matched there, the script falls back to
statement examples extracted from the problem text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasonrm.code_oracle import extract_statement_examples, iter_jsonl, normalize_text


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def load_groups(paths: Sequence[str]) -> List[Dict[str, Any]]:
    groups = []
    seen = set()
    for path in paths:
        for row in iter_jsonl(path):
            problem_id = row.get("problem_id")
            if problem_id in seen:
                continue
            seen.add(problem_id)
            groups.append(row)
    return groups


def load_raw_meta(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    meta_by_source_row = {}
    for row in iter_jsonl(path):
        source_row_index = row.get("source_row_index")
        if source_row_index is None:
            continue
        judge_meta = row.get("judge_meta") or {}
        meta = judge_meta.get("metadata") or {}
        meta_by_source_row[int(source_row_index)] = {
            "open_code_id": judge_meta.get("id"),
            "source_dataset": judge_meta.get("source_dataset"),
            **meta,
        }
    return meta_by_source_row


def hf_tests_to_list(test_obj: Any, source: str, max_count: int) -> List[Dict[str, Any]]:
    if not isinstance(test_obj, dict):
        return []
    inputs = test_obj.get("input") or []
    outputs = test_obj.get("output") or []
    tests = []
    for sample_input, sample_output in zip(inputs, outputs):
        if sample_input is None or sample_output is None:
            continue
        tests.append(
            {
                "input": str(sample_input),
                "output": str(sample_output),
                "source": source,
                "is_feedback_test": source == "public",
                "is_reward_test": source in {"private", "generated"},
            }
        )
        if max_count > 0 and len(tests) >= max_count:
            break
    return tests


def load_code_contests(
    cache_dir: str,
    splits: Sequence[str],
    max_generated_tests: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load deepmind/code_contests") from exc

    by_desc_hash: Dict[str, Dict[str, Any]] = {}
    by_split_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for split in splits:
        dataset = load_dataset("deepmind/code_contests", split=split, cache_dir=cache_dir)
        for index, row in enumerate(dataset):
            tests = []
            tests.extend(hf_tests_to_list(row.get("public_tests"), "public", max_count=0))
            tests.extend(hf_tests_to_list(row.get("private_tests"), "private", max_count=0))
            tests.extend(hf_tests_to_list(row.get("generated_tests"), "generated", max_count=max_generated_tests))
            record = {
                "match_source": "deepmind/code_contests",
                "match_split": split,
                "match_index": index,
                "name": row.get("name"),
                "cf_contest_id": row.get("cf_contest_id"),
                "cf_index": row.get("cf_index"),
                "time_limit": row.get("time_limit"),
                "memory_limit_bytes": row.get("memory_limit_bytes"),
                "tests": tests,
                "test_counts": {
                    "public": sum(1 for t in tests if t["source"] == "public"),
                    "private": sum(1 for t in tests if t["source"] == "private"),
                    "generated": sum(1 for t in tests if t["source"] == "generated"),
                },
            }
            description = row.get("description") or ""
            by_desc_hash.setdefault(text_hash(description), record)
            by_split_index[(split, index)] = record
    return by_desc_hash, by_split_index


def code_contests_match(
    group: Dict[str, Any],
    source_meta: Dict[str, Any],
    by_desc_hash: Dict[str, Dict[str, Any]],
    by_split_index: Dict[Tuple[str, int], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    problem = group.get("problem") or ""
    record = by_desc_hash.get(text_hash(problem))
    if record:
        return {**record, "match_method": "description_hash"}

    split = str(source_meta.get("split") or "")
    index_raw = source_meta.get("index")
    if split and index_raw is not None:
        try:
            index = int(index_raw)
        except (TypeError, ValueError):
            index = -1
        candidate = by_split_index.get((split, index))
        if candidate:
            return {**candidate, "match_method": "source_split_index"}
    return None


def build_statement_oracle(group: Dict[str, Any], max_examples: int) -> Dict[str, Any]:
    tests = extract_statement_examples(group.get("problem") or "", max_examples=max_examples)
    for test in tests:
        test["is_feedback_test"] = True
        test["is_reward_test"] = True
    return {
        "match_source": "statement",
        "match_method": "statement_examples",
        "tests": tests,
        "test_counts": {"statement_example": len(tests)},
    }


def truncate_tests(tests: Sequence[Dict[str, Any]], max_total: int) -> List[Dict[str, Any]]:
    if max_total <= 0 or len(tests) <= max_total:
        return list(tests)
    priority = {"private": 0, "generated": 1, "public": 2, "statement_example": 3}
    return sorted(tests, key=lambda test: priority.get(str(test.get("source")), 99))[:max_total]


def build_oracles(args: argparse.Namespace) -> List[Dict[str, Any]]:
    groups = load_groups(args.group_files)
    raw_meta = load_raw_meta(args.raw_file)

    by_desc_hash: Dict[str, Dict[str, Any]] = {}
    by_split_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not args.no_code_contests:
        by_desc_hash, by_split_index = load_code_contests(
            cache_dir=args.cache_dir,
            splits=args.code_contests_splits,
            max_generated_tests=args.max_generated_tests,
        )

    oracles = []
    for group in groups:
        source_row_index = (group.get("metadata") or {}).get("source_row_index")
        source_meta = raw_meta.get(int(source_row_index), {}) if source_row_index is not None else {}

        match = None
        if source_meta.get("dataset") == "code_contests" or not source_meta:
            match = code_contests_match(group, source_meta, by_desc_hash, by_split_index)

        if not match or not match.get("tests"):
            match = build_statement_oracle(group, max_examples=args.max_statement_examples)

        tests = truncate_tests(match.get("tests") or [], args.max_tests_total)
        oracle = {
            "problem_id": group.get("problem_id"),
            "source_row_index": source_row_index,
            "source_meta": source_meta,
            "oracle_source": match.get("match_source"),
            "match_method": match.get("match_method"),
            "tests": tests,
            "test_counts": {
                "public": sum(1 for t in tests if t.get("source") == "public"),
                "private": sum(1 for t in tests if t.get("source") == "private"),
                "generated": sum(1 for t in tests if t.get("source") == "generated"),
                "statement_example": sum(1 for t in tests if t.get("source") == "statement_example"),
                "total": len(tests),
                "reward": sum(1 for t in tests if t.get("is_reward_test")),
                "feedback": sum(1 for t in tests if t.get("is_feedback_test")),
            },
            "match_info": {
                key: match.get(key)
                for key in ("match_split", "match_index", "name", "cf_contest_id", "cf_index", "time_limit", "memory_limit_bytes")
                if key in match
            },
        }
        oracles.append(oracle)
    return oracles


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_files", nargs="+", default=["data/code_rm/train_groups.jsonl", "data/code_rm/dev_groups.jsonl"])
    parser.add_argument("--raw_file", default="data/code_rm/raw/code_clean_from_opencode.jsonl")
    parser.add_argument("--out", default="data/code_rm/code_test_oracles.jsonl")
    parser.add_argument("--cache_dir", default="data/oracle/.hf_cache")
    parser.add_argument("--code_contests_splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--max_generated_tests", type=int, default=64)
    parser.add_argument("--max_tests_total", type=int, default=128)
    parser.add_argument("--max_statement_examples", type=int, default=8)
    parser.add_argument("--no_code_contests", action="store_true")
    args = parser.parse_args()

    oracles = build_oracles(args)
    write_jsonl(args.out, oracles)

    summary: Dict[str, Any] = {
        "out": args.out,
        "num_oracles": len(oracles),
        "oracle_source_counts": {},
        "match_method_counts": {},
        "total_tests": 0,
        "total_reward_tests": 0,
    }
    for oracle in oracles:
        summary["oracle_source_counts"][oracle["oracle_source"]] = summary["oracle_source_counts"].get(oracle["oracle_source"], 0) + 1
        summary["match_method_counts"][oracle["match_method"]] = summary["match_method_counts"].get(oracle["match_method"], 0) + 1
        summary["total_tests"] += oracle["test_counts"]["total"]
        summary["total_reward_tests"] += oracle["test_counts"]["reward"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
