import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


POSITIVE_SENTINEL = "__positive__"
STRONG_CODE_PATTERNS = [
    re.compile(r"\bdef\s+"),
    re.compile(r"\bclass\s+"),
    re.compile(r"\breturn\b"),
    re.compile(r"\bimport\b"),
    re.compile(r"```"),
]
ANY_CODE_PATTERNS = STRONG_CODE_PATTERNS + [
    re.compile(r"\binput\s*\("),
    re.compile(r"\bprint\s*\("),
]
PLACEHOLDER_NEGATIVE_PATTERNS = [
    re.compile(r"^\[mock-", re.IGNORECASE),
    re.compile(r"mock-sonnet", re.IGNORECASE),
]
EXPLANATION_MARKERS = [
    "\n### Explanation",
    "\n## Explanation",
    "\n**Explanation:**",
]
DEFAULT_JUDGE_DIMENSIONS = [
    "task_understanding",
    "plan_quality",
    "step_coherence",
    "action_support",
    "non_leakage",
]


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def stable_hash_int(*parts):
    payload = "\0".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def deterministic_sample(groups, sample_size, seed):
    if sample_size is None:
        return []
    if sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    ordered = sorted(
        groups,
        key=lambda group: (
            stable_hash_int(seed, group.get("problem_id", "")),
            group.get("problem_id", ""),
        ),
    )
    return ordered[: min(sample_size, len(ordered))]


def split_train_dev(groups, args):
    if args.dev_output is None:
        return groups, []
    if args.dev_size is not None and args.dev_size < 0:
        raise ValueError("--dev-size must be non-negative")
    if not 0 <= args.dev_ratio < 1:
        raise ValueError("--dev-ratio must be in [0, 1)")
    if len(groups) <= 1:
        return groups, []

    if args.dev_size is not None:
        dev_size = args.dev_size
    else:
        dev_size = int(round(len(groups) * args.dev_ratio))
        if args.dev_ratio > 0 and dev_size == 0:
            dev_size = 1
    dev_size = min(dev_size, len(groups) - 1)

    ordered = sorted(
        groups,
        key=lambda group: (
            stable_hash_int(args.split_seed, group.get("problem_id", "")),
            group.get("problem_id", ""),
        ),
    )
    dev_problem_ids = {group.get("problem_id") for group in ordered[:dev_size]}
    train_groups = [group for group in groups if group.get("problem_id") not in dev_problem_ids]
    dev_groups = [group for group in groups if group.get("problem_id") in dev_problem_ids]
    return train_groups, dev_groups


def count_negative_kinds(groups):
    counts = Counter()
    for group in groups:
        for negative in group.get("negative_bank", []):
            counts[negative.get("negative_kind")] += 1
    return dict(sorted(counts.items()))


def count_negative_candidates(groups):
    return sum(len(group.get("negative_bank", [])) for group in groups)


def row_key(record):
    source_row_index = record.get("source_row_index")
    if source_row_index is not None:
        return str(source_row_index)
    judge_meta = record.get("judge_meta") or {}
    judge_id = judge_meta.get("id")
    if judge_id is None:
        raise KeyError("record is missing both source_row_index and judge_meta.id")
    return str(judge_id)


def parent_row_key(record):
    parent_source_row_index = record.get("parent_source_row_index")
    if parent_source_row_index is not None:
        return str(parent_source_row_index)
    return row_key(record)


def candidate_key(record):
    negative_kind = record.get("negative_kind")
    if negative_kind:
        return (parent_row_key(record), negative_kind, record.get("negative_index"))
    return (row_key(record), POSITIVE_SENTINEL, None)


def normalize_reasoning(reasoning, action_gt):
    text = (reasoning or "").strip()
    if not text:
        return ""

    think_end = text.find("</think>")
    if think_end != -1:
        text = text[:think_end]

    if action_gt:
        action_pos = text.find(action_gt)
        if action_pos != -1:
            text = text[:action_pos]

    fence_pos = text.find("```")
    if fence_pos != -1:
        text = text[:fence_pos]

    for marker in EXPLANATION_MARKERS:
        marker_pos = text.find(marker)
        if marker_pos != -1:
            text = text[:marker_pos]

    text = text.replace("<think>", "").strip()
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def has_pattern(text, patterns):
    return any(pattern.search(text) for pattern in patterns)


def is_placeholder_negative(text):
    return has_pattern(text or "", PLACEHOLDER_NEGATIVE_PATTERNS)


def bucketize_judge_score(raw_score):
    if raw_score is None:
        return None
    score = float(raw_score)
    bucket = int(score // 2.0)
    return min(4, max(0, bucket))


def convert_judge_record(record):
    rubric_scores = record.get("rubric_scores") or {}
    if not rubric_scores:
        return None

    rubric = {}
    for name in DEFAULT_JUDGE_DIMENSIONS:
        value = rubric_scores.get(name)
        label = bucketize_judge_score(value)
        if label is None:
            return None
        rubric[name] = label

    total_score = record.get("rubric_score")
    if total_score is None:
        return None
    rubric["total"] = round(max(0.0, min(1.0, float(total_score) / 10.0)), 6)
    return rubric


def load_judge_lookup(paths):
    lookup = {}
    seen_dimensions = Counter()
    for path in paths:
        for record in load_jsonl(path):
            key = candidate_key(record)
            rubric = convert_judge_record(record)
            if rubric is None:
                continue
            lookup[key] = {
                "rubric": rubric,
                "rubric_label": record.get("rubric_label"),
                "rubric_score_raw": record.get("rubric_score"),
                "rubric_reason": record.get("rubric_reason"),
            }
            for name in rubric:
                if name != "total":
                    seen_dimensions[name] += 1
    return lookup, sorted(seen_dimensions.keys())


def load_negative_lookup(paths, flat_paths):
    negatives_by_row = {}
    for path in paths:
        for record in load_jsonl(path):
            row = row_key(record)
            bucket = negatives_by_row.setdefault(row, [])
            for negative_index, negative in enumerate(record.get("generated_negatives") or []):
                if not negative.get("negative_kind"):
                    continue
                payload = dict(negative)
                payload.setdefault("parent_source_row_index", record.get("source_row_index"))
                payload.setdefault("negative_index", negative_index)
                bucket.append(payload)
    for path in flat_paths:
        for record in load_jsonl(path):
            if not record.get("negative_kind"):
                continue
            row = parent_row_key(record)
            negatives_by_row.setdefault(row, []).append(record)
    return negatives_by_row


def build_group(record, positive_rubric, negative_records, judge_lookup, args):
    positive_reason = normalize_reasoning(record.get("reason_clean"), record.get("action_gt"))
    positive_any_code = has_pattern(positive_reason, ANY_CODE_PATTERNS)
    positive_strong_code = has_pattern(positive_reason, STRONG_CODE_PATTERNS)
    if args.drop_positives_with_strong_code and positive_strong_code:
        return None, {
            "dropped_positive_strong_code": 1,
        }

    group = {
        "problem_id": "code_{}".format(str(row_key(record)).zfill(6)),
        "source": record.get("source", "code"),
        "problem": record.get("problem", ""),
        "reference_solution": record.get("action_gt", ""),
        "positive_pool": [],
        "negative_bank": [],
        "metadata": {
            "task_type": (record.get("judge_meta") or {}).get("task_type", "code"),
            "source_dataset": (record.get("judge_meta") or {}).get("source_dataset"),
            "source_row_index": record.get("source_row_index"),
            "positive_has_any_code_pattern": positive_any_code,
            "positive_has_strong_code_pattern": positive_strong_code,
        },
    }

    positive_candidate = {
        "reasoning": positive_reason,
        "label": record.get("label"),
        "raw_reason_length": len(record.get("reason_clean") or ""),
        "clean_reason_length": len(positive_reason),
    }
    if positive_rubric is not None:
        positive_candidate["rubric"] = positive_rubric["rubric"]
        positive_candidate["rubric_label"] = positive_rubric.get("rubric_label")
        positive_candidate["rubric_score_raw"] = positive_rubric.get("rubric_score_raw")
    group["positive_pool"].append(positive_candidate)

    stats = Counter()
    stats["positive_has_any_code_pattern"] += int(positive_any_code)
    stats["positive_has_strong_code_pattern"] += int(positive_strong_code)
    stats["positive_has_rubric"] += int(positive_rubric is not None)

    seen_negative_keys = set()
    for negative_record in sorted(
        negative_records,
        key=lambda item: (str(item.get("negative_kind") or ""), int(item.get("negative_index") or 0)),
    ):
        stats["negative_records_seen"] += 1
        negative_kind = negative_record.get("negative_kind")
        if not negative_kind:
            stats["dropped_negative_missing_kind"] += 1
            continue
        dedupe_key = (
            parent_row_key(negative_record),
            negative_kind,
            negative_record.get("negative_index"),
            negative_record.get("raw_negative_clean") or negative_record.get("reason_clean") or "",
        )
        if dedupe_key in seen_negative_keys:
            stats["dropped_negative_duplicate"] += 1
            continue
        seen_negative_keys.add(dedupe_key)
        negative_reason = normalize_reasoning(
            negative_record.get("raw_negative_clean") or negative_record.get("reason_clean"),
            negative_record.get("action_gt"),
        )
        negative_any_code = has_pattern(negative_reason, ANY_CODE_PATTERNS)
        negative_strong_code = has_pattern(negative_reason, STRONG_CODE_PATTERNS)
        if is_placeholder_negative(negative_reason):
            stats["dropped_negative_placeholder"] += 1
            continue
        if len(negative_reason.split()) < args.min_negative_words:
            stats["dropped_negative_too_short"] += 1
            continue
        if args.drop_negatives_with_strong_code and negative_strong_code:
            stats["dropped_negative_strong_code"] += 1
            continue

        negative_candidate = {
            "reasoning": negative_reason,
            "negative_kind": negative_kind,
            "negative_index": negative_record.get("negative_index"),
            "label": negative_record.get("label"),
            "raw_reason_length": len(
                negative_record.get("raw_negative_clean") or negative_record.get("reason_clean") or ""
            ),
            "clean_reason_length": len(negative_reason),
            "metadata": negative_record.get("judge_meta") or {},
        }
        negative_rubric = convert_judge_record(negative_record)
        if negative_rubric is not None:
            negative_rubric = {
                "rubric": negative_rubric,
                "rubric_label": negative_record.get("rubric_label"),
                "rubric_score_raw": negative_record.get("rubric_score"),
                "rubric_reason": negative_record.get("rubric_reason"),
            }
        if negative_rubric is None:
            negative_rubric = judge_lookup.get(candidate_key(negative_record))
        if negative_rubric is not None:
            negative_candidate["rubric"] = negative_rubric["rubric"]
            negative_candidate["rubric_label"] = negative_rubric.get("rubric_label")
            negative_candidate["rubric_score_raw"] = negative_rubric.get("rubric_score_raw")
            stats["negative_with_rubric"] += 1
        group["negative_bank"].append(negative_candidate)
        stats["negative_count"] += 1
        stats["negative_has_any_code_pattern"] += int(negative_any_code)
        stats["negative_has_strong_code_pattern"] += int(negative_strong_code)

    return group, stats


def main():
    parser = argparse.ArgumentParser(description="Prepare code-domain RM groups from raw clean/judge/negative files.")
    parser.add_argument("--positive-file", required=True, help="Positive clean jsonl file.")
    parser.add_argument(
        "--judge-files",
        nargs="*",
        default=[],
        help="Judge jsonl files. Current script joins on (source_row_index, negative_kind).",
    )
    parser.add_argument(
        "--merged-negative-files",
        nargs="*",
        default=[],
        help="Merged negative jsonl files containing generated_negatives.",
    )
    parser.add_argument(
        "--negative-files",
        nargs="*",
        default=[],
        help="Flat negative jsonl files. Judged negatives may include rubric fields directly.",
    )
    parser.add_argument("--proto-output", required=True, help="Output jsonl for proto groups.")
    parser.add_argument(
        "--train-output",
        required=True,
        help="Output jsonl for trainable groups with positive and negative rubrics.",
    )
    parser.add_argument(
        "--dev-output",
        default=None,
        help="Optional output jsonl for a deterministic dev split. If omitted, train-output receives all trainable groups.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Dev split ratio used when --dev-output is set and --dev-size is omitted.",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=None,
        help="Exact number of dev groups used when --dev-output is set.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=18,
        help="Seed for deterministic problem_id-based train/dev split.",
    )
    parser.add_argument(
        "--sample-train-output",
        default=None,
        help="Optional jsonl path for a deterministic small sample from the train split.",
    )
    parser.add_argument(
        "--sample-dev-output",
        default=None,
        help="Optional jsonl path for a deterministic small sample from the dev split.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=128,
        help="Maximum number of groups to write to each requested sample output.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=18,
        help="Seed for deterministic sample outputs.",
    )
    parser.add_argument("--summary-output", required=True, help="Output json for summary stats.")
    parser.add_argument(
        "--drop-positives-with-strong-code",
        action="store_true",
        help="Drop positives that still contain strong code patterns after cleanup.",
    )
    parser.add_argument(
        "--drop-negatives-with-strong-code",
        action="store_true",
        help="Drop negatives that still contain strong code patterns after cleanup.",
    )
    parser.add_argument(
        "--min-negative-words",
        type=int,
        default=40,
        help="Drop negatives whose cleaned reasoning has fewer words than this.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for debugging.",
    )
    args = parser.parse_args()

    positives = load_jsonl(args.positive_file)
    if args.limit is not None:
        positives = positives[: args.limit]

    judge_lookup, seen_dimensions = load_judge_lookup(args.judge_files)
    negatives_by_row = load_negative_lookup(args.merged_negative_files, args.negative_files)

    proto_groups = []
    train_groups = []
    summary = Counter()
    negative_kind_counts = Counter()

    for record in positives:
        positive_rubric = judge_lookup.get(candidate_key(record))
        group, stats = build_group(
            record=record,
            positive_rubric=positive_rubric,
            negative_records=negatives_by_row.get(row_key(record), {}),
            judge_lookup=judge_lookup,
            args=args,
        )
        summary.update(stats)
        summary["rows_seen"] += 1
        if group is None:
            continue

        for negative in group["negative_bank"]:
            negative_kind_counts[negative["negative_kind"]] += 1

        proto_groups.append(group)
        summary["proto_groups"] += 1
        summary["proto_groups_with_negatives"] += int(bool(group["negative_bank"]))

        positive_has_rubric = "rubric" in group["positive_pool"][0]
        negative_bank = group["negative_bank"]
        rubric_negatives = [neg for neg in negative_bank if "rubric" in neg]
        summary["negative_without_rubric"] += len(negative_bank) - len(rubric_negatives)
        if positive_has_rubric and rubric_negatives:
            train_group = dict(group)
            train_group["positive_pool"] = list(group["positive_pool"])
            train_group["negative_bank"] = rubric_negatives
            train_groups.append(train_group)
            summary["train_groups"] += 1
        elif positive_has_rubric:
            summary["groups_missing_negative_rubric"] += int(bool(negative_bank))
        else:
            summary["groups_missing_positive_rubric"] += 1

    trainable_groups_total = list(train_groups)
    train_groups, dev_groups = split_train_dev(trainable_groups_total, args)
    sample_train_groups = deterministic_sample(train_groups, args.sample_size, args.sample_seed)
    sample_dev_groups = deterministic_sample(dev_groups, args.sample_size, args.sample_seed)

    summary_payload = {
        "positive_file": args.positive_file,
        "judge_files": args.judge_files,
        "merged_negative_files": args.merged_negative_files,
        "negative_files": args.negative_files,
        "split_seed": args.split_seed,
        "dev_ratio": args.dev_ratio,
        "dev_size": args.dev_size,
        "rows_seen": summary["rows_seen"],
        "proto_groups": summary["proto_groups"],
        "proto_groups_with_negatives": summary["proto_groups_with_negatives"],
        "trainable_groups_total": len(trainable_groups_total),
        "train_groups": len(train_groups),
        "dev_groups": len(dev_groups),
        "groups_missing_positive_rubric": summary["groups_missing_positive_rubric"],
        "groups_missing_negative_rubric": summary["groups_missing_negative_rubric"],
        "positive_has_rubric": summary["positive_has_rubric"],
        "positive_has_any_code_pattern": summary["positive_has_any_code_pattern"],
        "positive_has_strong_code_pattern": summary["positive_has_strong_code_pattern"],
        "negative_records_seen": summary["negative_records_seen"],
        "negative_count": summary["negative_count"],
        "negative_with_rubric": summary["negative_with_rubric"],
        "negative_without_rubric": summary["negative_without_rubric"],
        "negative_has_any_code_pattern": summary["negative_has_any_code_pattern"],
        "negative_has_strong_code_pattern": summary["negative_has_strong_code_pattern"],
        "dropped_positive_strong_code": summary["dropped_positive_strong_code"],
        "dropped_negative_strong_code": summary["dropped_negative_strong_code"],
        "dropped_negative_placeholder": summary["dropped_negative_placeholder"],
        "dropped_negative_too_short": summary["dropped_negative_too_short"],
        "dropped_negative_duplicate": summary["dropped_negative_duplicate"],
        "dropped_negative_missing_kind": summary["dropped_negative_missing_kind"],
        "dimension_names": seen_dimensions,
        "negative_kind_counts": dict(sorted(negative_kind_counts.items())),
        "negative_kind_counts_trainable_all": count_negative_kinds(trainable_groups_total),
        "negative_kind_counts_train": count_negative_kinds(train_groups),
        "negative_kind_counts_dev": count_negative_kinds(dev_groups),
        "negative_count_trainable_all": count_negative_candidates(trainable_groups_total),
        "negative_count_train": count_negative_candidates(train_groups),
        "negative_count_dev": count_negative_candidates(dev_groups),
        "sample_train_output": args.sample_train_output,
        "sample_train_groups": len(sample_train_groups) if args.sample_train_output else 0,
        "sample_dev_output": args.sample_dev_output,
        "sample_dev_groups": len(sample_dev_groups) if args.sample_dev_output else 0,
    }

    write_jsonl(args.proto_output, proto_groups)
    write_jsonl(args.train_output, train_groups)
    if args.dev_output:
        write_jsonl(args.dev_output, dev_groups)
    if args.sample_train_output:
        write_jsonl(args.sample_train_output, sample_train_groups)
    if args.sample_dev_output:
        write_jsonl(args.sample_dev_output, sample_dev_groups)
    write_json(args.summary_output, summary_payload)


if __name__ == "__main__":
    main()
