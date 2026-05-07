import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def filter_group(group, min_negatives, require_negative_below_positive):
    if not group.get("positive_pool"):
        return None, "missing_positive"
    pos = group["positive_pool"][0]
    pos_rubric = pos.get("rubric") or {}
    if "total" not in pos_rubric:
        return None, "missing_positive_rubric"

    pos_total = float(pos_rubric["total"])
    negatives = []
    for negative in group.get("negative_bank") or []:
        rubric = negative.get("rubric") or {}
        if "total" not in rubric:
            continue
        if require_negative_below_positive and float(rubric["total"]) >= pos_total:
            continue
        negatives.append(negative)

    if len(negatives) < min_negatives:
        return None, "too_few_negatives"

    filtered = dict(group)
    filtered["positive_pool"] = list(group["positive_pool"])
    filtered["negative_bank"] = negatives
    return filtered, None


def summarize(rows):
    neg_dist = Counter()
    kind_counts = Counter()
    for row in rows:
        negatives = row.get("negative_bank") or []
        neg_dist[len(negatives)] += 1
        kind_counts.update(negative.get("negative_kind") for negative in negatives)
    return {
        "groups": len(rows),
        "negative_count_distribution": dict(sorted(neg_dist.items())),
        "negative_kind_counts": dict(sorted(kind_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Filter and split prepared code RM groups.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--smoke-train-output", default=None)
    parser.add_argument("--smoke-dev-output", default=None)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--smoke-train-size", type=int, default=32)
    parser.add_argument("--smoke-dev-size", type=int, default=8)
    parser.add_argument("--min-negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument(
        "--require-negative-below-positive",
        action="store_true",
        help="Drop negatives whose rubric total is not lower than the positive total.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    filtered_rows = []
    drop_reasons = Counter()
    for row in rows:
        filtered, reason = filter_group(
            row,
            min_negatives=args.min_negatives,
            require_negative_below_positive=args.require_negative_below_positive,
        )
        if filtered is None:
            drop_reasons[reason] += 1
            continue
        filtered_rows.append(filtered)

    rng = random.Random(args.seed)
    filtered_rows.sort(key=lambda item: str(item.get("problem_id", "")))
    rng.shuffle(filtered_rows)

    dev_size = min(args.dev_size, max(0, len(filtered_rows) // 10), len(filtered_rows))
    dev_rows = filtered_rows[:dev_size]
    train_rows = filtered_rows[dev_size:]

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)

    smoke = {}
    if args.smoke_train_output:
        smoke_train = train_rows[: args.smoke_train_size]
        write_jsonl(args.smoke_train_output, smoke_train)
        smoke["train"] = summarize(smoke_train)
    if args.smoke_dev_output:
        smoke_dev = dev_rows[: args.smoke_dev_size]
        write_jsonl(args.smoke_dev_output, smoke_dev)
        smoke["dev"] = summarize(smoke_dev)

    write_json(
        args.summary_output,
        {
            "input": args.input,
            "seed": args.seed,
            "min_negatives": args.min_negatives,
            "require_negative_below_positive": args.require_negative_below_positive,
            "input_groups": len(rows),
            "kept_groups": len(filtered_rows),
            "drop_reasons": dict(sorted(drop_reasons.items())),
            "train": summarize(train_rows),
            "dev": summarize(dev_rows),
            "smoke": smoke,
        },
    )


if __name__ == "__main__":
    main()
