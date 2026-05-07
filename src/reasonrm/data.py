import json
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .modeling_reward import DEFAULT_DIMENSION_NAMES


def _load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content[0] == "[":
        return json.loads(content)
    items = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def render_reasoning_text(group, candidate, task_name="code"):
    task_type = group.get("task_type")
    if task_type is None:
        task_type = group.get("metadata", {}).get("task_type") or group.get("source", "unknown")

    tests = group.get("tests_or_constraints") or group.get("tests") or ""
    parts = [
        "<task>{}</task>".format(task_name),
        "<type>{}</type>".format(task_type),
        "<problem>",
        group.get("problem", "").strip(),
        "</problem>",
    ]
    if tests:
        parts.extend(["<tests>", tests.strip(), "</tests>"])
    parts.extend(["<reasoning>", candidate.get("reasoning", "").strip(), "</reasoning>"])
    return "\n".join(parts)


class ProblemGroupDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.groups = _load_json_or_jsonl(path)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]


class ReasonRewardDataCollator(object):
    def __init__(
        self,
        tokenizer,
        max_length,
        num_negatives,
        task_name="code",
        dimension_names=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.task_name = task_name
        self.dimension_names = list(dimension_names or DEFAULT_DIMENSION_NAMES)

    def _sample_negative_bank(self, group):
        negatives = group.get("negative_bank", [])
        if len(negatives) >= self.num_negatives:
            return random.sample(negatives, self.num_negatives)
        if not negatives:
            raise ValueError("group {} has no negative_bank".format(group.get("problem_id")))
        return [random.choice(negatives) for _ in range(self.num_negatives)]

    def _rubric_to_labels(self, rubric):
        labels = []
        for name in self.dimension_names:
            value = rubric.get(name)
            if value is None:
                raise KeyError("missing rubric dimension '{}'".format(name))
            labels.append(int(value))
        return labels

    def __call__(self, features):
        texts = []
        dimension_labels = []
        gold_total = []
        candidate_is_positive = []
        group_sizes = []

        for group in features:
            positive_pool = group.get("positive_pool", [])
            if not positive_pool:
                raise ValueError("group {} has no positive_pool".format(group.get("problem_id")))
            pos = random.choice(positive_pool)
            negs = self._sample_negative_bank(group)
            candidates = [pos] + negs

            group_sizes.append(len(candidates))
            for candidate_index, candidate in enumerate(candidates):
                texts.append(render_reasoning_text(group, candidate, task_name=self.task_name))
                rubric = candidate.get("rubric", {})
                dimension_labels.append(self._rubric_to_labels(rubric))
                gold_total.append(float(rubric["total"]))
                candidate_is_positive.append(candidate_index == 0)

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["dimension_labels"] = torch.tensor(dimension_labels, dtype=torch.long)
        batch["gold_total"] = torch.tensor(gold_total, dtype=torch.float32)
        batch["candidate_is_positive"] = torch.tensor(candidate_is_positive, dtype=torch.bool)
        batch["group_sizes"] = torch.tensor(group_sizes, dtype=torch.long)
        return batch
