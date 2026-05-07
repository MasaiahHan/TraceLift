"""Reward utilities shared by TraceLift GRPO training scripts."""

from __future__ import annotations

from typing import Any, Dict


def classify_execution(execution: Dict[str, Any]) -> str:
    if execution.get("execution_ok"):
        return "passed"
    failures = execution.get("execution_failures") or []
    if not failures:
        return "no_tests"
    failure = failures[0]
    stderr = str(failure.get("stderr") or "")
    if failure.get("timeout"):
        return "timeout"
    if failure.get("returncode") not in (0, None):
        if "SyntaxError" in stderr or "IndentationError" in stderr:
            return "compile_error"
        return "runtime_error"
    return "failed_test"


def exec_reward(execution: Dict[str, Any], style: str) -> float:
    total = int(execution.get("execution_total") or 0)
    passed = int(execution.get("execution_passed") or 0)
    if style == "binary":
        return 1.0 if execution.get("execution_ok") else 0.0
    if style == "pass_rate":
        return float(passed) / float(total) if total > 0 else 0.0
    if style == "coderl":
        outcome = classify_execution(execution)
        if outcome == "passed":
            return 1.0
        if outcome == "compile_error":
            return -1.0
        if outcome in {"runtime_error", "timeout"}:
            return -0.6
        if outcome == "failed_test":
            return -0.3
        return -1.0
    if style == "hybrid":
        pass_part = float(passed) / float(total) if total > 0 else 0.0
        outcome = classify_execution(execution)
        validity_bonus = 0.0
        if outcome in {"passed", "failed_test"}:
            validity_bonus = 0.2
        elif outcome in {"runtime_error", "timeout"}:
            validity_bonus = 0.05
        return 0.8 * pass_part + validity_bonus
    raise ValueError("unknown exec reward style: {}".format(style))

