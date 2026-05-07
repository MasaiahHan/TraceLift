import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_python_code(text: str) -> str:
    text = text or ""
    fence = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        return fence.group(1).strip() + "\n"

    cut = len(text)
    for marker in ("\nI tested", "\nIn the ", "\nExplanation:", "\n###", "\nThe code"):
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip() + "\n"


def extract_statement_examples(problem: str, max_examples: int = 8) -> List[Dict[str, str]]:
    problem = problem or ""
    anchor = re.search(r"(?im)^\s*(?:Examples?|Sample Tests?|Sample Input)\s*$", problem)
    if not anchor:
        return []
    tail = problem[anchor.start() :]
    pattern = re.compile(
        r"(?:Sample\s+)?Input\s*\n+(.*?)\n\s*(?:Sample\s+)?Output\s*\n+(.*?)(?=\n\s*(?:Sample\s+)?Input\s*\n|\n\s*Note\b|\n\s*Explanation\b|\Z)",
        re.S | re.I,
    )

    tests = []
    for match in pattern.finditer(tail):
        sample_input = match.group(1).strip()
        sample_output = match.group(2).strip()
        if sample_input and sample_output:
            tests.append(
                {
                    "input": sample_input + "\n",
                    "output": sample_output + "\n",
                    "source": "statement_example",
                }
            )
        if len(tests) >= max_examples:
            break
    return tests


def normalize_output(text: str) -> List[str]:
    return (text or "").strip().split()


def compare_output(actual: str, expected: str) -> bool:
    return normalize_output(actual) == normalize_output(expected)


def _limit_resources(timeout_sec: float, memory_mb: int) -> None:
    try:
        import resource

        cpu_limit = max(1, int(math.ceil(timeout_sec)) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        if memory_mb > 0:
            memory_bytes = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    except Exception:
        # Resource limits are best-effort and unavailable on some platforms.
        pass


def run_python_code(
    code: str,
    stdin: str,
    timeout_sec: float = 2.0,
    memory_mb: int = 1024,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rrm_exec_") as tmpdir:
        script_path = Path(tmpdir) / "solution.py"
        script_path.write_text(code, encoding="utf-8")
        child_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        }
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                cwd=tmpdir,
                env=child_env,
                preexec_fn=lambda: _limit_resources(timeout_sec, memory_mb) if os.name == "posix" else None,
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "timeout": False,
            }
        except subprocess.TimeoutExpired:
            return {"returncode": None, "stdout": "", "stderr": "timeout", "timeout": True}


def select_oracle_tests(
    oracle: Optional[Dict[str, Any]],
    sources: Sequence[str],
    max_tests: int,
) -> List[Dict[str, str]]:
    if not oracle:
        return []

    wanted = set(sources)
    tests = []
    for test in oracle.get("tests", []):
        source = test.get("source")
        if "reward" in wanted and bool(test.get("is_reward_test")):
            tests.append(test)
        elif "feedback" in wanted and bool(test.get("is_feedback_test")):
            tests.append(test)
        elif source in wanted:
            tests.append(test)
        if max_tests > 0 and len(tests) >= max_tests:
            break
    return tests


def evaluate_completion(
    completion: str,
    tests: Sequence[Dict[str, str]],
    timeout_sec: float = 2.0,
    memory_mb: int = 1024,
) -> Dict[str, Any]:
    code = extract_python_code(completion)
    failures = []
    passed = 0
    for test_index, test in enumerate(tests):
        result = run_python_code(code, str(test.get("input") or ""), timeout_sec=timeout_sec, memory_mb=memory_mb)
        ok = (
            not result["timeout"]
            and result["returncode"] == 0
            and compare_output(str(result["stdout"]), str(test.get("output") or ""))
        )
        if ok:
            passed += 1
            continue
        failures.append(
            {
                "test_index": test_index,
                "source": test.get("source"),
                "returncode": result["returncode"],
                "timeout": result["timeout"],
                "stdout": str(result["stdout"])[:2000],
                "stderr": str(result["stderr"])[:2000],
                "expected": str(test.get("output") or "")[:2000],
            }
        )

    total = len(tests)
    return {
        "execution_ok": total > 0 and passed == total,
        "execution_passed": passed,
        "execution_total": total,
        "execution_failures": failures[:3],
    }


def load_oracle_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    oracles = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            oracle = json.loads(line)
            problem_id = oracle.get("problem_id")
            if problem_id:
                oracles[str(problem_id)] = oracle
    return oracles


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
