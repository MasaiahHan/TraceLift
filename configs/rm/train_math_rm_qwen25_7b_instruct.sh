#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TRAIN_FILE="${TRAIN_FILE:-data/math_rm/gsm8k_even_3000/train_groups_min4.jsonl}"
export EVAL_FILE="${EVAL_FILE:-data/math_rm/gsm8k_even_3000/dev_groups_min4.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen25-7b-instruct-math-rm}"
export DIMENSION_NAMES="${DIMENSION_NAMES:-problem_understanding,solution_strategy,step_coherence,calculation_correctness,answer_support}"
export DIMENSION_WEIGHTS="${DIMENSION_WEIGHTS:-0.2,0.2,0.2,0.2,0.2}"
export TASK_NAME="${TASK_NAME:-math}"

exec bash "${ROOT_DIR}/configs/rm/train_code_rm_qwen25_7b_instruct.sh" "$@"

