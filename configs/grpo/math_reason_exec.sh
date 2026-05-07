#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29654}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"
TRAIN_FILE="${TRAIN_FILE:-data/math_rm/gsm8k_even_3000/train_groups_min4.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/math-qwen25-7b-reason-exec}"
EXECUTOR_BASE_URL="${EXECUTOR_BASE_URL:-http://localhost:8000/v1}"
EXECUTOR_MODEL="${EXECUTOR_MODEL:-qwen2.5-7b}"
EXECUTOR_API_KEY="${EXECUTOR_API_KEY:-${OPENAI_API_KEY:-}}"
RM_SCORE_URL="${RM_SCORE_URL:-http://localhost:8001/score}"
RM_API_KEY="${RM_API_KEY:-}"

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  scripts/train_math_grpo_trl.py
  --model_path "${MODEL_PATH}"
  --train_file "${TRAIN_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --reward_mode reason_exec
  --e2e_reward_weight "${E2E_REWARD_WEIGHT:-0.5}"
  --reason_reward_weight "${REASON_REWARD_WEIGHT:-0.5}"
  --rm_score_url "${RM_SCORE_URL}"
  --num_generations "${NUM_GENERATIONS:-4}"
  --max_steps "${MAX_STEPS:-600}"
  --max_train_items "${MAX_TRAIN_ITEMS:-0}"
  --learning_rate "${LEARNING_RATE:-5e-6}"
  --max_prompt_length "${MAX_PROMPT_LENGTH:-2048}"
  --max_completion_length "${MAX_COMPLETION_LENGTH:-512}"
  --max_problem_chars "${MAX_PROBLEM_CHARS:-2500}"
  --max_reason_chars "${MAX_REASON_CHARS:-2000}"
  --executor_base_url "${EXECUTOR_BASE_URL}"
  --executor_model "${EXECUTOR_MODEL}"
  --executor_api_type "${EXECUTOR_API_TYPE:-completions}"
  --executor_extra_payload_json "${EXECUTOR_EXTRA_PAYLOAD_JSON:-{}}"
  --executor_repeats "${EXECUTOR_REPEATS:-3}"
  --baseline_repeats "${BASELINE_REPEATS:-3}"
  --uplift_exec_reward
  --executor_temperature "${EXECUTOR_TEMPERATURE:-0.5}"
  --executor_max_tokens "${EXECUTOR_MAX_TOKENS:-256}"
  --api_timeout "${API_TIMEOUT:-300}"
  --api_failure_reward "${API_FAILURE_REWARD:-0.0}"
  --rm_failure_score "${RM_FAILURE_SCORE:-0.5}"
  --max_api_failure_warnings "${MAX_API_FAILURE_WARNINGS:-20}"
  --use_lora "${USE_LORA:-true}"
  --lora_r "${LORA_R:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --save_steps "${SAVE_STEPS:-100}"
  --logging_steps "${LOGGING_STEPS:-1}"
  --seed "${SEED:-42}"
)

if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  CMD+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi
if [[ -n "${EXECUTOR_API_KEY}" ]]; then
  CMD+=(--executor_api_key "${EXECUTOR_API_KEY}")
fi
if [[ -n "${RM_API_KEY}" ]]; then
  CMD+=(--rm_api_key "${RM_API_KEY}")
fi
if [[ "${NO_PROXY_ALL:-0}" == "1" ]]; then
  CMD+=(--no_proxy)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"

