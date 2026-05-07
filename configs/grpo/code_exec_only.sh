#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29651}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"
TRAIN_FILE="${TRAIN_FILE:-data/code_rm/prepared/train_groups_seed18_new_full.jsonl}"
ORACLE_FILE="${ORACLE_FILE:-data/code_rm/code_test_oracles.full.statement.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/code-qwen25-7b-exec-only}"
EXECUTOR_BASE_URL="${EXECUTOR_BASE_URL:-http://localhost:8000/v1}"
EXECUTOR_MODEL="${EXECUTOR_MODEL:-qwen2.5-7b}"
EXECUTOR_API_KEY="${EXECUTOR_API_KEY:-${OPENAI_API_KEY:-}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bf16_grpo.json}"

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  scripts/train_code_grpo.py
  --model_path "${MODEL_PATH}"
  --train_file "${TRAIN_FILE}"
  --oracle_file "${ORACLE_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --reward_mode exec_only
  --exec_reward_style "${EXEC_REWARD_STYLE:-hybrid}"
  --num_generations "${NUM_GENERATIONS:-4}"
  --max_steps "${MAX_STEPS:-600}"
  --max_train_items "${MAX_TRAIN_ITEMS:-0}"
  --sample_mode "${SAMPLE_MODE:-random}"
  --learning_rate "${LEARNING_RATE:-1e-6}"
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
  --max_new_tokens "${MAX_NEW_TOKENS:-512}"
  --temperature "${TEMPERATURE:-0.7}"
  --top_p "${TOP_P:-0.95}"
  --max_problem_chars "${MAX_PROBLEM_CHARS:-3500}"
  --max_reason_chars "${MAX_REASON_CHARS:-2500}"
  --oracle_sources "${ORACLE_SOURCES:-reward}"
  --max_oracle_tests "${MAX_ORACLE_TESTS:-8}"
  --exec_timeout "${EXEC_TIMEOUT:-2.0}"
  --exec_memory_mb "${EXEC_MEMORY_MB:-1024}"
  --executor_base_url "${EXECUTOR_BASE_URL}"
  --executor_model "${EXECUTOR_MODEL}"
  --executor_api_type "${EXECUTOR_API_TYPE:-completions}"
  --executor_temperature "${EXECUTOR_TEMPERATURE:-0.5}"
  --executor_max_tokens "${EXECUTOR_MAX_TOKENS:-768}"
  --api_timeout "${API_TIMEOUT:-300}"
  --api_failure_reward "${API_FAILURE_REWARD:-0.0}"
  --max_api_failure_warnings "${MAX_API_FAILURE_WARNINGS:-20}"
  --bf16
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
  --save_steps "${SAVE_STEPS:-100}"
  --seed "${SEED:-42}"
  --print_metrics
)

if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  CMD+=(--deepspeed_config "${DEEPSPEED_CONFIG}")
fi
if [[ -n "${EXECUTOR_API_KEY}" ]]; then
  CMD+=(--executor_api_key "${EXECUTOR_API_KEY}")
fi
if [[ "${USE_LORA:-0}" == "1" || "${USE_LORA:-false}" == "true" ]]; then
  CMD+=(--use_lora --lora_r "${LORA_R:-16}" --lora_alpha "${LORA_ALPHA:-32}" --lora_dropout "${LORA_DROPOUT:-0.05}")
fi
if [[ "${NO_PROXY_ALL:-0}" == "1" ]]; then
  CMD+=(--no_proxy)
fi
if [[ -n "${SKIP_FINAL_SAVE:-}" ]]; then
  CMD+=(--skip_final_save)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"

