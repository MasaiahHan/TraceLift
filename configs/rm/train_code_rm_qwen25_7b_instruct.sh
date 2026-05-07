#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/code_rm/train_groups.jsonl}"
EVAL_FILE="${EVAL_FILE:-data/code_rm/dev_groups.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen25-7b-instruct-code-rm}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"

DIMENSION_NAMES="${DIMENSION_NAMES:-task_understanding,plan_quality,step_coherence,action_support,non_leakage}"
DIMENSION_WEIGHTS="${DIMENSION_WEIGHTS:-0.2,0.2,0.2,0.2,0.2}"
DIMENSION_LOSS_TYPE="${DIMENSION_LOSS_TYPE:-ce}"

USE_LORA="${USE_LORA:-True}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"

LAUNCH_CMD=("${PYTHON_BIN}" -m)
if [[ "${NPROC_PER_NODE}" != "1" ]]; then
  LAUNCH_CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    -m
  )
fi

CMD=(
  "${LAUNCH_CMD[@]}" reasonrm.train_reward_model
  --model_name_or_path "${BASE_MODEL}"
  --train_file "${TRAIN_FILE}"
  --eval_file "${EVAL_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --dimension_names "${DIMENSION_NAMES}"
  --dimension_weights "${DIMENSION_WEIGHTS}"
  --dimension_loss_type "${DIMENSION_LOSS_TYPE}"
  --task_name "${TASK_NAME:-code}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --logging_steps "${LOGGING_STEPS:-10}"
  --save_steps "${SAVE_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --eval_strategy steps
  --save_strategy steps
  --bf16 True
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-False}"
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS:-False}"
  --max_length "${MAX_LENGTH}"
  --num_negatives "${NUM_NEGATIVES:-4}"
  --use_lora "${USE_LORA}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --load_in_4bit "${LOAD_IN_4BIT}"
  --loss_dim_weight "${LOSS_DIM_WEIGHT:-1.0}"
  --loss_total_weight "${LOSS_TOTAL_WEIGHT:-0.5}"
  --loss_posneg_weight "${LOSS_POSNEG_WEIGHT:-0.7}"
  --loss_negneg_weight "${LOSS_NEGNEG_WEIGHT:-0.0}"
  --remove_unused_columns False
  --overwrite_output_dir "${OVERWRITE_OUTPUT_DIR:-False}"
  --report_to none
)

if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  CMD+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
