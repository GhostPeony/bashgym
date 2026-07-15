#!/usr/bin/env bash
set -euo pipefail

# Reproducible Ponyo launcher for the MemexAI positive-aware embedding campaign.
# Configuration is supplied through environment variables so the same staged
# script can run short proof jobs and resumable full candidates.

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/home/ponyo/bashgym-training/memexai-positive-aware-v1-20260712}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
MAX_PAIRS="${MAX_PAIRS:-696}"
MAX_STEPS="${MAX_STEPS:-0}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LOSS="${LOSS:-explicit_mnrl}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
NEGATIVES_PER_QUERY="${NEGATIVES_PER_QUERY:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
CHECKPOINT_SAVE_STEPS="${CHECKPOINT_SAVE_STEPS:--1}"
WALL_TIMEOUT_SECONDS="${WALL_TIMEOUT_SECONDS:-600}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-20}"
USE_BF16="${USE_BF16:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

case "$LOSS" in
  explicit_mnrl|cached_mnrl) ;;
  *)
    echo "LOSS must be explicit_mnrl or cached_mnrl, got: $LOSS" >&2
    exit 2
    ;;
esac

if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "RUN_NAME contains unsafe characters: $RUN_NAME" >&2
  exit 2
fi

PYTHON="$CAMPAIGN_ROOT/.venv/bin/python"
TRAINER="$CAMPAIGN_ROOT/scripts/train_embedding_retriever.py"
GROUPED="$CAMPAIGN_ROOT/inputs/positive-aware-train.jsonl"
CORPUS="$CAMPAIGN_ROOT/inputs/corpus-chunks.jsonl"
BASE_MODEL="/home/ponyo/models/embedding/qwen3-embedding-0.6b"
RUN_DIR="$CAMPAIGN_ROOT/runs/$RUN_NAME"

for required in "$PYTHON" "$TRAINER" "$GROUPED" "$CORPUS" "$BASE_MODEL"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required campaign input: $required" >&2
    exit 2
  fi
done

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 2
fi

available_bytes="$(awk '/MemAvailable:/ {print $2 * 1024}' /proc/meminfo)"
minimum_bytes="$((MIN_AVAILABLE_GIB * 1024 * 1024 * 1024))"
if (( available_bytes < minimum_bytes )); then
  echo "preflight rejected: available memory ${available_bytes}B is below ${MIN_AVAILABLE_GIB}GiB" >&2
  exit 3
fi

mkdir -p "$RUN_DIR"
printf 'epoch_utc,mem_used_bytes,mem_available_bytes,swap_used_bytes,gpu_util_pct,temp_c\n' \
  > "$RUN_DIR/resource_samples.csv"

sample_resources() {
  while true; do
    read -r mem_total mem_available swap_total swap_free < <(
      awk '
        /MemTotal:/ {mem_total=$2*1024}
        /MemAvailable:/ {mem_available=$2*1024}
        /SwapTotal:/ {swap_total=$2*1024}
        /SwapFree:/ {swap_free=$2*1024}
        END {print mem_total, mem_available, swap_total, swap_free}
      ' /proc/meminfo
    )
    gpu_sample="$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    printf '%s,%s,%s,%s,%s\n' \
      "$(date +%s)" \
      "$((mem_total - mem_available))" \
      "$mem_available" \
      "$((swap_total - swap_free))" \
      "$gpu_sample" \
      >> "$RUN_DIR/resource_samples.csv"
    sleep 2
  done
}

sample_resources &
sampler_pid=$!
cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

command=(
  "$PYTHON" "$TRAINER"
  --grouped-jsonl "$GROUPED"
  --corpus-jsonl "$CORPUS"
  --base-model-path "$BASE_MODEL"
  --output-dir "$RUN_DIR"
  --loss "$LOSS"
  --batch-size "$BATCH_SIZE"
  --mini-batch-size "$MINI_BATCH_SIZE"
  --max-pairs "$MAX_PAIRS"
  --negatives-per-query "$NEGATIVES_PER_QUERY"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --warmup-steps "$WARMUP_STEPS"
  --weight-decay 0.01
  --max-seq-length 1024
  --truncate-dim 768
  --temperature 0.02
  --seed 20260711
  --logging-steps "$LOGGING_STEPS"
  --checkpoint-save-steps "$CHECKPOINT_SAVE_STEPS"
)

if [[ "$USE_BF16" == "1" ]]; then
  command+=(--use-bf16)
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  command+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

printf '%q ' "${command[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

set +e
timeout --signal=TERM --kill-after=30s "$WALL_TIMEOUT_SECONDS" \
  /usr/bin/time -v "${command[@]}" \
  > >(tee "$RUN_DIR/training.log") \
  2> >(tee -a "$RUN_DIR/training.log" >&2)
exit_code=$?
set -e

printf '%s\n' "$exit_code" > "$RUN_DIR/exit_code"
exit "$exit_code"
