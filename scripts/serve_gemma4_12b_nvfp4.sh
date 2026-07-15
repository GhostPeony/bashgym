#!/usr/bin/env bash
set -euo pipefail

# Safe, local vLLM canary for the Blackwell NVFP4 deployment artifact. The
# matching training base is unsloth/gemma-4-12b-it; do not train this quant.
MODEL_REPO="${MODEL_REPO:-unsloth/gemma-4-12b-it-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-4c77b7a6786b83271b3cd3285c03d7191a7ca442}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/gemma-4-12b-nvfp4}"
VLLM_VENV="${VLLM_VENV:-$HOME/bashgym-training/envs/unsloth-nvfp4}"
VLLM_BIN="${VLLM_BIN:-$VLLM_VENV/bin/vllm}"
HF_BIN="${HF_BIN:-hf}"
PORT="${PORT:-8892}"
HOST="${HOST:-127.0.0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma4-12b-nvfp4}"
LORA_PATH="${LORA_PATH:-}"
LORA_NAME="${LORA_NAME:-bashgym-gemma4-12b-sft}"
LORA_RANK="${LORA_RANK:-16}"
CANARY_UNIT="${CANARY_UNIT:-bashgym-gemma4-12b-nvfp4.service}"
WATCHDOG_UNIT="${WATCHDOG_UNIT:-bashgym-gemma4-12b-nvfp4-watchdog.service}"
PROTECTED_ENDPOINTS="${PROTECTED_ENDPOINTS:-http://127.0.0.1:8888/health,http://127.0.0.1:8889/health}"
MIN_AVAILABLE_KIB="${MIN_AVAILABLE_KIB:-12582912}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/bashgym-training/cache/nvfp4}"

usage() {
  echo "Usage: $0 prepare|preflight|start|status|stop"
}

check_protected_endpoints() {
  local endpoint code
  IFS=',' read -r -a endpoints <<< "$PROTECTED_ENDPOINTS"
  for endpoint in "${endpoints[@]}"; do
    [[ -n "$endpoint" ]] || continue
    code=$(curl -sS -m 3 -o /dev/null -w '%{http_code}' "$endpoint" 2>/dev/null || true)
    if [[ "$code" != "200" ]]; then
      echo "Protected endpoint is unhealthy: $endpoint (HTTP ${code:-unreachable})" >&2
      return 1
    fi
  done
}

prepare() {
  command -v "$HF_BIN" >/dev/null
  mkdir -p "$MODEL_PATH"
  "$HF_BIN" download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_PATH"
}

preflight() {
  [[ -x "$VLLM_BIN" ]] || { echo "Missing vLLM runtime: $VLLM_BIN" >&2; return 1; }
  [[ -f "$MODEL_PATH/config.json" ]] || {
    echo "Missing pinned model at $MODEL_PATH; run '$0 prepare'." >&2
    return 1
  }
  command -v systemd-run >/dev/null
  command -v curl >/dev/null
  check_protected_endpoints
  "$VLLM_VENV/bin/python" -c \
    'import torch, vllm; print({"cuda_capability": torch.cuda.get_device_capability(), "vllm": vllm.__version__})'
}

start() {
  preflight
  if systemctl --user is-active --quiet "$CANARY_UNIT"; then
    echo "$CANARY_UNIT is already active."
    start_watchdog
    return 0
  fi
  local -a lora_args=()
  if [[ -n "$LORA_PATH" ]]; then
    [[ -f "$LORA_PATH/adapter_config.json" ]] || {
      echo "Missing LoRA adapter_config.json in $LORA_PATH" >&2
      return 1
    }
    lora_args=(
      --enable-lora
      --lora-modules "$LORA_NAME=$LORA_PATH"
      --max-lora-rank "$LORA_RANK"
      --max-loras 1
    )
  fi

  mkdir -p "$CACHE_ROOT"/{vllm,torchinductor,triton}
  systemd-run --user --no-block --unit="$CANARY_UNIT" --collect \
    --property=TimeoutStopSec=10s \
    --property=MemoryHigh=20G \
    --property=MemoryMax=24G \
    --property=MemorySwapMax=0 \
    --setenv=MAX_JOBS=2 \
    --setenv="PATH=$VLLM_VENV/bin:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --setenv=CUTE_DSL_ARCH=sm_121a \
    --setenv="XDG_CACHE_HOME=$CACHE_ROOT" \
    --setenv="VLLM_CACHE_ROOT=$CACHE_ROOT/vllm" \
    --setenv="TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/torchinductor" \
    --setenv="TRITON_CACHE_DIR=$CACHE_ROOT/triton" \
    "$VLLM_BIN" serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.16 \
    --kv-cache-memory-bytes 1G \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1024 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --generation-config vllm \
    --mm-processor-cache-gb 0 \
    "${lora_args[@]}"

  start_watchdog
}

start_watchdog() {
  if systemctl --user is-active --quiet "$WATCHDOG_UNIT"; then
    echo "$WATCHDOG_UNIT is already active."
    return 0
  fi
  local watchdog_script
  watchdog_script='while systemctl --user is-active --quiet "$CANARY_UNIT"; do
    avail=$(grep MemAvailable /proc/meminfo | tr -dc "0-9")
    unhealthy=0
    for endpoint in $(printf "%s" "$PROTECTED_ENDPOINTS" | tr "," " "); do
      [ -n "$endpoint" ] || continue
      code=$(curl -sS -m 2 -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null || true)
      [ "$code" = 200 ] || unhealthy=1
    done
    if [ -z "$avail" ] || [ "$avail" -lt "$MIN_AVAILABLE_KIB" ] || [ "$unhealthy" -ne 0 ]; then
      echo "SAFETY_STOP mem_available_kib=$avail protected_endpoint_failure=$unhealthy"
      systemctl --user kill --kill-whom=all --signal=KILL "$CANARY_UNIT"
      exit 1
    fi
    sleep 1
  done'
  systemd-run --user --no-block --unit="$WATCHDOG_UNIT" --collect \
    --setenv="CANARY_UNIT=$CANARY_UNIT" \
    --setenv="MIN_AVAILABLE_KIB=$MIN_AVAILABLE_KIB" \
    --setenv="PROTECTED_ENDPOINTS=$PROTECTED_ENDPOINTS" \
    /bin/bash -c "$watchdog_script"
}

status() {
  systemctl --user show "$CANARY_UNIT" \
    -p ActiveState -p SubState -p MainPID -p MemoryCurrent -p MemoryPeak \
    -p MemoryHigh -p MemoryMax -p NRestarts --no-pager 2>/dev/null || true
  free -h
  check_protected_endpoints || true
  curl -sS -m 3 "http://127.0.0.1:$PORT/health" || true
  echo
}

stop() {
  systemctl --user kill --kill-whom=all --signal=KILL "$WATCHDOG_UNIT" 2>/dev/null || true
  systemctl --user kill --kill-whom=all --signal=KILL "$CANARY_UNIT" 2>/dev/null || true
}

case "${1:-status}" in
  prepare) prepare ;;
  preflight) preflight ;;
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) usage; exit 2 ;;
esac
