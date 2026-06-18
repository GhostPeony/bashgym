#!/usr/bin/env bash
# Set up an ISOLATED serving/eval venv on the DGX Spark (GB10, sm_121, aarch64, CUDA 13).
#
# This is S8's remaining work, codified. It is NON-DESTRUCTIVE: it builds a fresh
# ~/bashgym-serve venv and never touches the working ~/bashgym-train (training) venv
# or the running Ollama server. Prove this venv, then optionally adopt it.
#
# Why: the consolidated stack (transformers 5.5 + Unsloth + vLLM + TRL) already
# coexists in bashgym-train, but with torch 2.10 (warns on sm_121), trl 1.0, and a
# cu12 vLLM build on a cu13 system. This venv gets the correct, current stack so vLLM
# can serve base+candidate models for the held-out eval runner (S3) and benchmarks (S6).
#
# Preflight first (read-only, installs nothing):
#                    bash ~/bashgym/scripts/setup_dgx_serve.sh --check
# Run ON the GX10:   bash ~/bashgym/scripts/setup_dgx_serve.sh
# or from desktop:   ssh ponyo@192.168.50.173 'cd ~/bashgym && git pull --ff-only && bash scripts/setup_dgx_serve.sh'
set -euo pipefail

VENV="${BASHGYM_SERVE_VENV:-$HOME/bashgym-serve}"
TORCH_CU130_INDEX="https://download.pytorch.org/whl/cu130"

# --check: read-only preflight of every hard prerequisite. Installs nothing,
# creates nothing, exits 0/1 on go/no-go. Verified green on GB10 2026-06-16.
if [ "${1:-}" = "--check" ]; then
  echo "==> Preflight (read-only; installs nothing)"
  ok=1
  if command -v python3.12 >/dev/null; then
    echo "  [ok]   python3.12: $(python3.12 --version 2>&1)"
    python3.12 -c "import venv" 2>/dev/null && echo "  [ok]   venv module present" \
      || { echo "  [FAIL] python3.12 has no venv module"; ok=0; }
  else
    echo "  [FAIL] python3.12 not on PATH (needed to build the venv)"; ok=0
  fi
  avail_gb=$(df -P "$HOME" | awk 'NR==2{printf "%.0f", $4/1024/1024}')
  if [ "${avail_gb:-0}" -ge 20 ]; then echo "  [ok]   free disk: ${avail_gb}G"; \
    else echo "  [FAIL] only ${avail_gb}G free (need ~20G for torch+vllm wheels)"; ok=0; fi
  curl -sfI -m 8 "$TORCH_CU130_INDEX/" >/dev/null \
    && echo "  [ok]   cu130 wheel index reachable" \
    || { echo "  [FAIL] $TORCH_CU130_INDEX unreachable"; ok=0; }
  if [ -x "$HOME/bashgym-train/bin/python" ]; then
    "$HOME/bashgym-train/bin/python" -c "import transformers,vllm; print('  [info] train venv already coexists: transformers', transformers.__version__, '+ vllm', vllm.__version__)" 2>/dev/null \
      || echo "  [info] train venv present (version probe skipped)"
  fi
  if [ "$ok" = 1 ]; then echo "==> Preflight OK — safe to run without --check"; exit 0; \
    else echo "==> Preflight found blocking issues (see [FAIL] above)"; exit 1; fi
fi

echo "==> Target venv: $VENV (training venv ~/bashgym-train is left untouched)"

if [ -d "$VENV" ] && [ "${1:-}" != "--force" ]; then
  echo "    $VENV already exists. Re-run with --force to recreate, or delete it first."
else
  rm -rf "$VENV" 2>/dev/null || true
  python3.12 -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

echo "==> Upgrading pip/wheel"
"$PIP" install -U pip wheel

echo "==> torch 2.11 (cu130 — proper sm_121/GB10 support; 2.10 only warns)"
"$PIP" install "torch==2.11.0" --index-url "$TORCH_CU130_INDEX"

echo "==> transformers 5.5 + trl 1.6 (AsyncGRPO) + peft/accelerate/datasets"
"$PIP" install "transformers==5.5.*" "trl==1.6.0" peft accelerate datasets

echo "==> vLLM >=0.22.1 (cu130 default wheels — logprob-capable serving)"
"$PIP" install "vllm>=0.22.1"

echo "==> Verification"
"$PY" - <<'PYCHECK'
import warnings; warnings.filterwarnings("ignore")
import torch, transformers, trl
print("torch        ", torch.__version__)
print("transformers ", transformers.__version__)
print("trl          ", trl.__version__)
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
print("cuda cap     ", cap, "(expect (12, 1) on GB10 — no 'max 12.0' warning means torch 2.11 is good)")
try:
    a = torch.randn(64, 64, device="cuda"); (a @ a).sum().item()
    print("GB10 matmul   OK")
except Exception as e:
    print("GB10 matmul   FAILED:", repr(e)[:160])
try:
    import vllm; print("vllm         ", vllm.__version__, "(import OK)")
except Exception as e:
    print("vllm import   FAILED:", repr(e)[:200]); raise
print("OK: serving venv ready")
PYCHECK

cat <<EOF

==> Done. The serving venv is at: $VENV
    Serve a merged checkpoint for the eval runner (S3) / benchmarks (S6):
        $VENV/bin/vllm serve <path-to-merged-ckpt> --port 8100
    Keep ~/bashgym-train as the training fallback until this venv is proven on a
    Gemma 4 + Qwen 3.6 fine-tune + serve.
EOF
