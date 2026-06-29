# GX10 Training Smoke Plan

Status: requires user approval before any SSH or remote command execution.

Purpose: prove the BashGym DPPO plus ECHO/RWML handoff on GX10 with the smallest
runtime attempt possible. This is a contract and integration proof, not a real
model-improvement run.

## Scope

Run only these phases:

1. Local verification of the current training CLI and smoke-bundle code.
2. Local replay readiness check against a real DPPO replay artifact, if one is
   available.
3. GX10 environment preflight on `ponyo`.
4. One-step installed-backend smoke if a backend is available.
5. Local analysis of returned artifacts.

Do not start a long training run in this pass.

## Local Commands

```bash
python -m pytest tests/cli/test_cli.py tests/gym/test_backend_smoke_bundle.py tests/gym/test_dppo_launcher.py tests/gym/test_world_model_trainer_adapter.py -q -o addopts=
python -m ruff check bashgym/cli.py bashgym/gym/backend_smoke_bundle.py bashgym/gym/dppo_launcher.py bashgym/gym/run_analysis.py tests/cli/test_cli.py tests/gym/test_backend_smoke_bundle.py tests/gym/test_dppo_launcher.py
python -m black --check bashgym/cli.py bashgym/gym/backend_smoke_bundle.py bashgym/gym/dppo_launcher.py bashgym/gym/run_analysis.py tests/cli/test_cli.py tests/gym/test_backend_smoke_bundle.py tests/gym/test_dppo_launcher.py
git diff --check
python -m bashgym.cli manifest --json
```

If a replay exists:

```bash
bashgym replay summarize data/dppo_replay/<run-id>.jsonl --json
bashgym training smoke-bundle \
  --replay data/dppo_replay/<run-id>.jsonl \
  --output-dir data/backend-smokes/<run-id> \
  --backend auto \
  --json
```

Local promotion gate:

- `contract_ready=true`
- `optimizer_ready=true` for real DPPO optimizer updates
- ECHO observation counts are non-zero when ECHO is enabled
- RWML transition counts are non-zero when RWML is enabled

If no real replay exists, stop before GX10 and create or export one from served
environment rollouts.

## GX10 Preflight Commands

Run only after approval:

```bash
ssh ponyo "set -e; hostname; python --version; nvidia-smi || true; free -h || true; df -h ."
ssh ponyo "python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())' || true"
ssh ponyo "python -c 'import verl; print(\"verl ok\")' || true"
ssh ponyo "python -c 'import skyrl; print(\"skyrl ok\")' || true"
ssh ponyo "test -d \"$TMAX_OPEN_INSTRUCT_DIR\" && echo open-instruct ok || true"
```

If the backend environment needs activation, rerun the import probes inside that
environment and record the activation command.

## Remote Layout

```text
~/bashgym-smokes/<run-id>/
  replay.jsonl
  backend_smoke_readiness.json
  dppo_replay_summary.json
  world_model_backend_probe.json
  dppo_launch_env.json
  launch_dppo_smoke.sh
  logs/
  outputs/
```

Copy artifacts only after local readiness is clean.

## One-Step Backend Smoke

```bash
ssh ponyo "cd ~/bashgym-smokes/<run-id> && mkdir -p logs outputs && bash launch_dppo_smoke.sh 2>&1 | tee logs/backend-smoke.log"
```

If no generated script is available, use a backend-specific command template
that preserves all `BASHGYM_DPPO_*` environment variables from
`dppo_launch_env.json`.

## Evidence To Bring Back

- `logs/backend-smoke.log`
- backend config or hydra output
- backend metrics JSONL or trainer state
- output/checkpoint directory listing
- `echo_loss`, if ECHO ran
- `rwml_pass_rate` and embedding-distance stats, if RWML ran
- DPPO mask telemetry and behavior/train logprob counts

Then run locally:

```bash
bashgym training analyze \
  --metrics data/models/<run-id>/metrics.jsonl \
  --replay data/dppo_replay/<run-id>.jsonl \
  --smoke-bundle data/backend-smokes/<run-id>/backend_smoke_readiness.json \
  --json
```

## Stop Conditions

Stop before another remote attempt if:

- replay schema is rejected
- behavior or train logprobs are missing for DPPO optimizer updates
- ECHO masks cannot be built
- RWML has zero targets
- backend imports fail outside a known missing environment activation
- backend starts but silently ignores ECHO/RWML hooks
- verifier, timeout, or tamper signals regress

Record the exact blocker as `backend_missing`, `contract_failed`, or
`runtime_failed`.
