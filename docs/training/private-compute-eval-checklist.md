# Private Compute Eval And Backend Smoke Checklist

Use this checklist when moving a BashGym training/eval run from local contract
proof to a private compute target or cloud GPU backend. The goal is to collect
enough evidence to know whether the external backend path works, without spending
a long training run first.

---

## Preconditions

Do not start private or cloud compute work until the local bundle exists:

```bash
bashgym training smoke-bundle \
  --replay data/dppo_replay/latest.jsonl \
  --output-dir data/backend-smokes/latest \
  --backend auto \
  --json
```

Required local state:

- `contract_ready=true`
- `optimizer_ready=true` for real DPPO optimizer updates
- `world_model_probe.batch.echo_observation_tokens > 0` when ECHO is enabled
- `world_model_probe.batch.rwml_transitions > 0` when RWML is enabled
- `dppo_launch_env.json` is saved

If `backend_launch_ready=false`, that is acceptable for local work. The private
or cloud target may have the backend installed even when the desktop does not.

---

## Files To Copy Or Sync

Minimum:

- DPPO replay JSONL.
- `backend_smoke_readiness.json`.
- `dppo_replay_summary.json`.
- `world_model_backend_probe.json`.
- `dppo_launch_env.json`.
- Optional `launch_dppo_smoke.sh`.
- The exact BashGym commit or branch containing the smoke-bundle code.

Recommended remote layout:

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

---

## Compute Target Environment Preflight

On the target machine, capture:

```bash
python --version
nvidia-smi || true
free -h || true
df -h .
git rev-parse HEAD
```

Backend-specific:

```bash
python -c "import verl; print('verl ok')" || true
python -c "import skyrl; print('skyrl ok')" || true
test -d "$TMAX_OPEN_INSTRUCT_DIR" && echo "open-instruct checkout ok" || true
```

If your target is reached through an operator-managed shell, capture the same
checks with your configured host alias:

```bash
ssh <compute-target> "set -e; hostname; python --version; nvidia-smi || true; free -h || true; df -h ."
ssh <compute-target> "cd ~/bashgym-smokes/<run-id> && git rev-parse HEAD || true"
ssh <compute-target> "python -c 'import verl; print(\"verl ok\")' || true"
ssh <compute-target> "python -c 'import skyrl; print(\"skyrl ok\")' || true"
```

Capture stdout/stderr. A failed backend import is not a BashGym replay failure by
itself; it means the installed backend environment is missing or not activated.

If the backend is installed in a custom environment, use
`--command-template` locally or edit the launch script on the target while
preserving the exported `BASHGYM_DPPO_*` variables.

---

## One-Step Backend Smoke

Start with one step:

```bash
set -euo pipefail
cd ~/bashgym-smokes/<run-id>
bash launch_dppo_smoke.sh 2>&1 | tee logs/backend-smoke.log
```

If using a command template:

```bash
bashgym training smoke-bundle \
  --replay replay.jsonl \
  --output-dir . \
  --backend verl \
  --command-template 'python -m verl.trainer.main_ppo data.train_files={replay_path} data.val_files={replay_path} actor_rollout_ref.model.path={base_model} trainer.total_epochs=1 trainer.default_local_dir={output_dir}' \
  --json
```

Keep the smoke small until these are true:

- Backend reads the replay file.
- Backend logs training/validation startup.
- ECHO hook can build/use observation masks when enabled.
- RWML reward path can score predictions when enabled.
- Output directory is written.
- The run exits cleanly or fails with an actionable backend/config error.

---

## Compute Decision Log

Save a short text or JSON note with each private/cloud compute attempt:

| Field | Example |
|---|---|
| `run_id` | `dppo-smoke-2026-06-24-001` |
| `local_commit` | Git SHA that produced the replay and smoke bundle. |
| `target_commit` | Git SHA checked out on the target, if BashGym code is used there. |
| `backend_env` | Conda/uv/venv name or module path used for verl/SkyRL/open-instruct. |
| `command` | Exact one-step smoke command or command template. |
| `verdict` | `passed`, `backend_missing`, `contract_failed`, or `runtime_failed`. |
| `next_action` | The smallest fix before another remote attempt. |

This note keeps the run useful even when it fails. The goal of the first compute
attempt is to locate the exact runtime boundary, not to produce a strong model.

---

## Evidence To Bring Back

Collect:

- `logs/backend-smoke.log`
- backend config/hydra output if present
- metrics JSONL or trainer state
- output/checkpoint directory listing
- any ECHO/RWML metrics:
  - `echo_loss`
  - `rwml_pass_rate`
  - `embedding_distance_mean`
  - `embedding_distance_p95`
  - `exit_code_accuracy`
  - `test_result_accuracy`
- any DPPO telemetry:
  - masked token rate
  - Binary-TV/KL threshold
  - behavior/train logprob counts

Then run local analysis:

```bash
bashgym training analyze \
  --metrics data/models/<run-id>/metrics.jsonl \
  --replay data/dppo_replay/<run-id>.jsonl \
  --smoke-bundle data/backend-smokes/<run-id>/backend_smoke_readiness.json \
  --json
```

Attach release evidence only after heldout/pass@k gates run.

---

## Stop Conditions

Stop and fix before scaling when:

- Replay schema is rejected.
- Behavior or train logprobs are missing.
- ECHO masks cannot be built.
- RWML has zero targets.
- Backend starts but silently ignores the reward/loss hooks.
- Verifier, timeout, or tamper signals regress.
- World-model metrics improve but pass@k or safety gets worse.

The first successful private/cloud compute smoke is a contract proof, not a
production training result. The next step after a clean smoke is a small
before/after pass@k run.
