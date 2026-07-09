---
name: training
description: "Operate BashGym training and evaluation runs for Hermes or another configured agent endpoint. Use when launching, monitoring, stopping, classifying, or evaluating SFT, DPO, GRPO/RLVR, Session Distillation, DPPO replay, ECHO/RWML, reward-model, or cascade training runs."
---

# Training

Manage training through BashGym, not ad-hoc shell scripts. A run that matters must be registered with BashGym, produce run artifacts, and pass the method-appropriate eval gate.

## Default Rules

- Use `scripts/api.sh` or `bashgym training ...` commands first.
- Do not start raw Unsloth/HF scripts unless the user explicitly asks for manual debugging.
- Classify every run by method from `run_state.json`/API config, not by model name.
- A completed train loss curve is not a full eval. Require heldout, environment, release-evidence, or RunCard artifacts as appropriate.
- Stopping a run is destructive. Confirm with the user before calling `POST /api/training/{run_id}/stop`.

For the full method matrix and eval gates, read `references/bashgym-methods-and-evals.md`.
For private remote-trainer details, use the operator's local runbook or saved compute-target profile. Do not assume a machine-specific runbook is checked into the repo.

## Preflight

```bash
scripts/api.sh GET /api/health
scripts/api.sh GET /api/system/info
scripts/api.sh GET /api/system/recommendations
scripts/api.sh GET /api/ssh/preflight
```

The helper discovers the active API base URL. Prefer setting `BASHGYM_API_URL`; if the local backend port changes, set `BASHGYM_API_PORT`, `BASHGYM_API_URLS`, or `BASHGYM_API_URL_FILE`, or write the live URL to `~/.bashgym/api_url`. Skills should keep using logical `/api/...` paths.

On private unified-memory or remote GPU compute targets, prefer BashGym-managed runs with `use_remote_ssh: true` and `load_in_4bit: false` when the saved profile supports it. Use `sft_backend: "unsloth"` for known-good SFT smoke runs; fall back to `sft_backend: "plain"` if Unsloth import or hardware support fails. Only stop model services for memory after user approval.

## Start Runs

Plan first when selecting a method:

```bash
bashgym training plan --strategy sft --hardware private_compute --data traces --json
bashgym training plan --strategy grpo --hardware private_compute --data terminal_envs --json
bashgym training plan --strategy session-distillation --hardware private_compute --data traces --json
```

Remote SFT:

```bash
scripts/api.sh POST /api/training/start '{
  "strategy": "sft",
  "dataset_path": "/path/to/train.jsonl",
  "base_model": "<model-id>",
  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_seq_length": 4096,
  "load_in_4bit": false,
  "sft_backend": "unsloth",
  "use_remote_ssh": true
}'
```

Terminal RL / RLVR:

```bash
scripts/api.sh POST /api/training/start '{
  "strategy": "rlvr",
  "dataset_path": "/path/to/prompts_or_env_specs.jsonl",
  "base_model": "<model-id>",
  "training_profile": "terminal_rl_tmax_like",
  "grpo_reward_mode": "verification",
  "grpo_group_size": 32,
  "filter_zero_std_groups": true,
  "active_sampling": true,
  "lm_head_fp32": true,
  "use_remote_ssh": true
}'
```

Session Distillation:

```bash
scripts/api.sh POST /api/training/start '{
  "strategy": "session_distillation",
  "dataset_path": "/path/to/session_distillation_records.jsonl",
  "base_model": "<model-id>",
  "session_distillation_mask_policy": "target_span_only",
  "session_distillation_context_mode": "hint_injected",
  "session_distillation_min_confidence": 0.6,
  "use_remote_ssh": true
}'
```

## Monitor And Control

```bash
scripts/api.sh GET /api/training
scripts/api.sh GET /api/training/runs
scripts/api.sh GET /api/training/{run_id}
scripts/api.sh GET "/api/training/{run_id}/log?tail=200"
scripts/api.sh GET /api/training/runs/{run_id}/metrics
scripts/api.sh GET /api/training/runs/{run_id}/analysis
scripts/api.sh GET /api/system/info
```

Pause/resume/stop work only for active in-memory runs:

```bash
scripts/api.sh POST /api/training/{run_id}/pause '{}'
scripts/api.sh POST /api/training/{run_id}/resume '{}'
scripts/api.sh POST /api/training/{run_id}/stop '{}'
```

## Required Post-Run Eval

Always analyze the saved run:

```bash
bashgym training analyze --run-id <run_id> --models-dir data/models --json
```

Minimum evidence:

- SFT/DPO/Session Distillation: metrics plus heldout behavior; terminal-facing models also need environment pass@k.
- GRPO/RLVR: rollout pass@k, holdout gate, reward-hacking canaries, and reward/zero-std sampling diagnostics.
- DPPO: rollout replay with logprobs, replay enrichment, smoke plan/bundle, then pass@k before/after comparison.
- ECHO/RWML: diagnostic world-model metrics only; do not promote from these without heldout pass@k/safety correlation.
- Reward model: strict reward examples plus heldout reward eval.
- Cascade: per-stage RunCards and eval gates before promotion.

Useful eval endpoints:

```bash
scripts/api.sh POST /api/eval/heldout '{...}'
scripts/api.sh GET /api/eval/heldout/{job_id}
scripts/api.sh POST /api/eval/environments/passk '{...}'
scripts/api.sh POST /api/eval/environments/holdout-gate '{...}'
scripts/api.sh POST /api/eval/environments/model-rollout-passk '{...}'
scripts/api.sh POST /api/eval/environments/reward-hacking-canaries '{...}'
scripts/api.sh POST /api/eval/environments/dppo-replay/enrich '{...}'
scripts/api.sh POST /api/eval/environments/dppo-replay/smoke-plan '{...}'
```

## RunCards

Create and validate a RunCard for any run that will be reused or promoted:

```bash
bashgym training runcard create --run-id <run_id> --training-method <method> --base-model <model-id> --compute-target <target-id> --output data/models/<run_id>/run_card.json --metrics data/models/<run_id>/metrics.jsonl --promotion --json
bashgym training runcard validate data/models/<run_id>/run_card.json --promotion
scripts/api.sh GET /api/training/runcards
scripts/api.sh GET "/api/training/runcards/validate?path=data/models/<run_id>/run_card.json&promotion=true"
```

## Run Classification

- `strategy: "sft"` plus `sft_backend: "unsloth"` is SFT, even when launched through BashGym.
- `training_profile: "default"` means it is not a TMax-style terminal-RL run.
- Empty `evaluation_history`, `heldout_evals`, and `environment_holdout_evals` mean no post-run eval evidence is attached.
- Missing `metrics.jsonl`, `model_profile.json`, or RunCard evidence means the run is incomplete for promotion.
