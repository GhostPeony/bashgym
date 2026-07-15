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
- Put the exact request config, including artifact retention and Hugging Face destination, into the durable run record before launch.
- For an official run, verify and pass a `tracking`/`tracking_context` object with project, experiment, model-version, dataset-version, environment, source revision, and digest IDs. If these are unknown, leave the run explicitly unassigned; never guess from conversational context.
- Keep Hugging Face repositories private unless the user explicitly approves a public release after license, provenance, and privacy review.
- Stopping a run is destructive. Confirm with the user before calling `POST /api/training/{run_id}/stop`.

For the full method matrix and eval gates, read `references/bashgym-methods-and-evals.md`.
Before launching any direct training strategy, read `references/bashgym-launch-recipes.md` and use its exact field names.
Before activating any local, SSH, or cloud compute, read `references/compute-target-activation.md`. It owns target preflight, executable submission, monitoring, cancellation, and artifact-return guidance.
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

Select and prove one activation lane before launch. `bashgym compute launch` is
dry-run planning only. `ssh:<device_id>` activates BashGym's remote trainer;
`cloud:nemo-customizer` activates hosted NeMo Customizer. NeMo RL uses a
registered private-compute campaign recipe. Hugging Face Jobs and managed-provider
fine-tunes use their own submission surfaces and must not be disguised as a
direct `/api/training/start` run.

Plan first when selecting a method. The plan output uses `TrainingRequest` field names and includes the default storage policy:

```bash
bashgym training plan --strategy sft --hardware private_compute --data traces --json
bashgym training plan --strategy dpo --hardware private_compute --data preference_pairs --json
bashgym training plan --strategy grpo --hardware private_compute --data terminal_envs --json
bashgym training plan --strategy distillation --hardware private_compute --data traces --json
bashgym training plan --strategy session-distillation --hardware private_compute --data traces --json
```

Direct `/api/training/start` strategies are `sft`, `dpo`, `grpo`, `rlvr`, `distillation`, and `session_distillation`. DPPO replay, reward-model training, ECHO/RWML diagnostics, and cascade/MOPD use their own workflows; do not relabel one of them as a direct strategy.

Use the CLI for a tracked launch when it is available. Put method-specific fields in a bounded JSON config and keep strategy, model, dataset, target, and provenance on the command line:

```bash
bashgym training start \
  --strategy sft \
  --model <model-id> \
  --dataset-path /path/to/train.jsonl \
  --compute-target ssh:<device-id> \
  --config run-config.json \
  --tracking-context tracking-context.json \
  --checkpoint-limit 1 \
  --artifact-retention adapter_only \
  --json
```

The direct agent tool accepts the same strategies. Put validated `TrainingRequest` overrides under `config`, including the storage and optional Hub fields:

```json
{
  "strategy": "sft",
  "model": "<model-id>",
  "dataset_path": "/path/to/train.jsonl",
  "compute_target": "ssh:<device-id>",
  "tracking_context": {
    "workspace_id": "<workspace-id>",
    "project_id": "<project-id>",
    "project_display_name": "<project name>",
    "experiment_id": "<experiment-id>",
    "experiment_name": "<experiment name>",
    "objective": "<measurable objective>",
    "task_type": "<task type>",
    "model_id": "<logical-model-id>",
    "model_version_id": "<model-version-id>",
    "model_source_uri": "<model source URI>",
    "dataset_version_id": "<dataset-version-id>",
    "dataset_id": "<logical-dataset-id>",
    "dataset_source_uri": "<dataset manifest URI>",
    "environment_id": "<environment-id>",
    "model_config_digest": "<sha256>",
    "dataset_content_digest": "<sha256>",
    "environment_runtime_digest": "<sha256>"
  },
  "config": {
    "num_epochs": 1,
    "checkpoint_limit": 1,
    "artifact_retention": "adapter_only",
    "auto_push_hf": false,
    "hf_private": true,
    "hf_upload_artifact": "auto"
  }
}
```

## Artifact And Hub Policy

- `adapter_only` is the default for routine experiments. Checkpoints remain available during the run and are pruned only after the final adapter saves successfully.
- Use `adapter_checkpoint` when a completed run must remain resumable or branchable.
- Use `deployable` only when a standalone merged model is needed for serving. This can add roughly another base-model-sized artifact.
- Use `full_run` only for audited/promoted work that must retain checkpoints and deployable artifacts.
- `checkpoint_limit` controls the maximum retained checkpoints for policies that keep them; start at `1`.
- `hf_upload_artifact: "adapter"` is the storage-efficient off-device default. `auto` uploads merged weights when present, otherwise the adapter. `merged` must fail if no merged artifact exists.
- `auto_push_hf: true` requires an approved repo destination. Keep `hf_private: true` unless public release is explicitly authorized.

For exact SFT, DPO, GRPO/RLVR, teacher-distillation, and Session Distillation payloads, use `references/bashgym-launch-recipes.md`. Do not invent aliases such as `epochs`; the API field is `num_epochs`.

API calls remain valid when the CLI is unavailable:

```bash
scripts/api.sh POST /api/training/start '{
  "strategy": "sft",
  "dataset_path": "/path/to/train.jsonl",
  "base_model": "<model-id>",
  "num_epochs": 1,
  "checkpoint_limit": 1,
  "artifact_retention": "adapter_only",
  "auto_push_hf": false,
  "hf_private": true,
  "hf_upload_artifact": "auto",
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

When the run belongs to a durable AutoResearch campaign, do not infer a campaign
decision from the training endpoint's completion status. Finish the pinned
evaluation and register its run/attempt/artifact/evaluation lineage. A completed
campaign-linked evaluation write automatically attempts ingestion; use
`campaign autoresearch-result --project <id> --evaluation-result <id>` only when
the write reports deferred ingestion or for exact replay. Do not author the
metric, cost, provenance, attempt IDs, or evidence references in a result JSON;
BashGym derives them from the campaign and experiment ledgers. A fake executor
or control smoke is derived as simulated and can never be reported as a real
baseline or quality improvement.

AutoResearch proposals request only `executor_kind: registered_training`. The
campaign controller resolves that logical request to the installation-owned
private-compute profile bound to the exact target-model digest. Do not replace
it with `cloud:nemo-customizer`, Hugging Face Jobs, a raw SSH command, or another fallback;
those are separate, explicitly authorized execution lanes.

For project history and agent synthesis, use `bashgym ledger projects`, `ledger
context`, `ledger runs`, `ledger run`, `ledger trend`, `ledger evaluations`,
`ledger compare`, `ledger events`, and `ledger health`. Compare results only when
they share the exact evaluation-suite ID. Keep detailed evidence in BashGym and
publish only bounded summaries and artifact references to external knowledge sinks.

After an evaluation or report completes, register its immutable suite, result,
artifact, and decision records through `/api/ledger/projects/{project_id}/*`,
then append one bounded cursor event referencing those IDs. Require
`experiment.ledger_write`; do not treat that capability as permission to open a
protected evaluation, promote a model, or publish an artifact.

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
