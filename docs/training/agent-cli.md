# Agent CLI Guide

Use `bashgym` as the machine-readable front door for training setup and model
analysis. Every inspection command below accepts `--json`, and JSON output is a
single object with `ok`, data fields, and `next` suggestions an agent can chain.

## Discover capabilities

```bash
bashgym manifest --json
bashgym training capabilities --json
```

Use the manifest first when an agent needs to know what BashGym commands and
docs exist. It returns:

- available commands
- training documentation topics
- doc file paths
- suggested next commands

Use `training capabilities` when an agent needs the structured spread of what
the platform can do. It returns:

- training methods and their readiness status
- data sources and artifact contracts
- model-family support, hardware profiles, and config axes
- platform surfaces that map capabilities to CLI commands, API endpoints, and UI screens
- metric catalog, recipe stages, and recommended education path
- evaluation/release gates and what blocks promotion
- backend stacks BashGym can interoperate with
- ecosystem methods that are source-backed but not first-class BashGym workflows
- source references used to ground stack claims
- recommended paths from first student to reward modeling, terminal RL, DPPO, and JEPA-style world models
- minimum evidence before routing a trained model

Model-family ids are compatibility profiles, not fixed checkpoint pins. For
example, `qwen3` means the Qwen3/Qwen3.6 family profile; choose the newest
compatible Qwen3.6, Qwen3-Coder, or provider-hosted Qwen3 checkpoint that fits
the target hardware and backend.

## Read training docs

```bash
bashgym training docs --json
bashgym training docs --topic overview --json
bashgym training docs --topic capabilities --json
bashgym training docs --topic methods-reference --json
bashgym training docs --topic session-distillation --json
bashgym training docs --topic artifacts --json
bashgym training docs --topic external-review --json
bashgym training docs --topic strategy --json
bashgym training docs --topic metrics --json
bashgym training docs --topic world-models --json
bashgym training docs --topic glossary --json
```

In JSON mode, a topic response includes the full markdown content so an agent can
read it without scraping files manually.

## Generate a starter training plan

```bash
bashgym training plan --strategy sft --json
bashgym training plan --strategy dpo --hardware local_24gb --json
bashgym training plan --strategy reward-model --json
bashgym training plan --strategy session-distillation --json
bashgym training plan --strategy grpo --hardware private_compute --data terminal_envs --json
bashgym training plan --strategy world-model --json
```

The plan returns:

- `when_to_use`
- `starting_settings`
- `settings_help` explaining what key input values mean and when to adjust them
- metrics to watch
- `metric_guide` explaining what each watched metric means and what to do when it is bad
- `recommended_next_steps` for operator actions after selecting the strategy
- `readiness_ladder` stages with required evidence before promotion
- `adjustment_rules` for common bad metric patterns
- docs to read next
- next command hints

Use the plan as a starting point, not a release guarantee. Evaluation gates still
decide whether a model is better.

Direct launch plans for SFT, DPO, GRPO/RLVR, teacher distillation, and Session
Distillation include the storage defaults `checkpoint_limit: 1`,
`artifact_retention: adapter_only`, `auto_push_hf: false`, `hf_private: true`,
and `hf_upload_artifact: auto`. The settings use the API field `num_epochs`, not
the informal alias `epochs`.

For reward-model, ORM, or PRM work, validate reward artifacts before trusting
the plan:

```bash
bashgym training reward-examples validate reward_examples.jsonl --strict --json
bashgym training reward-model smoke reward_examples.jsonl --output-dir data/reward-model-smokes/run-001 --json
bashgym training reward-eval evaluate reward_predictions.jsonl --output reward_eval.json --json
```

The reward-model smoke command trains a tiny dependency-free fixture scorer. It
writes `reward_model_fixture.json`, `reward_predictions.jsonl`, `reward_eval.json`,
`metrics.jsonl`, and a fixture report. Treat it as a contract smoke before a real
TRL/OpenRLHF reward-model backend, not as production reward training.

The reward-eval command expects reward examples with model prediction fields
such as `predicted_reward`, `predicted_score`, `model_score`, or
`reward_model_score`. It emits heldout pair accuracy, calibration, reward
margin, length bias, task-family breakdown, reward variance, and eval-only
leakage checks.

## Inspect an AutoResearch campaign

```bash
bashgym campaign autoresearch \
  --workspace-id <workspace> \
  --campaign <campaign-id> \
  --credential-ref <credential> \
  --json
```

The response joins the registered campaign specification, controller state,
proposals, immutable outcomes, code lineage, and a deterministic `diagnostics`
projection. Diagnostics identify low-signal or collapsed runs, compare every
registered checkpoint on the pinned evaluation suite, expose numeric error
slices, and rank bounded next hypotheses with evidence references and explicit
falsification criteria. Ranked hypotheses are advisory: the projection never
submits a candidate, spends campaign budget, or promotes a model.

To register an intermediate checkpoint without consuming an AutoResearch
proposal result, write its completed ledger evaluation with these slice fields:

```json
{
  "autoresearch_role": "checkpoint",
  "checkpoint_step": 80
}
```

Use the same evaluation suite and run lineage as the terminal evaluation. The
ledger retains the checkpoint result while the campaign controller excludes it
from attempt accounting; the diagnostics projection then orders checkpoint
metrics by step and reports signed improvement against both the previous
checkpoint and the verified baseline.

## Launch a tracked training run

First read the canonical
[compute-target activation contract](../../assistant/workspace/skills/training/references/compute-target-activation.md).
It distinguishes executable same-device, private SSH, NeMo, Hugging Face Jobs,
and managed-provider lanes. `bashgym compute launch --dry-run` only renders a
SkyPilot/dstack plan; it does not start compute.

The direct launch command supports `sft`, `dpo`, `grpo`, `rlvr`,
`distillation`, and `session_distillation` (`session-distillation` is accepted as
a CLI alias). Put additional validated `TrainingRequest` fields in a bounded JSON
config file:

```bash
bashgym training start \
  --strategy distillation \
  --model <student-model-id> \
  --dataset-path data/distillation/train.jsonl \
  --compute-target ssh:<device-id> \
  --config distillation-config.json \
  --checkpoint-limit 1 \
  --artifact-retention adapter_only \
  --json
```

For an official experiment, also pass `--tracking-context tracking-context.json`.
That file pins the workspace, project, experiment, model version, dataset version,
environment, source revisions, and content/runtime digests. Runs without it are
retained under an explicit unassigned project instead of being silently attached
to whichever project an agent last discussed.

## Inspect experiment history

```bash
bashgym ledger health --workspace-id <workspace> --credential-ref <credential> --json
bashgym ledger projects --workspace-id <workspace> --credential-ref <credential> --json
bashgym ledger context --workspace-id <workspace> --project <project> --credential-ref <credential> --json
bashgym ledger runs --workspace-id <workspace> --project <project> --credential-ref <credential> --json
bashgym ledger trend --workspace-id <workspace> --project <project> --run <run> --metric train.loss --credential-ref <credential> --json
bashgym ledger compare --workspace-id <workspace> --project <project> --run <baseline> --run <candidate> --credential-ref <credential> --json
bashgym ledger events --workspace-id <workspace> --project <project> --after-cursor 0 --credential-ref <credential> --json
```

The context response is an agent-ready, bounded synthesis of project health,
lineage, run history, evaluations, decisions, and evidence references. The event
response is the incremental curation boundary for GBrain or an optional cloud
sink; it is not a request to copy raw logs, datasets, checkpoints, or secrets.

Example `distillation-config.json`:

```json
{
  "teacher_model": "<teacher-model-id>",
  "teacher_temperature": 0.7,
  "distillation_alpha": 0.5,
  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_seq_length": 4096,
  "use_lora": true,
  "load_in_4bit": false
}
```

For approved off-device storage, add `--auto-push-hf --hf-repo-name <repo>
--hf-private --hf-upload-artifact adapter`. `--hf-public` is available but must
not be used without explicit public-release authority and license, provenance,
and privacy review. Use `deployable` or `full_run` only when the extra merged
model/checkpoint storage is intentional.

For a private device, `--compute-target ssh:<device_id>` now activates
`use_remote_ssh` and selects the matching registered device.
`cloud:nemo-customizer` activates hosted NeMo Customizer; it is not NeMo Gym or
NeMo RL. Local NeMo RL belongs behind the registered private-compute campaign
executor. Hugging Face Jobs is intentionally separate: verify
`hf auth whoami`, validate a remotely loadable dataset/script, submit with
`hf jobs uv run`, persist checkpoints/final artifacts to a private Hub repo, and
record the returned HF job id plus Trackio/Hub evidence in the BashGym session.
Do not pass `hf-jobs` to `bashgym training start`.

DPPO, ECHO/RWML, reward-model, and cascade/MOPD workflows are not aliases for a
direct training strategy. Use their replay, backend, reward, or cascade commands
and preserve a method-specific RunCard.

## Prepare source artifacts

```bash
bashgym sources list --json
bashgym sources recommend --goal dpo --json
bashgym sources inspect ultrafeedback_binarized --json
bashgym sources prepare ultrafeedback_binarized \
  --goal dpo \
  --input source_records.jsonl \
  --output-dir data/sources/ultrafeedback-smoke \
  --json
```

Without `--input`, `sources prepare` writes a source manifest only. With
`--input`, it converts local JSON/JSONL fixture records into BashGym artifacts
such as `training_examples.jsonl`, `dpo_pairs.jsonl`, `reward_examples.jsonl`,
`process_reward_examples.jsonl`, `eval_manifest.json`, or
`environment_specs.jsonl`. Training artifacts keep source id, split, prompt
hash, label source, quality, and decontamination-status metadata, then run the
strict DPO/reward validators where applicable. Evaluation sources that advertise
environment specs can export eval manifests plus BashGym `EnvironmentSpec`
JSONL for pass@k and release-gate workflows.

Eval-only sources remain blocked for training by default. Use eval-only sources
for release evidence unless an explicit override policy has been approved and
the override reason is saved into the source manifest and RunCard.

## Summarize DPPO replay

```bash
bashgym replay summarize data/dppo_replay/latest.jsonl --json
```

This reports DPPO replay readiness and world-model replay coverage:

- record count
- behavior/train logprob readiness
- `world_model_records`
- RWML transition counts and history depth
- ECHO action/observation character coverage

Coverage is not prediction quality. For model quality, run a backend smoke that
logs ECHO loss or RWML prediction reward and compare against heldout pass@k.

## Scrub trace replay

```bash
bashgym replay scrub data/traces/session.json --output data/traces/session.scrubbed.json --json
bashgym replay scrub data/dppo_replay/latest.jsonl --output data/dppo_replay/latest.scrubbed.jsonl --json
```

Use this before sharing traces, reviewing public examples, or handing replay
artifacts into training-data QA. The scrubber preserves JSON/JSONL shape while
redacting common API tokens/secrets and summarizing long `stdout`, `stderr`,
`output`, and transcript fields.

## Prepare a backend smoke bundle

```bash
bashgym training smoke-bundle \
  --replay data/dppo_replay/latest.jsonl \
  --output-dir data/backend-smokes/run-001 \
  --base-model <operator-selected-trainable-model> \
  --backend auto \
  --rwml-embedding-model qwen3-embedding \
  --json
```

This creates local readiness artifacts for the DPPO plus ECHO/RWML backend handoff:

- `backend_smoke_readiness.json` - contract, optimizer, and backend-launch verdict.
- `dppo_replay_summary.json` - behavior/train logprob and world-model coverage.
- `world_model_backend_probe.json` - ECHO mask and RWML target probe.
- `dppo_launch_env.json` - exact env vars the external backend consumes.
- `launch_dppo_smoke.sh` - only when an installed backend or command template is runnable.

Use this before private/cloud compute work. A `contract_ready=true` bundle means
the replay and world-model payloads are shaped correctly.
`backend_launch_ready=false` means the next step is backend
installation/configuration, not more replay plumbing.

## Analyze a training run

```bash
bashgym training analyze --run-id run-001 --json
bashgym training analyze --metrics data/models/run-001/metrics.jsonl --json
bashgym training analyze \
  --metrics data/models/run-001/metrics.jsonl \
  --replay data/dppo_replay/run-001.jsonl \
  --release-evidence artifacts/release-gate-run-001.json \
  --smoke-bundle data/backend-smokes/run-001/backend_smoke_readiness.json \
  --json
```

The analyzer combines persisted metrics, optional DPPO replay, and optional
release evidence. It returns:

- `training_metrics` summaries for loss, reward, reward variance, pass@k,
  timeout/tamper, and ECHO/RWML quality metrics
- `replay_summary` when a DPPO replay is provided
- `release_evidence` when a heldout/release-gate JSON artifact is provided
- `smoke_bundle` when a backend-smoke readiness JSON artifact is provided
- `findings` with conservative next actions
- `verdict.level`, such as `insufficient_evidence`, `needs_attention`, or
  `blocked`

Loss alone never proves the model is better. Use analyzer findings to decide
which heldout gate, replay export, or backend smoke should run next.

## Start the API server

```bash
bashgym serve --host 127.0.0.1 --port 8003
```

This delegates to the existing `python -m bashgym.main` server runner.
The wrapper accepts `--host`, `--port`, `--reload`, `--workers`,
`--log-level`, and `--env-file`; pass extra server args after `--`.

## Suggested agent flow

1. `bashgym manifest --json`
2. `bashgym training capabilities --json`
3. `bashgym training docs --topic overview --json`
4. `bashgym training docs --topic capabilities --json`
5. `bashgym training plan --strategy sft --json`
6. Read the compute-target activation contract and verify the selected lane.
7. Launch with `bashgym training start ... --json`, the matching API/agent tool,
   or the documented provider-specific surface,
   preserving the exact config and artifact policy.
8. Evaluate with pass@k and heldout gates.
9. If using terminal RL or DPPO, export replay and run
   `bashgym replay summarize <path> --json`.
10. If using DPPO/ECHO/RWML, run
   `bashgym training smoke-bundle --replay <path> --output-dir <dir> --json`.
11. Run `bashgym training analyze --run-id <run> --json` or pass explicit
   `--metrics`, `--replay`, `--smoke-bundle`, and `--release-evidence` paths.
12. Read `tmax-terminal-rl-recipe.md` before a real terminal-RL run.
13. Read `private-compute-eval-checklist.md` before moving artifacts to private/cloud compute.
14. Read `metrics-runbook.md` when behavior does not improve.

## Exit-code expectations

- `0`: command succeeded
- `2`: invalid command-line usage from `argparse`
- nonzero exception exit: missing file or invalid artifact; inspect stderr

## Read next

- [overview.md](overview.md)
- [capability-map.md](capability-map.md)
- [training-methods-reference.md](training-methods-reference.md)
- [external-review-packet.md](external-review-packet.md)
- [strategy-guide.md](strategy-guide.md)
- [world-models.md](world-models.md)
- [metrics-runbook.md](metrics-runbook.md)
- [tmax-terminal-rl-recipe.md](tmax-terminal-rl-recipe.md)
- [session-distillation.md](session-distillation.md)
- [private-compute-eval-checklist.md](private-compute-eval-checklist.md)
- [glossary.md](glossary.md)
