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
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
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
bashgym serve --host 127.0.0.1 --port 8000
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
6. Set up or inspect the run in the UI/API.
7. Evaluate with pass@k and heldout gates.
8. If using terminal RL or DPPO, export replay and run
   `bashgym replay summarize <path> --json`.
9. If using DPPO/ECHO/RWML, run
   `bashgym training smoke-bundle --replay <path> --output-dir <dir> --json`.
10. Run `bashgym training analyze --run-id <run> --json` or pass explicit
   `--metrics`, `--replay`, `--smoke-bundle`, and `--release-evidence` paths.
11. Read `tmax-terminal-rl-recipe.md` before a real terminal-RL run.
12. Read `private-compute-eval-checklist.md` before moving artifacts to private/cloud compute.
13. Read `metrics-runbook.md` when behavior does not improve.

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
