# Agent CLI Guide

Use `bashgym` as the machine-readable front door for training setup and model
analysis. Every inspection command below accepts `--json`, and JSON output is a
single object with `ok`, data fields, and `next` suggestions an agent can chain.

## Discover capabilities

```bash
bashgym manifest --json
```

Use this first when an agent needs to know what BashGym can do. It returns:

- available commands
- training documentation topics
- doc file paths
- suggested next commands

## Read training docs

```bash
bashgym training docs --json
bashgym training docs --topic overview --json
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
bashgym training plan --strategy grpo --hardware dgx --data terminal_envs --json
bashgym training plan --strategy world-model --json
```

The plan returns:

- `when_to_use`
- `starting_settings`
- metrics to watch
- docs to read next
- next command hints

Use the plan as a starting point, not a release guarantee. Evaluation gates still
decide whether a model is better.

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

## Analyze a training run

```bash
bashgym training analyze --run-id run-001 --json
bashgym training analyze --metrics data/models/run-001/metrics.jsonl --json
bashgym training analyze \
  --metrics data/models/run-001/metrics.jsonl \
  --replay data/dppo_replay/run-001.jsonl \
  --release-evidence artifacts/release-gate-run-001.json \
  --json
```

The analyzer combines persisted metrics, optional DPPO replay, and optional
release evidence. It returns:

- `training_metrics` summaries for loss, reward, reward variance, pass@k,
  timeout/tamper, and ECHO/RWML quality metrics
- `replay_summary` when a DPPO replay is provided
- `release_evidence` when a heldout/release-gate JSON artifact is provided
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
2. `bashgym training docs --topic overview --json`
3. `bashgym training plan --strategy sft --json`
4. Set up or inspect the run in the UI/API.
5. Evaluate with pass@k and heldout gates.
6. If using terminal RL or DPPO, export replay and run
   `bashgym replay summarize <path> --json`.
7. Run `bashgym training analyze --run-id <run> --json` or pass explicit
   `--metrics`, `--replay`, and `--release-evidence` paths.
8. Read `metrics-runbook.md` when behavior does not improve.

## Exit-code expectations

- `0`: command succeeded
- `2`: invalid command-line usage from `argparse`
- nonzero exception exit: missing file or invalid artifact; inspect stderr

## Read next

- [overview.md](overview.md)
- [strategy-guide.md](strategy-guide.md)
- [world-models.md](world-models.md)
- [metrics-runbook.md](metrics-runbook.md)
- [glossary.md](glossary.md)
