# Project-Isolated Experiment Ledger

BashGym keeps one local-first operational database for ML work. The experiment
ledger extends the existing campaign SQLite database; it is not a second source
of truth and does not require a hosted service.

## What it records

The durable identity chain is:

```text
workspace_id
  project_id
    experiment_id
      run_id
        attempt_id
        metric points
        evaluation_result_id -> evaluation_suite_id
        artifact_id
      decision_id

model_id -> model_version_id
dataset_id -> dataset_version_id
environment_id
source_system + source_run_id/source_event_id
correlation_id
```

IDs exist at reproducibility, retry, comparison, and isolation boundaries. A
model version identifies an exact base revision/checkpoint/adapter contract; a
dataset version identifies an immutable snapshot or manifest; an environment
identifies the compute/runtime contract. A run does not silently borrow any of
these from another project.

Every query requires both `workspace_id` and `project_id`. Cross-project reads
must be explicit. Direct launches without a complete tracking context are still
retained under `desktop-local / unassigned / unassigned`; they are marked
`context_status: unassigned` and cannot be mistaken for verified project
evidence.

## Storage and durability

The ledger uses the existing `campaigns/campaigns.sqlite3` database with:

- WAL journaling and a busy timeout;
- foreign-key enforcement;
- numbered, checksum-pinned migrations;
- immutable source identities and idempotent replay checks;
- bounded JSON metadata with secret-shaped and raw-data-shaped keys rejected;
- append-only metric and event identities;
- `PRAGMA quick_check`, schema-version, WAL-size, cursor, and record-count health;
- no raw datasets, checkpoints, credentials, or full logs inside ledger rows.

SQLite remains the right default for the open-source, same-device workflow. It
is portable, transactional, offline-capable, and needs no account or service.

Move to a Postgres/Supabase repository adapter only when the product actually
needs multiple concurrent writers across devices, a hosted team dashboard,
remote agents that cannot reach the local machine, centralized row-level access
control, or server-side analytics at a volume that exceeds the local store. The
stable IDs, contracts, and event cursor are intentionally database-neutral so
that change does not require rewriting training workflows.

## Official direct runs

For an official run, write a tracking context file before launch:

```json
{
  "workspace_id": "research-workspace",
  "project_id": "retrieval-project",
  "project_display_name": "Retrieval Project",
  "project_description": "Isolated retrieval model development.",
  "experiment_id": "contrastive-run-v2",
  "experiment_name": "Longer contrastive schedule",
  "objective": "Improve frozen development retrieval without latency regression.",
  "task_type": "retrieval",
  "model_id": "embedding-model",
  "model_version_id": "embedding-base-revision-abc",
  "model_source_uri": "hf://organization/model",
  "model_source_revision": "revision-abc",
  "model_config_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "dataset_id": "retrieval-pairs",
  "dataset_version_id": "retrieval-pairs-v2",
  "dataset_source_uri": "file://data/retrieval-pairs-v2.manifest.json",
  "dataset_content_digest": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "dataset_split_manifest": {"train": "train.jsonl", "dev": "dev.jsonl"},
  "dataset_row_counts": {"train": 800, "dev": 80},
  "environment_id": "gpu-runtime-v1",
  "environment_runtime_digest": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "environment_hardware": {"accelerator_family": "blackwell"},
  "owner_actor_id": "training-operator",
  "tags": ["retrieval", "contrastive"]
}
```

Then launch through the normal training surface:

```bash
bashgym training start \
  --strategy sft \
  --model organization/model \
  --dataset-path data/train.jsonl \
  --compute-target local \
  --tracking-context tracking-context.json \
  --config run-config.json \
  --json
```

The API and `start_training` agent tool expose the same object as `tracking` and
`tracking_context`. Queued, running, paused, resumed, completed, failed, and
stopped lifecycle changes update the ledger. Progress callbacks append canonical
loss, learning-rate, gradient, evaluation-loss, throughput, memory, utilization,
and Session Distillation metrics with replay-safe identities.

## Agent and canvas queries

Authenticated CLI reads:

```bash
bashgym ledger health --workspace-id <workspace> --credential-ref <credential> --json
bashgym ledger projects --workspace-id <workspace> --credential-ref <credential> --json
bashgym ledger context --workspace-id <workspace> --project <project> --credential-ref <credential> --json
bashgym ledger runs --workspace-id <workspace> --project <project> --credential-ref <credential> --json
bashgym ledger run --workspace-id <workspace> --project <project> --run <run> --credential-ref <credential> --json
bashgym ledger trend --workspace-id <workspace> --project <project> --run <run> --metric train.loss --credential-ref <credential> --json
bashgym ledger evaluations --workspace-id <workspace> --project <project> --credential-ref <credential> --json
bashgym ledger compare --workspace-id <workspace> --project <project> --run <baseline> --run <candidate> --credential-ref <credential> --json
bashgym ledger events --workspace-id <workspace> --project <project> --after-cursor 0 --credential-ref <credential> --json
```

Canvas/Hermes tools expose `list_experiment_projects`,
`get_experiment_context`, and `get_experiment_run`. The structured project
context includes lineage inventory, current and stale runs, status/method/task
counts, evaluation and artifact coverage, decisions, recent evidence IDs, and
health signals such as incomplete lineage or completed runs without evaluation.

Run comparison is conservative: values are compared only when both completed
results use the exact same `evaluation_suite_id`. BashGym does not treat similarly
named benchmark metrics from different contracts as comparable.

## Evaluation and evidence ingestion

Evaluation runners and report generators write through authenticated ledger
endpoints after their project and run identities have been verified:

```text
POST /api/ledger/projects/{project_id}/evaluation-suites
POST /api/ledger/projects/{project_id}/evaluation-results
POST /api/ledger/projects/{project_id}/artifacts
POST /api/ledger/projects/{project_id}/decisions
POST /api/ledger/projects/{project_id}/events
```

Each request also supplies `workspace_id` as a query parameter and repeats the
same workspace/project identity in its validated body. BashGym rejects any
mismatch. Immutable identities are replay-safe: sending the same record again
returns `replayed: true`, while reusing an ID with different evidence returns a
conflict. The caller needs the `experiment.ledger_write` capability. This lets a
development-eval operator record evidence without granting protected-evaluation,
promotion, or artifact-publication authority.

An evaluation producer should register its versioned suite first, write any
immutable report artifact, write the evaluation result, and finally append a
bounded event that references those IDs. That event is what incremental GBrain
or cloud curators consume. The ledger record remains the detailed source of
truth.

## GBrain and optional cloud sinks

BashGym owns detailed operational evidence. GBrain owns curated knowledge. The
`ledger events` response is an incremental `experiment_ledger_sync.v1` envelope
with a stable `next_cursor` and an explicit curation policy.

Store in GBrain or an optional cloud knowledge sink:

- goals and configuration decisions;
- model/dataset/environment lineage IDs and digests;
- milestones, anomalies, and bounded KPI snapshots;
- comparable evaluation summaries and conclusions;
- report/artifact references and follow-up work.

Keep in BashGym:

- raw logs and full metric streams;
- datasets and transcript bodies;
- checkpoints and model weights;
- credentials and private connection details.

A project product may keep its own hosted database. Exchange explicit handoff
records or consume the cursor feed; do not make that product database the hidden
required backend for BashGym, and do not duplicate the complete local ledger into
GBrain.
