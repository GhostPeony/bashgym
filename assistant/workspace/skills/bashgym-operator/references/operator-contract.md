# BashGym operator contract

## Sources of truth

| Concern | Authority |
|---|---|
| Current canvas, desktop campaigns, reports, allowed actions | Injected canvas context or reachable `GET /api/workspace/context` |
| Campaign goal, manifest, budget, attempts, durable cursor, evidence | BashGym campaign repository via REST/CLI |
| Local training-host processes and run artifacts | `scripts/operator_context.py` over `~/.hermes/training_runs.py` plus BashGym manifests |
| Personal/project history, decisions, preferences, follow-ups | GBrain |
| Raw logs, metric series, datasets, checkpoints, report files | BashGym artifact storage |

Never resolve a disagreement by copying live operational state into a new Hermes-only ledger.

Use this precedence for current claims: live runtime > durable BashGym ledger >
current workspace snapshot/project-local evidence > curated GBrain > conversation
memory. Include observation time and stable evidence IDs in the answer. If a lower
source disagrees, name the conflict instead of merging both claims.

## Context and campaign commands

Verify the live environment first:

```bash
python3 scripts/operator_context.py doctor
python3 scripts/operator_context.py context
python3 scripts/operator_context.py context --workspace-id "$WORKSPACE_ID" --project "$PROJECT_ID"
python3 scripts/operator_context.py workspace --workspace-id "$WORKSPACE_ID" --format markdown
```

The doctor must report `critical_skill_integrity.verified: true` before any
mutation or compute launch. The `bashgym`, `bashgym-operator`, and `training`
skills are source-managed; use a reviewed repository update and bundle deploy,
never an in-session self-improvement patch.

The `workspace` command requires a reachable `BASHGYM_API_BASE_URL`. In the canvas, prefer the prompt-injected workspace context. From a training-host agent, use local observer/manifests when the desktop API is unavailable.

Use the campaign CLI only when `doctor` reports it available. Add `--json` for agent parsing and run the relevant subcommand with `--help` before the first mutation in an environment:

```bash
bashgym campaign --json list ...
bashgym campaign --json status ...
bashgym campaign --json evidence ...
bashgym campaign --json events ...
bashgym campaign --json attempts ...
bashgym campaign --json comparisons ...
bashgym campaign --json metrics ...
bashgym campaign --json start|pause|resume|cancel|advance|conclude ...
bashgym campaign proposal lineage-prepare --campaign <id> --proposal <id> ...
bashgym campaign proposal lineage-capture --campaign <id> --proposal <id> ...
bashgym campaign --json export ...
```

If `doctor` reports `mutate_desktop_campaign: false`, do not run or fabricate these commands. A guarded local trainer may still create a real BashGym run manifest, but it does not mutate the desktop campaign ledger.

For a code-mutating AutoResearch proposal, the installation registry must pair
the registered private-compute stage with an approved code-lineage execution
binding. The binding names only a logical source-profile ID, an in-repository
Python entrypoint, and bounded archive policy. A launch is authoritative only
when its remote manifest records the binding digest, lineage record digest,
captured commit, patch digest, and uploaded archive digest.

The CLI credential must be a bounded secret reference. Never print or pass a raw refresh/access credential in command arguments or GBrain content.

The `context` command has no implicit project default. Its unscoped output is for
project discovery only. Load a task-specific profile only after the selected ledger
project requires it.

## Preflight outcome

Classify each field as `verified`, `missing`, `stale`, or `blocked`:

- goal and KPIs;
- task profile and base model revision;
- training/dev/protected dataset revisions;
- method and parameter envelope;
- compute target/capacity;
- development suite and baseline;
- budgets and stop rules;
- artifact/report destinations;
- promotion/publication/product-edit authority.

Launch only when required fields are verified. A missing optional report format is not a compute blocker; an ambiguous dataset split or evaluation boundary is.

## Compute activation

Use the [compute-target activation contract](../../training/references/compute-target-activation.md)
for exact local, SSH, NeMo, Hugging Face Jobs, and managed-provider surfaces.
The operator doctor reports lane readiness, but readiness is not launch proof:

- a BashGym `run_id` proves a native request was accepted;
- an upstream job id proves an HF/NeMo/managed request was submitted;
- `running` plus process/log/metric evidence proves compute actually started;
- `bashgym compute launch --dry-run` proves only that a provider plan was rendered.

Keep the upstream job id and durable artifact destination in the session ledger.
Do not route `hf-jobs` or `managed:<provider>` through `/api/training/start`.

## GBrain source and receipt

Read the two relevant knowledge scopes explicitly:

```bash
GBRAIN_BIN="${GBRAIN_BIN:-gbrain}"
BASHGYM_ACTIVITY_ROOT="${BASHGYM_ACTIVITY_ROOT:-$HOME/.local/share/bashgym/gbrain/bashgym-activity}"
"$GBRAIN_BIN" search "<project question>" --source default --limit 10
"$GBRAIN_BIN" search "<training question>" --source bashgym-activity --limit 10
```

Register once:

```bash
mkdir -p "$BASHGYM_ACTIVITY_ROOT"
"$GBRAIN_BIN" sources add bashgym-activity \
  --path "$BASHGYM_ACTIVITY_ROOT"
```

Sync after the curator writes a changed receipt:

```bash
"$GBRAIN_BIN" sync --source bashgym-activity
```

Desktop agents must use `scripts/gbrain_bridge.py` with a local, ignored profile
to publish rendered receipts over SSH. Preview is the default. The bridge may
atomically write a receipt and sync the configured source, but it must never copy,
symlink, or network-mount the GBrain database/index itself.

Receipt input uses `bashgym.activity.v1`:

```json
{
  "schema_version": "bashgym.activity.v1",
  "kind": "evaluation",
  "workspace_id": "memexai",
  "entity_id": "eval-20260713-a",
  "status": "completed",
  "occurred_at": "2026-07-13T18:20:00Z",
  "objective": "Measure retrieval quality against the frozen champion.",
  "summary": "Candidate missed the development gate.",
  "configuration": {"suite_id": "dev-v1", "candidate_digest": "..."},
  "metrics": {"mrr_delta": -0.012},
  "artifact_references": [{"kind": "report", "id": "export-...", "digest": "..."}],
  "decision": "Retain the champion.",
  "limitations": ["Protected test remained unopened."],
  "follow_up": ["Revise negative mining and repeat development evaluation."],
  "source_event_ids": ["evt-..."]
}
```

Use stable entity IDs so repeated curation updates the same page. Do not include local paths, raw log bodies, terminal transcripts, protected rows, secrets, or full metric arrays.

## Resume behavior

Discord and canvas are separate conversations. Resume by reading:

1. the relevant GBrain BashGym page;
2. the current workspace/campaign projection;
3. campaign events after the stored durable cursor;
4. the local run observer if compute is active on the registered training host.

Then report what changed since the last curated cursor and continue the same durable campaign/session.
