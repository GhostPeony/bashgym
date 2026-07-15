# Durable AutoResearch Campaigns

Status: installation-bound readiness and registered-training slice, July 2026.

For the NVIDIA-facing capability comparison and integration roadmap, see
[BashGym AutoResearch: Current Capability and NVIDIA NeMo Alignment](bashgym-autoresearch-nvidia-brief.md).

This is BashGym's authoritative path for new AutoResearch work. The older
`/api/autoresearch/*` endpoints remain prototype compatibility surfaces and are
explicitly non-durable.

## Contract

```mermaid
flowchart LR
  A["Portable policy + installation binding"] --> B["campaign doctor"]
  B --> C["DRAFT"]
  C --> D["Controller validation"]
  D --> E["READY"]
  E --> F["Authorized START"]
  F --> G["Real baseline proposal"]
  G --> H["Registered private compute + pinned evaluator"]
  H --> I["Durable result decision"]
  I -->|"keep/discard/crash"| J["One controlled candidate"]
  J --> H
  I -->|"budget/deadline/attempt/target"| K["EXHAUSTED"]
```

The control loop requires:

- an immutable objective, target-model contract, approved data scope, compute
  profile, evaluation plan, budget, and stop rules;
- controller-owned validation that stops at `READY`;
- a separate authenticated actor start gate;
- a real baseline before candidate search;
- exactly one changed variable and an incumbent parent for each candidate;
- exact proposal, study, attempt, artifact/evaluation, and result identities;
- explicit `real` versus `simulated` provenance;
- durable keep, discard, crash, and ineligible decisions;
- restart-safe state derived from SQLite, not conversation memory.

A fake executor can prove orchestration, sealing, metric ingestion, and restart
recovery. It cannot establish a baseline, become the incumbent, or support a
quality claim.

## First no-GPU test

Run the real campaign worker slice against an isolated temporary database:

```powershell
python -m pytest tests/campaigns/test_autoresearch_worker_slice.py -q
```

This test executes:

1. campaign creation and controller validation;
2. authenticated start;
3. explicit baseline submission;
4. scheduler selection and fake execution;
5. sealed artifact and loss-metric ingestion;
6. simulated result recording;
7. an `ineligible` decision;
8. repository reopen and state recovery.

The final state must still request a **real** baseline.

## Authenticated operator surface

The normal campaign CLI now exposes the durable path:

```bash
bashgym campaign templates \
  --workspace-id <workspace> --credential-ref <refresh-secret-ref> --json

bashgym campaign doctor \
  --workspace-id <workspace> --credential-ref <refresh-secret-ref> \
  --template <installation-template-id> --json

bashgym campaign create \
  --workspace-id <workspace> --credential-ref <refresh-secret-ref> \
  --template autoresearch-control-smoke-v1 \
  --campaign <campaign-id> --title "AutoResearch control smoke" \
  --idempotency-key <stable-create-key> --json

bashgym campaign autoresearch \
  --workspace-id <workspace> --credential-ref <refresh-secret-ref> \
  --campaign <campaign-id> --json

bashgym campaign start \
  --workspace-id <workspace> --credential-ref <refresh-secret-ref> \
  --campaign <campaign-id> --expected-version <ready-version> \
  --idempotency-key <stable-start-key> --json

bashgym campaign propose \
  --workspace-id <workspace> --credential-ref <refresh-secret-ref> \
  --campaign <campaign-id> --expected-version <version> \
  --proposal examples/autoresearch/control-smoke-baseline.json \
  --autoresearch-role baseline \
  --idempotency-key <stable-proposal-key> --json
```

No quality-claiming model template is selected by the package. A real campaign
must be materialized from an installation-owned binding that resolves an
operator-selected trainable model revision together with its approved data,
compute, and evaluation contracts. Missing or stale bindings fail closed; BashGym
does not fall back to an example model.

Installation-owned definitions live under
`~/.bashgym/campaigns/autoresearch-templates/*.json`. Each real definition must
identify an immutable trainable-base revision, approved dataset version, exact
ledger evaluation suite/primary metric, and logical compute contract. The
resident worker profile separately owns private transport, credentials, pinned
scripts and inputs, capacity policy, and budget reservation.

Create that definition without hand-authoring JSON:

```bash
bashgym campaign setup-autoresearch \
  --template <installation-template-id> \
  --objective "<measurable research objective>" \
  --model-ref 'hf://<organization>/<trainable-model>@<immutable-revision>' \
  --target-contract <model-contract-id> \
  --task <task-id> \
  --dataset-version <ledger-dataset-version-id> \
  --compute-profile <registered-private-compute-profile-id> \
  --project <ledger-project-id> \
  --evaluation-suite <ledger-evaluation-suite-id> \
  --primary-metric <exact-metric-id> \
  --metric-direction maximize \
  --budget-unit gpu_hours \
  --budget-limit <bounded-limit> \
  --max-attempts <bounded-count> \
  --minimum-improvement <minimum-delta> \
  --json
```

There is no model default. The command requires an exact 40/64-character content
revision (or SHA-256 digest), writes atomically, is idempotent by definition
digest, and refuses to overwrite a different binding unless `--replace` is
explicit. Its receipt contains the exact target-model digest and secret-free
ledger/evaluator/compute identities needed by the worker profile. It does not
store a host, user, key, remote path, or credential.

`campaign doctor` reports `materializable` only when the model, data, evaluator,
and registered compute profile all match. It reports `launch_ready` only when
those bindings match and the resident controller is online. A quality-claiming
template that is not materializable is rejected before campaign creation.

Proposals request only `executor_kind: registered_training`. The controller
resolves that logical request to an installation-owned private-compute profile
and persists the concrete executor contract. The profile is bound to the digest
of the full target-model contract; a different base revision or an inference
quant cannot silently satisfy it. Hosted compute is optional and is never a
fallback for this path.

After a real baseline becomes the incumbent, submit a candidate with:

```bash
--autoresearch-role candidate --parent-proposal <incumbent-proposal-id>
```

Recording a completed, campaign-linked evaluation through the experiment-ledger
API now automatically attempts authoritative AutoResearch ingestion. The ledger
write remains durable even when campaign prerequisites are temporarily
incomplete; its response reports `ingested`, `deferred`, or `not_applicable`.

Use the CLI only to reconcile a deferred result or replay an existing result:

```bash
bashgym campaign autoresearch-result \
  --workspace-id <workspace> --credential-ref <refresh-secret-ref> \
  --campaign <campaign-id> --project <ledger-project-id> \
  --evaluation-result <evaluation-result-id> \
  --idempotency-key <stable-result-key> --json
```

Neither path accepts caller-authored real metric/cost JSON. BashGym derives
the proposal role, study, run, action, all terminal attempts, primary metric,
evaluation suite, model/data/environment context, provenance, settled spend,
and sealed artifact hash match from the campaign and experiment ledgers. The
result ID and recorded time are server-owned. The old raw REST result boundary
is retained only for explicitly simulated compatibility results.

## Workspace canvas

The existing campaign canvas node remains the view layer. It reconstructs from
the campaign database after reload and projects:

- objective and campaign authority state;
- explicit real-baseline status;
- current hypothesis and falsification criterion;
- remaining budget;
- latest ledger or AutoResearch decision;
- current versus planned next action;
- attempts, metrics, sealed artifacts, events, and experiment-ledger evidence.
- resident-controller health as `online`, `stale`, or `offline`, independently
  from the campaign lifecycle.

The canvas does not maintain a second AutoResearch state machine. It reads the
same campaign/ledger projection used by CLI and API clients. Simulated outcomes
remain visible but never render as the baseline.

## What we adopted from NVIDIA

NVIDIA's workflow gets several operating principles right:

- validate the full model/data/runtime path with a smoke before long research;
- make the objective, method, environment, baseline, and time budget explicit;
- persist session context across compaction and disconnects;
- use one concrete hypothesis per experiment and preserve lineage;
- take the authoritative metric from the recipe/evaluator;
- record launcher, job, runtime, memory, metric, status, and artifact evidence;
- check stop rules before and after every run;
- never discard a meaningful idea based only on an underpowered smoke;
- keep the researcher responsible for goals, milestones, steering, and final
  interpretation.

Sources:

- [NVIDIA AutoResearch workflow article](https://developer.nvidia.com/blog/?p=119368)
- [NVIDIA NeMo RL Auto Research skill](https://github.com/NVIDIA/skills/blob/main/skills/nemo-rl-auto-research/SKILL.md)
- [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL)
- [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)

## Where BashGym is stronger

The NVIDIA skill uses per-hypothesis git branches plus an untracked TSV ledger.
BashGym keeps git lineage where code changes require it, but its operational
truth is a typed, authenticated, append-only campaign database with optimistic
concurrency, idempotency, authority checks, budget reservations, sealed
artifacts, cursor events, and workspace-scoped projections. This is safer for
Hermes/Codex coordination and more suitable for a product UI.

The BashGym operator and training skills also already encode project selection,
tracking identity, protected-evaluation policy, artifact retention, compute
activation, Hugging Face publication authority, and GBrain curation.

## Still required for a real quality iteration

The current slice intentionally does not pretend these are finished:

1. Extend the guided setup command beyond its completed definition installer so
   it can create/verify ledger records, a protected executor profile, and the
   resident-worker service without asking users to hand-author JSON.
2. Run one real baseline and controlled candidate against a fixed held-out suite
   using an explicitly selected current trainable model.
3. Commit the AutoResearch decision plus general-ledger decision/event in one
   transaction. Current evidence reads and the AutoResearch write are durable
   but not one atomic projection.
4. Add git worktree/branch/commit lineage for hypotheses that mutate trainer,
   gym, reward, or evaluator code. Recipe-only parameter changes need not create
   a branch.
5. Optionally connect NeMo Gym live environments and NeMo RL recipes as executor profiles;
   preserve exact rollout token IDs and behavior-policy logprobs, isolate rollout
   environments, support asynchronous generation/training, and synchronize
   weights explicitly. Keep BashGym as the control/evidence plane rather than
   replacing it.
6. Add low-signal detection, checkpoint comparison, error-slice analysis, and
   hypothesis ranking before allowing long autonomous campaigns.
7. Complete the fresh-clone/productization path: installer, secret setup,
   installation bindings, sample project, worker-service bootstrap, readiness
   validation, and first-run doctor.

Until the real baseline/candidate slice and atomic decision write have run, the
new ingestion path is a verified integration boundary—not yet a production
autonomous research campaign.
