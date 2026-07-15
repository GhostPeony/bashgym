---
name: bashgym-operator
description: Operate BashGym training and evaluation work from Hermes on Discord or the BashGym canvas. Use for ML session planning, preflight, campaign goals and KPIs, model training or fine-tuning, runtime monitoring, evaluation and benchmark comparison, report generation, resumable iteration, and curated GBrain updates across embedding, SFT, DPO, GRPO/RLVR, LoRA/QLoRA, distillation, and full-model workflows.
---

# BashGym Operator

Act as the same local Hermes operator whether the request arrives through Discord or the BashGym canvas. Treat those as separate conversations over one identity; resume from BashGym's durable campaign evidence and GBrain, not from assumed transcript sharing.

## Establish context

1. Run `python3 scripts/operator_context.py doctor` to verify the abilities available in the current Hermes environment. Do not infer API, CLI, or campaign access from documentation alone. If `critical_skill_integrity.verified` is false, stop before mutation or compute launch and report the mismatched source-managed files.
2. Select the exact project before loading task-specific evidence. Run `python3 scripts/operator_context.py context` or `bashgym ledger projects ... --json`; if more than one project is available, ask for the project ID instead of defaulting to the most familiar experiment.
3. Read BashGym live context for that identity:
   - In the canvas, use the injected `BashGym workspace context` block as the desktop workspace/campaign projection.
   - On the registered training host, run `python3 scripts/operator_context.py context --workspace-id <workspace> --project <project>` for live jobs, durable ledger state, and project-local evidence. Treat `task_profile` as supplemental evidence for that selected project only.
   - When `BASHGYM_API_BASE_URL` is reachable, run `python3 scripts/operator_context.py workspace --workspace-id <id> --format markdown`.
   - Never assume Discord can reach the desktop API merely because the canvas can inject its projection.
4. Query GBrain for the selected project's goal, prior decisions, accepted datasets/models, recent findings, and open follow-ups. Select `--source bashgym-activity` explicitly for curated BashGym activity; use `--source default` for broader project context. A GBrain page never proves what is currently running.
5. Reconcile all sources using this precedence: live runtime > durable BashGym ledger > current workspace snapshot/project-local evidence > curated GBrain > conversation memory. Report source timestamps and conflicts; do not silently blend them.
   - Run `bashgym ledger projects ... --json` before choosing a project when the request is ambiguous.
   - Run `bashgym ledger context --project <id> ... --json` to load structured health, lineage, recent runs, eval coverage, decisions, and evidence IDs.
   - Use `bashgym ledger run`, `trend`, `evaluations`, and `compare` for evidence. A comparison is valid only when the evaluation-suite ID matches.
6. Ask only for missing, stale, ambiguous, or safety-critical facts.

Read [references/operator-contract.md](references/operator-contract.md) before mutating a campaign, launching compute, evaluating protected data, promoting a model, or configuring GBrain curation.

Before any direct LLM training launch, read the sibling [training skill](../training/SKILL.md), [exact launch recipes](../training/references/bashgym-launch-recipes.md), and [compute-target activation contract](../training/references/compute-target-activation.md). The training skill owns executable strategy/config/target guidance; this operator skill owns session continuity, authority, monitoring, evaluation, reporting, and GBrain curation.

Machine- or project-specific execution maps belong in a local operator profile
outside the reusable skill. Load one only when the doctor identifies it; never
apply an embedding-specific profile to general LLM campaigns.

## Verify the session contract

Before launch, identify or confirm:

- objective and measurable success criteria/KPIs;
- workspace, project, experiment, run, attempt, model-version, dataset-version, environment, evaluation-suite, artifact, and correlation identities;
- base model and task profile;
- approved dataset revisions and contamination boundaries;
- method and hyperparameter envelope;
- development evaluation suite, baseline, comparison gates, and protected-test policy;
- compute target, capacity, credentials/readiness, time/cost budget, and stop rules;
- checkpoint cadence/limit, artifact retention, remote download/cleanup responsibility, and report destinations;
- Hugging Face repository, private/public visibility, adapter/merged upload choice, and publication/promotion authority.

Use the existing campaign ledger as the durable training-session record. Use `general` for task-general LLM work and a named profile such as `embedding_retrieval` only when its task-specific executor and metrics apply. Never make embeddings the platform-wide default.

## Run the cycle

1. Inspect the workspace context, campaign evidence, recent cursor, attempts, runtime state, and reports.
2. State the verified plan and unresolved gates. Do not relaunch work that already exists.
3. Preview and persist the exact strategy/config, including `checkpoint_limit`, `artifact_retention`, and Hugging Face destination fields. Select a doctor-verified activation lane and perform the next allowed action through its executable surface. A generated SkyPilot/dstack plan is not a launch; an HF Jobs id is not a native BashGym run id. If the doctor reports `launch_general_training: false`, do not pretend the documented CLI is executable there: continue planning/inspection and request a reachable desktop API or an updated isolated checkout. If desktop campaign mutation is unavailable but a project-specific guarded training script exists, use that script and its manifest only for that named project profile; do not pretend a desktop campaign was advanced or create a parallel campaign ledger.
   - For direct runs, pass `--tracking-context <json>` or the agent tool's `tracking_context`. If lineage is incomplete, record an unassigned smoke/ad-hoc run and resolve it before using the result for a project decision.
4. Monitor at a cadence appropriate to the run. Record milestones and anomalies, not every metric point.
5. Evaluate with the declared suite, compare against the pinned baseline and gates, and distinguish smoke/runtime evidence from model-quality evidence.
6. Export Markdown, charts, DOCX, and PDF only after the required full run and evaluation complete. Preserve report/export IDs and hashes.
7. Curate the milestone into GBrain with `scripts/curate_activity.py`, then sync the `bashgym-activity` source.
   - Prefer the incremental `bashgym ledger events --after-cursor <cursor>` envelope. Persist the returned cursor only after the curated write succeeds.
8. Recommend and, within authority and budget, execute the next bounded iteration. Otherwise stop and ask for the specific missing authority.

## Run a durable AutoResearch campaign

Use the campaign control plane for any new multi-iteration research loop. The
legacy `/api/autoresearch/*` surface is prototype compatibility only and must not
be used as the authoritative campaign record.

1. Discover source-managed templates with `bashgym campaign templates ... --json`.
   For real research, run `bashgym campaign setup-autoresearch --help`, supply
   the explicitly approved immutable trainable-base revision and logical
   model/data/evaluator/private-compute/source-repository IDs, then run `bashgym campaign doctor`
   on the installed template. Never select a packaged example model or put
   private transport/credential details in the definition.
2. For a no-GPU orchestration proof, create from
   `autoresearch-control-smoke-v1`. This template is explicitly ineligible for
   quality claims. For real research, proceed only when the doctor reports both
   `materializable: true` and `launch_ready: true`; it must resolve the exact
   model-contract digest, ledger bindings, pinned smoke/full training materials,
   and resident controller.
3. Inspect `bashgym campaign autoresearch --campaign <id> ... --json`. Creation
   ends at `READY`; use the authenticated `campaign start` command as the
   separate authority gate.
4. Submit the first proposal with `campaign propose --autoresearch-role baseline`.
   After a real baseline is accepted, submit exactly one candidate at a time with
   `--autoresearch-role candidate --parent-proposal <incumbent-id>`. Do not use
   the generic proposal route for an AutoResearch campaign.
   If the candidate's primary variable starts with `trainer.`, `algorithm.`,
   `gym.`, `environment.`, `reward.`, `evaluator.`, or `verifier.`, Git lineage
   is mandatory. Hermes does not receive `experiment.code_mutate`; hand the
   proposal to a capability-authorized Codex operator. That operator must call
   `campaign proposal lineage-prepare`, edit only the returned private worktree,
   then call `campaign proposal lineage-capture`. Never edit the user's branch,
   bypass the approved path scope, or merge the hypothesis branch automatically.
   Scalar recipe variables remain ledger-native and do not use this flow.
5. Launch and evaluate through the training skill. The primary metric must come
   from the pinned evaluation suite, not a training-loss proxy. Register exact
   run, attempt, artifact, and evaluation lineage. A completed campaign-linked
   evaluation write automatically attempts ingestion and reports its status;
   invoke `campaign autoresearch-result --project <id> --evaluation-result <id>`
   only for deferred reconciliation or exact replay. BashGym derives the metric,
   settled cost, provenance, campaign attempts, and evidence IDs; do not submit
   those facts as agent-authored JSON.
6. Re-read AutoResearch state after every result. Simulated/fake results prove
   orchestration only and cannot establish the baseline or become the incumbent.
7. Stop only when the durable state reports a stop rule, or when the user invokes
   an authorized pause/cancel. Preserve the campaign, ledger, branches, and sealed
   evidence for restart and review.

Use `bashgym campaign doctor`, the authenticated campaign API, and the durable
ledger projection as the executable AutoResearch contract. Do not rely on local
planning documents as runtime authority.

## Curate GBrain

Use the deterministic helper instead of pasting raw logs:

```bash
GBRAIN_BIN="${GBRAIN_BIN:-gbrain}"
BASHGYM_ACTIVITY_ROOT="${BASHGYM_ACTIVITY_ROOT:-$HOME/.local/share/bashgym/gbrain/bashgym-activity}"
python scripts/curate_activity.py context workspace-context.json \
  --output-root "$BASHGYM_ACTIVITY_ROOT"
"$GBRAIN_BIN" sync --source bashgym-activity
```

For a decision or milestone not present in workspace context, create a `bashgym.activity.v1` JSON receipt and run:

```bash
python scripts/curate_activity.py receipt receipt.json \
  --output-root "$BASHGYM_ACTIVITY_ROOT"
"$GBRAIN_BIN" sync --source bashgym-activity
```

The helper is idempotent and strips secret-shaped fields, high-volume content, and local absolute paths. Curate goals, configuration decisions, lineage IDs/digests, milestones, anomalies, KPI snapshots, comparisons, conclusions, follow-ups, and report references. Keep raw datasets, checkpoints, transcripts, full logs, and metric series in BashGym.

For a desktop-to-remote handoff, render a `bashgym.session-handoff.v1` input with
`curate_activity.py handoff`, then preview `scripts/gbrain_bridge.py publish`.
Only add `--execute --sync` after the rendered document and local bridge profile
have been reviewed. The bridge writes one bounded Markdown receipt with an atomic
rename and asks the authoritative remote GBrain to sync that source; it never copies
or mounts the live index.

## Boundaries

- Do not introduce MCP or another daemon when local API/CLI/filesystem access works.
- Do not merge BashGym and MemexAI repositories. Exchange versioned artifacts and contracts.
- Do not open protected evaluation data, publish, promote, expand budget, or edit a product repository without the corresponding authority.
- Do not claim quality findings from smoke runs.
- Do not silently switch work from the selected compute target to paid external compute.
- Do not create a public Hugging Face repository, retain a merged/full run, or purge resumable artifacts without the matching session authority.
- Keep Discord updates concise: current phase, latest milestone/KPI, anomaly or decision, next action, and artifact/report reference.
- Treat `bashgym`, `bashgym-operator`, and `training` as source-managed critical skills. Never let a self-improvement review or Skill Lab call rewrite them; propose a reviewed repository change and redeploy the bundle instead.
