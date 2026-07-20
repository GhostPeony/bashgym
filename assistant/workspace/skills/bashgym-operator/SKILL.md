---
name: bashgym-operator
description: Operate BashGym training, evaluation, and durable AutoResearch from Codex, Claude Code, Hermes, or another compatible agent. Use for guided campaign preparation and resumption, ML session planning, preflight, goals and KPIs, model training or fine-tuning, runtime monitoring, evaluation and benchmark comparison, report generation, bounded iteration, and curated GBrain updates across embedding, SFT, DPO, GRPO/RLVR, LoRA/QLoRA, distillation, and full-model workflows.
---

# BashGym Operator

Act as the current operator inside an already-running Codex, Claude Code, Hermes, or another compatible agent session. The agent host is an interface over BashGym, not a launcher or a separate campaign system. Resume from BashGym's durable campaign evidence and GBrain rather than assumed transcript sharing between interfaces.

## Establish context

1. Run `bashgym operator doctor` to verify the abilities available in the current agent environment. Do not infer API, CLI, or campaign access from documentation alone. If `critical_skill_integrity.verified` is false, stop before mutation or compute launch and report the mismatched source-managed files.
2. Select the exact workspace and project before loading task-specific evidence. Run `bashgym operator context --workspace-id <workspace>` or `bashgym ledger projects --workspace-id <workspace> --json`; if more than one project is available, ask for the project ID instead of defaulting to the most familiar experiment. There is no implicit second workspace.
3. Read BashGym live context for that identity:
   - In the canvas, use the injected `BashGym workspace context` block as the desktop workspace/campaign projection.
   - On the registered training host, run `bashgym operator context --workspace-id <workspace> --project <project>` for live jobs and project-isolated durable ledger state. The reusable helper never guesses or loads a task profile.
   - When the BashGym API is reachable, run `bashgym operator workspace --workspace-id <id> --format markdown`. Set `BASHGYM_API_BASE` to the backend's `/api` URL when it is not `http://localhost:8003/api`.
   - Never assume Discord can reach the desktop API merely because the canvas can inject its projection.
4. Query GBrain for the selected project's goal, prior decisions, accepted datasets/models, recent findings, and open follow-ups. Select `--source bashgym-activity` explicitly for curated BashGym activity; use `--source default` for broader project context. A GBrain page never proves what is currently running.
5. Reconcile all sources using this precedence: live runtime > durable BashGym ledger > current workspace snapshot > explicitly selected local-profile evidence > curated GBrain > conversation memory. Report source timestamps and conflicts; do not silently blend them.
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

Use the existing campaign ledger as the durable training-session record. Use `general` for task-general work and a named profile only when its separately installed executor and evaluation contract apply. Never make one task profile the platform-wide default.

## Run the cycle

1. Inspect the workspace context, campaign evidence, recent cursor, attempts, runtime state, and reports.
2. State the verified plan and unresolved gates. Do not relaunch work that already exists.
3. Preview and persist the exact strategy/config, including `checkpoint_limit`, `artifact_retention`, and Hugging Face destination fields. Select a doctor-verified activation lane and perform the next allowed action through its executable surface. A generated SkyPilot/dstack plan is not a launch; an HF Jobs id is not a native BashGym run id. If the doctor reports `launch_general_training: false`, do not pretend the documented CLI is executable there: continue planning/inspection and request a reachable desktop API or an updated isolated checkout. Never imply that local project tooling advanced the desktop campaign ledger.
   - For direct runs, pass `--tracking-context <json>` or the agent tool's `tracking_context`. If lineage is incomplete, record an unassigned smoke/ad-hoc run and resolve it before using the result for a project decision.
4. Monitor at a cadence appropriate to the run. Record milestones and anomalies, not every metric point.
5. Evaluate with the declared suite, compare against the pinned baseline and gates, and distinguish smoke/runtime evidence from model-quality evidence.
6. Export Markdown, charts, DOCX, and PDF only after the required full run and evaluation complete. Preserve report/export IDs and hashes.
7. Curate the milestone into GBrain with `bashgym operator curate`, then sync the `bashgym-activity` source.
   - Prefer the incremental `bashgym ledger events --after-cursor <cursor>` envelope. Persist the returned cursor only after the curated write succeeds.
8. Recommend and, within authority and budget, execute the next bounded iteration. Otherwise stop and ask for the specific missing authority.

## Run a durable AutoResearch campaign

Use the campaign control plane for any new multi-iteration research loop. The
legacy `/api/autoresearch/*` surface is prototype compatibility only, is hidden
unless `BASHGYM_ENABLE_LEGACY_AUTORESEARCH=true`, and must not be used as the
authoritative campaign record.

An initial natural-language request to start AutoResearch authorizes discovery and preparation only. It does not authorize compute launch. Do not require the user to fill a configuration form or repeat decisions already present in the registered BashGym context.

1. Run `bashgym campaign setup-context --help`, then
   `bashgym campaign setup-context --workspace-id <workspace> --json`. Use its
   registered templates, installations, model/data/compute/evaluation bindings,
   and resumable setup session as the bounded choice set. Never scan private
   caches, invent binding IDs, select a packaged example model, or expose
   transport and credential details.
2. Resume the returned setup session when one exists. Otherwise create a setup
   session through the same command surface. Process the ordered steps with
   `bashgym campaign setup-step --help`: template, installation, model, data,
   compute, then evaluation. If a step has exactly one eligible registered
   choice, select it and continue. If there are zero registered choices, report
   the missing registration or activation action. If there are multiple registered choices,
   show their safe labels and ask for only that choice.
   Re-read `setup-context` after each sealed step receipt; do not ask again for a
   choice already durably recorded.
3. Run `bashgym campaign setup-doctor --help` and the read-only doctor for the
   completed draft. Resolve only reported missing, stale, ambiguous, or blocked
   facts. For real research, continue only when doctor reports
   `materializable: true` and `launch_ready: true`; it must resolve the exact
   model-contract digest, ledger bindings, pinned smoke/full training materials,
   and resident controller.
4. Run `bashgym campaign setup-validate --help` to seal the exact prepared draft.
   Then run `bashgym campaign setup-create --help` with the returned validation
   receipt. Creation is atomic, consumes that receipt once, and ends at `READY`.
   If the campaign already exists, inspect and resume it instead of creating a
   replacement.
5. When state is `READY`, present the exact campaign ID, model, data, evaluation, compute, budget, and stop rules,
   plus unresolved warnings and evidence IDs.
   Then **STOP and wait for a subsequent explicit Start confirmation**. Never run `bashgym campaign start` in the preparation turn.
   Only a later user message
   that clearly confirms Start authorizes the authenticated Start command for
   that exact READY campaign; reconfirm if the manifest or readiness changed.
6. For a no-GPU orchestration proof, create from
   `autoresearch-control-smoke-v1`. This template is explicitly ineligible for
   quality claims.
7. After explicit Start, inspect
   `bashgym campaign autoresearch --campaign <id> ... --json`. Submit the first
   proposal with `campaign propose --autoresearch-role baseline`.
   After a real baseline is accepted, submit exactly one candidate at a time with
   `--autoresearch-role candidate --parent-proposal <incumbent-id>`. Do not use
   the generic proposal route for an AutoResearch campaign.
   If the candidate's primary variable starts with `trainer.`, `algorithm.`,
   `gym.`, `environment.`, `reward.`, `evaluator.`, or `verifier.`, Git lineage
   is mandatory. If the current agent lacks `experiment.code_mutate`, hand the
   proposal to a capability-authorized operator. That operator must call
   `campaign proposal lineage-prepare`, edit only the returned private worktree,
   then call `campaign proposal lineage-capture`. Never edit the user's branch,
   bypass the approved path scope, or merge the hypothesis branch automatically.
   Before launch, `campaign doctor` must report
   `code_lineage_execution_binding_ready`. The installation-owned remote stage
   profile must bind the same logical source profile to an in-repository Python
   entrypoint; BashGym packages the exact captured commit and verifies its digest
   again before upload. Scalar recipe variables remain ledger-native and do not
   use this flow.
8. Launch and evaluate through the training skill. The primary metric must come
   from the pinned evaluation suite, not a training-loss proxy. Register exact
   run, attempt, artifact, and evaluation lineage. A completed campaign-linked
   evaluation write automatically attempts ingestion and reports its status;
   invoke `campaign autoresearch-result --project <id> --evaluation-result <id>`
   only for deferred reconciliation or exact replay. BashGym derives the metric,
   settled cost, provenance, campaign attempts, and evidence IDs; do not submit
   those facts as agent-authored JSON.
9. Re-read AutoResearch state after every result. Simulated/fake results prove
   orchestration only and cannot establish the baseline or become the incumbent.
10. Stop only when the durable state reports a stop rule, or when the user invokes
    an authorized pause/cancel. Preserve the campaign, ledger, branches, and sealed
    evidence for restart and review.

Use `bashgym campaign doctor`, the authenticated campaign API, and the durable
ledger projection as the executable AutoResearch contract. Do not rely on local
planning documents as runtime authority.

## Curate GBrain

Use the deterministic helper instead of pasting raw logs:

```text
bashgym operator curate context workspace-context.json --output-root <activity-root>
gbrain sync --source bashgym-activity
```

For a decision or milestone not present in workspace context, create a `bashgym.activity.v1` JSON receipt and run:

```text
bashgym operator curate receipt receipt.json --output-root <activity-root>
gbrain sync --source bashgym-activity
```

The helper is idempotent and strips secret-shaped fields, high-volume content, and local absolute paths even when a path is embedded in prose. Curate goals, configuration decisions, lineage IDs/digests, milestones, anomalies, KPI snapshots, comparisons, conclusions, follow-ups, and report references. Keep raw datasets, checkpoints, transcripts, full logs, and metric series in BashGym.

For a desktop-to-remote handoff, render a `bashgym.session-handoff.v1` input with
`bashgym operator curate handoff`, then preview with
`bashgym operator gbrain-bridge --profile <ignored-profile> publish --file <receipt> --relative <remote-relative-path>`.
Only add `--execute --sync` after the rendered document and local bridge profile
have been reviewed. The bridge writes one bounded Markdown receipt with an atomic
rename and asks the authoritative remote GBrain to sync that source; it never copies
or mounts the live index.

## Boundaries

- Do not introduce MCP or another daemon when local API/CLI/filesystem access works.
- Keep project repositories separate from BashGym. Exchange versioned artifacts and contracts.
- Do not open protected evaluation data, publish, promote, expand budget, or edit a product repository without the corresponding authority.
- Do not claim quality findings from smoke runs.
- Do not silently switch work from the selected compute target to paid external compute.
- Do not create a public Hugging Face repository, retain a merged/full run, or purge resumable artifacts without the matching session authority.
- Keep operator updates concise: current phase, latest milestone/KPI, anomaly or decision, next action, and artifact/report reference.
- Treat `bashgym`, `bashgym-operator`, and `training` as source-managed critical skills. Never let a self-improvement review or Skill Lab call rewrite them; propose a reviewed repository change and redeploy the bundle instead.
