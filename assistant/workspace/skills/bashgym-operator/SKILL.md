---
name: bashgym-operator
description: Run repeated BashGym model experiments from Codex, Cursor, Claude Code, Hermes, or another compatible agent. Use when an agent should establish a baseline, inspect failures, change data or training, fine-tune a candidate, evaluate it on fixed criteria, compare it, and decide the next iteration or report.
---

# BashGym Operator

Act inside an already-running Codex, Cursor, Claude Code, Hermes, or another
compatible agent session. Use the host's native goal, plan, tools, and subagents
for scientific work. Use BashGym to execute and record the experiments.

When explaining BashGym, describe the experiment loop first. Do not lead with
setup state, execution infrastructure, persistence, authority, budget, or the
observation UI. Discuss those details only when they affect the current action
or the user asks for them. Use plain technical language; do not invent slogans.

## Drive the experiment loop

1. Read `bashgym research state` and put its objective, fixed evaluation,
   latest comparison, stop conditions, and next action into the host's goal and
   plan.
2. If no baseline exists, evaluate the registered starting model on the fixed
   held-out suite.
3. Inspect failed tasks, traces, metric slices, dataset checks, and prior
   interventions. Form one falsifiable hypothesis.
4. Change one supported input: data, training recipe, reward, evaluator, or
   approved training code. Submit one candidate against the current reference.
5. Let BashGym schedule and run the registered stages. Use `research wait` with
   its durable cursor, then refresh `research state` when progress or an agent
   action is reported.
6. Compare the candidate with the baseline and current reference. Inspect
   aggregate gains, protected regressions, and the tasks behind both.
7. Repeat only when the evidence supports a specific next intervention.
   Otherwise stop and request the report.

Keep user-facing updates in this order: objective, current experiment, latest
comparison, latest finding, next action. Add budget only when it constrains the
decision. Link the BashGym experiment view only when the user asks for the
visual record or needs to inspect detailed logs or artifacts.

## Establish context

1. Run `bashgym operator doctor` to verify the abilities available in the current agent environment. Do not infer API, CLI, or campaign access from documentation alone. If `critical_skill_integrity.verified` is false, stop before mutation or compute launch and report the mismatched source-managed files.
2. Select the exact workspace and project before loading task-specific evidence. Run `bashgym operator context --workspace-id <workspace>` or `bashgym ledger projects --workspace-id <workspace> --json`; if more than one project is available, ask for the project ID instead of defaulting to the most familiar experiment. There is no implicit second workspace.
3. Read BashGym live context for that identity:
   - In the canvas, use the injected `BashGym workspace context` block as the current workspace/campaign projection.
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

## Use the host's native capabilities

- In Codex, use `/goal` for a long-running, verifiable campaign objective and
  keep the task plan synchronized with `research state`.
- In Cursor, use Agent planning and todos. Install the project-local Agent
  Skills bundle with `bashgym operator skills install --host agents`.
- In Claude Code, use its native task list and resume/continue workflow.
- Use editors, terminals, browsers, and subagents to inspect evidence, curate a
  dataset, change an approved recipe or source file, and analyze evaluations.
- Refresh `research state` after every completed result and on every resume.
  Do not store host session IDs, goal IDs, todos, or conversational scratch work
  in the campaign ledger.
- Do not add another BashGym planner. The host agent supplies experimental
  judgment; BashGym supplies composable research actions and durable facts.

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

Use the durable campaign API for any new multi-iteration research loop. The
legacy `/api/autoresearch/*` surface is prototype compatibility only, is hidden
unless `BASHGYM_ENABLE_LEGACY_AUTORESEARCH=true`, and must not be used as the
authoritative campaign record.

An initial request to begin AutoResearch authorizes preparation, not training.
Do not make the user repeat choices that already exist in registered context.

1. Run `bashgym research prepare` with the workspace, credential reference, and
   `--json`. Treat it as context discovery:
   read the registered template, installation, model, data, compute, and
   evaluation choices and resume any existing setup session. Ask only when a
   required choice is missing or ambiguous.
2. Once the choices are exact, create a private
   `autoresearch_onboarding_contract.v1` that references the reviewed
   definition, activation request, and immutable target-model request. Preview
   it with `bashgym research onboard --contract <file> --json`. This returns the
   ordered plan without applying it.
3. Apply the same contract by adding `--apply` to `bashgym research onboard`.
   The command may register an existing target model or acquire the explicitly
   pinned model on the selected execution target; it records
   metadata rather than copying weights to the API host. It then activates the
   installation, installs or starts the resident API and worker, syncs the
   registry, completes guided setup, and creates one campaign. Step receipts
   make an exact retry resume from the last completed step.
4. Require `campaign_status: ready` and
   `next_action: explicit_start_confirmation_required`. Read `research state`
   for that campaign, then present the exact
   campaign ID, model, data, evaluation, compute, budget, stop rules, warnings,
   and evidence IDs. Stop and wait for a later explicit Start confirmation.
   Never run `research start` in the preparation turn.
5. After that confirmation, run `bashgym research start` for the unchanged
   `READY` campaign. Read `research state`, then submit the evaluation-only
   baseline with `research submit-iteration --role baseline`.
6. While a stage is running, call `bashgym research wait` with the campaign and
   last `next_cursor`. It is read-only and returns the latest state
   plus `next_cursor`. On `changed`, `agent_action_required`, or `terminal`,
   synchronize the host's native goal and plan. On `timeout`, call it again with
   the returned cursor when continued monitoring is useful.
7. After a real baseline is accepted, submit one candidate at a time with
   `research submit-iteration --role candidate` and the incumbent proposal as
   `--parent-proposal`. Do not use the generic proposal route. For changes to
   trainer, algorithm, gym, environment, reward, evaluator, or verifier code,
   use `campaign proposal lineage-prepare`, edit only the returned worktree,
   and finish with `campaign proposal lineage-capture`. Scalar recipe changes
   remain ledger-native.
8. Evaluate every candidate on the pinned suite and ingest the exact run,
   attempt, artifact, and evaluation lineage. The primary metric comes from the
   evaluator, not training loss. Smoke or simulated results prove wiring only;
   they cannot establish a baseline or incumbent.
9. Re-read `research state` after each result. Continue only with a specific
   evidence-backed hypothesis. Stop on the durable stop rule or an authorized
   pause/cancel, then request the final artifacts with `research report`.

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
