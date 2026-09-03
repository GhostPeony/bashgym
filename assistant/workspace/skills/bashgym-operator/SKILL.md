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
3. Read `bashgym research failures` to compare bounded evaluator-authored
   behavioral categories for the reference and latest candidate. Combine those
   summaries with traces, numeric slices, dataset checks, and prior
   interventions, then form one falsifiable hypothesis. Failure summaries are
   development evidence; they never authorize access to protected rows.
4. Choose the intervention mode deliberately. Use a `controlled` candidate for
   one declared scientific path when a narrow causal test is possible. Use an
   `exploratory` candidate for a bounded, fully declared bundle when coupled
   changes are the hypothesis; give that branch a hypothesis-family ID and do
   not attribute its result to one component.
   Read `decision_packet.method_selection` before choosing SFT, DPO, verifier
   RL, or distillation. Its persisted thresholds and readiness statuses are
   advisory; the host agent still makes the scientific choice, and only a
   validated installed runner can execute it. Read
   `recommended_intervention_families` before spending training compute. When
   prompt/context, retrieval/tooling, or serving parity could distinguish the
   failure more cheaply, run that bounded control first. A weight update remains
   available when its method evidence is eligible; the projection advises the
   host and never chooses or launches an intervention.
5. When the next method choice depends on a measurable unknown, submit an
   agent-designed diagnostic with `research submit-iteration --role diagnostic
--parent-proposal <reference>`. The agent chooses the probe family, question,
   hypothesis, measurements, and bounded parameters; no pre-registered design
   ID is required. The installed runner and campaign still bound the accessible
   data, sample count, measurement count, runtime, outputs, and cost. Treat
   `unsupported` as evidence that this runner cannot answer the question—never
   substitute a different probe or method silently. Diagnostics spend campaign
   budget and proposal rounds, but do not consume a model-candidate attempt,
   change the retained reference, or produce KEEP/DISCARD.
   Read `research state.diagnostic_capabilities` first. Treat it as the
   installed runner's measurement matrix, not an experiment menu. Passive
   failure, trajectory, dataset-quality, and training-metric diagnostics should
   be used directly without scheduling another action. If the active request is
   outside the matrix, it may still be scientifically valid, but the runner
   must return `unsupported`; never rewrite it into a supported probe silently.
   When the runner identity is `bashgym-scientific-diagnostics`, it consumes
   pinned aggregate receipts only: matched fixed-budget plasticity probes,
   existing reward-integrity canaries, preference-integrity counts, exact
   teacher/student suite comparisons, or paired no-hint/hinted recovery counts.
   It does not run a generic training probe, teacher inference, or session
   evaluation, and it does not inspect raw examples. Missing aggregate evidence
   is `unsupported`. Use an installation-owned runner for a novel executable
   measurement while keeping the agent's hypothesis and design in the proposal.
   For repeated fine-tuning lineages, do not call an endpoint regression
   plasticity loss. A `plasticity_probe` is interpretable only when the runner
   exposes all required measurements and executes the same recipe digest, seed,
   sample scope, optimizer contract, and fixed step budget against at least two
   exact parent checkpoints. Read `decision_packet.plasticity`: retention decline
   and reduced adaptation efficiency are separate observations, and the
   campaign-supplied retention tolerance and minimum efficiency ratio determine
   whether either concern is material.
   Before GRPO or RLVR, require a completed `reward_integrity_probe` when the
   installed runner exposes it. Bind the probe to the exact decomposed reward
   spec, inspect each named component distribution, and predeclare any
   non-negotiable component bound. The existing adversarial reward-hacking
   canaries must run through the same environment guardrails. Reward variance
   alone is not readiness: a failed hard constraint, an unguarded canary, a
   verifier error, or missing campaign thresholds keeps verifier RL
   `blocked`/`diagnostic_needed`. The host agent may propose a different named
   component weight or reward definition as an experiment, but the installed
   verifier executable and evaluation boundary remain fixed unless the proposal
   carries approved code lineage.
   Before DPO, use a completed `preference_integrity_probe` when the installed
   runner exposes it. Bind the exact preference dataset and labeling contract;
   inspect agreement, ambiguous-pair rate, position-order bias, contradictory
   labels, and held-out overlap. Row-format validation or a large pair count is
   not enough. Missing campaign thresholds keeps DPO `diagnostic_needed`; a
   measured breach suggests a labeling or data-revision experiment, not an
   automatic switch to another training method.
   Before teacher distillation, bind `teacher_gap_probe` to one exact evaluation
   suite, metric direction, teacher and student model digests, and output-
   validation contract. A positive signed gap and acceptable output rate make
   the method eligible only when campaign thresholds and an installed teacher-
   distillation runner also agree. Before session distillation, bind
   `recovery_trace_probe` to the exact recovery dataset and reader contract;
   require paired outcomes for the same cases and use the derived lower
   confidence bound rather than an asserted recovery lift. Either diagnostic is
   readiness evidence, not the post-training heldout decision.
6. Let BashGym schedule and run the registered stages. Use `research wait` with
   its durable cursor, then refresh `research state` when progress or an agent
   action is reported.
7. Compare the candidate with the baseline and current reference. Inspect
   aggregate gains, protected regressions, and the tasks behind both.
8. Repeat only when the evidence supports a specific next intervention.
   Otherwise stop and request the report.

Keep user-facing updates in this order: objective, current experiment, latest
comparison, latest finding, next action. Add budget only when it constrains the
decision. Link the BashGym experiment view only when the user asks for the
visual record or needs to inspect detailed logs or artifacts.

### Preserve idea diversity

Before selecting the next expensive run, keep a small hypothesis portfolio in
the host agent's native goal or plan:

- one **exploit** idea that improves the current reference using the strongest
  measured signal;
- one **near-miss** idea that revisits a promising failure with a materially
  different, evidence-backed intervention;
- one **structural** idea from error analysis, profiling, an advisor, a
  subagent, or cited external research.

Use a cheap diagnostic or evaluation probe when it can distinguish these ideas
before training. Execute one declared candidate at a time, evaluate it on the
fixed suite, then update or retire the portfolio from the recorded result. A
completed real ancestor may be used as a verified branch parent, but
KEEP/DISCARD still compares the candidate with the current retained reference.
The portfolio is research scratch state, not a second scheduler: it does not
permit parallel training, undeclared changes, promotion without evaluation, or
training from an unverified checkpoint.

After each completed result and before compaction or handoff, update one
**bounded scientific handoff** in the host's native goal, plan, or curated
research context.
Record the current question, three or four portfolio entries, each entry's
status (`active`, `tested`, `retired`, or `deferred`), one evidence reference,
the next discriminating probe, and why no experiment was selected when the
agent stops. On resume, load this handoff after live campaign state and before
new brainstorming. It preserves scientific intent without becoming another
campaign scheduler.

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
   - For a claim-bearing full-training profile, declare and seal a compact
     `training_metrics.jsonl` when the trainer supports it. If the recipe cannot
     emit metrics, record `loss unavailable for this recipe`; do not present an
     empty loss chart as observed training behavior.
   - When the installed runner and fixed evaluator implement checkpoint
     trajectory evidence, set `--intermediate-checkpoint-limit N` during
     activation (maximum 8). Treat retained checkpoint scores as diagnostic
     evidence only; do not promote an intermediate checkpoint implicitly.
   - To test whether an apparent improvement repeats, submit separate candidate
     studies with distinct declared training seeds and one shared
     `hypothesis_family_id`. Hold the intervention and evaluation contract fixed.
     Read the family mean, sample standard deviation, standard error, and range
     as descriptive evidence; do not claim replication from one run or from
     multiple checkpoints of the same run.
   - Read `experiment_power` before making a robustness claim. An observed
     evaluation count is not proof of adequate power. Accept `sufficient` or a
     sequential-stopping result only when the fixed evaluator supplied the
     typed predeclared criterion. Between-seed standard error is between-run
     evidence, not a per-example confidence interval. If the packet says
     `not_assessed` or `not_predeclared`, propose the missing evidence rather
     than inventing a universal threshold.
   - Once every candidate in a `hypothesis_family_id` has a result, explicitly
     conclude that family as `supported`, `exhausted`, or `inconclusive` with
     `bashgym research conclude-family` or
     `research_conclude_hypothesis_family`. This is separate from KEEP/DISCARD.
     Record a new follow-up family ID and hypothesis when useful, but do not
     treat the conclusion as a generated plan: the agent may still open any new
     family justified by the evidence.
4. Monitor at a cadence appropriate to the run. Record milestones and anomalies, not every metric point.
5. Evaluate with the declared suite, compare against the pinned baseline and gates, and distinguish smoke/runtime evidence from model-quality evidence.
6. Export Markdown, charts, DOCX, and PDF only after the required full run and evaluation complete. Preserve report/export IDs and hashes.
7. Curate the milestone into GBrain with `bashgym operator curate`, then sync the `bashgym-activity` source.
   - Prefer the incremental `bashgym ledger events --after-cursor <cursor>` envelope. Persist the returned cursor only after the curated write succeeds.
8. Recommend and, within authority and budget, execute the next bounded iteration. Otherwise stop and ask for the specific missing authority.

## Run a durable AutoResearch campaign

Use the durable campaign API for any new multi-iteration research loop. The
campaign record is the single authoritative AutoResearch state.

An initial request to begin AutoResearch authorizes preparation, not training.
Do not make the user repeat choices that already exist in registered context.

1. Run `bashgym research prepare` with the workspace, credential reference, and
   `--json`. Treat it as context discovery:
   read the registered template, installation, model, data, compute, and
   evaluation choices and resume any existing setup session. Ask only when a
   required choice is missing or ambiguous.
2. Once the choices are exact, create a private
   `autoresearch_onboarding_contract.v1` that references the reviewed
   definition, activation request, and immutable target-model request. Require
   the user to choose `max_attempts`, `budget_unit`, `max_total_cost`, and
   `minimum_improvement`; attempts include the baseline and candidates. Never
   infer these values from a prior campaign or silently use the template
   policy. `attempts_used` counts scientific experiments only: an
   infrastructure-, permission-, or configuration-class crash does not consume
   an attempt, though its measured spend still counts against
   `max_total_cost`. Preserve the template's fixed primary metric, direction, and
   protected gates. Preview
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
7. After a real baseline is accepted, either submit one candidate at a time with
   `research submit-iteration --role candidate` and the incumbent proposal as
   `--parent-proposal`, or run a bounded diagnostic first when a measured result
   can distinguish competing interventions. Do not use the generic proposal
   route. For changes to
   trainer, algorithm, gym, environment, reward, evaluator, or verifier code,
   use `campaign proposal lineage-prepare`, edit only the returned worktree,
   and finish with `campaign proposal lineage-capture`. Scalar recipe changes
   remain ledger-native.
   To build a candidate proposal from a prior study instead of writing one
   from scratch, clone it: run `research clone-study --campaign <id> --study
<id> --proposal-id <new-id> --set
training_recipe='{"schema_version":"recipe.v1","seed":23}' --output
proposal.json`; review the printed diff and edit `proposal.json` so
   `primary_variable` and `controlled_variables` declare the changed path and
   the variables held constant; then submit with `research submit-iteration
--proposal proposal.json --role candidate --parent-proposal <source
proposal>`, where `<source proposal>` is the cloned study's original
   proposal ID (`source.proposal_id` in the clone response). Clone submits
   nothing, and a replication clone changes only the seed. The MCP equivalent
   is `research_clone_study(campaign_id, study_id, proposal_id, changes)`,
   which returns the same `source`, `submission`, and `diff` fields; call
   `research_submit_iteration` to submit the reviewed candidate.
8. Evaluate every candidate on the pinned suite and ingest the exact run,
   attempt, artifact, and evaluation lineage. The primary metric comes from the
   evaluator, not training loss. Smoke or simulated results prove wiring only;
   they cannot establish a baseline or incumbent. When an attempt or completion
   event carries `reused_from_attempt_id`, that stage was not executed again:
   a data build on registered compute matched the content key of one already
   completed in the workspace, so the attempt costs zero and its bytes are the
   producing attempt's. Read it as the same result, not as new evidence of
   determinism. Evaluation and training stages always execute.
9. Re-read `research state` and `research failures` after each result. Continue only with a specific
   evidence-backed hypothesis. On a crash, read
   `decision_packet.outcome_assessment.failure_kind`: it names the failure
   class, so `infrastructure`, `permission`, or `configuration` means fix the
   environment and rerun the same experiment rather than revising the
   hypothesis. Stop on the durable stop rule or an authorized
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
