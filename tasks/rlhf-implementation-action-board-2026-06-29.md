# RLHF Implementation Action Board

Date: 2026-06-29

This is the short handoff/action-board view of the RLHF research work. It
summarizes whether the current BashGym direction is still aligned with the
earlier RLHF Handbook, TMax terminal-RL, and JEPA/ECHO/RWML research, then
segments the remaining implementation plan into actionable build lanes.

For product positioning, keep this action board paired with
[platform-flywheels.md](../docs/training/platform-flywheels.md). The core
BashGym flywheel is still:

```text
coding work -> trace extraction -> dataset generation -> training method
  -> eval/routing -> more coding work
```

The lanes below are implementation workstreams around that core loop, not
separate product concepts.

## Alignment Verdict

Yes. The current plan is in line with the earlier RLHF research.

The research-backed shape is:

```text
source data
  -> curated train/eval artifacts
  -> SFT, preference, reward, rejection-sampling, or RL method
  -> independent heldout/eval evidence
  -> safety and reward-hacking checks
  -> reproducible run card and conservative promotion claim
```

BashGym's current direction matches that sequence:

- Source Library and Data Designer cover source discovery, provenance, split
  policy, artifact conversion, and dataset cards.
- DPO/reward validators cover the preference and reward-data quality gap called
  out by the RLHF Handbook.
- Reward-model fixture smoke, `reward_eval.json`, RewardBench/CUARewardBench
  ingest, and learned-reward evidence cover the missing reward-model lane.
- TMax/DPPO work keeps terminal-agent RL tied to executable environments,
  rollouts, verifier rewards, pass@k, and backend replay contracts.
- Trace capture now has an explicit agent-status marker protocol and replay
  scrubber, so training-data curation can use structured events instead of
  brittle terminal-log status guesses.
- JEPA/ECHO/RWML work is correctly diagnostic for now: useful for auxiliary
  learning, curriculum, and insight, but not a release blocker until correlated
  with heldout pass@k and safety.
- Compute targets and RunCards keep scale-out training reproducible without
  pretending BashGym should replace TRL, SkyRL, verl, or OpenRLHF immediately.

## Implementation Segments

### Segment A: Evidence, RunCards, And Claim Tiers

Research reason: RLHF metrics do not prove product quality by themselves. A
model promotion needs independent behavior evidence, safety evidence, and
reproducible artifacts.

Already in place:

- RunCard schema and CLI.
- Promotion validation for required artifacts.
- Source manifest checks.
- Strict DPO and reward-example evidence checks.
- Claim-tier scaffolding for local smoke, narrow routing, and broad public
  claim.
- Learned-reward evidence is surfaced as diagnostic release evidence.
- Training UI now lists local RunCards, validates selected cards for promotion,
  and surfaces claim-tier blockers, warnings, diagnostics, and artifact
  presence in-product.
- RunCard validation now includes a compact promotion explanation with failed
  gates and next actions, reused by CLI/API/UI.

Actionable next items:

- Choose numeric/default thresholds for local smoke, narrow routing, and broad
  public claim.

Definition of done:

- A run cannot be presented as a serious claim without a RunCard, source
  manifests, metrics, release evidence, and method-specific artifacts.
- Missing evidence is explicit and actionable.

Human decision:

- Approve the three-tier claim policy and starter thresholds.

### Segment B: Source Adapters And Domain Library

Research reason: RLHF quality starts with data quality. Public sources must
preserve license, split, contamination, and provenance before they become
training inputs.

Already in place:

- SourceCard schema and curated P0 registry.
- CLI/API source list, inspect, recommend, and prepare.
- Eval-only guardrails.
- Data Designer `from_source` path with source manifests and dataset cards.
- Local JSON/JSONL source adapters for SFT examples, DPO pairs, reward examples,
  process-reward examples, eval manifests, and environment specs through
  CLI/API/Data Designer.
- Hugging Face-backed source fetch orchestration through CLI/API/Factory UI/Data
  Designer, writing capped local `source_records.jsonl` files and
  `source_fetch_report.json` before adapter conversion.
- Source fetch policy now records request fingerprints, reuses matching local
  fetch caches, supports force refresh, and requires an approval reason before
  larger-than-default Hugging Face pulls across helper/API/CLI/UI/Data Designer.
- Source-specific P0 mappers now normalize UltraFeedback Binarized and
  HelpSteer2 into BashGym DPO/reward artifacts, including HelpSteer2
  `response_1`/`response_2` preference rows and paired scored-response rows,
  with mapper reports surfaced in Source Library output.

Actionable next items:

- Source Library UI handoff for prepared artifacts is now in Factory. Next,
  expand public-source policy and source expansion priorities.
- Keep RewardBench, CUARewardBench, BFCL, tau-bench, Terminal-Bench, and
  SWE-bench eval-only by default.

Definition of done:

- A source can be inspected, converted, validated, traced through Data Designer,
  and attached to a RunCard without losing metadata.
- Eval-only sources cannot silently become training data.

Human decision:

- Choose the first source expansion set: terminal/coding-agent sources first, or
  preference/reward sources first.

Recommendation:

- If BashGym's public story is "terminal-agent training," choose terminal and
  coding-agent sources first.
- If the fastest method-completeness win matters more, choose preference and
  reward-model sources first.

### Segment C: Reward-Model Lane

Research reason: the RLHF Handbook, TRL, and OpenRLHF treat learned reward
models as a core lane, not an optional extra.

Already in place:

- Reward-model, ORM, and PRM training-plan support.
- Strict reward-example validation.
- Dependency-free reward-model fixture smoke.
- `reward_eval.json` with heldout pair accuracy, calibration, reward margin,
  length bias, variance, task-family breakdown, and eval-only leakage checks.
- RewardBench/CUARewardBench external result normalization.
- Diagnostic release surfacing for learned reward evidence.

Actionable next items:

- Add real TRL/OpenRLHF-style reward backend integration.
- Add threshold policy for learned-reward claims.
- Add reward-model UI guidance explaining which metrics matter and which are
  only diagnostics.
- Add reward model as a scorer option for rejection sampling and trajectory
  audits after thresholds exist.

Definition of done:

- A user can train/evaluate a real reward model, attach `reward_eval.json`, and
  understand whether it is safe for selection, auditing, or downstream RL.

Human decision:

- Choose whether learned-reward thresholds should start conservative and
  diagnostic-only, or become blockers for reward-model promotion claims.

### Segment D: Rejection Sampling With Matched Controls

Research reason: reward-selected data should be compared against random-selected
data with the same sample budget before trusting the reward signal.

Already in place:

- RunCard and source metadata can carry selected-vs-random evidence.
- Verifier, judge, and learned reward evidence surfaces are available or in
  progress.

Actionable next items:

- Sample N rollouts per environment.
- Score rollouts with verifier, judge, or reward model.
- Export selected traces and random-control traces.
- Train both variants with the same budget.
- Compare heldout pass@k, timeout/tamper rate, reward-hacking canaries, and
  external benchmark evidence.

Definition of done:

- BashGym can prove whether a scorer improves behavior before online RL uses
  that scorer.

Human decision:

- Pick the first scorer to test: verifier, judge, or learned reward model.

### Segment E: TMax Terminal RL, DPPO, And GX10 Proof

Research reason: BashGym's strongest differentiator is executable terminal-agent
training, not generic chat-completion RLHF.

Already in place:

- Terminal environment specs and materialization.
- Local and served-model rollouts.
- Environment pass@k.
- Holdout, spurious-reward, canary, and paired comparison gates.
- DPPO replay and behavior/train logprob contracts.
- Smoke-bundle readiness checks.

Actionable next items:

- Pick one installed backend first: SkyRL, verl, or OpenRLHF.
- Run one tiny DPPO smoke on the GX10.
- Save launch command, environment, logs, metrics, output listing, replay, and
  RunCard evidence.
- Attach before/after heldout pass@k and safety evidence.

Definition of done:

- One installed backend consumes BashGym replay and emits auditable DPPO metrics
  plus behavior evidence.

Human decision:

- Choose the first backend.

Recommendation:

- SkyRL first if the goal is to validate the terminal-agent platform story.
- verl first if the goal is to validate broad scale-out RL compatibility.
- OpenRLHF second or third, unless reward-model/PPO completeness becomes the
  immediate priority.

### Segment F: JEPA/ECHO/RWML Diagnostics

Research reason: JEPA-style predictive state learning can improve
representation quality and curriculum insight, but BashGym should not release
gate on it until correlation is proven.

Already in place:

- ECHO/RWML replay contracts.
- Backend adapter hooks.
- Replay-level coverage telemetry.
- Training config settings and docs.
- Diagnostic world-model release-evidence lane.

Actionable next items:

- Wire/test ECHO/RWML hooks inside the chosen installed backend.
- Log ECHO/RWML quality metrics from real backend runs.
- Correlate world-model metrics with heldout pass@k, timeout rate, command
  count, tamper rate, and safety controls.
- Keep world-model metrics diagnostic-only until correlation is proven.

Definition of done:

- ECHO/RWML metrics are attached to RunCards and dashboards, and BashGym can
  say whether they correlate with actual terminal-agent behavior.

Human decision:

- Decide whether to prioritize terminal-specific SkyRL proof or broader verl
  proof for the first backend integration.

### Segment G: Data Designer And AutoResearch Data Recipes

Research reason: good post-training often comes from better data mixtures and
curricula, not only different hyperparameters.

Already in place:

- Data Designer `from_source` handoff.
- `DataRecipeSearchSpace`.
- Data recipe proposal/status/stop/export API.
- AutoResearch dashboard panel for data recipe proposals.
- Eval-only guardrails during simulated recipe search.

Actionable next items:

- Export adapter-created source artifacts into Data Designer pipelines.
- Add richer comparison charts for candidate recipes.
- Run a small real two-source search.
- Track cost per improved pass@k point.
- Export the best recipe into a training plan and RunCard.

Definition of done:

- AutoResearch can propose and compare source mixes that preserve guardrails and
  produce measurable improvement/cost evidence.

Human decision:

- Pick the first two safe training-eligible sources for the real recipe search.

### Segment H: Cloud/GX10 Compute Usability

Research reason: scale should be explicit, reproducible, and auditable. BashGym
should orchestrate and record training, not hide remote execution.

Already in place:

- ComputeTarget schema.
- Local, SSH/GX10, SkyPilot, and dstack dry-run launch plans.
- RunCard compute metadata.
- Secret redaction checks in dry-run outputs.

Actionable next items:

- Add Training Config compute target selection.
- Add provider setup checklist and secret health display.
- Add cost/budget fields.
- Add artifact sync plan for run card, source manifest, training config,
  metrics, checkpoints, and release evidence.
- Add logs/status commands after the launcher path is trusted.

Definition of done:

- A user can preflight a local, GX10, or cloud target, inspect the exact launch
  config, and understand synced artifacts before any remote or billable action.

Human decision:

- Require explicit approval before any remote or billable action.
- Choose the first cloud launcher to polish.

Recommendation:

- Keep explicit approval before remote/billable work.
- Polish SkyPilot first, dstack second, and provider-specific launchers later.

### Segment I: Metrics, Recipes, And Education

Research reason: RLHF systems fail when users tune values they do not
understand. The platform needs guidance, expected metrics, and failure labs.

Already in place:

- Training methods reference.
- Strategy guide.
- Metrics runbook.
- Glossary.
- Agent CLI docs.
- GX10 eval checklist.
- TMax terminal-RL recipe.
- Reward-model fixture guidance.

Actionable next items:

- Add beginner recipes for SFT, DPO, reward model, rejection sampling,
  GRPO/RLVR, DPPO smoke, and ECHO/RWML diagnostic runs.
- Add hardware-specific starter defaults for local 12 GB, local 24 GB, GX10/DGX,
  and common cloud GPU classes.
- Add UI help for technical input fields and failure modes.
- Add failure labs for contamination, length bias, zero reward variance, reward
  hacking, over-optimization, and unreliable single-pass wins.

Definition of done:

- A new user can choose a method, understand every required input value, run a
  small smoke, and know which metrics must move before scaling.

Human decision:

- Approve starter thresholds and recommended hardware presets.

## Recommended Execution Order

1. Source adapter implementation for training-eligible preference/reward/SFT
   artifacts.
2. Human decisions for eval-only overrides, claim tiers, first backend, first
   source expansion set, first cloud launcher, and billable approval boundary.
3. GX10 installed-backend proof for DPPO plus ECHO/RWML diagnostics.
4. Real reward-model backend integration.
5. Rejection sampling with matched controls.
6. Cloud/GX10 Training Config polish.
7. Education, recipes, metric explanations, and failure labs.
8. Real two-source AutoResearch data-recipe loop.

## Immediate Build Tickets

### Ticket 1: Source Adapter Layer

Status: completed for local/fixture JSON/JSONL inputs.

- Added `bashgym/sources/adapters.py`.
- Defined adapter outputs for SFT examples, DPO pairs, reward examples,
  process-reward examples, and eval manifests.
- Preserved source id, split, prompt hash, quality score, label source,
  decontamination status, and eval-only flags.
- Added tests proving eval-only sources cannot export training artifacts by
  default.

Remaining: public-source expansion policy and new source-specific mappings for
whatever source family is approved next.

### Ticket 2: CLI/API/Data Designer Adapter Wiring

Status: completed for local/fixture JSON/JSONL inputs.

- Extended `bashgym sources prepare` with adapter input and limit options.
- Returned artifact paths and validation reports in the source prepare payload.
- Threaded prepared artifacts into Data Designer source preparation.
- Added CLI/API/Data Designer tests for trainable source artifact preparation.

### Ticket 3: GX10 Backend Smoke RunCard

- Choose backend.
- Generate tiny replay and smoke bundle.
- Run one installed-backend smoke.
- Save logs, launch env, metrics, output listing, and RunCard.
- Attach release evidence and summarize blockers.

### Ticket 4: Claim-Tier Threshold Policy

- Convert recommended claim tiers into validation rules.
- Add UI/API explanation for each failed promotion gate.
- Keep diagnostic-only evidence separate from blocking evidence.

### Ticket 5: Reward Backend Integration

- Add TRL/OpenRLHF-style reward backend plan or launcher.
- Consume validated reward examples.
- Emit `reward_eval.json`.
- Attach learned-reward evidence to RunCard and release gate.

### Ticket 6: Education Pass

- Add a beginner path per method.
- Add setting help and suggested defaults.
- Add failure labs and metric interpretation examples.
- Link source adapters, RunCards, and GX10/cloud setup into one guided flow.

## Open Human Decisions

- Eval-only override policy.
- First installed backend for GX10 proof.
- First cloud launcher to polish.
- First public-source expansion set.
- Remote/billable approval boundary.
- Claim-tier thresholds.
- First scorer for rejection sampling.
- First safe sources for real AutoResearch data-recipe search.
