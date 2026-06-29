# RLHF-Aligned Implementation Segments And Action Items

Date: 2026-06-29

This document segments the BashGym implementation plan after the RLHF Handbook
comparison, TMax terminal-RL research, JEPA/ECHO/RWML work, and the new source
library / compute-target implementation.

## Alignment Check

Yes: the current direction is in line with the earlier RLHF research.

The RLHF research says post-training credibility comes from a disciplined loop:

```text
source data -> curated artifacts -> training method -> independent eval
  -> safety/reward-hacking checks -> reproducible run evidence -> conservative routing
```

The recent BashGym work supports exactly that loop:

| RLHF lesson | BashGym implementation direction |
|---|---|
| Treat RLHF as a sequence, not one algorithm. | Source cards, Data Designer handoff, training plans, compute targets, run cards, and release evidence connect the stages. |
| Preference/reward data quality dominates method choice. | Source manifests and future strict DPO/RM validators preserve provenance, split, quality, and contamination metadata. |
| Reward models are a missing lane. | UltraFeedback/HelpSteer2 source cards and future RM/ORM/PRM artifact contracts give this a concrete path. |
| Rejection sampling needs matched controls. | Data recipe search and run cards can require selected-vs-random control manifests. |
| Eval independence matters. | Eval-only guardrails block Terminal-Bench, SWE-bench, BFCL, tau-bench, RewardBench, and CUARewardBench from default training use. |
| Loss/reward curves are not enough. | Run cards and release evidence push claims toward pass@k, holdouts, canaries, spurious controls, external benchmark manifests, and known limitations. |
| External trainers should own scale until proven. | Compute targets and dry-run launch configs make BashGym an orchestration/evidence layer around TRL, SkyRL, verl, OpenRLHF, and cloud/GX10 runs. |

The main caveat remains unchanged: DPPO and ECHO/RWML are implemented as
contracts/adapters/diagnostics, but not yet proven as training improvements
until an installed-backend GX10 smoke run is recorded.

## Segmented Implementation Plan Snapshot

Use this section as the handoff view. It separates the RLHF research-backed
work into execution lanes, highlights what is already landed, and keeps the
remaining work tied to explicit verification gates.

| Lane | Why it matches the RLHF research | Already landed | Next actionable items | Verification gate | Human decision needed |
|---|---|---|---|---|---|
| Evidence and release contracts | RLHF claims need reproducible artifacts, independent eval, and fail-closed promotion. | RunCards, source manifests, claim-tier scaffolding, strict DPO/reward evidence requirements, `reward_eval.json` enforcement. | Wire RunCard requirements deeper into product-facing promotion flows; turn claim-tier defaults into concrete thresholds; add clearer UI/API status for missing/failed/warning/diagnostic evidence. | `bashgym training runcard validate --promotion` reports exact blockers and refuses unsupported claims. | Claim-tier thresholds. |
| Preference and reward data quality | Pair/reward quality dominates DPO, RM, ORM, PRM, rejection sampling, and later RL. | Strict DPO pair validation, strict reward-example validation, richer pair provenance, RunCard promotion checks, local source adapters that emit validated DPO/reward artifacts. | Add pair/reward quality summaries in UI/docs and widen public-source coverage. | Fixture/public source converts into valid artifacts with provenance, split, decontamination, and quality metadata intact. | First source expansion set. |
| Reward-model lane | RLHF Handbook, TRL, and OpenRLHF all treat learned reward models as a core method family. | Reward-model/ORM/PRM training plans, artifact contracts, fixture smoke path, metric guidance, heldout reward eval artifact, RewardBench/CUARewardBench eval adapters, diagnostic release surfacing, RunCard enforcement. | Add real TRL/OpenRLHF-style reward backend integration and threshold policy for learned-reward claims. | A fixture RM/ORM/PRM run can train/evaluate, emit `reward_eval.json`, and attach evidence without eval-only leakage. | Calibration and claim thresholds for learned rewards. |
| TMax terminal RL and DPPO | Terminal-agent RL should train/evaluate on executable environments, not loose chat completions. | Environment specs, materialization, local/model rollouts, pass@k, holdouts, spurious controls, canaries, external ingest, DPPO replay/logprob contracts. | Choose SkyRL or verl; run one tiny installed-backend DPPO smoke on GX10; save mask telemetry, logs, outputs, RunCard, and before/after pass@k. | Backend consumes BashGym replay and emits auditable DPPO metrics plus behavior evidence. | First installed backend. |
| JEPA/ECHO/RWML world-model diagnostics | JEPA-style predictive state learning can help diagnostics, curriculum, and future reranking, but should not be release-gating yet. | ECHO/RWML contracts, replay payloads, adapter hooks, smoke-bundle probes, dashboard/doc support, diagnostic release lane. | Wire/test hooks inside the chosen backend; correlate ECHO/RWML quality with pass@k, timeouts, command count, and safety. | World-model metrics are attached as diagnostic evidence and only promoted if correlation evidence exists. | Whether to prioritize SkyRL-style terminal integration or broader verl-scale proof. |
| Rejection sampling with matched controls | The RLHF research stresses selected-vs-random controls before trusting reward optimization. | Source/RunCard/metric contracts are ready to carry selection evidence. | Sample N rollouts, score with verifier/judge/RM, create selected and random-control manifests, train both, compare pass@k and safety. | Selected-vs-random evidence proves the scorer helps before online RL uses it. | Which scorer to test first: verifier, judge, or reward model. |
| Source and domain library | Public data and evals need provenance, license, split, and eval-only handling before they feed training. | SourceCard registry, source CLI/API, recommendation, prepare flow, eval-only guardrails, Data Designer `from_source`, local SFT/DPO/reward/process/eval-manifest/environment-spec adapters. | Implement network/HF source download orchestration and broader Source Library UI handoff. | Eval-only sources cannot become training artifacts by default; training-eligible sources keep manifests through Data Designer. | Eval-only override policy. |
| Cloud/GX10 compute | Scale should be handled by explicit compute targets and saved launch artifacts, not hidden remote execution. | ComputeTarget schema, local/SSH/GX10 dry-run plans, SkyPilot/dstack config writers, RunCard compute metadata. | Add compute target selection in Training Config; add secret health, budget fields, status/log commands, and artifact sync plan. | User can preflight, inspect exact launch config, and understand synced artifacts before remote/billable action. | Remote/billable approval boundary and first cloud launcher. |
| Metrics, recipes, and education | Operators need to understand method choice, settings, failure modes, and what metrics can or cannot prove. | Training docs, capability map, metrics runbook, training-plan readiness ladders, starter settings, agent-readable CLI. | Add beginner recipes, failure labs, hardware-specific defaults, claim-tier explanations, and UI guidance for every technical setting. | A new user can choose SFT, DPO, reward-model, GRPO/RLVR, DPPO smoke, or ECHO/RWML diagnostic path and understand the required inputs. | Exact public default thresholds and recommended hardware presets. |
| AutoResearch data recipes | Data mix and curriculum search should respect eval-only guardrails and report cost per improvement. | `DataRecipeSearchSpace`, propose/status/stop/export API, UI proposal panel, simulated safe-source search. | Run a small real two-source search, add comparison charts, export best recipe into Data Designer/training plan, track cost per pass@k point. | AutoResearch proposes source mixes that preserve guardrails and produce measurable improvement/cost evidence. | Which two safe sources to use first. |

Immediate execution recommendation:

1. Add UI/release surfacing for learned-reward evidence now that fixture smoke
   and reward-model eval adapters are in place.
2. In parallel, prepare the GX10 backend-smoke run card and pick SkyRL or verl
   so DPPO/ECHO/RWML can move from contract to measured evidence.
3. After those are green, implement rejection sampling with selected-vs-random
   controls so learned rewards are tested before they influence policy updates.
4. Then widen source adapters, cloud launch polish, and education/failure labs.

## Segment 0: Completed Foundation From This Slice

Purpose: make the platform coherent enough that data, compute, and evidence can
be inspected before a training run.

Completed action items:

- SourceCard schema and curated P0 source registry.
- Source CLI/API for list, inspect, recommend, and prepare.
- Eval-only vs training-eligible guardrails.
- Data Designer `from_source` handoff with source manifests and dataset cards.
- AutoResearch `DataRecipeSearchSpace` for source/domain/quality mixes.
- ComputeTarget schema for local, SSH/GX10, SkyPilot, and dstack dry-run plans.
- RunCard schema and CLI for create, validate, and attach-evidence.

Next verification action:

- Add a short UI/API smoke once the frontend Source Library and compute target
  surfaces exist.

## Segment 1: P0 Evidence Enforcement

Purpose: turn BashGym's research discipline into release behavior that fails
closed when required artifacts are missing.

Action items:

- Require a RunCard for serious training runs and promotion attempts.
- Make release validation distinguish `missing`, `failed`, `warning`, and
  `diagnostic_only` evidence.
- Require source manifests for any public-source-backed training artifact.
- Require split/decontamination metadata for any claim involving public data.
- Add claim tiers:
  - local smoke
  - narrow routing
  - broad public claim
- Define mandatory evidence for each tier.

Definition of done:

- A promotion attempt without run card, metrics, source manifest, or release
  evidence reports exactly what is missing and does not silently pass.

Implementation note, 2026-06-29:

- `bashgym training runcard validate --promotion` now checks required artifact
  file presence, source manifest verdicts, release evidence ship/no-ship status,
  and keeps world-model quality marked as diagnostic-only.
- RunCards now include a `claim_tier` field. Promotion validation supports
  `local_smoke`, `narrow_routing`, and `broad_public_claim` checks so stronger
  claims require stronger evidence while final threshold policy remains a human
  decision.

## Segment 2: P0 Preference And Reward Data Contracts

Purpose: close the highest-risk RLHF data-quality gap before scaling DPO or
reward-model work.

Action items:

- Add strict DPO pair schema:
  - `pair_id`
  - `prompt_hash`
  - chosen/rejected trace ids
  - source ids
  - pair-generation method
  - annotator/judge/verifier provenance
  - label strength
  - tie/noise flags
  - quality scores
  - chosen/rejected length ratio
  - split and decontamination metadata
- Add warning mode for lightweight local experiments and strict mode for serious
  runs.
- Add fixture tests for invalid pairs, contaminated pairs, weak labels, and
  extreme length-ratio pairs.
- Thread accepted pair metadata into run cards and release evidence.

Definition of done:

- DPO cannot be presented as serious evidence unless the pair contract is
  satisfied.

Implementation note, 2026-06-29:

- `bashgym training dpo-pairs validate` now supports lightweight and strict
  validation. Strict mode fails missing pair provenance, label source/strength,
  quality scores, domain/task-family, split metadata, and decontamination
  metadata.
- New DataFactory DPO pairs include richer provenance fields such as pair id,
  prompt hash, chosen/rejected trace ids, pair generation method, label source,
  length metadata, and task-family hints.
- DPO RunCard promotion validation now requires `preference_pairs_path` and
  runs strict preference-pair validation before considering the run promotable.

## Segment 3: P0 GX10 / Installed-Backend Proof

Purpose: turn DPPO and ECHO/RWML from honest contracts into measured backend
evidence.

Action items:

- Choose one backend first:
  - SkyRL if the priority is terminal-agent environments.
  - verl if the priority is broader scale-out RL infrastructure.
- Generate a tiny replay from 2-5 executable environments.
- Include behavior logprobs, train-policy logprob replay, verifier rewards, and
  world-model payloads.
- Run a one-step DPPO smoke.
- Log mask telemetry:
  - Binary-TV or Binary-KL threshold
  - masked update fraction
  - behavior/train logprob readiness
  - reward before/after
- Wire ECHO/RWML into the same backend path and log:
  - `echo_loss`
  - `rwml_pass_rate`
  - `embedding_distance_mean`
  - `embedding_distance_p95`
  - `exit_code_accuracy`
  - `test_result_accuracy`
- Follow with before/after environment pass@k and safety checks.
- Save all configs, commands, logs, manifests, metrics, and output listings.

Definition of done:

- A RunCard plus release evidence can prove that one installed backend consumed
  BashGym replay and produced auditable results.

## Segment 4: P1 Reward-Model Lane

Purpose: add the main method gap versus the RLHF Handbook, TRL, and OpenRLHF.

Action items:

- Add reward artifact formats:
  - preference reward examples
  - outcome reward examples
  - process reward examples
- Add training-plan support for:
  - Preference RM
  - ORM
  - PRM
- Add source adapters for UltraFeedback/HelpSteer2-style preference data.
- Add RewardBench/CUARewardBench-style eval adapters as eval-only by default.
- Add metrics:
  - heldout pair accuracy
  - calibration
  - reward margin
  - length bias
  - task-family breakdown
- Add optional use sites:
  - best-of-N selection
  - rejection sampling
  - trajectory scoring
  - reward audits

Definition of done:

- A user can create a reward-model run plan, train/evaluate on fixture data, and
  attach reward-model evidence without mixing eval-only sources into training.

Implementation note, 2026-06-29:

- `bashgym training reward-examples validate` now validates reward-model,
  outcome-reward, and process-reward artifacts in lightweight or strict mode.
  Strict mode requires reward type, reward scale/label schema, label source,
  source provenance, quality/confidence metadata, domain/task-family, split, and
  decontamination metadata.
- RunCards now include `reward_examples_path`. Promotion validation for
  `reward_model`, `rm`, `preference_rm`, `orm`, `prm`, and process-reward
  methods requires strict reward-example validation before the run can be
  presented as promotable evidence.
- `bashgym training plan --strategy reward-model --json` now produces a
  Preference RM / ORM / PRM starter recipe, readiness ladder, adjustment rules,
  setting help, metric guidance, and next commands. The structured capability
  matrix and training docs also list the reward-model lane.
- `bashgym training reward-eval evaluate` now computes `reward_eval.json`
  evidence from reward examples with prediction fields, including heldout pair
  accuracy, calibration, reward margin, length bias, task-family breakdown,
  reward variance, and eval-only leakage. Reward-model RunCard promotion now
  requires this artifact alongside strict reward examples.
- RewardBench and CUARewardBench result JSON now normalize through external
  benchmark ingest as eval-only reward-model release evidence.
- `bashgym training reward-model smoke` now runs a dependency-free fixture scorer
  over strict reward examples and writes a fixture model, predictions,
  `metrics.jsonl`, `reward_eval.json`, and a report.
- Held-out Gate release evidence now accepts `learned_reward_evidence` and shows
  learned reward health as diagnostic evidence in the combined release gate.
- Local Source Library adapters now convert JSON/JSONL source-card records into
  validated DPO and reward artifacts through CLI/API/Data Designer while keeping
  eval-only sources blocked from training by default.
- Remaining Segment 4 work: real TRL/OpenRLHF-style reward backend integration
  and threshold policy for learned-reward claims.

## Segment 5: P1 Rejection Sampling With Matched Controls

Purpose: provide a cheaper and safer improvement loop before expensive online
RL.

Action items:

- Sample N terminal rollouts per environment.
- Score each rollout with verifier, judge, or reward model.
- Select top traces for SFT or preference construction.
- Create a random-selection control set with the same sample budget.
- Train selected and random-control variants.
- Compare with:
  - heldout pass@k
  - timeout/tamper deltas
  - reward-hacking canaries
  - external benchmark evidence when appropriate
- Attach selected-vs-random manifests to the RunCard.

Definition of done:

- BashGym can tell whether a reward signal is useful before relying on it for
  policy-gradient training.

## Segment 6: P1 Source Adapters And Domain Library

Purpose: turn the source registry from catalog into reusable training/eval
inputs.

Action items:

- Add a Hugging Face dataset adapter for one training-eligible source.
- Add a preference-source adapter for DPO/RM data.
- Add a terminal-environment adapter for Harbor/Terminal-Bench-style manifests.
- Add BFCL/tau-bench eval adapters for tool-use reliability.
- Add SWE-bench / SWE-agent / OpenHands trajectory research adapter once split
  and contamination policies are explicit.
- Add dataset, benchmark, and environment cards for converted artifacts.
- Add a resource-library UI/API that helps users find public data sources by:
  - method
  - domain
  - license
  - risk
  - artifact type
  - training/eval eligibility

Definition of done:

- A public source can be inspected, transformed, traced through Data Designer,
  and attached to a training or eval plan with provenance intact.

## Segment 7: P1 Cloud/GX10 Compute Setup

Purpose: make training on real GPUs easier without hiding costs, secrets, or
reproducibility details.

Action items:

- Add Training Config compute target selection.
- Add local, GX10/SSH, SkyPilot, and dstack setup checklists.
- Add provider credential health checks that expose missing secret names, not
  secret values.
- Add cost/budget fields and billable-action confirmation.
- Add dry-run config download for SkyPilot and dstack.
- Add artifact sync plan:
  - run card
  - source manifest
  - training config
  - metrics
  - checkpoints
  - release evidence
- Add logs/status commands after the launcher path is trusted.

Definition of done:

- A user can choose a compute target, run preflight, see the exact launch config,
  and understand what will be synced back before any remote/billable action.

## Segment 8: P1/P2 Metrics, Recipes, And Education

Purpose: make the platform teachable, not just technically capable.

Action items:

- Add guided recipes for:
  - first SFT student
  - DPO preference refinement
  - GRPO/RLVR terminal RL
  - rejection sampling
  - DPPO backend smoke
  - ECHO/RWML diagnostic pass
- Add recommended settings by hardware class:
  - local 12 GB
  - local 24 GB
  - GX10/DGX
  - cloud A10/L4/A100/H100-style targets
- Add metric explanations:
  - training health
  - signal quality
  - behavior evidence
  - safety/reward-hacking evidence
  - diagnostic-only world-model evidence
- Add failure labs for common mistakes:
  - no reward contrast
  - contaminated benchmark use
  - DPO length bias
  - reward hacking
  - over-optimization
  - passing once but not reliably
- Add suggested thresholds per claim tier.

Definition of done:

- A beginner can choose a strategy, understand each input value, and know which
  metrics must move before scaling or promoting a model.

## Segment 9: P2 AutoResearch Expansion

Purpose: let BashGym discover better data recipes and curricula instead of only
manual hyperparameter settings.

Action items:

- Add data-recipe API endpoints:
  - propose
  - status
  - stop
  - export
- Add UI for candidate source mixes and domain weights.
- Run a simulated search over fixture sources.
- Run a small real search over two safe training sources.
- Export the best recipe into Data Designer and training-plan input.
- Track cost per improved pass@k point.

Definition of done:

- AutoResearch can propose and compare source mixes while preserving eval-only
  guardrails and evidence artifacts.

Implementation note, 2026-06-29:

- `POST /api/autoresearch/data-recipe/propose` now runs a bounded simulated
  search over safe source-card candidates, blocks eval-only sources for training
  goals by default, returns excluded-source reasons, and can export the
  reproducible proposal JSON for Data Designer handoff.
- `GET /api/autoresearch/data-recipe/status`,
  `POST /api/autoresearch/data-recipe/stop`, and
  `POST /api/autoresearch/data-recipe/export` now expose latest-run state,
  future-ready stop handling, and explicit export of the latest proposal.
- The AutoResearch dashboard now has a Data Recipe proposal panel for goal,
  source ids, domain filter, quality floor, eval-only inclusion, latest status,
  and export controls.
- Remaining Segment 9 work: add richer candidate comparison charts, a small
  real two-source search, and cost per improved pass@k tracking.

## Current Execution Matrix

This matrix turns the research-aligned segments into handoff-sized build
items. The order is deliberate: preserve evidence first, prove backend claims
second, then widen method and source coverage.

| Priority | Implementation lane | Actionable items | Main surfaces | Dependencies | Verification |
|---|---|---|---|---|---|
| P0 | GX10/backend proof | Pick SkyRL or verl, run one tiny DPPO smoke, enable ECHO/RWML payloads, save launch env, logs, metrics, output listing, and RunCard evidence. | `bashgym training smoke-bundle`, DPPO launch planner, GX10 checklist, RunCard evidence. | Human backend choice; accessible GX10/backend checkout. | Saved RunCard plus release evidence shows backend consumed BashGym replay and emitted DPPO/ECHO/RWML metrics. |
| P0 | Claim-tier evidence policy | Finalize thresholds for local smoke, narrow routing, and broad public claims; wire the chosen thresholds into RunCard promotion validation. | `bashgym/run_cards.py`, CLI validation output, docs/training metrics guidance. | Human threshold decision. | Promotion failures list exact missing or failed evidence by claim tier. |
| P1 | Reward-model lane | Completed fixture smoke and diagnostic release surfacing; next add real reward backend integration and threshold policy. | reward validation, RunCards, training docs, tests. | Reward-example validators, reward-model training plan, fixture smoke, and `reward_eval.json` evidence are already in place. | User can train/evaluate on fixture reward data and attach reward-model evidence without eval-only leakage. |
| P1 | Reward-model eval adapters | Completed: RewardBench/CUARewardBench-style result JSON normalizes as eval-only release evidence through external benchmark ingest. | `bashgym/sources`, `bashgym/eval`, external ingest API, RunCards. | Source-card eval-only policy; reward metric schema. | Eval adapters cannot export training data by default and can attach benchmark manifests to evidence. |
| P1 | Rejection sampling controls | Sample N rollouts, score with verifier/judge/RM, create selected and random-control manifests, compare downstream heldout pass@k and safety. | Environment rollouts, Data Designer, RunCards, release gate. | Usable verifier or reward signal with non-zero contrast. | Selected-vs-random evidence proves the scorer helps before online RL. |
| P1 | Source adapters | Local SFT, DPO/preference, reward-example, process-reward, eval-manifest, and environment-spec adapters are implemented. Next: network/HF orchestration and Source Library UI handoff. | `bashgym/sources`, Data Designer `from_source`, dataset/source cards. | Existing SourceCard registry and artifact validators. | Fixture/public source converts into valid artifact with provenance, split, and decontamination metadata intact. |
| P1 | Cloud/GX10 usability | Polish Training Config compute-target selection; add launcher status/log commands after dry-run launcher path is trusted. | Compute targets, Training Config UI, run-card compute metadata. | Human launcher priority; billable-action approval boundary. | User can preflight a target, inspect exact config, and see synced artifacts without secret leakage. |
| P1/P2 | Education and recipes | Add beginner recipes, setting explanations, metric runbooks, failure labs, and suggested safe defaults by hardware class. | `docs/training`, Training Guides UI, CLI `training plan`. | Claim thresholds and method defaults. | A new user can pick SFT, DPO, GRPO/RLVR, reward model, DPPO smoke, or ECHO/RWML diagnostic path and understand every required input. |
| P2 | AutoResearch real loop | Add richer candidate comparison UI, run a small real two-source search, export best recipe into Data Designer/training plan, and track cost per pass@k point. | AutoResearch API/UI, Data Designer, RunCards. | At least two safe training-eligible sources and evaluation target. | Recipe proposal survives eval-only guardrails and produces measurable improvement/cost evidence. |

## Immediate Next Slice

Recommended next implementation slice:

1. Prepare the GX10 backend-smoke checklist for the chosen backend.
2. Add real TRL/OpenRLHF-style reward backend integration.
3. Define learned-reward claim thresholds before making reward evidence a blocker.
4. Leave claim thresholds, eval-only override, and billable remote launch policy
   as explicit human-decision gates until the choices are made.

## Human Decisions

These choices should be made before the next product-facing slice:

1. Eval-only override policy.
   - Recommendation: hard block by default; admin override only with required
     reason saved to source manifest and run card.

2. First installed backend for GX10 proof.
   - Recommendation: SkyRL first if terminal-agent environments are the main
     story; verl first if scale-out RL infrastructure is the main story.

3. First cloud launcher to polish.
   - Recommendation: SkyPilot first, dstack second, RunPod/Modal later.

4. First source expansion set.
   - Recommendation: terminal/coding-agent sources if BashGym's public story is
     executable agent training; preference/reward sources if the fastest RLHF
     method-completeness path matters more.

5. Remote/billable approval boundary.
   - Recommendation: require explicit approval before any remote or billable
     action until trusted-target budgets and expirations exist.

6. Claim-tier thresholds.
   - Recommendation: define local-smoke, narrow-routing, and broad-claim gates
     before public model-performance claims.

## Recommended Execution Order

1. Evidence enforcement: run-card requirement and fail-closed release gate.
2. Strict DPO/preference pair contract.
3. GX10 installed-backend proof for DPPO plus ECHO/RWML diagnostics.
4. Reward-model lane design and fixture implementation.
5. Rejection sampling with matched controls.
6. Source adapter expansion.
7. Cloud/GX10 compute UI polish and launcher status.
8. Education/recipe pass with beginner-safe defaults and failure labs.
9. AutoResearch data-recipe API/UI loop.

This order keeps the platform honest first, proves the riskiest training claims
second, and then expands capability breadth.
