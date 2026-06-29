# Source Library, Data Recipes, And Cloud Compute Action Plan

Date: 2026-06-29

This plan segments the next BashGym training-platform improvements that grew
out of the RLHF Book comparison, TMax terminal-RL research, JEPA/world-model
work, and the current BashGym capability map.

The short version:

```text
Source Library + Data Designer + AutoResearch + Cloud Compute Targets
  -> better training data
  -> safer preference/reward data
  -> stronger eval evidence
  -> easier local, GX10, and cloud training
```

This is in line with the RLHF research. The RLHF Book frames post-training as a
sequence of data, preferences/rewards, optimization, and evaluation. BashGym's
next platform step should make those ingredients operational: users should be
able to find a source, understand whether it is for training or eval, transform
it into a BashGym artifact, run the right training method, and attach evidence.

---

## RLHF Alignment

| RLHF lesson | BashGym implication | Implementation response |
|---|---|---|
| RLHF is a pipeline, not one algorithm. | Users need a path from data source to trainable artifact to eval evidence. | Build a Source Library with adapters, source cards, and recipe handoff. |
| Preference data quality dominates method choice. | DPO/RM/PRM data needs provenance, pair identity, label strength, and contamination metadata. | Add strict source cards and downstream pair/reward-example validators. |
| Reward models are a missing lane. | Public preference/reward data should flow into Preference RM, ORM, and PRM workflows. | Add reward-model source types and Data Designer transforms. |
| Rejection sampling needs matched controls. | Sources and generated rollouts need random-vs-selected control manifests. | Make source recipes produce selection and control groups. |
| Evaluation and over-optimization must be first-class. | Benchmarks must be marked eval-only unless explicitly safe for training. | Source cards must separate training-eligible data from eval-only harnesses. |
| External trainers should own scale until proven. | BashGym needs launch targets, run cards, and evidence capture, not a custom cloud trainer first. | Add cloud compute targets around existing scripts/backends. |

---

## Workstream A: Source And Domain Library

Goal: give users and agents a curated, inspectable library of public data,
evals, benchmarks, and domain sources that can be safely used in BashGym.

### Deliverables

1. Add `bashgym/sources/` with:
   - `SourceCard`
   - `SourceUse`
   - `SourceRisk`
   - `SourceAdapter`
   - curated registry loader
   - source-card validator

2. Add source-card fields:
   - `id`
   - `name`
   - `homepage`
   - `repo`
   - `huggingface_id`
   - `domain`
   - `task_family`
   - `artifact_types`
   - `training_eligible`
   - `eval_only`
   - `license`
   - `data_size`
   - `input_format`
   - `adapter`
   - `recommended_uses`
   - `not_recommended_for`
   - `known_risks`
   - `decontam_notes`
   - `split_policy`
   - `source_quality_notes`

3. Add CLI:
   - `bashgym sources list --json`
   - `bashgym sources inspect <id> --json`
   - `bashgym sources recommend --domain coding --goal sft --json`
   - `bashgym sources prepare <id> --output <dir> --json`

4. Add API:
   - `GET /api/sources`
   - `GET /api/sources/{source_id}`
   - `POST /api/sources/recommend`
   - `POST /api/sources/{source_id}/prepare`

5. Add UI:
   - Factory -> Source Library tab
   - filters for domain, method, artifact type, license, risk, and training/eval use
   - source detail view with "Send to Data Designer", "Create eval job", and "Create training plan"

### Initial curated sources

P0 registry:

- Harbor / Terminal-Bench: terminal-agent eval and future environment source.
- SWE-bench: software-engineering eval, narrow training use only with strict decontamination.
- BFCL: function/tool-call eval and tool-schema reliability.
- tau-bench: tool-use/business workflow eval.
- RewardBench 2: reward-model eval.
- CUARewardBench: computer-use reward-model eval.
- UltraFeedback binarized: preference/DPO seed data.
- HelpSteer2: preference/reward-model seed data.

P1 registry:

- OpenHands/SWE-rebench trajectories.
- SWE-agent trajectories.
- SWE-Gym / SWE-Smith style environment sources.
- WebArena / WebArena-Verified.
- OSWorld.
- EvalPlus / HumanEval-style coding evals.
- The Stack / StarCoderData as raw-code reference only, not default training.

### Verification

- Unit tests for source-card schema validation.
- CLI tests proving every registry source is inspectable.
- Adapter tests for at least one Hugging Face dataset and one benchmark manifest.
- A "no eval-only source used for training without override" guardrail.

---

## Workstream B: Source Adapters And Artifact Contracts

Goal: turn public sources into BashGym artifacts without losing provenance.

### Deliverables

1. Add adapter interfaces:
   - `SourceAdapter.prepare()`
   - `SourceAdapter.sample()`
   - `SourceAdapter.to_training_examples()`
   - `SourceAdapter.to_dpo_pairs()`
   - `SourceAdapter.to_reward_examples()`
   - `SourceAdapter.to_environment_specs()`
   - `SourceAdapter.to_eval_manifest()`

2. Normalize outputs into existing or new artifacts:
   - `training_examples.jsonl`
   - `dpo_pairs.jsonl`
   - `reward_model_examples.jsonl`
   - `process_reward_examples.jsonl`
   - `environment_specs.jsonl`
   - `benchmark_manifest.json`
   - `dataset_card.json`
   - `source_manifest.json`

3. Add guardrails:
   - license and terms warning
   - eval-only training block
   - required source manifest
   - hash/decontamination manifest
   - source split/grouping metadata

### Verification

- Fixture-backed adapters for one SFT source, one DPO/preference source, one eval-only benchmark, and one environment source.
- Tests that source metadata survives all conversions.
- Tests that eval-only sources cannot be exported as training data by default.

Status, 2026-06-29:

- Implemented strict validation contracts for `dpo_pairs.jsonl`,
  reward-model examples, outcome-reward examples, and process-reward examples.
- Implemented reward-model/ORM/PRM starter training-plan support through
  `bashgym training plan --strategy reward-model --json`, including capability
  matrix and docs coverage.
- Implemented `reward_eval.json` generation with heldout reward-model metrics
  and RunCard promotion enforcement for learned-reward evidence.
- Implemented RewardBench/CUARewardBench external-result normalization as
  eval-only benchmark evidence for reward-model release review.
- Implemented `bashgym training reward-model smoke` as a dependency-free
  fixture scorer that writes a fixture model, predictions, `metrics.jsonl`,
  `reward_eval.json`, and a report from validated reward examples.
- Implemented Held-out Gate `learned_reward_evidence` surfacing so reward eval
  artifacts appear as diagnostic release evidence.
- Implemented local/fixture source adapters through `bashgym sources prepare
  --input ... --output-dir ...`, `/api/sources/{source_id}/prepare`, and Data
  Designer `prepare_source`. These convert JSON/JSONL source-card records into
  `training_examples.jsonl`, `dpo_pairs.jsonl`, `reward_examples.jsonl`,
  `process_reward_examples.jsonl`, `eval_manifest.json`, or
  `environment_specs.jsonl`, preserve source metadata, and run strict
  DPO/reward/environment validators where applicable.
- Remaining: network/Hugging Face download orchestration and broader
  public-source expansion policy. The Factory Source Library now covers source
  recommendations, eval-only guardrails, local artifact preparation, and
  Training path handoff for prepared artifacts.

---

## Workstream C: Data Designer Source Pipelines

Goal: let Data Designer use source cards as first-class seeds.

### Deliverables

1. Add/extend Data Designer pipelines:
   - `from_source`
   - `source_to_sft`
   - `source_to_dpo`
   - `source_to_reward_model`
   - `source_to_process_reward`
   - `source_to_terminal_env`
   - `source_to_eval_cases`

2. Update existing `from_external` behavior so it can consume a `SourceCard`,
   not only an ad hoc local file.

3. Add generated dataset-card output:
   - source ids
   - transform recipe
   - judge/model used
   - quality thresholds
   - row counts
   - split policy
   - decontamination settings

4. UI handoff:
   - Source Library -> Data Designer prefilled pipeline
   - Data Designer -> Training Plan
   - Data Designer -> Eval Job

### Verification

- Fixture source -> Data Designer -> training examples.
- Fixture source -> Data Designer -> DPO pairs with pair metadata.
- Fixture source -> terminal environment drafts with verifier fields.
- Dataset-card snapshot tests.

Status, 2026-06-29:

- Implemented Data Designer `prepare_source(..., input_path=...)` artifact
  conversion for local source records. Dataset cards now include adapter report
  paths and artifact summaries.
- Implemented terminal-environment source conversion into
  `environment_specs.jsonl` for eval-capable source cards.
- Remaining: richer dataset-card snapshot tests and full Source Library UI
  workflows.

---

## Workstream D: AutoResearch Data Recipe Search

Goal: expand AutoResearch from hyperparameter/schema search into data-recipe
search.

### Deliverables

1. Add `DataRecipeSearchSpace` with mutation axes:
   - source ids
   - source weights
   - domain weights
   - sample size
   - quality threshold
   - synthetic multiplier
   - DPO pair difficulty
   - reward-label source
   - decontamination threshold
   - environment difficulty
   - eval target
   - cost budget

2. Add objective metrics:
   - heldout pass@1
   - heldout pass@k
   - tool-call validity
   - timeout/tamper rate
   - reward variance
   - preference accuracy
   - cost per improved point

3. Add API:
   - `POST /api/autoresearch/data-recipe/propose`
   - `GET /api/autoresearch/data-recipe/status`
   - `POST /api/autoresearch/data-recipe/stop`

4. Add UI:
   - AutoResearch -> Data Recipe panel
   - source mix chart
   - candidate recipes table
   - best recipe export
   - "Create training plan from recipe"

### Verification

- Simulated search test over fixture sources.
- Real small-run smoke with two source mixes.
- Exported recipe can feed Data Designer and `bashgym training plan`.
- AutoResearch never mutates eval-only sources into training data without explicit override.

---

## Workstream E: Cloud Compute Targets

Goal: make cloud GPU setup easy while preserving reproducibility and evidence.

### Deliverables

1. Add `ComputeTarget` schema:
   - `id`
   - `provider`
   - `launcher`
   - `gpu_type`
   - `gpu_count`
   - `region`
   - `image`
   - `python_version`
   - `cuda_version`
   - `disk_gb`
   - `max_budget_usd`
   - `env_vars`
   - `secret_refs`
   - `dataset_mount`
   - `output_sync`
   - `preflight_command`

2. Start with launch modes:
   - local
   - remote SSH / GX10
   - SkyPilot
   - dstack

3. Add later launch modes:
   - RunPod
   - Modal
   - Lambda Cloud
   - Vast.ai

4. Add CLI:
   - `bashgym compute targets --json`
   - `bashgym compute preflight --target <id> --json`
   - `bashgym compute launch --target <id> --plan <plan.json> --json`
   - `bashgym compute logs --run <run_id>`

5. Add UI:
   - Training Config -> Compute target
   - provider setup checklist
   - GPU/memory fit estimate
   - estimated cost
   - secret health
   - output sync status

6. Add run-card integration:
   - compute target id
   - provider
   - image
   - GPU type/count
   - region
   - backend version
   - environment variables used
   - artifacts synced back

### Verification

- Local and SSH preflight tests.
- SkyPilot/dstack config generation snapshot tests.
- Secret redaction tests.
- Run card includes compute target metadata.
- Dry-run launch writes the exact script/config that would run remotely.

---

## Workstream F: Run Cards And Release Evidence

Goal: make every serious run reproducible and reviewable.

### Deliverables

1. Add `RunCard` schema:
   - source recipe
   - data artifacts
   - compute target
   - training method
   - backend
   - model family/profile
   - thresholds
   - metrics path
   - release evidence path
   - known limitations

2. Add commands:
   - `bashgym training runcard create`
   - `bashgym training runcard validate`
   - `bashgym training runcard attach-evidence`

3. Add release-gate behavior:
   - fail closed when required artifacts are missing
   - distinguish missing, failed, warning, and diagnostic-only evidence

### Verification

- Run-card schema tests.
- Missing-evidence release gate tests.
- CLI creates a run card from a training plan plus source recipe.

---

## Prioritized Implementation Order

### P0: Make the system coherent and safe

1. Source-card schema and registry loader.
2. Curated P0 source registry.
3. CLI/API source listing and inspection.
4. Eval-only vs training-eligible guardrail.
5. ComputeTarget schema with local/SSH/GX10 support.
6. RunCard schema with source and compute metadata.
7. Data Designer `from_source` fixture path.

### P1: Make it useful for training

1. Hugging Face dataset adapter.
2. Source-to-SFT and source-to-DPO Data Designer pipelines.
3. Reward-model and PRM artifact formats.
4. AutoResearch `DataRecipeSearchSpace`.
5. Training Config cloud compute target UI.
6. SkyPilot/dstack dry-run launch config generation.

### P2: Make it powerful

1. Terminal-Bench/Harbor environment adapter.
2. SWE-bench/SWE-agent/OpenHands trajectory adapters.
3. BFCL/tau-bench eval adapters.
4. RewardBench/CUARewardBench reward-model eval adapters. (Done for external
   result normalization; remaining work is UI surfacing and any future upstream
   harness command generation.)
5. Data Recipe AutoResearch real small-run loop.
6. Direct RunPod/Modal/Lambda launchers.
7. Source quality scoring and source refresh checks.

---

## Actionable Tickets

### Ticket 1: SourceCard Core

- Add `bashgym/sources/catalog.py`.
- Add `SourceCard`, `SourceUse`, `SourceArtifactType`, `SourceRisk`.
- Add registry JSON fixtures.
- Add validator.
- Add tests.

Done when: `python -m pytest tests/sources -q -o addopts=` passes and every P0
source card validates.

### Ticket 2: Source CLI/API

- Add `bashgym sources list`.
- Add `bashgym sources inspect`.
- Add `bashgym sources recommend`.
- Add `/api/sources`.
- Add `/api/sources/{source_id}`.

Done when: CLI and API both return the same source registry.

### Ticket 3: Eval-Only Guardrail

- Add source-use validation before export.
- Block training export for eval-only sources unless override is explicit.
- Include override reason in source manifest.

Done when: tests prove BFCL/Terminal-Bench-style eval cards cannot be exported
as training examples by default.

### Ticket 4: Data Designer `from_source`

- Add source-card seed loader.
- Extend `from_external` or add `from_source`.
- Emit dataset card and source manifest.

Done when: one fixture source becomes valid SFT JSONL with preserved metadata.

Status, 2026-06-29:

- Implemented local source input conversion for Data Designer preparation,
  including validated DPO/reward artifacts, eval manifests, and environment
  specs.
- Remaining: full Source Library UI handoff.

### Ticket 5: ComputeTarget Core

- Add compute target schema.
- Add local and SSH/GX10 target resolvers.
- Add secret redaction.
- Add dry-run launch script generation.

Done when: dry-run creates a launch plan and run card without contacting a
remote provider.

### Ticket 6: Cloud Launch Dry Runs

- Add SkyPilot config writer.
- Add dstack config writer.
- Add CLI preflight output.

Done when: `bashgym compute launch --dry-run` emits provider-native configs and
no secrets leak in JSON/log output.

### Ticket 7: Data Recipe AutoResearch

- Add `DataRecipeSearchSpace`.
- Add simulated objective over fixture source cards.
- Add exported recipe JSON.

Done when: AutoResearch can propose a source mix and produce a Data Designer
input recipe.

Status, 2026-06-29:

- Implemented: `DataRecipeSearchSpace` plus
  `POST /api/autoresearch/data-recipe/propose` for bounded simulated proposals,
  eval-only training guardrails, excluded-source reporting, and proposal JSON
  export.
- Implemented: `GET /api/autoresearch/data-recipe/status`,
  `POST /api/autoresearch/data-recipe/stop`, and
  `POST /api/autoresearch/data-recipe/export` for latest-run status, future-ready
  stop handling, and explicit export of the latest proposal.
- Implemented: AutoResearch dashboard Data Recipe proposal panel for source mix,
  guardrail, latest-status, and export controls.
- Remaining: richer UI comparison charts, small real two-source search, and
  cost-per-improved-pass@k accounting.

### Ticket 8: RunCard And Evidence Gate

- Add run-card schema.
- Add create/validate CLI.
- Require source recipe and compute target fields for serious runs.

Done when: release validation reports exactly which artifacts are missing and
does not silently pass absent evidence.

Status, 2026-06-29:

- Implemented RunCard promotion checks for required artifacts, source manifests,
  strict DPO/preference pairs, strict reward examples, release evidence, and
  claim-tier gates.
- Remaining: UI surfacing of RunCard promotion failures and real installed-backend
  evidence attachment from GX10/cloud runs.

---

## Non-Goals For This Pass

- Do not train on benchmark test sets by default.
- Do not make world-model quality a release gate.
- Do not build a custom distributed trainer before proving SkyRL/verl/OpenRLHF
  integration.
- Do not ingest huge raw-code corpora by default.
- Do not store cloud secrets in source cards, run cards, logs, or git.

---

## Recommended First Build Slice

Build these together:

1. SourceCard core.
2. P0 source registry.
3. Source CLI/API.
4. Data Designer `from_source` fixture path.
5. ComputeTarget schema with local/SSH dry-run.
6. RunCard schema stub.

This slice gives BashGym a concrete loop:

```text
choose source -> inspect risk -> prepare artifact -> create training plan
  -> choose compute target -> write run card -> evaluate evidence
```

That is the platform-level version of the RLHF loop.
