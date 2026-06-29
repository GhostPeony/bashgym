# BashGym → Model-Agnostic Pipeline — Task Tracker

Master plan: `tasks/roadmap-2026-06-15-model-agnostic-pipeline.md` (9 segments).
Reviews: `tasks/status-review-2026-06-11.md`, `tasks/eval-and-data-strategy-2026-06-11.md`.
Archive: `tasks/archived-completed-before-2026-06-22.md` contains completed items older than one week that were cleared from this live tracker.

## Active — RLHF Handbook comparison and BashGym gap plan (2026-06-28)
- [x] Crawl `natolambert/rlhf-book` and extract RLHF/post-training lessons
- [x] Compare handbook practices against BashGym docs, CLI/API/UI, training, eval, and world-model surfaces
- [x] Answer or classify open external-review questions
- [x] Write a synthesized strengths/gaps/action-plan doc for BashGym
- [x] Verify final claims against local files and primary web sources

## Active — Source library, data recipes, and cloud compute plan (2026-06-29)
- [x] Segment RLHF-aligned implementation plan (`tasks/source-library-cloud-compute-action-plan-2026-06-29.md`)
- [x] Segment implementation plans and actionable items across evidence, data contracts, backend proof, reward modeling, source adapters, compute, education, and AutoResearch, including a current execution matrix with dependencies and verification gates (`tasks/rlhf-aligned-implementation-segments-2026-06-29.md`)
- [x] Add compact RLHF implementation action board for handoff/review (`tasks/rlhf-implementation-action-board-2026-06-29.md`)
- [x] Define segmented BashGym platform flywheels so the original trace-to-training loop stays distinct from reward, terminal-RL, JEPA, source, compute, AutoResearch, and education loops (`docs/training/platform-flywheels.md`)
- [x] Add SourceCard schema, curated source registry, and validation tests
- [x] Add source CLI/API discovery and recommendation commands
- [x] Add eval-only vs training-eligible guardrails for source exports
- [x] Add Data Designer `from_source` pipeline path with source manifests and dataset cards
- [x] Add AutoResearch data-recipe search space for source/domain/quality mixes
- [x] Add `/api/autoresearch/data-recipe/propose` with eval-only guardrails, bounded simulated search, and proposal export
- [x] Add AutoResearch data-recipe status, stop, and latest-proposal export controls
- [x] Add AutoResearch dashboard Data Recipe proposal panel for source mix, guardrail, status, and export controls
- [x] Add ComputeTarget schema, local/SSH/GX10 dry-run preflight, and cloud launcher config writers
- [x] Add RunCard schema tying sources, compute, training config, metrics, and release evidence together
- [x] Add fail-closed RunCard promotion validation for required artifact files, source manifests, release verdicts, and diagnostic-only world-model evidence
- [x] Add strict DPO/preference pair artifact validator with lightweight/strict modes, CLI, and richer DataFactory pair provenance
- [x] Require strict DPO/preference pair artifact evidence for DPO RunCard promotion validation
- [x] Add strict reward-model/ORM/PRM example validator, CLI, and RunCard promotion evidence checks
- [x] Add reward-model/ORM/PRM training-plan recipe with readiness ladder, metrics guidance, capability-map reporting, and docs
- [x] Add reward-model heldout eval metric artifact (`reward_eval.json`), CLI evaluator, and RunCard promotion enforcement
- [x] Add RewardBench/CUARewardBench external eval adapters for reward-model release evidence
- [x] Add dependency-free reward-model fixture smoke path that writes model, prediction, metrics, and `reward_eval.json` artifacts
- [x] Surface learned-reward evidence in Held-out Gate release review as diagnostic evidence
- [x] Add RunCard claim-tier scaffolding for local smoke, narrow routing, and broad public claim evidence checks
- [x] Add local/fixture Source Library adapters that convert JSON/JSONL source-card records into SFT, DPO, reward, process-reward, eval-manifest, or environment-spec artifacts through CLI/API/Data Designer while preserving source metadata and eval-only guardrails
- [ ] Human decision: choose eval-only override policy, first installed backend, first cloud launcher priority, first public-source expansion set, remote/billable approval boundary, and claim-tier thresholds (`tasks/human-decisions-source-compute-2026-06-29.md`)

## Active — Dashboard UI/UX quality pass (2026-06-24)
- [x] Map dashboard routes, shared components, and current design primitives
- [x] Audit button hover/pressed states for clipping, odd movement, and inconsistent shadows
- [x] Audit text inputs that should be preset selects, segmented controls, or dropdowns
- [x] Apply scoped UI fixes across dashboard pages while preserving Botanical Brutalism
- [x] Verify with frontend checks and browser smoke screenshots

## Active — Dashboard hover-depth correction (2026-06-24)
- [x] Remove global card/container hover depth changes
- [x] Move hover shadow compression out of button and dashboard controls
- [x] Verify real hover keeps depth stable and active/click owns press feedback

## Active — PR #24 CI fix (2026-06-23)
- [x] Fix Python Lint failure
  - [x] Run Black over `bashgym/`
  - [x] Verify `ruff check bashgym/`
  - [x] Verify `black --check bashgym/`
- [x] Fix Python Tests failure
  - [x] Add missing `psutil` runtime dependency for system-info/model recommendation tests
  - [x] Reproduce CI command locally: `pytest tests/ -x -v --tb=short --timeout=30`
  - [x] Push CI fix commit to `codex/feat-tmax-terminal-rl-platform`
  - [x] Re-check GitHub Actions status

## Active — Training platform button-up plan (2026-06-23)
- [x] Inventory current BashGym training/eval/config/docs surfaces
- [x] Research current open-model training stacks and JEPA/world-model implications
- [x] Write consolidated plan: `tasks/training-platform-button-up-plan-2026-06-23.md`
- [x] Promote JEPA/training education scratchpad context into tracked handoff docs (`tasks/jepa-training-docs-handoff-2026-06-23.md`)
- [x] Add stable/experimental/requires-backend capability status to docs and UI
- [x] Add backend-health affordance for frontend "failed to fetch" states
- [x] Add public training capability map and agent-readable CLI docs topic
- [x] Add local DPPO/ECHO/RWML backend-smoke readiness bundle and CLI preflight
- [x] Teach `bashgym training analyze` to ingest smoke-bundle readiness and report GX10 blockers
- [x] Add TMax-style terminal RL recipe and GX10 eval/test checklist docs
- [x] Add structured training-plan readiness ladders and metric adjustment rules to CLI/docs
- [x] Add structured training/eval/backend capability matrix CLI for agents
- [x] Source-ground capability matrix with ecosystem methods and backend/export boundaries
- [x] Expose strategy-specific operator next steps in training-plan JSON
- [x] Add data-source and artifact-contract coverage to training capabilities
- [x] Add model-family, hardware-profile, and config-axis coverage to training capabilities
- [x] Add CLI/API/UI platform-surface coverage to training capabilities
- [x] Add metric catalog, recipe checkpoints, and beginner-facing plan guidance
- [x] Write explicit GX10 training-smoke execution plan (`tasks/gx10-training-smoke-plan-2026-06-24.md`)
- [ ] Run one real installed-backend DPPO smoke and save artifacts
- [ ] Run one real installed-backend ECHO/RWML smoke and save quality metrics
- [ ] Keep world-model quality diagnostic-only until correlated with heldout pass@k and safety

## Active — Training config modal UI pass (2026-06-23)
- [x] Widen and reorganize the training setup experience
  - [x] Convert Training Configuration from a skinny single column into a wider two-column workspace
  - [x] Broaden shared modal sizing defaults where appropriate
  - [x] Loosen Training Monitor page width and responsive grid spans
  - [x] Run frontend checks (`npm run typecheck`, `npm run lint`, `git diff --check` -> clean; browser screenshot smoke unavailable because no Chrome/Edge executable or Playwright install was available)

## Active — Training guidance UI integration (2026-06-23)
- [x] Integrate rendered training guidance into the platform UI
  - [x] Add a rendered Training Guides surface inside the Training overlay
  - [x] Surface setup-time best practices and tooltips in Training Configuration
  - [x] Connect the guidance to source-backed platform patterns from HF TRL, Unsloth, OpenAI RFT, verl, and SkyRL
  - [x] Run focused frontend verification (`npm run typecheck`, `npm run lint` -> clean; Playwright smoke rendered Guides and Sources subtabs with screenshots in `artifacts/training-guides-*-smoke.png`)

## Active — TMax → BashGym action plan (2026-06-22)
- [x] Read Lambert Substack, TMax paper/blog/repo, and referenced RL papers
- [x] Map research takeaways onto current BashGym architecture and roadmap
- [x] Write a concrete implementation action plan with priorities and verification gates (`tasks/tmax-bashgym-action-plan-2026-06-22.md`)
- [~] Implement Phase 0/1 foundation: first-class terminal environment contracts, TMax-style import, metrics, decontamination, and materialization helpers
  - [x] Add `bashgym/environments/` package and tests
  - [x] Add fixture-backed TMax/Harbor-like importer tests
  - [x] Register `terminal_env_generation` Data Designer pipeline skeleton
  - [x] Add `/api/environments/*` routes for pipeline metadata, normalize/import, decontamination, and materialization
  - [x] Add Factory → Environment Lab UI for importing, inspecting, filtering, and materializing executable terminal environments
  - [x] Run focused environment/Data Designer suite (`python -m pytest tests/environments tests/factory/test_data_designer.py -q -o addopts=` → 100 passed)
  - [x] Run API/environment/Data Designer verification (`python -m pytest tests/api/test_environment_routes.py tests/environments tests/factory/test_data_designer.py -q -o addopts=` → 106 passed)
  - [x] Run frontend checks (`npm run typecheck`, `npm run lint` → clean)
- [~] Wire environment pass@k into eval service and Factory workflows
  - [x] Add `bashgym/eval/environment_passk.py` with attempt records, pass@k reports, timeout/status/token telemetry
  - [x] Add `POST /api/eval/environments/passk` and optional model-registry benchmark recording
  - [x] Add Environment Lab pass@k panel for pasted/JSONL rollout attempts and failed-baseline smoke reports
  - [x] Run focused eval/API/environment verification (`python -m pytest tests/eval/test_environment_passk.py tests/eval/test_passk.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/environments -q -o addopts=` → 50 passed)
  - [x] Add local persistent-shell command-script rollouts that materialize an `EnvironmentSpec`, execute commands, run the verifier, and emit pass@k-ready telemetry
  - [x] Add `POST /api/eval/environments/local-rollout-passk` plus Environment Lab local rollout controls
  - [x] Run focused rollout/API/frontend verification (`python -m pytest tests/environments/test_rollout.py tests/eval/test_environment_passk.py tests/api/test_eval_routes.py -q -o addopts=`, `npm run typecheck`, `npm run lint` → clean)
  - [x] Run broad relevant gate (`python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_passk.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/factory/test_data_designer.py -q -o addopts=` → 142 passed)
  - [x] Connect served-model policy rollouts to the local rollout/pass@k harness through `POST /api/eval/environments/model-rollout-passk`
  - [x] Add Environment Lab model rollout controls for endpoint, model, attempts, tool-call mode, and verifier-backed pass@k output
  - [x] Run served-model rollout verification (`python -m pytest tests/environments/test_rollout.py tests/eval/test_service.py tests/api/test_eval_routes.py -q -o addopts=` → 44 passed; `npm run typecheck`, `npm run lint` → clean)
  - [x] Run broad served-rollout gate (`python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_passk.py tests/eval/test_service.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/factory/test_data_designer.py -q -o addopts=` → 166 passed)
  - [x] Add protected-file manifest checksums for verifier scripts, tests, private fixtures, and `env.json`
  - [x] Audit local/model rollout workspaces before verification and surface tamper status in Environment Lab
  - [x] Run tamper guardrail verification (`python -m pytest tests/environments/test_builder.py tests/environments/test_rollout.py -q -o addopts=` → 17 passed; broader environment/DPPO gate → 142 passed; frontend checks → clean)
  - [x] Expose served-rollout observation prompt budget in API metadata and Environment Lab while preserving raw rollout logs
  - [x] Run observation-budget verification (`python -m pytest tests/eval/test_service.py tests/api/test_eval_routes.py -q -o addopts=` → 41 passed; broader environment/DPPO gate → 143 passed; Ruff/frontend checks → clean)
  - [x] Add built-in reward-hacking canary suite for verifier, tests, private fixture, and task-manifest tamper attempts
  - [x] Add `/api/eval/environments/reward-hacking-canaries` and Environment Lab Guardrail Canaries panel
  - [x] Run canary verification (`python -m pytest tests/environments/test_canaries.py tests/environments/test_builder.py tests/environments/test_rollout.py tests/api/test_eval_routes.py -q -o addopts=` → 43 passed; broader environment/DPPO gate → 149 passed; Ruff/frontend checks → clean)
  - [x] Add deterministic environment holdout splits with content-hash contamination manifests and release-gate thresholds
  - [x] Add `/api/eval/environments/holdout-gate` and Environment Lab Holdout Gate controls for grouped split, pass@k, leakage, timeout, and tamper verdicts
  - [x] Run focused holdout-gate verification (`python -m pytest tests/eval/test_environment_holdout.py tests/eval/test_environment_passk.py tests/api/test_eval_routes.py -q -o addopts=` → 34 passed; Ruff/frontend checks → clean)
  - [x] Run broad environment/eval/API/DPPO gate with holdout coverage (`python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` → 157 passed; `git diff --check` → clean)
  - [x] Smoke Environment Lab in Chrome against local backend/frontend; Holdout Gate panel rendered (`artifacts/environment-holdout-gate-smoke.png`, ignored)
  - [x] Store environment holdout gate manifests/verdicts on model profiles and expose latest environment holdout verdict through `/api/eval/verdict/{model_id}`
  - [x] Add Environment Lab registry controls for recording holdout gates against a model id
  - [x] Add Harbor-native Terminal-Bench command generation alongside legacy `tb run`
  - [x] Run registry/Harbor gate verification (`python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/eval/test_heldout_registry.py tests/eval/test_benchmarks_ext.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` → 177 passed; Ruff/frontend checks → clean; `git diff --check` → clean)
  - [x] Smoke Environment Lab in Chrome; Holdout Gate registry controls rendered with no console errors (`artifacts/environment-holdout-registry-controls-smoke.png`, ignored)
  - [x] Add Olmo-style spurious-reward negative-control audit for environment holdouts (`bashgym/eval/environment_spurious_reward.py`, `/api/eval/environments/spurious-reward-control`, Environment Lab panel)
  - [x] Run spurious-control verification (`python -m pytest tests/eval/test_environment_spurious_reward.py tests/api/test_eval_routes.py -q -o addopts=` → 32 passed; broader environment/eval/API/DPPO gate → 184 passed; Ruff/frontend checks and `git diff --check` → clean)
  - [x] Smoke Environment Lab in Chrome; Spurious Reward Control panel rendered with no console errors (`artifacts/environment-spurious-reward-control-smoke.png`, ignored), stubbed UI action flow completed (`artifacts/environment-spurious-reward-control-action-smoke.png`, ignored), and live backend endpoint returned `bashgym.environment_spurious_reward_control.v1`
  - [x] Add paired-bootstrap base-vs-candidate environment holdout comparison gate (`bashgym/eval/environment_holdout_comparison.py`, `/api/eval/environments/holdout-comparison`, Environment Lab panel)
  - [x] Run holdout-comparison verification (`python -m pytest tests/eval/test_environment_holdout_comparison.py tests/api/test_eval_routes.py -q -o addopts=` → 33 passed; broader environment/eval/API/DPPO gate → 190 passed; Ruff/frontend checks and `git diff --check` → clean)
  - [x] Smoke Environment Lab in Chrome; Holdout Comparison action flow completed with no console errors (`artifacts/environment-holdout-comparison-action-smoke.png`, ignored) and live backend endpoint returned `bashgym.environment_holdout_comparison.v1`
  - [x] Fold precomputed environment evidence into `/api/eval/heldout` release verdicts (`bashgym/eval/release_gate.py`), including holdout, holdout-comparison, and spurious-reward gates plus optional pass@k support
  - [x] Run unified release-gate verification (`python -m pytest tests/eval/test_release_gate.py tests/api/test_eval_routes.py -q -o addopts=` → 35 passed; broader environment/eval/API/DPPO gate → 196 passed; Ruff and `git diff --check` → clean)
  - [x] Add Evaluator → Held-out Gate release-evidence UI for attaching environment pass@k, holdout, holdout-comparison, and spurious-control JSON to `/api/eval/heldout`
  - [x] Verify release-evidence UI (`npm run typecheck`, `npm run lint` → clean; Chrome smoke rendered `Release evidence` with no console errors and saved `artifacts/evaluator-release-evidence-smoke.png`; focused backend release-gate tests → 35 passed)
  - [x] Add standalone external benchmark result ingest for Harbor/Terminal-Bench, BFCL, SWE-bench, and other public harness JSON (`/api/eval/benchmarks/external-ingest`) with tolerant score/pass-rate/trial-reward normalization and model-registry recording
  - [x] Add Evaluator → Held-out Gate external benchmark ingest UI for recording pasted public harness results after command execution
  - [x] Verify external benchmark ingest (`python -m pytest tests/eval/test_benchmarks_ext.py tests/eval/test_service.py tests/api/test_eval_routes.py -q -o addopts=` → 69 passed; broader environment/eval/API/DPPO gate → 201 passed; Ruff clean; `npm run typecheck`, `npm run lint` → clean; live backend route returned normalized `harbor_terminal_bench`; Chrome smoke rendered and posted sample Harbor-style trial JSON with no console errors; `git diff --check` → clean)
  - [x] Fold external benchmark reports/manifests into `/api/eval/heldout` release verdicts with optional minimum-score thresholds and explicit external pass/hold status
  - [x] Add Evaluator release-evidence field for normalized external benchmark reports
  - [x] Verify external benchmark release evidence (`python -m pytest tests/eval/test_release_gate.py tests/api/test_eval_routes.py -q -o addopts=` → 40 passed; broader environment/eval/API/DPPO gate → 205 passed; Ruff/frontend checks → clean; Chrome smoke rendered `External benchmarks` release-evidence field with no console errors; `git diff --check` → clean)
- [~] Implement Phase 2 training stability profile: `terminal_rl_tmax_like`, active sampling, zero-std filtering, group-size gates, and UI telemetry
  - [x] Add pure terminal-RL profile and active-sampling helpers
  - [x] Thread profile settings through `TrainerConfig`, API schema, generated GRPO scripts, and run metadata
  - [x] Add Training Config controls and GRPO telemetry parsing/display hooks
  - [x] Run focused backend/frontend verification
  - [x] Wire active-sampling enforcement into served-model environment rollout batches with API/UI telemetry
  - [x] Add served-rollout behavior logprob capture plumbing for future DPPO validation
  - [x] Add unit-tested DPPO Binary-TV/Binary-KL mask math and readiness telemetry
  - [x] Add DPPO backend capability probe/config and explicit GRPO fallback reporting
  - [~] Feed sampled non-zero-std rollout batches into the DPPO optimizer loop
    - [x] Export sampled served-model rollouts as DPPO replay JSONL artifacts for train-logprob replay
    - [x] Add train-policy logprob replay enrichment with DPPO mask telemetry and API/UI adapter
    - [x] Add backend-specific launcher/script planner for `verl`/SkyRL/TMax smoke training
    - [x] Add local `bashgym training smoke-bundle` readiness report for replay/logprob/world-model/backend launch artifacts
    - [ ] Run a real backend smoke on an installed DPPO stack and record saved artifacts
- [~] Fold JEPA/ECHO/RWML world-model learning into the training platform
  - [x] Record Claude-session ECHO/RWML handoff (`tasks/jepa-worldmodel-hardware-handoff-2026-06-23.md`)
  - [x] Dispatch focused JEPA, Yann LeCun, BashGym-world-model, and HF/Unsloth settings research
  - [x] Thread ECHO/RWML settings through training API and frontend run config
  - [x] Add in-product setup guidance for SFT, terminal RL, and world-model metrics/settings
  - [x] Expose world-model DPPO replay export and smoke-launch config plumbing
  - [x] Write JEPA/BashGym action plan (`tasks/jepa-bashgym-action-plan-2026-06-23.md`)
  - [x] Verify schema, frontend type safety, and lint gates
  - [x] Add replay-level ECHO/RWML coverage telemetry to DPPO summaries and Environment Lab
  - [x] Add concrete training curriculum docs for how the gym works and operator best practices
    - [x] Create `docs/training/` docs covering overview, strategy selection, world models, metrics/runbook, and glossary
    - [x] Add agent-facing CLI guide for machine-readable setup and replay analysis
    - [x] Link the new docs from README and existing training docs
    - [x] Verify markdown links and source-backed defaults
  - [x] Add dependency-free `bashgym` CLI entrypoint for manifest, training docs, training plans, replay summaries, and server launch
    - [x] Fix `bashgym serve --host/--port/--log-level` forwarding and backend startup under `LOG_FORMAT=json`
  - [x] Add agent-facing `bashgym training analyze` for metrics, replay, release-evidence, and world-model coverage diagnostics
  - [x] Add backend-facing adapter utilities for ECHO masks and RWML reward inputs from DPPO replay payloads
  - [x] Add trainer-facing backend adapter hooks for ECHO `compute_loss` and RWML reward functions
  - [x] Add local backend probe that tokenizes ECHO spans, counts RWML targets, and writes launch env artifacts
  - [ ] Wire/test those hooks inside an installed DPPO/GRPO backend checkout
  - [x] Add world-model quality metrics to dashboards and release evidence
    - [x] Add diagnostic `world_model_quality` release-evidence lane for ECHO/RWML metrics
    - [x] Add Training Monitor world-model quality panel for backend-emitted ECHO/RWML stats
    - [x] Keep world-model quality diagnostic-only until correlated with heldout pass@k and safety

## Carry-forward branch and GX10 follow-ups
- [ ] **Promote `integration/unify-2026-06-15` → `feat/...` + push** (unifies desktop + GX10 on one branch) — *needs user go (affects shared remote)*
- [ ] **Sync GX10**: after push, `git fetch && checkout` the unified branch on ponyo
- [x] Consolidate untracked `train_gemma4_*.py` → `scripts/` — no matching files exist in the current worktree; no local consolidation remains

## Segments 1–8 — PENDING (see roadmap for full cards)
- [~] **S1** Model-agnostic core — headline delivered:
  - [x] core `bashgym/families/` registry + select_backend + apply_patches (`4477a99`, 28 tests)
  - [x] GRPO generator consumes profile `target_modules` (`ee37dbe`, behavior-preserving)
  - [x] **GRPO backend dispatch + plain-transformers generator** (`acf1b6f`) — Unsloth↔plain switch; plain path profile-driven (patches/excludes/attn via apply_patches); Unsloth output byte-identical to baseline; +4 dispatch tests
  - [x] **model-agnostic `scripts/train_model.py --base-model`** (`4cee6af`); retired 4 `train_gemma4_*.py`; +6 tests
  - [ ] *(follow-up)* Liger FLCE opt-in — **DGX-dependent** (needs Liger Gemma 4 patch, linkedin/Liger-Kernel#1186; verify on Spark)
  - [ ] *(follow-up, design note)* SFT/DPO backend dispatch — bigger lift; SFT/DPO read `config.lora_target_modules` (a user knob) so a profile swap would break it. Right design: profile as default, config as override — defer until a model with non-standard targets needs it.
- [~] **S2** GGUF + Ollama Modelfile export — core+wiring done:
  - [x] `bashgym/export/gguf.py`: build_modelfile (always emits TEMPLATE), ollama-modelfile parser, base-template reuse, check_template_roundtrip (`ffa08d7`, 8 tests)
  - [x] deploy_gguf_to_ollama emits correct TEMPLATE + reuses base via `ollama show`; `TrainerConfig.ollama_base_tag` (`f566ecf`) — fixes the real #1 deploy bug (was template-less)
  - [ ] *(DGX runtime)* end-to-end smoke: export merged checkpoint → deploy → render HF-template vs Ollama → check_template_roundtrip passes; wire the roundtrip check as a deploy gate
- [~] **S3** Held-out trace eval — core done (`6cd57cb`, `bashgym/eval/`, 31 tests): tool-call metrics (name + per-arg F1), session-clustered paired bootstrap, contamination-free session/repo split + manifest, deploy gate (min delta + CI-excl-0 + max forgetting). Remaining *(DGX/runtime)*: eval runner (serve base+candidate via vLLM, episode pass@k in sandbox), registry gating wiring, fix the contaminated `eval_finetuned.py`.
- [~] **S4** HF public dataset ingestion — core done (`b91ea13`, 23 tests): `decontaminate.py` (13-gram + 3-gram-Jaccard benchmark gates — closes the no-decontamination gap), `mixer.py` (self-fraction ratio mixing, up/down-sample), `converters.normalize_public_messages` (OpenHands/SWE-agent/ShareGPT → our messages, JSON-string tool args → dicts). Remaining *(runtime/HF)*: promote ws2 research scanner simulate→real + `hf_ingest.py` download orchestration for SWE-rebench/Kwai-Klear/Nemotron-SWE/Toucan/SWE-chat.
- [~] **S5** Data pipeline quality upgrades — partial (`ac367fa`, `d95b3b2`, current strict-validator slice): converters carry a continuous `quality_score` [0,1] in example metadata (SERA-style soft signal, not just gold/failed bucket); `masking.py` = SFT loss policy (train on assistant turns only, never observations); decision-level DPO is wired into the factory; strict DPO/preference pair artifact validation now exists and is required for DPO RunCard promotion checks. Remaining: SFT trainer assistant-only-loss application *(DGX)*; per-ModelProfile tool sanitization (Gemma4 `<|"|>`).
- [~] **S6** External benchmark evals (lm-eval/Terminal-Bench 2.0/BFCL-V4/SWE-bench)
  - [x] Generate lm-eval, legacy Terminal-Bench, Harbor Terminal-Bench, BFCL, and SWE-bench commands from the evaluator service
  - [x] Ingest standalone external harness result JSON into normalized model-registry benchmark records
  - [x] Add benchmark-version/result-manifest attachment to release verdict records through external benchmark release evidence
  - [x] Add first-class BFCL/SWE-bench result adapters if their local harness outputs need richer per-category drilldown
- [ ] **S8** DGX env consolidation *(APPROVAL-GATED)*

## Future (from research; not yet scheduled)
- [ ] Explicit agent-status protocol to replace regex terminal parsing; session replay scrubber over trace events
- [ ] OpenPipe-style relabel/prune ops; Fireworks-style token-level loss-mask rendering (needs tokenizer in backend)
