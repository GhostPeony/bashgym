# BashGym → Model-Agnostic Pipeline — Task Tracker

Master plan: `tasks/roadmap-2026-06-15-model-agnostic-pipeline.md` (9 segments).
Reviews: `tasks/status-review-2026-06-11.md`, `tasks/eval-and-data-strategy-2026-06-11.md`.
Archive: `tasks/archived-completed-before-2026-06-22.md` contains completed items before/through 2026-06-22 that were cleared from this live tracker.

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
- [x] Add Factory Source Library UI for source recommendations, eval-only guardrails, local artifact preparation, and Training handoff
- [x] Add Hugging Face-backed source fetch orchestration through Source Library CLI/API/UI/Data Designer, capped local `source_records.jsonl` outputs, fetch reports, and fetch-to-artifact conversion
- [x] Add Source Library remote-fetch approval and cache policy for larger Hugging Face pulls across helper/API/CLI/UI/Data Designer
- [x] Add source-specific schema mapping for existing P0 preference/reward source cards (UltraFeedback Binarized and HelpSteer2) with mapper reports surfaced in Source Library output
- [x] Add Training RunCard Evidence UI/API for local RunCard discovery, promotion validation, claim-tier blockers, and artifact presence
- [x] Add compact RunCard promotion explanations with failed gates and next actions across CLI/API/UI
- [ ] Human decision: choose eval-only override policy, first installed backend, first cloud launcher priority, first public-source expansion set, remote/billable compute approval boundary, and claim-tier thresholds (`tasks/human-decisions-source-compute-2026-06-29.md`)

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

## Active — TMax / JEPA carry-forward (2026-06-22)
Archived completed 2026-06-22 TMax terminal-RL and JEPA platform details to
`tasks/archived-completed-before-2026-06-22.md`; the detailed technical record
also remains in `tasks/tmax-bashgym-action-plan-2026-06-22.md` and
`tasks/jepa-bashgym-action-plan-2026-06-23.md`.

- [~] DPPO optimizer/backend proof
  - [x] Local terminal-RL, rollout, pass@k, holdout, canary, spurious-control, paired-comparison, release-evidence, replay/logprob, backend-plan, and smoke-bundle foundations are implemented and archived
  - [ ] Run a real backend smoke on an installed DPPO stack and record saved artifacts
- [~] JEPA/ECHO/RWML world-model proof
  - [x] Local ECHO/RWML config, replay telemetry, guidance docs, diagnostics, release-evidence lane, and launch-probe plumbing are implemented and archived
  - [ ] Wire/test ECHO/RWML hooks inside an installed DPPO/GRPO backend checkout
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
