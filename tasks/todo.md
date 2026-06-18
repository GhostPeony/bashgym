# BashGym → Model-Agnostic Pipeline — Task Tracker

Master plan: `tasks/roadmap-2026-06-15-model-agnostic-pipeline.md` (9 segments).
Reviews: `tasks/status-review-2026-06-11.md`, `tasks/eval-and-data-strategy-2026-06-11.md`.

## Segment 0 — Reconcile divergent branches & land finished work ✅ COMPLETE (2026-06-15)
- [x] Safety: tag `backup/pre-reconcile-2026-06-15` + origin `backup/desktop-frontend-2026-06-15` (rescued 6 desktop-only commits)
- [x] Verified GX10 (ponyo) fully backed up: ws2-hf-research = origin; `ee1c1dc` already in ws2
- [x] gitignore generated data dirs; commit finished work (`69963e6`, `698d78e`, `554f4f9`)
- [x] Fix broken `test_grpo_script.py` (grpo_use_vllm/grpo_backend config) — 9/9 green
- [x] Merge ws2 ↔ desktop on `integration/unify-2026-06-15` (`569d039`): 5 conflicts resolved
      (trainer.py = ws2 Unsloth GRPO + my config params; routes.py = both endpoint sets; 3 frontend = union)
- [x] Fixed latent ws2 bug: unescaped `{state.global_step}` in generated GRPO script (NameError)
- [x] Verified: GRPO test 9/9; frontend tsc + eslint clean; 1530 collect; 190/194 pure-logic pass

### S0 follow-ups (carry forward)
- [ ] **Promote `integration/unify-2026-06-15` → `feat/...` + push** (unifies desktop + GX10 on one branch) — *needs user go (affects shared remote)*
- [ ] **Sync GX10**: after push, `git fetch && checkout` the unified branch on ponyo
- [ ] Consolidate untracked `train_gemma4_*.py` → `scripts/` (S1 supersedes with model-agnostic generator)

## New findings → tracked tasks
- [x] **4 pre-existing cascade preflight tests fail** — FIXED (`5e…` converter handles messages format + simulate skips empty-domain preflight); 57 cascade tests green. Also fixed test_pipeline_builders drift.
- [x] **Test suite is non-hermetic** — FIXED (`e1bc45b`): `--timeout=120` bounds every test, `conftest.py` skips `@pytest.mark.network` by default (`--run-network` to include); confirmed hangers marked (factory_routes/e2e_pipeline/remote_integration). Non-network bulk green (484). Remaining fast (non-hanging) drift failures tracked → task #13.

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
- [~] **S5** Data pipeline quality upgrades — partial (`ac367fa`, 8 tests, regression green): converters carry a continuous `quality_score` [0,1] in example metadata (SERA-style soft signal, not just gold/failed bucket); `masking.py` = SFT loss policy (train on assistant turns only, never observations). Remaining: wire dormant decision-level DPO into the factory; SFT trainer assistant-only-loss application *(DGX)*; per-ModelProfile tool sanitization (Gemma4 `<|"|>`).
- [ ] **S6** External benchmark evals (lm-eval/Terminal-Bench 2.0/BFCL-V4/SWE-bench)
- [ ] **S7** Flywheel automation (MOPD unstub, threshold→cascade trigger, Hermes importer)
- [ ] **S8** DGX env consolidation *(APPROVAL-GATED)*

## Future (from research; not yet scheduled)
- [ ] Explicit agent-status protocol to replace regex terminal parsing; session replay scrubber over trace events
- [ ] OpenPipe-style relabel/prune ops; Fireworks-style token-level loss-mask rendering (needs tokenizer in backend)
