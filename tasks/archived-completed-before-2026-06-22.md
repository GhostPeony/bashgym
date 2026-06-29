# Archived Completed Tasks Before And Through 2026-06-22

Archived on 2026-06-29 from `tasks/todo.md` to keep the active tracker focused.

## Segment 0 — Reconcile Divergent Branches And Land Finished Work

Completed 2026-06-15.

- [x] Safety: tag `backup/pre-reconcile-2026-06-15` + origin `backup/desktop-frontend-2026-06-15` (rescued 6 desktop-only commits)
- [x] Verified GX10 (ponyo) fully backed up: ws2-hf-research = origin; `ee1c1dc` already in ws2
- [x] gitignore generated data dirs; commit finished work (`69963e6`, `698d78e`, `554f4f9`)
- [x] Fix broken `test_grpo_script.py` (grpo_use_vllm/grpo_backend config) — 9/9 green
- [x] Merge ws2 ↔ desktop on `integration/unify-2026-06-15` (`569d039`): 5 conflicts resolved
      (trainer.py = ws2 Unsloth GRPO + my config params; routes.py = both endpoint sets; 3 frontend = union)
- [x] Fixed latent ws2 bug: unescaped `{state.global_step}` in generated GRPO script (NameError)
- [x] Verified: GRPO test 9/9; frontend tsc + eslint clean; 1530 collect; 190/194 pure-logic pass

## Completed New Findings

- [x] **4 pre-existing cascade preflight tests fail** — FIXED (`5e…` converter handles messages format + simulate skips empty-domain preflight); 57 cascade tests green. Also fixed test_pipeline_builders drift.
- [x] **Test suite is non-hermetic** — FIXED (`e1bc45b`): `--timeout=120` bounds every test, `conftest.py` skips `@pytest.mark.network` by default (`--run-network` to include); confirmed hangers marked (factory_routes/e2e_pipeline/remote_integration). Non-network bulk green (484). Remaining fast (non-hanging) drift failures tracked → task #13.

## Segment 7 — Flywheel Automation

- [x] Confirm `/api/cascade/distill` runs real MOPD distillation instead of the old placeholder
- [x] Wire pipeline gold-threshold cascade triggers into a CascadeStartRequest-compatible payload and scheduler queue
- [x] Add Pipeline config/API/UI fields for cascade base model, mode, stage steps, min examples, remote SSH, and repo-domain triggers
- [x] Keep Hermes SQLite `~/.hermes/state.db` importer fixture-backed and aligned with current Hermes session storage
- [x] Let AutoResearch mutate environment recipe axes and export reproducible proposals

## TMax Terminal-RL And JEPA Platform Work

Archived on 2026-06-29 from `tasks/todo.md` after the detailed 2026-06-22
TMax block became historical context rather than active work. Full technical
details remain in `tasks/tmax-bashgym-action-plan-2026-06-22.md` and
`tasks/jepa-bashgym-action-plan-2026-06-23.md`.

Completed foundations:

- [x] Read Lambert Substack, TMax paper/blog/repo, and referenced RL papers
- [x] Map research takeaways onto current BashGym architecture and roadmap
- [x] Write the TMax BashGym implementation action plan
- [x] Implement first-class terminal environment contracts, TMax/Harbor-like import fixtures, decontamination, materialization, and Factory Environment Lab import/inspect/materialize workflows
- [x] Wire environment pass@k through eval service and Factory workflows for pasted attempts, local rollouts, served-model rollouts, and registry recording
- [x] Add environment holdout gates, content-hash contamination manifests, tamper checks, reward-hacking canaries, spurious-reward controls, paired holdout comparison, and release-gate evidence integration
- [x] Add external benchmark command generation, result ingest, and release-evidence handling for Harbor/Terminal-Bench, BFCL, SWE-bench, and related harnesses
- [x] Implement terminal-RL stability plumbing: active sampling, zero-std filtering, DPPO Binary-TV/KL mask math, behavior/train logprob replay, backend launch planners, and local smoke-bundle readiness artifacts
- [x] Record the Claude-session JEPA/ECHO/RWML handoff and research synthesis
- [x] Thread ECHO/RWML settings through training API and frontend run config
- [x] Add replay-level ECHO/RWML coverage telemetry, backend-facing ECHO/RWML adapter utilities, trainer hook shims, local backend probes, world-model quality dashboards, and diagnostic release-evidence surfacing
- [x] Add in-product training guidance and docs for SFT, terminal RL, world models, metrics, runbooks, and glossary material
