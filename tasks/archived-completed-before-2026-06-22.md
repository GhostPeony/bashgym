# Archived Completed Tasks Before 2026-06-22

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
