# Changelog

All notable changes to Bash Gym are documented here.

## [0.2.0.0] - 2026-03-25

### Added
- **AutoCurriculum Compiler** — self-evolving training data pipeline
  - SchemaResearcher: evolutionary optimization of Data Designer pipeline configs with two-stage evaluation (judge filter + micro-training)
  - SearchSpace ABC generalizing AutoResearcher for pluggable optimization targets
  - Template library with failure-driven schema selection from failed trace analysis
  - Embedding-based dedup via NIM API with diversity scoring
  - Schema research API endpoints (start/stop/pause/resume/status/quality)
  - Frontend: 3-mode AutoResearch (hyperparam/trace/schema) with Evolution, Quality, and Templates tabs
- **Cascade RL Scheduler** — sequential domain-by-domain GRPO training inspired by Nemotron Cascade 2
  - 4-domain taxonomy: file_operations (syntax reward), bash_commands (execution), search_and_navigate (execution), multi_step_reasoning (verification)
  - Per-domain checkpoint chaining between stages
  - MOPD (Multi-domain On-Policy Distillation) merges domain experts into unified student
  - Cascade API endpoints (start/stop/status/distill) with WebSocket progress
  - Frontend: Cascade RL strategy in TrainingConfig with domain selection checklist
- **Data Designer upgrade** — multi-provider support, feature detection, shared builder infrastructure
  - ProviderSpec for per-column provider assignment (e.g., NIM for code, Anthropic for judge)
  - Feature detection via hasattr for graceful degradation across DD versions
  - Shared build_base_config() extracted from 5 pipeline builders (DRY refactor)
  - CLI passthrough endpoints for `data-designer agent context` and `validate`
- **Nemotron models** added to base model dropdown (Cascade-2-30B-A3B, Nano-4B, Mini-4B)
- **Roadmap** section added to README with shipped/up-next/backlog items
- 236 new tests across 9 test files covering factory, gym, and API layers

### Changed
- Training strategy selector: 5-column grid → 3-column for text overflow fix
- TrainerConfig: added task_domain, cascade_stage, cascade_run_id fields for cascade metadata
- remote_trainer.py: parameterized script_name (was hardcoded to train_sft.py)
- TODOS.md restructured as a proper roadmap

## [0.1.0] - 2026-03-20

### Added
- Initial release: trace capture, curation, SFT/DPO/GRPO/RLVR training
- NVIDIA NeMo Data Designer integration with 5 pipeline types
- AutoResearch evolutionary hyperparameter search
- Provider abstraction (Anthropic, NIM, Ollama)
- Remote SSH training on DGX Spark
- Electron desktop app with React frontend
