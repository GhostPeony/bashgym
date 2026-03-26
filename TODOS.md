# TODOS

## Deferred from AutoCurriculum Compiler (Phase 1+2)

### Phase 3: Cascade RL Scheduler
- **What:** Sequential domain RL training (CascadeScheduler) + MOPD distillation inspired by Nemotron Cascade 2
- **Why:** Cascade 2 showed sequential domain-by-domain RL with on-policy distillation produces dramatically better results than single-stage training
- **Depends on:** Phase 1+2 complete, proven schema evolution loop
- **Context:** Design doc at `~/.gstack/projects/GhostPeony-bashgym/Cade-feat-training-strategies-device-mgmt-design-20260325-210645.md`. Domain taxonomy: file_operations, bash_commands, search_and_navigate, multi_step_reasoning.

### Schema Sharing & Community Recipes
- **What:** Export/import evolved schemas with provenance metadata
- **Why:** Once winning schemas exist, sharing them accelerates others' training
- **Depends on:** Winning schemas from SchemaSearchSpace evolution
- **Context:** CEO review deferred this — build value first, share later

### Micro-Training Model Caching Optimization
- **What:** Keep base model loaded in VRAM between SchemaSearchSpace Stage 2 evaluations, swap only LoRA adapters
- **Why:** Saves ~30s per evaluation (5 evaluations = 2.5 min per generation)
- **Cons:** Risk of gradient leakage if LoRA reinitialization is imperfect. Unsloth's FastLanguageModel may not support repeated adapter swaps cleanly. CUDA memory fragmentation after many cycles.
- **Depends on:** Proven correct sequential evaluation first
- **Context:** Outside voice flagged this (eng review 2026-03-25). Deferred because correctness > speed for evolutionary search.

### Provider Config Reconciliation
- **What:** Sync BashGym ProviderRegistry configuration with Data Designer multi-provider entries automatically
- **Why:** Currently users must maintain two independent provider configs. If they disagree (different API keys, model selections), confusing behavior results.
- **Depends on:** Multi-provider support in Phase 1
- **Context:** Outside voice flagged (eng review 2026-03-25). For now, document the dual-config clearly.
