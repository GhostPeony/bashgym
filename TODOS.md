# Roadmap

## Shipped

### AutoCurriculum Compiler — Phase 1+2 (v0.2.0)
Multi-provider Data Designer upgrade, SchemaResearcher with evolutionary schema optimization, embedding-based dedup, training data quality dashboard.
- SearchSpace ABC generalizing AutoResearcher for pluggable optimization targets
- SchemaSearchSpace: two-stage evaluation (judge filter + micro-training)
- Template library with failure-driven schema selection
- 199 tests

### AutoCurriculum Compiler — Phase 3 (v0.2.0)
Cascade RL Scheduler with sequential domain-by-domain GRPO training, MOPD distillation.
- 4-domain taxonomy: file_operations, bash_commands, search_and_navigate, multi_step_reasoning
- Per-domain reward functions (syntax, execution, verification)
- Checkpoint chaining between stages
- MOPD merges domain experts into unified student
- 37 tests

---

## Up Next

### Black-Box On-Policy Distillation (Phase 3c)
- **What:** Real-time teacher inference during training — student generates rollouts, Claude/NIM labels them, student learns from the comparison
- **Why:** Offline distillation uses pre-generated teacher outputs which go stale. On-policy produces fresher, harder examples targeting the student's actual weaknesses
- **Complexity:** High — adds API billing during training loops, requires async teacher inference integration
- **Depends on:** Cascade RL proven end-to-end

### Schema Sharing & Community Recipes
- **What:** Export/import evolved schemas with provenance (which model, what eval scores, training lineage)
- **Why:** Once SchemaResearcher discovers winning schemas, sharing them lets others skip the search
- **Depends on:** Winning schemas from real SchemaSearchSpace runs

---

## Backlog

### Micro-Training Model Caching
Keep the base model loaded in VRAM between SchemaSearchSpace evaluations, swap only LoRA adapters. Saves ~2.5 min per generation but risks gradient leakage if adapter reinitialization is imperfect. Deferred until sequential evaluation is proven correct.

### Provider Config Reconciliation
Auto-sync BashGym ProviderRegistry with Data Designer's multi-provider entries. Currently users maintain two independent provider configs — document the dual-config for now, automate later.

### Cascade Domain Discovery
Automatically discover domain taxonomy from trace analysis rather than using fixed 4 domains. Analyze tool usage patterns across gold traces to suggest new domains.

### Cascade Ordering Optimization
Integrate cascade stage ordering into AutoResearch — evolve the order of domain stages alongside hyperparameters. Some orderings may produce better final models than others.
