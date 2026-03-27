# NVIDIA Data Designer Integration Feedback

**From:** BashGym (Ghost Peony LLC)
**Date:** March 25, 2026
**Data Designer Version:** >=0.5.4
**Contact:** cade@ghostpeony.com

---

## Context

BashGym is an open-source platform that trains personal reasoning language models from AI coding sessions. We integrated NVIDIA NeMo Data Designer as the synthetic data generation engine — 5 pipeline types producing SFT, DPO, and tool-use training data from execution traces.

In v0.2, we built the **AutoCurriculum Compiler**: an evolutionary search system that mutates Data Designer pipeline configs to find schemas that produce measurably better training data than hand-tuned presets. This document captures our integration experience, what worked well, and what we'd love to see.

---

## What Worked Well

### Column DAG Architecture
The column-based pipeline model (`SamplerColumnConfig` → `LLMTextColumnConfig` → `LLMStructuredColumnConfig` → `LLMJudgeColumnConfig` → `ExpressionColumnConfig`) maps perfectly to training data generation. Each column has clear inputs/outputs, and the DAG execution handles dependencies automatically. This made it straightforward to build 5 different pipeline presets with different column topologies.

### Structured Output via Pydantic
`LLMStructuredColumnConfig` with Pydantic models for output enforcement is excellent. Our `AgentSolution` and `ToolUseConversation` schemas produce consistently structured tool-use trajectories that feed directly into NeMo-format training data. The structured output quality is dramatically better than prompt-only approaches.

### LLM-as-Judge with Score Objects
`LLMJudgeColumnConfig` with named `Score` dimensions (correctness, tool_usage, completeness) provides granular quality signals. We use these scores for:
1. Filtering low-quality examples in the pipeline (`quality_score.correctness >= 3`)
2. SchemaResearcher evaluation (average judge scores as a fast filter before micro-training)
3. Quality dashboard analytics (score distributions across domains)

### Expression Columns
Jinja2 templating in `ExpressionColumnConfig` is the glue that makes everything work. Flattening structured `AgentSolution` objects into training-ready text, computing DPO chosen/rejected from dual judge scores — expression columns handle the reshape without custom code.

### Config Serialization (YAML/JSON)
`DataDesignerConfigBuilder.from_config()` accepting YAML/JSON configs was essential for our SchemaResearcher. Evolved schemas are YAML-serializable config dicts that we can mutate, validate, serialize, and version without any custom serialization layer.

### Agent Context CLI
`data-designer agent context` providing the full API surface in one read dramatically reduces the integration burden. We proxy this through a REST endpoint so the frontend can discover capabilities at runtime.

---

## What We'd Love to See

### 1. Validation Column for Code Execution
**Priority: High**

`ValidationColumnConfig` with `CodeValidatorParams` exists in the docs but we couldn't get reliable code execution validation in our pipelines. We'd love:
- A `CodeExecutionValidator` that runs generated code in a sandbox and returns pass/fail + output
- Integration with Docker containers for safe execution (we already have sandbox infrastructure in BashGym)
- Timeout and resource limits as config parameters

This would replace our manual post-pipeline verification step.

### 2. Per-Column Provider Assignment (First-Class)
**Priority: High**

We achieved multi-provider by adding multiple `ModelProvider` entries and assigning model aliases to different providers. It works, but it's implicit — you have to know that `"code-model"` maps to NVIDIA NIM and `"judge-model"` maps to Anthropic by reading the `ModelConfig` alias assignments.

**What we'd love:** Explicit per-column `provider` field on `LLMTextColumnConfig`, `LLMStructuredColumnConfig`, etc. Something like:
```python
dd.LLMStructuredColumnConfig(
    name="solution",
    provider="nvidia-nim",  # Explicit provider for this column
    model="qwen/qwen2.5-coder-32b-instruct",
    ...
)
```

### 3. Embedding Column for Dedup
**Priority: Medium**

`EmbeddingColumnConfig` is documented but we ended up building our own `EmbeddingDeduplicator` with NIM API calls because we needed:
- Batch embedding computation (not per-row)
- Pairwise similarity comparison across the full generated dataset
- A diversity score metric

If EmbeddingColumnConfig could output vectors that we could then pass to a post-processor for dedup, that would integrate more cleanly.

### 4. Pipeline Metrics / Stats Output
**Priority: Medium**

After `designer.generate()`, we get a DataFrame but no metadata about the generation process:
- How many rows were filtered by processors?
- What was the average judge score before filtering?
- How long did each column take?
- What was the token cost per column?

We compute some of this ourselves for the quality dashboard, but having it built into the generation result would save work and be more accurate.

### 5. Incremental / Streaming Generation
**Priority: Medium**

For SchemaResearcher evaluation, we generate 25 examples just to check judge scores (Stage 1 fast filter). Currently this runs the full pipeline. It would be useful to:
- Generate N rows and stop early
- Stream rows as they're generated (for real-time UI updates)
- Cancel a generation mid-pipeline if early results look bad

### 6. Schema Diffing / Versioning
**Priority: Low**

Our SchemaResearcher evolves configs over generations. We track diffs manually (which params changed, by how much). Built-in config diffing or versioning would help:
- `dd.diff_configs(config_a, config_b)` → list of changes
- Semantic diff (not just structural) — "judge threshold changed from 3 to 4" vs "JSON key changed"

---

## Integration Patterns We Developed

These might be useful for other Data Designer integrations or future documentation:

### Shared Builder Pattern
We extracted a `build_base_config(PipelineConfig)` function that constructs the `ModelConfig` + `ModelProvider` entries shared across all 5 pipelines. Each pipeline adds its specific columns on top. This DRY pattern reduced ~500 lines of duplication.

### Feature Detection
Instead of version-checking `data_designer.__version__`, we use `hasattr(dd, 'ValidationColumnConfig')` to detect which features are available. This is more robust across patch versions and works even if the version scheme changes.

### Schema as Genome
For evolutionary optimization, we represent pipeline configs as YAML-serializable dicts (the "genome"). Mutations toggle booleans, perturb floats within bounds, and swap discrete choices. The evaluate function generates actual training data from the mutated config and measures downstream training loss. This treats Data Designer configs as a first-class search space.

### Two-Stage Evaluation
Generating full training data for every candidate schema is expensive. We use a two-stage filter:
1. **Stage 1 (fast):** Generate 25 examples, check average judge scores. Filter 50% of candidates.
2. **Stage 2 (slow):** Generate full dataset from top candidates, micro-train for 50 steps, measure loss.

This reduces evaluation cost ~5x while maintaining signal quality.

---

## Numbers

| Metric | Value |
|--------|-------|
| Pipeline types built | 5 (SFT, DPO, tool-use, external, unstructured) |
| Column types used | 6 (Sampler, LLMText, LLMStructured, LLMJudge, Expression, Processor) |
| Training strategies integrated | 6 (SFT, DPO, GRPO, RLVR, Distillation, Cascade RL) |
| Gold traces available | 3,139 |
| Tests covering DD integration | 117 (factory layer) |
| Schema evolution generations tested | 20+ per search run |
| Cascade RL domains | 4 (file_ops, bash, search, multi_step) |

---

## Summary

Data Designer is the right abstraction for training data generation. The column DAG model, structured output enforcement, and LLM-as-judge scoring are exactly what's needed for building self-improving training pipelines. The main gaps are around execution validation, per-column provider ergonomics, and generation observability.

We're happy to discuss any of this in more detail or contribute upstream if useful.

— Cade Russell, Ghost Peony LLC
