# Synthetic Data Generator Design

> Turn seed traces into larger, high-quality datasets for LoRA fine-tuning

**Date:** 2026-01-25
**Status:** Approved
**Goal:** Amplify 84+ seed traces into 500-2000+ synthetic training examples while preserving repo-specific patterns

---

## Overview

The Synthetic Data Generator creates training data from real execution traces using NVIDIA NIM. It supports three strategies:

1. **Trace-Seeded** (primary) - Extract patterns from traces, generate new tasks following those patterns
2. **Augmented** - Generate N variations of each trace's prompt
3. **Schema-Driven** - Generate tasks from repo structure definitions

### Why This Matters

- **LoRA minimum**: 100 examples for reasonable results
- **LoRA optimal**: 1,000+ examples for best performance
- **Current state**: 84 seed traces
- **Gap**: Need 6-25x more data

Sources:
- [Unsloth Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- [AMD: 10x Fine-Tuning with Synthetic Data](https://www.amd.com/en/developer/resources/technical-articles/2025/10x-model-fine-tuning-using-synthetic-data-with-unsloth.html)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SyntheticDataGenerator                        │
│                                                                  │
│  Config:                                                         │
│  • repo_filter: "single" | "multi" | "all"                      │
│  • selected_repos: ["bashgym", "other-repo"]                  │
│  • strategy: "trace_seeded" | "augmented" | "schema_driven"     │
│  • preset: "quick_test" | "balanced" | "production" | "custom"  │
│  • provider: "nim" | "anthropic"                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ TRACE_SEEDED │ │  AUGMENTED   │ │SCHEMA_DRIVEN │
│              │ │              │ │              │
│ Extract      │ │ Take each    │ │ Build schema │
│ patterns  →  │ │ trace prompt │ │ from repo    │
│ Generate new │ │ → N variants │ │ structure →  │
│ tasks from   │ │              │ │ Generate     │
│ patterns     │ │              │ │ tasks        │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┴────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   NIM / Anthropic     │
            │   (generation LLM)    │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Synthetic Dataset    │
            │  (NeMo JSONL format)  │
            └───────────────────────┘
```

---

## Component 1: Pattern Extractor

Analyzes seed traces to understand what kinds of work you do and how you do it.

### Data Structures

```python
@dataclass
class TracePatterns:
    # Task classification
    task_types: Dict[str, float]  # {"feature": 0.4, "bugfix": 0.3, "refactor": 0.2, "spec": 0.1}

    # File patterns - which files get touched together
    file_clusters: List[FileCluster]  # [("src/*.py", "tests/*.py"), ("*.md", "docs/*")]
    common_paths: List[str]           # ["src/", "bashgym/", "frontend/src/"]
    languages: Dict[str, float]       # {"python": 0.6, "typescript": 0.3, "markdown": 0.1}

    # Tool sequences - common workflows
    tool_patterns: List[ToolSequence]  # [read→edit→bash, glob→read→edit→read]
    avg_tool_calls: int                # 12

    # Prompt patterns - how tasks are described
    prompt_templates: List[str]  # ["Fix the {issue} in {file}", "Add {feature} to {component}"]
    prompt_keywords: List[str]   # ["implement", "fix", "add", "refactor", "update"]

    # Repo context
    repo_name: str
    framework_hints: List[str]  # ["fastapi", "react", "pytest"]
```

### Extraction Process

1. **Task Classification** - Analyze prompt text + tool usage to classify each trace
   - "fix", "bug", "error" → bugfix
   - "add", "implement", "create" → feature
   - "refactor", "clean", "improve" → refactor
   - "spec", "design", "plan" → spec

2. **File Clustering** - Find files that are frequently modified together
   - If `src/foo.py` and `tests/test_foo.py` always change together → cluster them

3. **Tool Sequence Mining** - Find common N-gram patterns in tool calls
   - "Read → Edit → Bash(pytest)" appears in 60% of traces → strong pattern

4. **Prompt Template Extraction** - Generalize real prompts into templates
   - "Fix the bug in utils.py" → "Fix the {issue_type} in {file_path}"

---

## Component 2: Synthetic Task Generator

Uses extracted patterns to generate new realistic tasks via NIM.

### Data Structures

```python
@dataclass
class SyntheticTask:
    task_id: str
    prompt: str                    # The user request
    target_files: List[str]        # Files likely to be touched
    task_type: str                 # feature/bugfix/refactor/spec
    expected_tools: List[str]      # Predicted tool sequence
    source_pattern_id: str         # Which seed trace inspired this
    repo: str                      # Target repo
```

### Generation Process

1. **Sample from distributions** - Pick task type based on extracted weights
   - 40% feature, 30% bugfix, 20% refactor, 10% spec

2. **Select file cluster** - Pick a realistic set of files to target
   - Weighted by how often each cluster appeared in seeds

3. **Build prompt via NIM** - Send pattern + context to LLM:
   ```
   You are generating coding tasks for a {repo} codebase.

   Task type: {task_type}
   Target files: {file_cluster}
   Framework: {frameworks}

   Example real tasks from this repo:
   - "{seed_prompt_1}"
   - "{seed_prompt_2}"
   - "{seed_prompt_3}"

   Generate a new, realistic task that:
   - Matches the style of the examples
   - Targets the specified files
   - Is a {task_type} task

   Return only the task prompt, nothing else.
   ```

4. **Validate & dedupe** - Ensure generated task isn't too similar to seeds

---

## Component 3: Generation Presets

Research-based presets for optimal LoRA training.

### Dataset Size Guidelines

| Threshold | Examples | Notes |
|-----------|----------|-------|
| Minimum | 100 | Bare minimum for reasonable results |
| Recommended | 1,000+ | Where "more data = better" kicks in |
| Diminishing returns | 3,000+ | Quality matters more than quantity beyond this |

### Preset Definitions

```typescript
type GenerationPreset = "quick_test" | "balanced" | "production" | "custom"

const PRESETS = {
  quick_test: {
    label: "Quick Test",
    description: "Fast iteration, minimal generation",
    targetExamples: 100,
    multiplier: "auto",
  },
  balanced: {
    label: "Balanced (Recommended)",
    description: "Good quality/time tradeoff for LoRA",
    targetExamples: 500,
    multiplier: "auto",
  },
  production: {
    label: "Production",
    description: "Full dataset for best results",
    targetExamples: 2000,
    multiplier: "auto",
  },
  custom: {
    label: "Custom",
    description: "Set your own target",
    targetExamples: null,
    multiplier: null,
  }
}
```

### Auto-Multiplier Logic

```
multiplier = ceil(targetExamples / seedCount)

Example: 84 seeds, "balanced" preset (500 target)
→ multiplier = ceil(500 / 84) = 6
→ Generates ~504 examples
```

---

## Component 4: Output Format

### Directory Structure

```
data/synthetic/
├── {generation_run_id}/
│   ├── metadata.json          # Generation config, stats, seed info
│   ├── patterns.json          # Extracted patterns (for debugging/reuse)
│   ├── train.jsonl            # Training examples (90%)
│   ├── val.jsonl              # Validation examples (10%)
│   └── source_mapping.json    # Links synthetic → seed traces
```

### Example Format (NeMo-compatible JSONL)

```json
{
  "messages": [
    {"role": "system", "content": "You are a coding assistant working on the bashgym project..."},
    {"role": "user", "content": "Add retry logic to the API client when NIM requests fail"},
    {"role": "assistant", "content": "I'll add retry logic with exponential backoff..."}
  ],
  "metadata": {
    "synthetic": true,
    "strategy": "trace_seeded",
    "source_pattern": "pattern_003",
    "seed_traces": ["trace_abc", "trace_def"],
    "task_type": "feature",
    "repo": "bashgym",
    "generated_at": "2026-01-25T..."
  }
}
```

### Merge Options

- `synthetic_only` - Train purely on synthetic data
- `mixed` - Combine with real gold traces (default)
- `synthetic_weighted` - 2:1 or 3:1 synthetic:real ratio

---

## Component 5: API Design

### Endpoints

```python
POST /api/factory/synthetic/generate
  Request:
    - strategy: "trace_seeded" | "augmented" | "schema_driven"
    - repo_filter: "single" | "multi" | "all"
    - selected_repos: ["bashgym", ...]
    - preset: "quick_test" | "balanced" | "production" | "custom"
    - target_examples: int (if custom)
    - multiplier: int (if custom)
    - provider: "nim" | "anthropic"
    - merge_mode: "synthetic_only" | "mixed" | "synthetic_weighted"
  Response:
    - job_id: str
    - status: "queued"

GET /api/factory/synthetic/jobs/{job_id}
  Response:
    - status: "running" | "completed" | "failed"
    - progress: { current: 45, total: 500 }
    - patterns_extracted: 8
    - examples_generated: 45
    - estimated_remaining_seconds: 120

GET /api/factory/synthetic/runs
  Response:
    - List of past generation runs with stats

POST /api/factory/synthetic/runs/{run_id}/merge
  - Merge a synthetic run into training batches
```

---

## Component 6: Frontend UI

### Generator Panel

```
┌─────────────────────────────────────────────────────────────────┐
│  Synthetic Data Generator                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Strategy:  [Trace-Seeded ▼]  [Augmented]  [Schema-Driven]      │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Repo Filter:  ○ All  ○ Multi  ● Single                         │
│  Selected:     [bashgym ▼]                                    │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Preset:  [Quick Test]  [Balanced ✓]  [Production]  [Custom]    │
│                                                                  │
│  Target Examples: 500        Seeds Available: 84                │
│  Auto Multiplier: 6x         Est. Output: ~504 examples         │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Provider:     ● NIM (qwen-32b)  ○ Anthropic (claude-sonnet)    │
│  Merge Mode:   ○ Synthetic Only  ● Mixed  ○ Weighted (3:1)      │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Est. Time: ~8 min    Est. Cost: ~$0.40 (NIM)                   │
│                                                                  │
│            [ Generate Synthetic Dataset ]                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Progress View

```
┌─────────────────────────────────────────────────────────────────┐
│  Generating: run_20260125_143022                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [████████████░░░░░░░░░░░░░░░░░░] 42%                           │
│                                                                  │
│  Step 1: Extract patterns ✓ (8 patterns found)                  │
│  Step 2: Generate tasks   ⟳ 210/500                             │
│  Step 3: Validate & save  ○ pending                             │
│                                                                  │
│  Elapsed: 3m 22s    Remaining: ~4m 30s                          │
│                                                                  │
│            [ Cancel ]                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

| Component | Files | Effort | Priority |
|-----------|-------|--------|----------|
| PatternExtractor | `bashgym/factory/pattern_extractor.py` | New | P0 |
| SyntheticGenerator | `bashgym/factory/synthetic_generator.py` | New | P0 |
| API routes | `bashgym/api/factory_routes.py` | Extend | P0 |
| Frontend UI | `frontend/src/components/factory/SyntheticGenerator.tsx` | New | P1 |
| Augmented strategy | Wire existing `augment_example()` | Low lift | P2 |
| Schema-driven strategy | Wire existing `SchemaBuilder` | Low lift | P2 |

### Strategy Comparison

| Strategy | Input | Output Quality | Effort |
|----------|-------|----------------|--------|
| `trace_seeded` | Patterns from traces | High fidelity to workflow | New work |
| `augmented` | Trace prompts directly | Good, close to originals | Wire existing |
| `schema_driven` | Repo structure only | More varied, may drift | Wire existing |

---

## Configuration

Uses existing environment variables:

```bash
NVIDIA_API_KEY=nvapi-...           # Required for NIM
NIM_ENDPOINT=https://integrate.api.nvidia.com/v1
NIM_MODEL=qwen/qwen2.5-coder-32b-instruct
AUGMENTATION_PROVIDER=nim          # or "anthropic"
```

---

## Success Criteria

1. Generate 500+ synthetic examples from 84 seeds in <10 minutes
2. Synthetic tasks pass human review for realism (spot-check 20 samples)
3. LoRA trained on synthetic data shows improvement on held-out eval set
4. All three strategies functional and selectable in UI
