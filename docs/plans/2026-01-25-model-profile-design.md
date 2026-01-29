# Model Profile & Registry Design

> Agent profile pages for viewing trained models and their benchmarks

**Date:** 2026-01-25
**Status:** Draft

---

## Overview

A comprehensive model lifecycle management system that enables:
- **Model Selection**: Pick the best model to deploy based on performance
- **Training History**: Track improvement over time across training runs
- **Debugging**: Understand why models perform well/poorly via lineage and training data

---

## Data Model

### Model Profile Schema

```python
@dataclass
class ModelProfile:
    # Identity
    model_id: str                    # e.g., "bashgym-coder-v3"
    run_id: str                      # e.g., "run_20260125_091944"
    display_name: str                # User-editable name
    description: str                 # User-editable description
    tags: List[str]                  # e.g., ["production", "coding", "v3"]
    starred: bool                    # Pinned to top of list
    created_at: datetime

    # Lineage
    base_model: str                  # e.g., "Qwen2.5-Coder-1.5B"
    training_strategy: str           # SFT, DPO, GRPO, distillation
    teacher_model: Optional[str]     # If distillation
    training_traces: List[str]       # Which gold trace IDs were used
    parent_model: Optional[str]      # If fine-tuned from another trained model

    # Training
    config: Dict[str, Any]           # Full TrainerConfig snapshot
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    loss_curve: List[Dict]           # [{step, loss, val_loss?}, ...]
    final_metrics: Dict[str, float]  # {final_loss, epochs, samples}

    # Artifacts
    checkpoints: List[Dict]          # [{path, step}, ...]
    final_adapter_path: str
    merged_path: Optional[str]
    gguf_exports: List[Dict]         # [{path, quantization}, ...]

    # Evaluations
    benchmarks: Dict[str, BenchmarkResult]  # {benchmark_name: result}
    custom_evals: Dict[str, EvalResult]     # {eval_set_id: result}
    evaluation_history: List[Dict]          # Timestamped for trends

    # Operational
    model_size_bytes: int
    inference_latency_ms: Optional[float]
    status: str                      # ready, needs_eval, regression_detected
    deployed_to: Optional[str]       # e.g., "ollama:bashgym-v3"
```

Persisted as `model_profile.json` in each model directory and indexed in a central registry.

---

## Custom Eval Generation

### Approach

Generate evaluation sets from gold traces in two modes:

**1. Exact Replay Evals**
- Extract original `user_initial_prompt` from gold trace
- Extract verification criteria (test files, expected outputs)
- Run trained model with same prompt
- Score: pass/fail + similarity to original solution

**2. Variation Evals**
- Generate variations using Claude:
  - **Paraphrase**: Same task, different wording
  - **Parameter tweak**: Modify requirements slightly
  - **Complexity shift**: Add/remove constraints
- Each variation inherits verification from original
- Tests generalization vs. memorization

### Eval Set Schema

```python
@dataclass
class CustomEvalSet:
    eval_set_id: str
    source_trace_id: str
    eval_type: str                   # "replay" | "variation"
    prompt: str                      # Original or generated
    verification: EvalVerification
    difficulty: str                  # "same" | "easier" | "harder"
    created_at: datetime

@dataclass
class EvalVerification:
    method: str                      # "test_file" | "output_match" | "llm_judge"
    test_commands: List[str]         # e.g., ["pytest test_sort.py"]
    expected_patterns: List[str]     # Output markers
```

Auto-generate eval sets when traces are promoted to gold.

---

## UI Design

### Model Browser (List Page)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Models                                      [+ Train New] [Compare]    │
├─────────────────────────────────────────────────────────────────────────┤
│  Filters: [All Strategies ▼] [All Base Models ▼] [Tags... ] [Search...]│
│           Sort by: [Custom Eval ▼]  View: [Grid] [List] [Leaderboard]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────┐  ┌─────────────────────────┐              │
│  │ bashgym-v3         ★   │  │ bashgym-v3-dpo         │              │
│  │ Qwen-1.5B • SFT         │  │ Qwen-1.5B • DPO         │              │
│  │                         │  │                         │              │
│  │ Custom: 84.0%  ▲ +8.0   │  │ Custom: 81.2%  ▲ +5.2   │              │
│  │ HumanEval: 67.2%        │  │ HumanEval: 68.4%        │              │
│  │                         │  │                         │              │
│  │ Jan 25 • 142ms • 1.2GB  │  │ Jan 25 • 156ms • 1.2GB  │              │
│  │ [Production]            │  │ [Experimental]          │              │
│  │                         │  │                         │              │
│  │ [View] [Compare] [···]  │  │ [View] [Compare] [···]  │              │
│  └─────────────────────────┘  └─────────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- Grid view (cards), list view, or leaderboard view toggle
- Star/pin important models to top
- Tags: production, experimental, archived
- Quick metrics visible without clicking in
- Baseline model card (untrained) for comparison reference

### Model Profile Page

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ← Back to Models                                    [Compare] [Export] │
│                                                                         │
│  bashgym-coder-v3                                          ● Production│
│  Based on Qwen2.5-Coder-1.5B • SFT • Trained Jan 25, 2026              │
│  "Third iteration with expanded trace set"                              │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  QUICK STATS                                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  Custom  │ │ Benchmark│ │   Size   │ │ Latency  │ │ Training │      │
│  │  84.0%   │ │  68.5%   │ │  1.2GB   │ │  142ms   │ │  2h 34m  │      │
│  │  ▲ +8.0  │ │  ▲ +3.2  │ │  1.5B    │ │  ▼ -23ms │ │          │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                        (compared to previous version)   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ▼ Training Details                                                     │
│  ▼ Benchmark Results                                                    │
│  ▼ Custom Evaluations                                                   │
│  ▼ Lineage & Traces                                                     │
│  ▼ Artifacts & Export                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Quick Stats:**
| Stat | Description |
|------|-------------|
| Custom Eval Pass Rate | Your traces - the metric you care most about |
| Overall Benchmark Score | Aggregate across standard benchmarks |
| Model Size | Params + file size |
| Inference Latency | Avg ms per request |
| Training Time | How long it took |

### Expanded Sections

**▼ Training Details**
- Loss curve chart (step vs. loss)
- Full config table: learning rate, batch size, LoRA rank, epochs
- Training logs summary
- Hardware used: GPU, duration, samples/second

**▼ Benchmark Results**
- Table with all benchmark scores, sortable
- Radar chart across benchmark categories
- Drill-down to individual test cases
- Historical trend line per benchmark

**▼ Custom Evaluations**
- Replay results: X/Y gold traces reproduced
- Variation results: X/Y variations passed
- Failure analysis with diffs
- "Re-run evaluation" button

**▼ Lineage & Traces**
- Visual tree: base model → parent → this model
- Training data breakdown by repo
- Trace list with quality scores
- Teacher model link if distillation

**▼ Artifacts & Export**
- File tree of model directory
- One-click export: GGUF Q4, Q8, push to Ollama
- Deployment status
- Download links

### Comparison Views

**Side-by-Side**
```
┌─────────────────┬─────────────────┬─────────────────┐
│  bashgym-v2    │  bashgym-v3    │  bashgym-v3-dpo│
├─────────────────┼─────────────────┼─────────────────┤
│  Custom: 76.0%  │  84.0% ✓ best   │  81.2%          │
│  HumanEval: 63% │  67.2%          │  68.4% ✓ best   │
│  Latency: 165ms │  142ms ✓ best   │  156ms          │
└─────────────────┴─────────────────┴─────────────────┘
```

**Leaderboard**
- Sortable table of all models
- Default sort by custom eval score
- Filter by strategy, base model, date, tags

**Trend Charts**
- X-axis: time or model version
- Y-axis: selected metric
- Multiple metrics overlaid
- Annotations for key events

---

## API Endpoints

### Model Registry

```
GET    /api/models                    # List all models (with filters)
GET    /api/models/{model_id}         # Full model profile
POST   /api/models/{model_id}         # Update metadata
DELETE /api/models/{model_id}         # Archive/delete model
POST   /api/models/{model_id}/star    # Pin to top
```

### Evaluations

```
POST   /api/models/{model_id}/evaluate           # Run all benchmarks
POST   /api/models/{model_id}/evaluate/custom    # Run custom evals only
GET    /api/models/{model_id}/evaluations        # Evaluation history
```

### Custom Eval Sets

```
GET    /api/eval-sets                            # List custom eval sets
POST   /api/eval-sets/generate                   # Generate from gold traces
POST   /api/eval-sets/{set_id}/run               # Run against a model
```

### Comparison

```
POST   /api/models/compare                       # Compare multiple models
GET    /api/models/leaderboard                   # Ranked list by metric
GET    /api/models/trends                        # Metrics over time
```

### Artifacts

```
POST   /api/models/{model_id}/export/gguf        # Export to GGUF
POST   /api/models/{model_id}/deploy/ollama      # Push to Ollama
GET    /api/models/{model_id}/artifacts          # List all files
```

---

## Implementation Structure

### New Files

```
bashgym/
├── models/
│   ├── __init__.py
│   ├── registry.py          # ModelRegistry class
│   ├── profile.py           # ModelProfile dataclass
│   └── evaluator.py         # Custom eval generation & running
│
├── api/
│   └── models_routes.py     # /api/models/* endpoints

frontend/src/
├── components/models/
│   ├── ModelBrowser.tsx     # List page
│   ├── ModelCard.tsx        # Card component
│   ├── ModelProfile.tsx     # Profile page
│   ├── ModelComparison.tsx  # Side-by-side
│   ├── ModelLeaderboard.tsx # Ranked table
│   ├── ModelTrends.tsx      # Trend charts
│   ├── EvalResults.tsx      # Custom eval display
│   └── LineageTree.tsx      # Visual lineage
│
├── services/
│   └── api.ts               # Model API functions
```

### Files to Modify

| File | Changes |
|------|---------|
| `trainer.py` | Save config + loss curve to profile on completion |
| `benchmarks.py` | Link results to model registry |
| `routes.py` | Include models_routes router |
| `websocket.ts` | Model evaluation progress events |
| `App.tsx` | Add /models routes |

---

## Implementation Phases

### Phase 1: Model Registry & Profile (Foundation)
- Create `ModelProfile` dataclass with full schema
- Create `ModelRegistry` class to index existing models
- Scan `data/models/` and build initial registry
- Save `model_profile.json` to each model directory
- Basic API: list, get, update

### Phase 2: Backend Evaluation Integration
- Hook trainer to save full config + loss curve
- Hook benchmarks to write results to profile
- Add evaluation history tracking
- API: trigger evaluation, get results

### Phase 3: Custom Eval Generation
- Build eval set generator from gold traces (replay)
- Add variation generator using Claude API
- Eval runner for custom sets
- API: generate, run, view results

### Phase 4: Frontend Model Browser
- Model list page with grid/list views
- Filtering, sorting, search
- Model cards with quick stats
- Navigation to profile

### Phase 5: Frontend Model Profile
- Profile page with header + quick stats
- Collapsible detail sections
- Training details with loss curve chart
- Benchmark results display

### Phase 6: Frontend Comparison & Trends
- Side-by-side comparison view
- Leaderboard table
- Trend charts over time
- Lineage visualization

---

## Success Criteria

- [ ] All trained models indexed in registry with profiles
- [ ] Training saves full config and loss curve
- [ ] Benchmark results linked to model profiles
- [ ] Custom eval sets generated from gold traces
- [ ] Model browser shows all models with quick stats
- [ ] Profile page displays full model details
- [ ] Side-by-side comparison works for 2-3 models
- [ ] Leaderboard ranks models by any metric
- [ ] Trend charts show improvement over time
