# Legacy AutoResearch Compatibility: Detailed Technical Outline

> **Historical, hidden, unsupported compatibility architecture.** This document
> describes the earlier in-process hyperparameter, trace, schema, and
> data-recipe researchers. Its frontend panel and Zustand store are no longer
> official product surfaces. The `/api/autoresearch/*` routes are absent by
> default and register only when `BASHGYM_ENABLE_LEGACY_AUTORESEARCH=true`; they
> are never a durable campaign path or Control Room feature. New AutoResearch
> work uses durable campaigns, the Training → AutoResearch Control Room,
> registered model/data/evaluator/compute bindings, and `/api/campaigns/*`; see
> [Durable AutoResearch Campaigns](training/autoresearch-campaign.md).

## What this historical compatibility material documents

This historical implementation was BashGym's automated experimentation loop. It
mutated hyperparameter and data-curation settings, then:

1. Takes your current configuration as a starting point
2. Mutates it slightly
3. Runs a short training experiment
4. Measures the result
5. Keeps the mutation if it improved, reverts if it didn't
6. Repeats

It operates in two independent modes that search different spaces.

---

## Legacy level 1: Hyperparameter search

**Goal:** Find the best training hyperparameters for your model.

### Search Space (9 parameters)

| Parameter | Type | Range | Notes |
|---|---|---|---|
| `learning_rate` | float | 1e-6 to 1e-3 | Log-scale mutations (orders of magnitude, not linear) |
| `lora_r` | int | 4, 8, 16, 32, 64, 128 | Discrete choices, biased toward neighbors |
| `lora_alpha` | int | 8 to 256 | Continuous int range |
| `lora_dropout` | float | 0.0 to 0.3 | Linear mutations |
| `warmup_ratio` | float | 0.0 to 0.3 | Linear mutations |
| `gradient_accumulation_steps` | int | 1, 2, 4, 8, 16, 32, 64 | Discrete choices |
| `batch_size` | int | 1, 2, 4, 8, 16, 32 | Discrete choices |
| `max_seq_length` | int | 512, 1024, 2048, 4096, 8192 | Discrete choices |
| `load_in_4bit` | bool | true/false | Toggles QLoRA quantization |

The user selects which of these parameters to include in the search via checkboxes in the UI. Unselected parameters stay fixed at the base config values.

### The Mutation Algorithm

For each experiment, every selected parameter has a probability of being mutated (controlled by `mutation_rate`, default 0.3 = 30% chance per param):

- **Floats (log-scale):** Mutate in log space. `log(current) + gaussian(0, scale)`, then exponentiate back. This means a learning rate of 2e-4 might become 3e-4 or 8e-5 — it explores orders of magnitude naturally rather than making tiny linear adjustments.
- **Floats (linear):** `current + gaussian(0, current * mutation_scale)`. Clamped to the parameter's min/max range.
- **Ints (discrete choices):** 70% chance to move one step up or down in the choices list, 30% chance to jump to a random choice. So LoRA rank 16 usually becomes 8 or 32, but occasionally jumps to 128.
- **Ints (continuous):** `current + randint(-delta, +delta)` where delta is `current * mutation_scale`.
- **Bools:** Simply flipped.

The `mutation_scale` (default 0.2) controls how aggressive mutations are. 0.05 = conservative small tweaks, 1.0 = wild swings.

### Experiment Execution

Each experiment:

1. Deep-copies the current best config
2. Applies mutations to the copy
3. Runs the experiment in a thread executor (so it doesn't block the async event loop)
4. Currently **simulated** — sleeps 2-3 seconds and returns a realistic loss value based on the config
5. When real training is enabled: would train for N steps on a subset of the data and measure validation loss

### The Simulated Loss Function

The simulation models real hyperparameter dynamics so the search behaves realistically for demos:

- **Learning rate sweet spot** around 2e-5 (log10 = -4.7). Penalty grows quadratically as you move away. Extreme LR (>5e-4 or <1e-7) adds large penalties.
- **LoRA rank:** Diminishing returns on a log2 curve. Rank 16 is decent, 64 is good, 128 is marginally better.
- **Alpha/rank ratio:** Sweet spot at 2.0. Penalty for deviating.
- **Warmup:** Small warmup (0.03-0.1) helps. No warmup adds 0.15 penalty. Too much (>0.2) hurts.
- **4-bit quantization:** Full precision is slightly better (-0.05).
- **Base loss** starts at ~2.1, so defaults land around 2.3-2.5.
- **Progress bonus:** -0.15 over the full run (the search naturally improves over time).
- **Gaussian noise** (std=0.12) so not every experiment improves and the chart looks realistic.
- Loss clamped to 0.3-5.0.

### Keep or Revert

After each experiment:

```
if metric < best_metric:
    best_config = candidate    # Keep the mutation
    best_metric = metric
else:
    pass                       # Revert (candidate is discarded)
```

This is a greedy hill-climbing strategy. The noise in the evaluation function provides implicit exploration — sometimes a config that's actually better will look worse due to noise, and vice versa, which prevents getting stuck in local optima.

---

## Legacy level 2: Trace mining (data-pipeline optimization)

**Goal:** Find the best way to curate training data from your collected traces.

### Why This Exists

Hyperparameter tuning only gets you so far. Research consistently shows that **data quality matters more than model architecture or hyperparameters**. Trace mining optimizes the data pipeline itself: which traces to include, how to segment them, what quality thresholds to apply, and how to balance diversity vs. quality.

### Search Space (10 parameters)

| Parameter | Type | Range | Default | What It Controls |
|---|---|---|---|---|
| `min_success_rate` | float | 0.3 - 0.95 | 0.7 | How successful a trace's tool calls need to be for inclusion |
| `min_quality_score` | float | 0.2 - 0.9 | 0.5 | Minimum composite quality score |
| `max_steps_per_example` | int | 5 - 100 | 50 | Maximum tool calls in a single training example |
| `min_steps_per_example` | int | 1 - 20 | 3 | Minimum tool calls (filters trivial examples) |
| `include_cognitive` | bool | - | true | Include `<thinking>`, `<plan>`, `<reflection>` tags in training data |
| `include_failed_as_dpo` | bool | - | false | Use failed traces as DPO "rejected" examples |
| `time_gap_threshold_minutes` | float | 1.0 - 30.0 | 5.0 | Time gap that triggers a new segment boundary |
| `silver_inclusion_ratio` | float | 0.0 - 1.0 | 0.0 | What fraction of borderline ("silver") traces to include |
| `dedup_similarity_threshold` | float | 0.5 - 1.0 | 0.85 | How similar examples must be to be considered duplicates |
| `max_examples_per_repo` | int | 10 - 1000 | 500 | Cap per repo to prevent one project dominating the dataset |

### How These Parameters Interact (What the Simulation Models)

The simulated evaluation captures real data curation dynamics:

**Quality vs. Quantity Trade-off:**
- Lowering `min_success_rate` and `min_quality_score` includes more traces but adds noise
- The simulation generates more examples when thresholds are lenient (`quality_factor = 1.0 + (0.7 - min_success_rate) * 2`)
- But training quality peaks with moderate strictness — too strict means too little data, too lenient means noisy data

**Optimal Zones (what the search discovers):**
- `min_success_rate` ~ 0.8 (penalty grows linearly from optimal)
- `min_quality_score` ~ 0.6
- `silver_inclusion_ratio` ~ 0.2 (some silver helps diversity, too much adds noise)
- `time_gap_threshold_minutes` ~ 5.0 (too small fragments tasks, too large mixes unrelated work)
- `dedup_similarity_threshold` ~ 0.85

**Binary Switches:**
- `include_cognitive = true` gives a -0.15 bonus (teaching the model to reason, not just output)
- `include_failed_as_dpo = true` gives a -0.10 bonus (contrastive learning from mistakes)

**Data Volume Effects:**
- More data helps on a log scale: `-0.2 * log(examples / 100)`
- Per-repo caps under 200 add a -0.05 bonus (prevents overfitting to one codebase)
- Longer examples (higher `max_steps_per_example`) help up to ~50 steps

### Mutation Algorithm

Same evolutionary approach as hyperparameter search, but adapted for the data pipeline parameter types:

- **Floats:** `current + uniform(-delta, +delta)` where `delta = (max - min) * mutation_scale`. Clamped and rounded to 4 decimal places.
- **Ints:** `current + randint(-delta, +delta)` where `delta = (max - min) * mutation_scale`. Minimum delta of 1.
- **Bools:** Simply flipped.

### Experiment Output

Each trace mining experiment produces not just a metric value but also data statistics:
- **examples_generated:** How many training examples the pipeline config would produce
- **unique_repos:** How many different repos are represented
- **avg_example_length:** Average tool calls per example

These are displayed in the UI alongside the metric value, so you can see the data quality/quantity trade-off in real time.

---

## Legacy system architecture

### Full Stack Flow

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND                             │
│                                                          │
│  AutoResearchPanel.tsx                                   │
│    ├── Mode toggle: [Hyperparams] [Trace Mining]        │
│    ├── Config UI (param checkboxes, sliders, inputs)    │
│    ├── Start/Pause/Resume/Stop buttons                  │
│    ├── Progress bar + best metric display               │
│    ├── SVG mini-chart (experiment dots + best line)     │
│    ├── Terminal-style experiment log                     │
│    └── Best config comparison table                     │
│                                                          │
│  autoresearchStore.ts (Zustand)                          │
│    ├── Hyperparam state (status, experiments, best...)   │
│    ├── Trace research state (separate status/experiments)│
│    └── activeMode toggle                                 │
│                                                          │
│  api.ts                                                  │
│    └── autoresearchApi.start/stop/pause/resume/status    │
│    └── autoresearchApi.startTraceResearch/stop/...       │
│                                                          │
│  websocket.ts                                            │
│    ├── autoresearch:experiment -> addExperiment()         │
│    ├── autoresearch:status -> setStatus()                │
│    ├── autoresearch:trace-experiment -> addTraceExp()    │
│    └── autoresearch:trace-research-complete -> setTrace()│
└──────────────────────────┬──────────────────────────────┘
                           │  REST + WebSocket
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     BACKEND                              │
│                                                          │
│  autoresearch_routes.py (FastAPI router)                 │
│    ├── POST /api/autoresearch/start                      │
│    ├── POST /api/autoresearch/stop                       │
│    ├── POST /api/autoresearch/pause                      │
│    ├── POST /api/autoresearch/resume                     │
│    ├── GET  /api/autoresearch/status                     │
│    ├── POST /api/autoresearch/trace-research/start       │
│    ├── POST /api/autoresearch/trace-research/stop        │
│    ├── POST /api/autoresearch/trace-research/pause       │
│    ├── POST /api/autoresearch/trace-research/resume      │
│    └── GET  /api/autoresearch/trace-research/status      │
│                                                          │
│  autoresearch.py (AutoResearcher class)                  │
│    ├── SEARCH_SPACE definition                           │
│    ├── mutate_config() — evolutionary mutation           │
│    ├── run_experiment() — evaluate one config            │
│    ├── run_loop() — async main loop                      │
│    └── stop/pause/resume controls                        │
│                                                          │
│  trace_researcher.py (TraceResearcher class)             │
│    ├── DATA_SEARCH_SPACE definition                      │
│    ├── mutate_pipeline() — evolutionary mutation          │
│    ├── evaluate_pipeline() — evaluate one pipeline       │
│    ├── run_loop() — async main loop                      │
│    └── stop/pause/resume controls                        │
└─────────────────────────────────────────────────────────┘
```

### Concurrency Model

- The main search loop runs as an **asyncio background task** (`asyncio.create_task`)
- Each experiment's compute (the simulated sleep + loss calculation) runs in a **thread executor** (`loop.run_in_executor`) so it doesn't block the event loop
- Pause is implemented as a `while self._paused: await asyncio.sleep(0.5)` loop
- Stop sets `self._running = False`, checked at the top of each iteration
- The WebSocket **callback** is called after every experiment, broadcasting results to all connected clients in real time

### State Management

- Backend: `AutoResearcher` / `TraceResearcher` instances stored on `app.state` (FastAPI application state)
- Frontend: Zustand store with completely separate state for each mode (status, experiments, bestMetric, bestConfig for hyperparam; traceStatus, traceExperiments, traceBestMetric, traceBestPipeline for trace mining)
- The `activeMode` field controls which mode the UI displays, but both can technically run simultaneously since they use separate backend instances

### WebSocket Message Types

| Message Type | Direction | Payload |
|---|---|---|
| `autoresearch:experiment` | Server -> Client | experiment_id, total_experiments, config_snapshot, metric_value, best_metric, improved, duration_seconds |
| `autoresearch:complete` | Server -> Client | status, total_experiments, best_metric, error |
| `autoresearch:failed` | Server -> Client | status, total_experiments, best_metric, error |
| `autoresearch:trace-experiment` | Server -> Client | experiment_id, total_experiments, config_snapshot, examples_generated, unique_repos, avg_example_length, metric_value, best_metric, improved, duration_seconds |
| `autoresearch:trace-research-complete` | Server -> Client | status, total_experiments, best_metric, best_data_stats, error |
| `autoresearch:trace-research-failed` | Server -> Client | status, total_experiments, best_metric, error |

---

## Retired user-facing controls

### Configuration (Before Starting)

**Hyperparameter mode:**
- Checkbox grid to select which params to search (default: learning_rate, lora_r, lora_alpha)
- Max experiments (5-500, default 50)
- Train steps per experiment (10-5000, default 100)
- Mutation scale (0.05-1.0, default 0.2)
- Mutation rate slider (0.1-0.5, default 0.3)

**Trace mining mode:**
- Checkbox grid to select which pipeline params to search (default: min_quality_score, min_success_rate, silver_inclusion_ratio)
- Max experiments (5-200, default 30)
- Mutation scale (0.05-1.0, default 0.2)
- Mutation rate slider (0.1-0.5, default 0.3)

### During Execution

- **Progress bar** with current/total experiment count
- **Best metric card** (loss for hyperparam, quality score for trace mining)
- **Extra stats for trace mining:** examples generated, unique repos, avg example length
- **SVG chart:** Each experiment plotted as a dot (accent color if improved, gray if not), with a step-function line tracking the best metric over time
- **Experiment log** in a terminal-style scrollable container with auto-scroll, showing: experiment number, improved/not icon, metric value, abbreviated config diff, duration

### After Completion

- **Best configuration comparison table:** Shows each searched parameter with its starting value and best-found value, with changed rows highlighted
- **Copy button** to export the best config as JSON to clipboard
- **New Search button** to reset and try again

---

## Historical limitations and future work

### Current: Simulated
- Experiments don't run real training — they simulate realistic loss dynamics with configurable sleep times (2-4 seconds per experiment)
- The simulation is designed to reward configurations that match known best practices, so the search produces plausible results

### Future: Real Training
- Replace `run_experiment()` with actual short training runs using Unsloth on the GPU
- Use `dataset_subset_ratio` (default 10%) to train on a fraction of data for speed
- Replace `evaluate_pipeline()` with real trace processing + short training + validation loss measurement
- Enable the `train_minutes` budget to cap wall-clock time per experiment
- Multi-objective optimization (loss + inference speed + memory)

### Integration Points
- Best config from autoresearch can be copied and pasted into the Training Config panel
- Future: "Apply Best Config" button that pre-fills the training configuration
- Future: "Apply Best Pipeline" button that updates the data factory settings
