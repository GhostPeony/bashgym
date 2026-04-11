# BashGym Workstreams

Three parallel lanes of work to move bashgym from "pipeline runs end-to-end but is missing the last mile" to "trainable platform with auto-research + frontend in sync." Each section below is self-contained and designed to be **pasted into a fresh Claude Code window** as the initial prompt.

## How to use this document

1. Open a new Claude Code window in `/home/ponyo/bashgym`.
2. Pick one of the three workstreams below.
3. Copy the entire section (from `# Workstream N` through the next `---`) into the conversation.
4. Claude reads the briefing, knows what to do, and does not need to re-discover context.
5. When that workstream is done, it writes back to this file with a "COMPLETED" marker and any handoff notes for the other lanes.

---

## Shared context — READ FIRST in every workstream window

You are working on **bashgym**, an open-source cascade RL orchestration platform for training code/bash agents. It already has most of its pieces; the remaining work is wiring, hardening, and filling gaps.

**Repo**: `/home/ponyo/bashgym` on branch `feat/training-strategies-device-mgmt`.

**Platform state as of 2026-04-10** (verified by codebase inventory):

- **Five trainers in `bashgym/gym/trainer.py`**: `train_sft()` at line 445, `train_dpo()` at line 704, `train_grpo()` at line 2068, `train_distillation()` at line 1238, `train_rlvr()` at line 1444. All share the same subprocess+log-streaming+checkpoint plumbing.
- **Script generators**: `_generate_unsloth_sft_script()` at line 939, `_generate_unsloth_dpo_script()` at line 1612, `_generate_grpo_script()` at line 2373.
- **API**: `POST /api/training/start` (routes.py:881) accepts a `strategy` field. `POST /api/cascade/start` (cascade_routes.py:102) runs multi-stage cascades. WebSocket at `/ws` for log streaming.
- **Data Designer**: `bashgym/factory/data_designer.py` with `DataDesignerPipeline` class at line 107. Five registered pipelines in `designer_pipelines/`: `coding_agent_sft`, `coding_agent_dpo` (uses LLM judge for pair selection), `tool_use_sft`, `from_external`, `from_unstructured`.
- **DPO pair generator**: `pair_failures_for_dpo()` in `bashgym/factory/dpo_pairer.py:93` — builds preference pairs from gold + failed traces via embeddings.
- **Cascade scheduler**: `bashgym/gym/cascade_scheduler.py::CascadeScheduler` — sequential domain-by-domain stages with checkpoint chaining. Currently hardcoded to GRPO per stage (see Workstream 1).
- **Dataset validator**: `bashgym/datasets/validator.py` + `FormatContract` enum in `contracts.py` (SFT, DPO, GRPO, DISTILLATION).
- **Frontend**: Vite + React + Electron at `frontend/`. Cascade UI at `frontend/src/components/training/TrainingConfig.tsx:877`. API service at `frontend/src/services/api.ts`. WebSocket client for live training logs.

**Fixes already applied to the GRPO path on 2026-04-10** (these need to be understood to backport them to SFT/DPO):

- `_install_gc_compat()` — wraps `model.gradient_checkpointing_enable` to accept both positional and kwargs calling conventions, fixing the Unsloth/TRL contract mismatch. In `_generate_grpo_script()` at trainer.py ~line 2440.
- Persistent `training.log` written inside `_run_grpo_loop()` — `log_file = open(run.output_path / "training.log", "w", ...)` then tee stdout lines into it.
- `_parse_trl_stats()` — dict parser with `ast.literal_eval` primary + regex fallback, handles `nan`/`inf`, used to extract real per-step metrics from TRL's stringified dict output.
- `DegenerateRewardStop` TrainerCallback — in-script early-stop that fires when `frac_reward_zero_std > 0.95` for 3 consecutive logs after step 5, writes `EARLY_STOPPED` sentinel file, cleanly stops training via `control.should_training_stop = True`.
- `_reward_dataset_mismatch()` preflight in `cascade_scheduler.py` — validates reward ↔ dataset compatibility before any training starts, refuses the run with a specific error if the data has no field the reward function needs.
- Cascade `_run_stage` raises `RuntimeError` if `run.status == "failed"` so silent failures don't get marked "completed".
- `CascadeStartRequest` now accepts `repo_domains_enabled`, `repo_domains_dir`, `repo_domains_filter`, `grpo_reward_mode` overrides.

**Hardware**: NVIDIA DGX Spark (GB10, sm_121, CUDA 13.0), 121 GB shared memory, aarch64.

**Models**: Gemma 4 family (E4B, etc.). There's a previously-trained Unsloth checkpoint (check `~/.bashgym/models/`, `~/.bashgym/cascade/`, or `~/bashgym-training/` for the most recent) that should be used as the base for continued training instead of the raw `unsloth/gemma-4-E4B-it`.

**Datasets**: 3,143 gold traces at `~/.bashgym/gold_traces_local/` and 460 failed traces at `~/.bashgym/failed_traces_local/`. Both are Claude Code agent traces (not pure Python code — they are multi-turn bash/tool-use conversations).

**Known pitfall from today's debugging**: if you choose a reward function that cannot discriminate between model outputs on your data, GRPO runs will complete successfully but produce `frac_reward_zero_std ≈ 1.0` and learn nothing. The preflight check catches this before training starts. Don't bypass it.

---

# Workstream 1 — Training loop hardening + multi-strategy cascade

## Goal

Make the cascade scheduler dispatch to **any** training strategy per stage (SFT → DPO → GRPO), and ensure all three subprocess loops have the same observability/safety fixes that GRPO got today.

Current cascade hardcodes GRPO at `cascade_scheduler.py::_run_stage()`. SFT and DPO subprocess loops in `trainer.py` probably don't have the persistent log, metrics parser, or early-stop that GRPO has — this needs auditing.

## What success looks like

1. `CascadeStartRequest` accepts `stage_strategies: list[str]` (e.g. `["sft", "dpo", "grpo"]`) and the cascade runs those strategies in order.
2. Each stage's checkpoint becomes the next stage's `base_model`.
3. SFT and DPO have: persistent `training.log`, metrics parser for their respective stats dict formats, and the `DegenerateRewardStop` or equivalent (probably just "loss not decreasing" check for SFT).
4. Running `POST /api/cascade/start` with `stage_strategies: ["sft", "dpo"]` on the real gold + failed traces kicks off SFT → DPO end-to-end with real metrics visible and real checkpoints saved.

## Tasks in order

### 1. Audit the SFT subprocess loop

- Read `bashgym/gym/trainer.py::train_sft()` starting at line 445 and the subprocess loop it calls into (look for `_run_sft_loop` or inline Popen — the exact structure may differ from GRPO).
- Read `_generate_unsloth_sft_script()` at line 939 to understand what the generated SFT script prints and what config it uses.
- Check if the loop writes a persistent `training.log` file (GRPO does — see `_run_grpo_loop` pattern).
- Check if it parses per-step metrics (SFT stats dict keys are different from GRPO: `loss`, `learning_rate`, `epoch`, `grad_norm`, `num_tokens`, but no `reward` or `frac_reward_zero_std`).
- Check if it raises on non-zero exit code. Check if it has a silent-failure path like the one we fixed in the cascade today.

### 2. Audit the DPO subprocess loop

- Same exercise for `train_dpo()` at line 704 and `_generate_unsloth_dpo_script()` at line 1612.
- DPO stats dict keys include: `loss`, `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/chosen`, `logps/rejected`. These are the ones the metrics parser should extract.
- DPO has its own degenerate condition: if `rewards/accuracies` stays at exactly 0.5 for many steps, the model isn't learning preferences. Early-stop threshold for DPO: `accuracy < 0.52` sustained after step 10 (research-verified threshold before adding).

### 3. Backport GRPO fixes to SFT and DPO

- Extract the GRPO fixes into helper functions if they're duplicated. The dict parser `_parse_trl_stats()` is already general-purpose — move it to a module-level helper if it isn't already.
- Add persistent `training.log` to both SFT and DPO loops (identical to the GRPO implementation).
- Add a `_run_training_loop()` common helper if it simplifies things — otherwise just ensure each loop has the same shape: open log file → Popen → stream stdout → parse + write + callback → handle exit code → close log file in finally.

### 4. Make the cascade scheduler strategy-aware

- Add a `strategy: str = "grpo"` field to `CascadeStage` dataclass in `cascade_scheduler.py`.
- Add `stage_strategies: list[str]` to `CascadeConfig` and to `CascadeStartRequest` in `cascade_routes.py`.
- Modify `_run_stage()` to dispatch based on `stage.strategy`: call `trainer.train_sft()`, `trainer.train_dpo()`, or `trainer.train_grpo()` with the right arguments. Each trainer returns a `TrainingRun` object; the downstream logic (`run.metrics`, `run.status`, checkpoint detection) should work uniformly.
- Update the stage construction loop so `stage.strategy = config.stage_strategies[i]` if provided, else default to `"grpo"` (preserve current behavior).

### 5. Strategy-aware preflight

- Extend `_reward_dataset_mismatch()` in `cascade_scheduler.py` into a more general `_strategy_dataset_mismatch(strategy, reward_mode, example)`:
  - **SFT** requires `prompt` and `completion` (or `messages`) per example. If missing, fail preflight with a clear message.
  - **DPO** requires `chosen` and `rejected` per example. If they're missing, also check if there's a `dpo_pairer` path that can generate them from `gold` + `failed` traces, and fail with a hint pointing at `pair_failures_for_dpo()`.
  - **GRPO** keeps the existing `_reward_dataset_mismatch` logic.

### 6. Fix checkpoint chaining

- In `cascade_scheduler.py::run_cascade()` around line 494-504 where `stage.checkpoint_path` is set, the current logic looks at `stage.output_path / "final"` but that's wrong — the actual checkpoint lives at `{stage.output_path}/{run_id}/final` or `.../merged`. Fix it to walk the output directory and find the actual saved checkpoint, preferring `merged` (full model) over `final` (LoRA adapter only).
- Next stage's `stage.base_model` needs to be updated from the previous stage's checkpoint path. Check if this is already happening at line ~458 (`if prev_checkpoint: stage.base_model = prev_checkpoint`) and verify the types match (path string, not Path object).

### 7. Integration test

- Build a tiny test dataset with 10 examples (5 gold, 5 failed) — enough to run through both SFT and DPO stages quickly.
- Start the backend: `python run_backend.py`.
- `POST /api/cascade/start` with `stage_strategies: ["sft", "dpo"]`, `mode: "real"`, `max_steps: 20` for each stage.
- Verify: stage 1 produces a checkpoint, stage 2 loads it as `base_model`, stage 2 produces a final checkpoint. Both `training.log` files are written with real metrics. Cascade status transitions: idle → running → completed with no silent failures.

## Files you'll touch

- `bashgym/gym/trainer.py` (several edit locations around SFT line 445, DPO line 704, generators at 939 and 1612)
- `bashgym/gym/cascade_scheduler.py` (`_run_stage`, `run_cascade`, `CascadeStage`, `CascadeConfig`, `_reward_dataset_mismatch`)
- `bashgym/api/cascade_routes.py` (`CascadeStartRequest` schema + wiring)

## Don't do these

- Don't introduce new abstractions (e.g. a "UniversalTrainer" class) — just make the three existing trainers callable from the scheduler with uniform return types.
- Don't break the existing GRPO-only cascade path. Default `stage_strategies` to `["grpo"]` when not provided.
- Don't rewrite `_generate_unsloth_sft_script()` or `_generate_unsloth_dpo_script()` unless they're actively broken. If they were working before today, leave them alone.
- Don't run full-scale training as a final test — 20 steps each is enough to verify the plumbing works.

## Completion handoff

When done, append a section at the bottom of this file:

```markdown
## Workstream 1: COMPLETED <date>

- Commits: <list>
- Files touched: <list>
- How to run an SFT→DPO cascade: <one-line curl command>
- Known limitations: <anything>
- Handoff to Workstream 3 (frontend): the new `stage_strategies` field needs to be exposed in TrainingConfig.tsx
```

---

# Workstream 2 — HuggingFace auto-research agent

## Goal

Build an automated agent that scans HuggingFace Hub for datasets relevant to training code/bash agents, scores them, and produces a ranked report with download commands we can feed into bashgym's `DataDesigner.from_dataset()` flow.

The agent should run as a **one-shot CLI script** first (simplest), then optionally as a scheduled cron job that refreshes the report weekly. No plumbing into the cascade scheduler yet — just a research tool that the user can invoke and review the output of.

## Why this matters

Your 3,143 gold traces are thin for a multi-stage cascade. The Data Designer (`bashgym/factory/data_designer.py::DataDesignerPipeline.from_dataset()`) can pull any HuggingFace dataset and convert it into bashgym training format. Having a curated, scored list of candidate datasets means you can augment your real data with the best available public data without spending hours manually searching HF Hub.

## What success looks like

1. A script at `bashgym/research/hf_dataset_scanner.py` that runs `python bashgym/research/hf_dataset_scanner.py` and produces a ranked report at `~/.bashgym/research/hf_datasets_report.md`.
2. The report contains, for each candidate dataset:
   - HF repo ID (e.g. `SWE-Gym/SWE-Gym`)
   - Size (row count, disk size)
   - License (Apache/MIT/etc. — filter out non-commercial)
   - Last updated date
   - Schema (what columns it has)
   - Compatibility score with bashgym formats (SFT, DPO, GRPO) — does it have `prompt`+`completion`? `chosen`+`rejected`? `test_cases`?
   - Relevance score 0-10 based on match criteria
   - A one-line **download command** using `DataDesigner.from_dataset()` that would import it into bashgym
3. The top 20 candidates printed to stdout with `pandas.to_markdown()` formatting.
4. A small JSON cache at `~/.bashgym/research/hf_datasets_cache.json` so repeat runs don't re-query the Hub for datasets that haven't changed.

## Search criteria (what "relevant" means)

The agent should score datasets on:

| Dimension | Weight | Signal |
|---|---|---|
| **Task match** | 30% | Tagged with `code`, `code-generation`, `agentic`, `tool-use`, `bash`, `shell`, `swe`, `software-engineering`, or description matches |
| **License** | 15% | Apache-2.0, MIT, BSD pass. CC-BY-SA pass with warning. CC-NC-* fail. Unknown fail. |
| **Size** | 15% | 100-50,000 examples ideal for our compute budget. < 10 fail (too small). > 500k downweight (takes too long to process). |
| **Freshness** | 10% | Updated in the last 12 months is full marks. 2+ years old downweight. |
| **Schema compatibility** | 20% | Has columns that map cleanly to bashgym's `SFT`/`DPO`/`GRPO` contracts (see `bashgym/datasets/contracts.py`) |
| **Popularity** | 10% | Download count on HF Hub as a weak signal — popular datasets are usually higher quality |

Hard filters (exclude entirely):
- Non-commercial licenses
- Datasets gated behind a form/approval
- Datasets flagged as `pii` without masking
- Datasets < 10 examples or > 1M examples

## Tasks in order

### 1. Explore the HuggingFace Hub API

- Use the `huggingface_hub` Python package (already installed — verify with `pip show huggingface_hub`).
- Read the docs for `HfApi.list_datasets()` — it accepts `filter`, `sort`, `direction`, `limit`, `full` parameters.
- The `filter` argument can take tags like `code-generation`, `task_categories:text-generation`, etc. Experiment with a few queries interactively to see what comes back.
- The `full=True` flag returns richer metadata including card data (description, license, tags).

### 2. Build the search query list

Start with these query combinations (test each one returns 10-50 results):

- `filter="code"` + `sort="downloads"` + `limit=200`
- `search="bash agent"` + `limit=50`
- `search="swe-bench"` + `limit=50`
- `search="tool use"` + `filter="code"`
- `filter=["code-generation", "text-generation"]`
- `search="coding agent"` + `limit=100`
- `filter="task_categories:text-generation"` + `search="python code"`

Dedupe the results by `dataset.id`.

### 3. Enrich each candidate

For each candidate, fetch the dataset's card data (`HfApi.dataset_info(repo_id)`) and extract:

- `cardData.license` (often missing → flag as `unknown`)
- `cardData.dataset_info.splits` (row counts per split)
- `cardData.dataset_info.download_size` (bytes)
- `cardData.dataset_info.features` (column schema)
- `lastModified`
- `downloads` (all-time download count)
- `tags`

### 4. Score and rank

Implement a `score_dataset(info) -> dict` function that returns:

```python
{
    "repo_id": "...",
    "score": 8.3,         # 0-10 weighted sum
    "reasons": [          # why this score
        "+2.5 task match: tagged 'code-generation'",
        "+1.5 license: apache-2.0",
        "+0.8 size: 4,200 examples (ideal)",
        ...
    ],
    "warnings": ["license unknown for one sub-config"],
    "bashgym_format": "sft",  # inferred best fit
    "download_command": "DataDesigner.from_dataset('...', ...)",
}
```

### 5. Write the report

Generate `~/.bashgym/research/hf_datasets_report.md` with:

- Top 20 by score, as a markdown table
- For each, a section with the `reasons` list, warnings, and the download command
- A "rejected" section summarizing why common candidates failed (license, size, etc.) so the user can verify the filters are sane

### 6. (Optional) Schedule it

Use the `schedule` skill (already installed) or a simple systemd timer to re-run weekly and diff against the previous report — only notify on new high-scoring entries.

## Files to create

- `bashgym/research/__init__.py`
- `bashgym/research/hf_dataset_scanner.py`
- `bashgym/research/scoring.py` (the `score_dataset()` function, unit-testable)
- `bashgym/research/contracts.py` (the criteria thresholds as constants for easy tuning)
- `tests/research/test_scoring.py` (pure unit tests, no network calls, ~10 test cases with mocked DatasetInfo objects)

## Don't do these

- Don't download the datasets themselves — just their metadata. Actual download is the user's decision after reading the report.
- Don't invoke the `DataDesigner.from_dataset()` import in this script — it's a heavy import chain. Keep the research module standalone and just emit the Python *expression* as a string.
- Don't use an LLM to score the datasets — use deterministic rule-based scoring. Faster, repeatable, and auditable.
- Don't hardcode the HF API token. Use `os.environ.get("HF_TOKEN")` and fall back to anonymous (HF public endpoints work without auth).
- Don't overfit to "bash agent" specifically. Code/tool-use datasets are also valuable for bashgym's multi-strategy cascade.

## Completion handoff

When done, append:

```markdown
## Workstream 2: COMPLETED <date>

- Script location: `bashgym/research/hf_dataset_scanner.py`
- Example report: `~/.bashgym/research/hf_datasets_report.md`
- Top 3 recommended datasets from first run: <list>
- Handoff to Workstream 1: consider importing <top dataset> via DataDesigner for the next SFT run
- Handoff to Workstream 3: report location should be linked from the frontend "Data Sources" panel
```

---

# Workstream 3 — Frontend refinement & backend wiring

## Goal

Surface everything the backend can now do in the frontend so the user can drive bashgym entirely through the UI — no curl commands, no tailing backend.log. This includes the new multi-strategy cascade from Workstream 1, the Data Designer pipelines (already built in the backend but barely exposed), and live training metrics from the fixes applied today.

## Current frontend state

- **Location**: `frontend/` (Vite + React + Electron, TypeScript).
- **Cascade UI exists** at `frontend/src/components/training/TrainingConfig.tsx:877` — has domain checkboxes, steps per stage, simulate/real toggle.
- **Training Dashboard** at `frontend/src/components/training/TrainingDashboard.tsx`.
- **API service**: `frontend/src/services/api.ts` — has `cascadeApi.start(config)` calling `POST /api/cascade/start`.
- **WebSocket**: `/ws` endpoint connects from the frontend via a hook somewhere in `frontend/src/services/` or `frontend/src/hooks/` — needs investigation.

## What's missing or broken in the frontend

1. **No multi-strategy cascade config UI**. You can pick domains and GRPO knobs, but not "run stage 1 as SFT, stage 2 as DPO, stage 3 as GRPO." Requires Workstream 1 to ship first (or stub the field with a default of `["grpo"]` and update once W1 lands).
2. **Preflight error display**. The cascade API now returns specific preflight errors (reward ↔ dataset mismatch). The frontend currently just shows a generic "started" → "failed" status. It should surface the actual error message inline so the user sees "your verification reward doesn't match your dataset" instead of silent failure.
3. **Real per-step metrics**. The training log stream now emits `reward_std`, `frac_reward_zero_std`, `grad_norm`, `kl` per step. The frontend probably doesn't chart these — it should, because that's how the user can tell a run is healthy mid-training.
4. **`EARLY_STOPPED` banner**. When a run is killed by `DegenerateRewardStop`, the frontend should show a prominent red banner with the reason, not just "failed".
5. **Data Designer pipelines are invisible**. The backend has 5 registered pipelines and a `POST /api/factory/synthetic/generate` endpoint, but there's no frontend page to pick a pipeline, configure it, kick it off, watch it run, or review the generated examples.
6. **Training log viewer** — the generated `training.log` files are on disk at `~/.bashgym/cascade/.../training.log`. There should be a way to view them from the frontend (tail the last N lines, search, download).
7. **Checkpoint browser** — after a run finishes, the frontend should let the user see the saved checkpoint directory, its size, and a "set as base model for next run" button that pre-fills the next cascade's `base_model` field.

## Tasks in order

### 1. Map the existing frontend → backend wires

Start by reading these files (in order) to build a mental model:

- `frontend/src/services/api.ts` — what API endpoints are already wrapped
- `frontend/src/services/` (any WebSocket client)
- `frontend/src/hooks/` (look for `useTraining`, `useCascade`, `useWebSocket` hooks)
- `frontend/src/components/training/TrainingConfig.tsx` — existing cascade config
- `frontend/src/components/training/TrainingDashboard.tsx` — existing dashboard
- `frontend/src/components/training/TrainingLogs.tsx` (if it exists)

Produce a written mental model: "the cascade config UI writes to `cascadeApi.start()`, which POSTs to `/api/cascade/start`, the scheduler broadcasts events over `/ws` with message types `cascade:stage-started`, `cascade:stage-log`, `cascade:stage-completed`, `cascade:completed`, `cascade:failed`, which are consumed by `useCascadeStream` in `hooks/useCascadeStream.ts` and rendered in `TrainingDashboard`."

### 2. Surface preflight errors inline

When the cascade API call returns `{ status: "started" }` but `/api/cascade/status` later returns `{ status: "failed", error: "Cascade preflight: reward_mode='verification' cannot work..." }`, the frontend should:

- Poll `/api/cascade/status` every 1s while the cascade is running OR
- Listen for a `cascade:failed` WebSocket event with the error payload
- Render the error in a red inline banner on the `TrainingConfig` page with a "Fix it" button that pre-fills suggestions (e.g. "switch reward mode to execution" or "enable DPO pair generation from failed traces")

### 3. Wire per-step metrics into a live chart

The cascade's `stage-log` WebSocket events now carry lines from `training.log`. Parse them client-side (or have the backend pre-parse and emit a `stage-metric` event with `{step, loss, reward, reward_std, frac_reward_zero_std, grad_norm, kl}`). Render a live-updating chart using Recharts or similar (check `package.json` for existing charting libs).

Four small charts on one panel:
1. Loss (descending line)
2. Reward + reward_std band (line with shaded variance)
3. `frac_reward_zero_std` (line with red threshold at 0.95)
4. `kl` divergence (line)

The user should be able to tell at a glance whether the run is learning.

### 4. Add multi-strategy cascade UI (after W1 lands)

Once Workstream 1 exposes `stage_strategies` in `CascadeStartRequest`:

- Replace the single "GRPO" assumption in `TrainingConfig.tsx` with a list of stage slots. Each slot is a dropdown: `[SFT | DPO | GRPO]`.
- Per-stage hyperparam overrides (max_steps, learning_rate) below each slot.
- Display the stage chain visually: `SFT → DPO → GRPO → Final` with arrows.

### 5. Build a Data Designer page

New route/component at `frontend/src/pages/DataDesigner.tsx`:

- List the 5 registered pipelines from `GET /api/factory/synthetic/presets`
- For each, show a card with: name, description, required inputs
- Form to pick one and configure it (`num_records`, source dataset path or trace dir)
- "Generate" button calls `POST /api/factory/synthetic/generate`
- Job list view showing running and completed jobs with job IDs, polling `GET /api/factory/synthetic/jobs/{job_id}`
- On completion, a preview of the first 5 generated examples (JSON tree viewer)
- "Use this dataset" button that pre-fills a new cascade run with the generated data path

### 6. Training log viewer

New component at `frontend/src/components/training/TrainingLogViewer.tsx`:

- Input: run_id
- Calls a new backend endpoint `GET /api/training/{run_id}/log?tail=500` that returns the last N lines of `training.log` from disk (add this to `bashgym/api/routes.py`)
- Monospace code view with search (regex), severity highlighting, and a "download full log" button

### 7. Checkpoint browser

New component at `frontend/src/components/training/CheckpointBrowser.tsx`:

- Calls `GET /api/training/checkpoints` (add this endpoint — scans `~/.bashgym/cascade/` and `~/.bashgym/models/` for `final/`, `merged/`, and `checkpoint-*/` directories)
- Shows each checkpoint with: run_id, date, size, base model it was trained from
- "Use as base model" button that navigates to `TrainingConfig` with the path pre-filled
- "Delete" button with confirmation (calls `DELETE /api/training/checkpoints/{id}`)

## Files you'll touch

- `frontend/src/services/api.ts` (add new API wrappers)
- `frontend/src/hooks/useCascadeStream.ts` (or create if missing)
- `frontend/src/components/training/TrainingConfig.tsx` (multi-strategy UI)
- `frontend/src/components/training/TrainingDashboard.tsx` (live charts)
- `frontend/src/pages/DataDesigner.tsx` (new)
- `frontend/src/components/training/TrainingLogViewer.tsx` (new)
- `frontend/src/components/training/CheckpointBrowser.tsx` (new)
- `bashgym/api/routes.py` (new endpoints: `/api/training/{run_id}/log`, `/api/training/checkpoints`, `/api/training/checkpoints/{id}` DELETE)

## Don't do these

- Don't redesign existing working components — only modify them to surface new data.
- Don't add dependencies the frontend doesn't already have. Use existing charting, form, and state libraries.
- Don't bypass the API — every frontend feature should correspond to a documented backend endpoint. If you need something the backend doesn't expose, add the endpoint first.
- Don't hardcode URLs. Use `VITE_API_URL` / `VITE_WS_URL` env vars (already set up in `frontend/.env.local`).
- Don't ship breaking changes to the existing cascade UI before W1 is ready. Gate the multi-strategy UI behind a feature flag or a separate tab.

## Completion handoff

When done, append:

```markdown
## Workstream 3: COMPLETED <date>

- New pages: DataDesigner, CheckpointBrowser, TrainingLogViewer
- New API endpoints added: <list>
- How to access the new features: <URLs>
- Known limitations: <anything>
- Screenshots: <paths or URLs>
```

---

# Coordination rules

- **Workstream 1 blocks Workstream 3's multi-strategy UI section** (task 4). The rest of W3 can proceed in parallel.
- **Workstream 2 is fully independent.** It can run in parallel with either of the others.
- **If any workstream discovers the briefing is wrong** (a file path doesn't exist, a function signature changed, an assumption is false), it should **stop, update this file with a correction**, and **ask the user before continuing**. Don't rediscover silently.
- **All three must respect the existing GRPO fix in the generator** (`_install_gc_compat()` in the generated script). Do not remove or modify the gradient_checkpointing patch — it's load-bearing on GB10/sm_121.
- **All three should leave the existing `feat/training-strategies-device-mgmt` branch clean.** Create feature branches from it: `ws1-cascade-multi-strategy`, `ws2-hf-research`, `ws3-frontend-refinement`.

## Current date and branch

- Date: 2026-04-10
- Repo: `/home/ponyo/bashgym`
- Branch: `feat/training-strategies-device-mgmt` at `d4b8442`
- Recent uncommitted work (from today): `bashgym/gym/trainer.py`, `bashgym/gym/cascade_scheduler.py`, `bashgym/api/cascade_routes.py`. Commit these to `feat/training-strategies-device-mgmt` before branching so all three workstreams start from the same base.

---

## Workstream 1: COMPLETED 2026-04-10

- **Commits**:
  - `ee1c1dc` (base) — GRPO hardening + cascade preflight + datasets package (pre-existing uncommitted work committed per briefing)
  - `d842a04` — multi-strategy dispatch + SFT/DPO loop hardening
- **Branch**: `ws1-cascade-multi-strategy` (off `feat/training-strategies-device-mgmt`)
- **Files touched**:
  - `bashgym/gym/trainer.py` — `_parse_trl_stats` module-level, SFT/DPO loops rewritten to mirror GRPO (persistent `training.log`, TRL stats, EARLY_STOPPED sentinel), `train_dpo` signature gains `log_callback`/`pid_callback`, generated SFT/DPO scripts gain `LossPlateauStop`/`DegenerateAccuracyStop` TrainerCallbacks
  - `bashgym/gym/cascade_scheduler.py` — `CascadeStage.strategy`, `CascadeConfig.stage_strategies`, strategy-aware `_run_stage`/`_filter_dataset`, `_strategy_dataset_mismatch`, `_find_stage_checkpoint` prefers `merged/` over `final/`
  - `bashgym/api/cascade_routes.py` — `CascadeStartRequest.stage_strategies`, start response echoes resolved strategies
  - `tests/gym/test_cascade_scheduler.py` — 17 new tests (strategy assignment, preflight dispatch, checkpoint chaining)

- **How to run an SFT→DPO cascade**:
  ```bash
  curl -X POST http://localhost:8003/api/cascade/start \
    -H 'Content-Type: application/json' \
    -d '{
      "domains": ["file_operations", "bash_commands"],
      "stage_strategies": ["sft", "dpo"],
      "base_model": "unsloth/gemma-4-E4B-it",
      "dataset_path": "/home/ponyo/.bashgym/gold_traces_local",
      "train_steps_per_stage": 20,
      "mode": "real"
    }'
  ```
  DPO stage requires pre-paired `chosen`/`rejected` examples — generate via `bashgym.factory.dpo_pairer.pair_failures_for_dpo()` first. The strategy-aware preflight will refuse the run with a clear error message if the dataset can't support the stage.

- **Known limitations**:
  - DPO preflight now requires chosen/rejected per-example but does **not** auto-pair from gold+failed — user must run the pairer upstream. Adding `pair_failures_for_dpo()` integration to `_filter_dataset` was out of scope for this workstream (touches the factory layer).
  - `LossPlateauStop` for SFT uses fixed thresholds (`min_step=10, patience=5, min_delta=1e-4`). Not config-driven yet.
  - `DegenerateAccuracyStop` threshold 0.52 is a heuristic informed by [TRL issue #2194](https://github.com/huggingface/trl/issues/2194) — may need tuning based on real DPO run telemetry.
  - The 3 pre-existing `TestCascadeRun` / `TestFilterDataset::test_filters_by_domain` failures in `tests/gym/test_cascade_scheduler.py` are **not** caused by this workstream (present in `ee1c1dc`, the base commit). Test data needs to be updated to match the new trace-conversion flow — out of scope.

- **Handoff to Workstream 3 (frontend)**:
  - The new `stage_strategies: list[str]` field on `CascadeStartRequest` needs a UI control in `TrainingConfig.tsx` — a list of stage slots, each with a dropdown `[SFT | DPO | GRPO]`.
  - `scheduler.get_status()` now includes `stage.strategy` in each stage dict — surface the badge in the stage list.
  - The `cascade:stage-log` WebSocket payload now carries a `strategy` field alongside `stage`, `domain`, and `line` — useful for per-strategy metric chart selection.
  - Error messages from `_strategy_dataset_mismatch` are intentionally verbose and actionable — render them verbatim in the preflight error banner (don't truncate).

---

## Workstream 2: COMPLETED 2026-04-10

- **Branch**: `ws2-hf-research` (off `ws1-cascade-multi-strategy` at commit `e97d0de` — W1's completion marker. W1 was already merged-in at that point, so W2 stacks on W1.)
- **Script location**: `bashgym/research/hf_dataset_scanner.py`
- **CLI usage**:
  ```bash
  python -m bashgym.research.hf_dataset_scanner                   # full scan, uses cache
  python -m bashgym.research.hf_dataset_scanner --limit 50        # cap results per query
  python -m bashgym.research.hf_dataset_scanner --no-cache        # force re-enrichment
  python -m bashgym.research.hf_dataset_scanner --max-candidates 100
  ```
- **Output paths**:
  - Report: `~/.bashgym/research/hf_datasets_report.md`
  - Cache: `~/.bashgym/research/hf_datasets_cache.json` (keyed by repo_id; `--no-cache` to bypass)

- **Module layout**:
  - `bashgym/research/contracts.py` — scoring weights, thresholds, license policy, schema patterns, column-mapping hints. Tuning constants live here, not in scoring.py.
  - `bashgym/research/scoring.py` — pure `score_dataset(DatasetMetadata) -> ScoredDataset`. Hard filters + 6 weighted dimensions (task match, license, size, schema, freshness, popularity) + format inference + download-command string builder. Fully unit-tested with mocked inputs, no network calls.
  - `bashgym/research/hf_client.py` — thin `HFResearchClient` wrapper over `huggingface_hub.HfApi`. Verified against `huggingface_hub` 1.8.0 on 2026-04-10 — see the module docstring for the attribute-access quirks (`card_data.dataset_info` is a dict, `DatasetCardData` is a dataclass not a bare dict, license sometimes lives in `tags` as `license:xxx`).
  - `bashgym/research/report.py` — pure markdown renderer (header + top-20 summary table + per-dataset details + grouped rejected section).
  - `bashgym/research/hf_dataset_scanner.py` — argparse CLI that orchestrates discover → cache-check → enrich → score → render → write.

- **Tests**: `tests/research/test_scoring.py` (22 tests), `tests/research/test_report.py` (6 tests). 28/28 passing. No network. Run with `venv/bin/python -m pytest tests/research/ -v`.

- **Top 3 recommended datasets from first smoke run** (2026-04-10, `--limit 15 --max-candidates 60`):
  1. `SWE-bench/SWE-smith-trajectories` — **score 9.24**, SFT, 76k rows, MIT, updated 2025-07-19. Trajectories from the SWE-smith repo fixing dataset — directly relevant for bash/code-agent training.
  2. `JackYoung27/humaneval-s0-train` — score 8.14, SFT, 48 rows, MIT, updated 2026-04-08. Tiny but very fresh; useful as an eval set or a bootstrap seed, not a main training corpus.
  3. `SWE-bench/SWE-smith` — score 7.99, 59k rows, MIT, updated 2025-12-14. Raw SWE-smith issues (companion to the trajectories above). Format not auto-detected — would need a column mapping pass before ingestion.

- **Known limitations**:
  - **Format detection** has two fallback layers: (1) `card_data.dataset_info` YAML (fast, local), (2) HF `datasets-server.huggingface.co/info` API (works for any auto-converted parquet dataset). After both, the scorer runs a heuristic pass that matches prompt-like columns (`text`, `problem_statement`, `raw_problem`, `instruction`, `query`, …) against response-like columns (`code`, `solution`, `patch`, `completion`, `answer`, …) to catch idiomatic layouts like mbpp's `text`+`code`+`test_list` → GRPO or SWE-bench's `problem_statement`+`patch`+`FAIL_TO_PASS` → GRPO. Datasets that expose only a single raw `content` column (e.g. `GunA-SD/bash_code`, `1stvamp/py_ast`) still cannot be auto-formatted — they need a user-written column mapping.
  - **The discovery query list is tuned for code/tool-use keywords only** (`bashgym/research/hf_client.py::SEARCH_QUERIES`). Change those constants if you want to probe other domains (e.g. DevOps, sysadmin, ML training logs).
  - **Scheduled re-runs not implemented.** The WORKSTREAMS task 6 "optional" step to diff against a previous report and notify on new high-scoring entries is deferred — use the `schedule` skill or a systemd timer separately when needed.
  - **No LLM-based scoring** — per the plan, scoring is deterministic rule-based for repeatability and auditability. Reviewing the "rejected" section manually is still recommended after the first run to sanity-check the filters.
  - **`description` field is truncated to 500 chars** on enrichment to keep the cache small.
  - **License fallback uses `tags` only when `card.license` is missing.** If both are missing the dataset is hard-rejected as "unknown license" per the WORKSTREAMS briefing — this is deliberate.

- **Handoff to Workstream 1**: Consider ingesting **`SWE-bench/SWE-smith-trajectories`** via `DataDesignerPipeline.from_dataset(source='SWE-bench/SWE-smith-trajectories', split='train')` as augmentation for the next SFT stage — the format already maps to bashgym SFT (has `messages` column).

- **Handoff to Workstream 3**: The report path `~/.bashgym/research/hf_datasets_report.md` should be linked from a frontend "Data Sources" panel. Two natural UIs:
  - A button next to the cascade config that opens the report in a modal.
  - A background endpoint `GET /api/research/latest-report` that reads and returns the file (no need to re-run the scanner from the frontend — it's a CLI tool, running it on demand would block the request for 30-120 seconds).

- **Commits on this branch** (all prefixed `feat(research):`, `test(research):`, `chore(research):`, or `docs(research):`):
  - `feat(research): add bashgym.research package skeleton`
  - `feat(research): add scoring contracts — weights, license policy, size windows, schema patterns`
  - `test(research): add hard-filter test cases for score_dataset`
  - `feat(research): implement hard filters in score_dataset`
  - `test(research): add weighted dimension, schema inference, and download command tests`
  - `feat(research): implement weighted scoring, schema inference, download command builder`
  - `feat(research): add HFResearchClient wrapper for discovery + enrichment`
  - `test(research): add render_report unit tests`
  - `feat(research): add markdown report renderer`
  - `feat(research): add hf_dataset_scanner CLI entrypoint`
  - `chore(research): pin huggingface_hub>=0.19.0 for dataset scanner`
  - `docs(research): mark Workstream 2 complete`
