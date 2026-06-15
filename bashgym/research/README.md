# bashgym.research — HuggingFace dataset scanner

A standalone, read-only research tool that scans HuggingFace Hub for code/tool-use training datasets, scores them with deterministic rules, and emits a ranked markdown report with ready-to-paste `DataDesignerPipeline.from_dataset()` download commands. No LLMs, no dataset downloads — metadata only. Review the report, pick what you want, then run the download commands manually.

## Run it

```bash
# Full scan (uses cache, ~30-120 seconds on first run)
venv/bin/python -m bashgym.research.hf_dataset_scanner

# Cap results per query (faster smoke test)
venv/bin/python -m bashgym.research.hf_dataset_scanner --limit 20 --max-candidates 60

# Force re-enrichment (ignore cache)
venv/bin/python -m bashgym.research.hf_dataset_scanner --no-cache
```

**Outputs** (written to `~/.bashgym/research/`):

- `hf_datasets_report.md` — ranked markdown report, top 20 with download commands
- `hf_datasets_cache.json` — raw metadata keyed by `repo_id`, reused on subsequent runs

**Auth:** Uses `HF_TOKEN` env var if set, otherwise anonymous (public datasets work fine without auth).

## Pipeline

```
┌─────────────┐   ┌──────────────┐   ┌───────────┐   ┌────────────┐
│  Discover   │──▶│   Enrich     │──▶│   Score   │──▶│   Report   │
│             │   │              │   │           │   │            │
│ list_datasets│   │ dataset_info │   │weighted 6 │   │ markdown + │
│ 10 queries  │   │ + fallback   │   │dimensions │   │ top-20 + cli│
└─────────────┘   └──────────────┘   └───────────┘   └────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  JSON cache     │
                 │ ~/.bashgym/     │
                 │   research/     │
                 └─────────────────┘
```

### 1. Discover — `hf_client.py::HFResearchClient.discover_candidates`

Runs ~10 curated search queries against `HfApi.list_datasets()`, dedupes by `repo_id`. Queries are tuned in `SEARCH_QUERIES` at the top of `hf_client.py`:

```python
SEARCH_QUERIES = [
    {"filter": "code-generation", "limit": 200},
    {"search": "code generation", "limit": 100},
    {"search": "swe-bench", "limit": 50},
    {"search": "humaneval", "limit": 50},
    {"search": "mbpp", "limit": 50},
    ...
]
```

Edit that list to probe different domains (DevOps, sysadmin, SQL, etc.).

### 2. Enrich — `hf_client.py::HFResearchClient.enrich`

For each candidate, builds a `DatasetMetadata` dataclass. Has **three-layer metadata extraction** because HF dataset cards are inconsistent:

1. **Primary:** `HfApi.dataset_info(repo_id).card_data.dataset_info` — parses the YAML block at the top of the dataset's README. Fast, no extra network call. Often empty on community uploads.
2. **Fallback 1:** `datasets-server.huggingface.co/info?dataset={repo_id}` — the same API that powers HF's dataset viewer. Populated for any dataset that's been auto-converted to parquet (most public ones). Fills in `num_rows` and `features` when the card is thin.
3. **License fallback:** if `card.license` is None, scan `info.tags` for a `license:xxx` entry.

Result: a `DatasetMetadata` with `repo_id`, `tags`, `license`, `num_rows`, `features` (dict of column→dtype), `last_modified`, `downloads`, `gated`, `description`.

### 3. Score — `scoring.py::score_dataset`

Two phases: hard filters (pass/fail) then weighted scoring.

**Hard filters** — immediate rejection if any trip:

| Filter | Threshold | Source |
|---|---|---|
| `gated=True` | any | `info.gated` |
| Non-commercial license | prefix in `BLOCKED_LICENSE_PREFIXES` | `contracts.py` |
| Unknown license | `license is None` | (both card_data and tags failed) |
| Too small | `num_rows < SIZE_MIN_HARD` (10) | `contracts.py` |
| Too large | `num_rows > SIZE_MAX_HARD` (1,000,000) | `contracts.py` |

**Weighted dimensions** — each returns a 0-1 subscore; the final score is `sum(subscore[i] * WEIGHTS[i]) * 10`:

| Dimension | Weight | What it rewards |
|---|---|---|
| `task_match` | 30% | `code-generation`, `agentic`, `tool-use`, `bash`, `swe`, `software-engineering` tags (direct) or description (half-credit). See `TASK_TAGS` in `contracts.py`. |
| `schema` | 20% | Features map to a bashgym format (`SFT`/`DPO`/`GRPO`). Exact matches (`SCHEMA_PATTERNS`) score 1.0, heuristic matches score 0.6. |
| `license` | 15% | `PERMISSIVE_LICENSES` (apache-2.0, mit, bsd, cc-by-4.0, cc0-1.0) score 1.0. `WARN_LICENSES` (share-alike) score 0.6 and emit a warning. |
| `size` | 15% | `SIZE_IDEAL_MIN`-`SIZE_IDEAL_MAX` (100-50k rows) scores 1.0. Linear decay toward the hard cap. |
| `freshness` | 10% | Updated within `FRESHNESS_FULL_DAYS` (365d) scores 1.0. Linear decay to `FRESHNESS_STALE_DAYS` (730d). |
| `popularity` | 10% | `downloads / POPULARITY_SATURATION_DOWNLOADS` (10k), clamped to 1.0. Weak signal. |

**Schema detection** — two layers, in order:

1. **Exact `SCHEMA_PATTERNS`** (`contracts.py`) — checks if the column set is a superset of a known bashgym contract:
   - `{messages}` → SFT
   - `{prompt, chosen, rejected}` → DPO
   - `{prompt, tests}` or `{prompt, test_cases}` → GRPO
   - `{prompt, completion}` / `{instruction, output}` / `{question, answer}` → SFT

2. **Heuristic fallback** — scans column names against semantic sets in `contracts.py`:
   - `PROMPT_LIKE_COLS`: `prompt`, `instruction`, `text`, `problem`, `problem_statement`, `raw_problem`, `query`, `task`, …
   - `RESPONSE_LIKE_COLS`: `completion`, `response`, `output`, `code`, `solution`, `patch`, `raw_solution`, `answer`, …
   - `TEST_LIKE_COLS`: `tests`, `test_list`, `test_code`, `FAIL_TO_PASS`, `PASS_TO_PASS`, …
   - `PREFERENCE_COLS`: `chosen`, `rejected`

   A prompt-like + preference → DPO. A prompt-like + test-like → GRPO. A prompt-like + response-like → SFT. This is how `mbpp` (`text + code + test_list`) gets detected as GRPO and `SWE-bench/SWE-smith` (`problem_statement + patch + FAIL_TO_PASS`) gets detected as GRPO.

Heuristic matches are labeled `(heuristic)` in the reasons list and score 0.6 schema credit vs 1.0 for exact matches — eyeball the report to see which detections to trust.

### 4. Report — `report.py::render_report`

Pure function: takes `(accepted, rejected)` lists and returns a markdown string. Writes to `~/.bashgym/research/hf_datasets_report.md`. Layout:

- Header with generation timestamp, accepted/rejected counts
- Top 20 summary table (rank, repo, score, format, rows, license, updated)
- Details section per top-20 entry with reason breakdown and a ready-to-paste `DataDesignerPipeline.from_dataset(...)` download command
- Rejected section grouped by rejection reason

## Module layout

| File | Role |
|---|---|
| `contracts.py` | Scoring weights, thresholds, license policy, schema patterns, column-mapping hints. **All tuning knobs live here** — no logic. |
| `scoring.py` | Pure `score_dataset(DatasetMetadata) -> ScoredDataset`. Hard filters + weighted dimensions + format inference + download-command builder. Fully unit-tested with no network calls. |
| `hf_client.py` | `HFResearchClient` wraps `huggingface_hub.HfApi` for discovery and enrichment. Includes the datasets-server fallback. **The only module that makes network calls.** |
| `report.py` | Pure markdown renderer. Input: `list[ScoredDataset]`. Output: `str`. |
| `hf_dataset_scanner.py` | CLI entrypoint. Ties discover → cache-check → enrich → score → render → write. Handles argparse and JSON cache I/O. |

## Tuning

**Change scoring weights** — edit `contracts.py::WEIGHTS`. Must sum to 1.0. E.g. to prioritize freshness over popularity:

```python
WEIGHTS = {
    "task_match": 0.30,
    "license":    0.15,
    "size":       0.15,
    "schema":     0.20,
    "freshness":  0.15,   # +5
    "popularity": 0.05,   # -5
}
```

**Change size window** — edit `SIZE_IDEAL_MIN` / `SIZE_IDEAL_MAX` in `contracts.py` to shift what counts as "ideal size."

**Add search queries** — append to `hf_client.py::SEARCH_QUERIES`. Each query is a dict of kwargs passed to `HfApi.list_datasets()`. Valid keys include `search` (free text), `filter` (tag), `task_categories`, `author`, `language`, `limit`. Test new queries interactively before committing:

```python
from huggingface_hub import HfApi
api = HfApi()
list(api.list_datasets(search="devops automation", limit=10))
```

**Add a new format detector** — if you want to recognize, say, `{input_code, target_code}` as SFT, add a `frozenset` entry to `SCHEMA_PATTERNS`. If you want looser matching, add the column names to `PROMPT_LIKE_COLS` / `RESPONSE_LIKE_COLS` and they'll participate in the heuristic.

**Add a column mapping hint** — `COLUMN_MAP_HINTS` maps HF column names to `DataDesigner` seed columns. When the scorer builds the `download_command` string, any dataset column listed here gets translated. Add new entries if you ingest a dataset with an unusual column name.

**Tune the heuristic schema credit** — in `scoring.py::_score_schema`, the heuristic match currently scores 0.6 and the exact match scores 1.0. Adjust those if you want to penalize/reward heuristic detections more.

## Cache

The JSON cache at `~/.bashgym/research/hf_datasets_cache.json` is keyed by `repo_id`. Each entry is the raw `DatasetMetadata` as a serialized dict. On subsequent runs the scanner reuses cached metadata for datasets it's already seen and only hits the network for new discoveries.

The cache is **not invalidated by `lastModified` changes** — if you want fresh metadata, use `--no-cache`. A future improvement could compare cached `last_modified` against the discovery result's `last_modified` and selectively re-enrich, but it's not worth the complexity until cache staleness becomes a real problem.

To clear the cache entirely:

```bash
rm ~/.bashgym/research/hf_datasets_cache.json
```

## Testing

```bash
venv/bin/python -m pytest tests/research/ -v
```

35 tests, no network calls, all inputs mocked. Test files:

- `tests/research/test_scoring.py` — hard filters, weighted dimensions, schema inference (exact + heuristic), download command
- `tests/research/test_report.py` — markdown generation, top-N limiting, ordering, rejected sections

## Known limitations

- **Datasets with only a raw `content` column** (e.g. `GunA-SD/bash_code`, `1stvamp/py_ast`) cannot be auto-formatted — they need a user-written column mapping. The scanner will still score and rank them on the other dimensions but `bashgym_format` will be `None`.
- **Gated datasets are always rejected**, even if the current user has access. The briefing from WORKSTREAMS.md treats gating as a compliance concern, not an auth concern.
- **Unknown license = rejection.** If both `card.license` and the `license:xxx` tag fallback are missing, the dataset is hard-rejected. This is deliberate — we don't want to accidentally ingest unlicensed data.
- **Non-English datasets are not deprioritized.** The task_match dimension doesn't check language. If that matters, add a language filter to `SEARCH_QUERIES` or a language penalty to `_score_task_match`.
- **The `popularity` signal saturates at 10k downloads.** Superstar datasets (HumanEval, MBPP) don't dominate the ranking just by being popular.
- **Scheduled re-runs not implemented.** Use the `schedule` skill or a systemd timer separately when you want weekly refreshes.

## How the download commands get generated

Each `ScoredDataset.download_command` is a string built by `scoring.py::_build_download_command`. The pattern:

```python
DataDesignerPipeline(
    PipelineConfig(pipeline='coding_agent_sft', num_records=1000)
).from_dataset(
    source='org/repo_id',
    split='train',
    column_mapping={'text': 'seed_task', 'code': 'seed_response'},  # from COLUMN_MAP_HINTS
)
```

The `pipeline` slot is picked from:
- `sft` → `coding_agent_sft`
- `dpo` → `coding_agent_dpo`
- `grpo` → `coding_agent_sft` (GRPO-compatible seed ingestion via the SFT pipeline)

The `column_mapping` is built from `COLUMN_MAP_HINTS` — any dataset column whose name is a key in that dict gets translated to the corresponding DataDesigner seed column.

**This command is a string, not an executed call.** The scanner never imports `DataDesignerPipeline` because that would pull in the heavy NVIDIA NeMo DataDesigner dependency. Copy the command from the report into your own script or REPL to actually run it.

## Empirical dataset ranking (optional, GPU required for real mode)

The rule-based scanner produces a static ranking. If you want to know which datasets *actually* help your model — not just which ones look good on paper — there's a second tool that runs short SFT training experiments per top-N candidate and ranks them by measured eval loss.

```bash
# Fast dev loop — no training, just verifies the pipeline works
venv/bin/python -m bashgym.research.dataset_research_runner --top-n 5 --mode simulate

# Real runs — trains briefly on each dataset (GPU required)
venv/bin/python -m bashgym.research.dataset_research_runner \
    --top-n 10 --mode real \
    --train-steps 100 --num-records 500 \
    --base-model unsloth/gemma-4-E4B-it
```

**How it works:**

1. Loads the scanner cache (`hf_datasets_cache.json`) — run `hf_dataset_scanner` first.
2. Filters to non-rejected SFT candidates, sorts by static score, takes top-N.
3. For each candidate: materializes via `DataDesignerPipeline.from_dataset()` + `export_nemo()`, trains a short SFT run (`train_steps=100` by default), captures `eval_loss`.
4. Ranks empirically by eval loss (lower = better).
5. Writes `~/.bashgym/research/dataset_empirical_ranking.md`.

**What you need:**

- `data-designer>=0.5.0` installed (real mode only). Simulate mode has no deps beyond the scanner itself.
- A GPU capable of running the base model (real mode only).
- The scanner cache populated — run `hf_dataset_scanner` first.

**How it integrates with autoresearch:**

The runner injects a custom `DatasetSearchSpace` (a concrete impl of the existing `SearchSpace` ABC from `bashgym/gym/autoresearch.py`) into `AutoResearcher`. Instead of mutating hyperparameters against a fixed dataset, it holds hyperparameters fixed and iterates through the candidate list. The orchestrator's status tracking, pause/resume, and experiment history all work unchanged. If you want an API endpoint, the existing autoresearch routes already know how to expose the orchestrator.

**Scope:** SFT-format datasets only. DPO and GRPO candidates are filtered out because their training loops have different eval metrics. Adding those means: extend the filter in `_load_candidates`, pick the right `strategy` on TrainerConfig, and read the right metric from `run.metrics`.

**Module layout:**

| File | Role |
|---|---|
| `dataset_search_space.py` | `DatasetSearchSpace(SearchSpace)` — cursor-based dataset enumeration with simulate/real evaluate modes |
| `dataset_research_runner.py` | CLI entrypoint — loads candidates, wires AutoResearcher, writes ranking report |

## Extending the scanner

If you want to:

- **Add a new scoring dimension** — add the weight to `WEIGHTS`, write a `_score_foo()` helper in `scoring.py`, and call it from `score_dataset()`. Add tests to `tests/research/test_scoring.py`.
- **Change the report format** — `report.py::render_report` is pure. Replace it with a Jinja template or a JSON emitter without touching the scorer.
- **Persist historical runs** — write to `~/.bashgym/research/history/{timestamp}.md` from `hf_dataset_scanner.py::run` after the report is written. Diff new runs against the latest history entry to highlight newcomers.
- **Ingest the top-N automatically** — take the top-20 `ScoredDataset` list, iterate, and invoke `DataDesignerPipeline(...).from_dataset(...)` directly. Don't do this from the scanner module itself (keeps the import chain light) — do it from a separate script under `scripts/`.
