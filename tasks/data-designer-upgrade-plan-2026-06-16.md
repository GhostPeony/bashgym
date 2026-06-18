# NVIDIA NeMo Data Designer — Upgrade & Capability-Surfacing Plan

*Compiled 2026-06-16. Research grounded in the live NeMo Data Designer docs (`docs.nvidia.com/nemo/datadesigner/latest`), PyPI, and the open-source `NVIDIA-NeMo/DataDesigner` repo, cross-referenced against a full code-level map of our current integration.*

> **Goal:** iterate and improve our training pipeline and BashGym's interface to Data Designer by adopting the capabilities that shipped between the **v0.5.x** API we target today and the **current v0.6.1** release — and surface them in the UI so the platform can drive them.

---

## 0. TL;DR — the gap in one paragraph

We integrate Data Designer at the **v0.5.x** API level (`data-designer>=0.5.0`, optional/commented in `requirements.txt:30`) with five hand-built column-DAG pipelines and a hand-rolled seed extractor. Since then, **v0.6.1** (current; Apache-2.0; package split into `data-designer-config` + `data-designer-engine`; open-sourced as `NVIDIA-NeMo/DataDesigner`) added a stack of features that map almost 1:1 onto problems we already documented as weaknesses: **native agent-trace ingestion** (`AgentRolloutSeedSource` reading `~/.claude/projects/*.jsonl` directly), a **turnkey trace-distillation recipe**, **real tool execution via MCP** during generation, **execution/validation columns** (`ValidationColumnConfig` with local-callable/remote hooks → run our own sandbox verifier), **in-pipeline embeddings** (move dedup before generation spend), **workflow chaining** (multi-stage/curriculum generation + schema-transform processors that emit chat/messages format), and a **plugin system** (we can ship our verifier + quality scorer as first-class DD components). None of these are wired in today.

---

## 1. Current state (what we have)

*From a full code map (`bashgym/factory/data_designer.py`, `designer_pipelines/`, `api/factory_routes.py`, `frontend/.../DataDesignerTab.tsx`, `bashgym/research/`).*

| Area | Status | Location |
|---|---|---|
| Dependency | `data-designer>=0.5.0`, **optional & commented** | `requirements.txt:28-31` |
| Imports | `import data_designer.config as dd` + `from data_designer.interface import DataDesigner` | `data_designer.py:26-32` |
| Feature detection | `hasattr` flags for 6 column types (`LLMStructured/Judge/Validation/Embedding/Custom/Expression`) | `data_designer.py:34-48` |
| Entry points | `from_traces / from_dataset / from_unstructured / from_config`, `preview / validate / export_nemo / push_to_hub` | `data_designer.py:143-409` |
| Seed extraction | **Hand-rolled** `_extract_seeds_from_traces` (task/tools/complexity/language) | `data_designer.py:451-503` |
| Pipelines (5) | `coding_agent_sft` (default), `coding_agent_dpo` (dual-temp + judge), `tool_use_sft`, `from_external`, `from_unstructured` | `designer_pipelines/` |
| Columns used | Sampler (mostly Category), LLM-Text, LLM-Structured, LLM-Judge, Expression, filter Processor | `designer_pipelines/*.py` |
| Multi-provider | `ProviderSpec` + `build_base_config` (NVIDIA/Anthropic/OpenAI/OpenRouter env resolution) | `designer_pipelines/__init__.py:43-110` |
| API (8 routes) | `preview / create / jobs / pipelines / validate / from-hf / push-to-hub / agent-context / validate-schema` | `factory_routes.py:496-771` |
| Frontend | `DataDesignerTab.tsx` — pipeline cards, num_records/seed form, preview, job polling | `frontend/src/components/factory/` |
| Research tie-in | `DatasetSearchSpace` materializes HF candidates via `from_dataset()` for empirical ranking | `bashgym/research/` |
| In-flight (uncommitted) | `InferenceParameters → ChatCompletionInferenceParams` rename — **targets v0.5.x** | `designer_pipelines/{__init__,coding_agent_dpo,tool_use_sft}.py` |

**Already-documented DD-specific weaknesses** (from `tasks/status-review-2026-06-11.md`, the "20 weaknesses"):
- **#6** no code-execution validation (judge *reads*, doesn't *run*)
- **#7** implicit per-column provider assignment
- **#8** no incremental/streaming generation
- **#9** embedding dedup is **post-hoc** → wastes generation budget
- **#10** no pipeline observability (no row-filter counts / timings / token costs)

---

## 2. What's new in Data Designer **v0.6.1** (capability inventory)

All confirmed against `docs.nvidia.com/nemo/datadesigner/latest` and verified against the `NVIDIA-NeMo/DataDesigner` repo at the **v0.6.1 tag** (a deep-research adversarial pass confirmed 22/25 sampled claims; corrections noted inline).

### 2.0 Release line, engine, and the two product distributions *(verified)*
- **v0.6.1 is current** (2026-06-01); no 0.6.2/0.7.0 exists. Ladder: 0.5.4–0.5.9 (Apr) → **0.6.0** (May 13) → **0.6.1** (Jun 1). Our `>=0.5.0` target predates *every* agent-rollout feature (rollout ingestion landed v0.5.4; ATIF/Hermes/Pi added v0.5.6).
- **Async engine is the DEFAULT path as of v0.6.0** — cell-level, overlaps independent columns, **adapts concurrency per (provider, model)**. Just bumping the version gets us throughput + reasoning-token usage tracking + **run-resume** for interrupted jobs — i.e. partial wins on weaknesses **#8** (incremental/throughput) and **#10** (observability) *for free*. Legacy sync engine survives one transitional release via `DATA_DESIGNER_ASYNC_ENGINE=0`.
- **Plugin system graduated to stable** in v0.6.0 (typed configs + entry-point discovery) — Phase 5 is building on stable ground, not experimental.
- v0.6.1 adds (verified, with PRs): workflow chaining (#636), reasoning-token tracking (#670), audio/video context (#701), multimodal MCP tool-result preservation (#689), dropped-column preservation toggle (#691), bounded-borrow task admission (#693), configurable async in-flight cap (#699).
- **TWO product lines, versioned differently** — this matters for deployment:
  1. **Standalone library** — `NVIDIA-NeMo/DataDesigner` repo + PyPI `data-designer` **0.6.1** (← *our line*).
  2. **NeMo Microservices PySDK** — calendar-versioned **25.12.0 / 26.3.0** (microservice/NMP execution).
  The `data_designer.config` module is **context-agnostic**: "all `config_builder` code works identically; only the execution interface changes." → our pipelines port from library to microservice **without rewrite** (de-risks Phase 6).

### 2.1 Native agent-trace ingestion — `AgentRolloutSeedSource`
Reads coding-agent rollouts directly off disk and normalizes them, **no importer code required**.
```python
dd.AgentRolloutSeedSource(
    format=dd.AgentRolloutFormat.CLAUDE_CODE,  # CLAUDE_CODE | CODEX | HERMES_AGENT | PI_CODING_AGENT | ATIF
    path=None, recursive=True, file_pattern="*.jsonl",  # all optional overrides
)
```
- **CLAUDE_CODE** → `~/.claude/projects/*.jsonl` (preserves `agentId`, `isSidechain`) — i.e. *our exact trace source*.
- **HERMES_AGENT** → `~/.hermes/sessions/*.json*` (our deploy-loop trace source).
- Normalizes every format to one schema: `trace_id, source_kind, source_path, root_session_id, agent_id, is_sidechain, cwd, project_path, git_branch, started_at, ended_at, messages, source_meta, message_count, tool_call_count, final_assistant_message`.
- Feeds downstream columns via `with_seed_dataset()`; fields available in Jinja (`{{ messages | random }}`, `{{ trace_id }}`) and custom generators (explode `messages`→tool-call rows).
- **Overlaps our hand-rolled importers** (`trace_capture/importers/`, `data_designer.py:451-503`).

### 2.2 Turnkey trace-distillation recipe (*Agent Rollout Trace Distillation*)
End-to-end: ingest → `AgentRolloutTraceDigest` (LLM-structured digest) → `AgentRolloutFinetuningRecord` (standalone instruction/response) → `LLMJudgeColumnConfig` with `SFT_JUDGE_SCORES` (5 dims: groundedness, standalone clarity, response quality, faithfulness, training utility, 0-4) → flatten to `sft_instruction / sft_response / sft_skill_tags` + `recommended_for_sft`. Recipe teacher: `nvidia/nemotron-3-super-120b-a12b`.

### 2.3 Real tool use via MCP (during generation)
LLM columns can **call real tools / MCP servers** while generating — real trajectories, not simulated.
```python
dd.LocalStdioMCPProvider(name="bashgym-sandbox", command="python", args=["-m","bashgym.mcp.sandbox_server"])
dd.ToolConfig(tool_alias="sandbox", providers=["bashgym-sandbox"])
dd.LLMTextColumnConfig(name="answer", prompt="...", model_alias="...", tool_alias="sandbox")
```
Also `dd.MCPProvider` (remote SSE + auth). Governance: **allowlists, budgets, timeouts**. Directly addresses weakness **#6** for tool-use data.

### 2.4 Validators — `ValidationColumnConfig`
```python
dd.ValidationColumnConfig(name="...", target_columns=["solution"],
    validator_type="code" | "local_callable" | "remote",
    validator_params=..., batch_size=10, drop=False)
```
- `CodeValidatorParams(code_lang=dd.CodeLang.PYTHON | SQL_POSTGRES | ...)` — **lints** (Ruff/SQLFluff), returns `is_valid`, `python_linter_score`(0-10), messages. *Does not execute.*
- `LocalCallableValidatorParams(validation_function, output_schema)` — arbitrary Python over the DataFrame → **our hook to run pytest in the Docker sandbox and gate on real pass/fail**.
- `RemoteValidatorParams(endpoint_url, timeout, max_retries, ...)` — HTTP validator → call our verifier service. Closes weakness **#6** properly.

### 2.5 In-pipeline embeddings — `EmbeddingColumnConfig` *(⚠ feature-gate)*
Generate embeddings as a column → dedup/filter **inside** the DAG. Would close weakness **#9** (no more post-hoc dedup of already-paid-for rows).
> **Caveat (verified):** `EmbeddingColumnConfig` (and `ImageColumnConfig`, `CustomColumnConfig`) appear in the standalone repo's `concepts/columns.md` but were **NOT confirmed** in the NeMo Microservices PySDK 25.12.0 column-config reference (which lists ~10: Sampler, Expression, LLMText, LLMCode, LLMStructured, LLMJudge+Score, SeedDataset, Validation + the abstract base). Treat Embedding/Image/Custom as **present-but-verify** — gate via the existing `hasattr` feature detection and keep the post-hoc `EmbeddingDeduplicator` as fallback when absent.

### 2.6 Workflow chaining (experimental)
```python
wf = data_designer.compose_workflow(name="coding-curriculum")
wf.add_stage("draft", draft_builder, num_records=500, output_processors=[...])
wf.add_stage("to_chatml", chatml_builder, output="processor:chatml")
res = wf.run(); res.load_dataset()
```
Stages pass parquet via `LocalFileSeedSource`; processor-only stages allowed. **Linear only; no stage-resume yet.** Enables multi-stage / curriculum generation aligned with our Cascade RL.

### 2.7 Processors (batch transforms) — `DropColumnsProcessorConfig`, `SchemaTransformProcessorConfig`
`SchemaTransformProcessorConfig(template={"messages":[{"role":"user","content":"{{prompt}}"},{"role":"assistant","content":"{{solution}}"}]})` **replaces our hand-written `_write_nemo_jsonl`** (`data_designer.py:589-629`). Lifecycle hooks: `process_before/after_batch`, `process_after_generation`.

### 2.8 New column types & samplers we don't use
- `LLMCodeColumnConfig` — **confirmed** — code in 15+ langs with markdown extraction/parsing (better than generic LLM-Text for solution code). Inference is via **LiteLLM**, so any OpenAI-compatible endpoint (Ollama/NIM/vLLM) works per-column.
- `SeedDatasetColumnConfig` — **confirmed**. `CustomColumnConfig` (`@custom_column_generator`) / `ImageColumnConfig` — *present-but-verify* (see §2.5 caveat).
- Samplers beyond Category (**confirmed** in 25.12.0): `Subcategory, Person, Uniform, Gaussian, Poisson, Binomial, Bernoulli(+Mixture), Scipy, Datetime, Timedelta`.

### 2.9 Plugin system + CLI
- Three plugin types: `PluginType.COLUMN_GENERATOR | SEED_READER | PROCESSOR` (configs inherit `SingleColumnConfig/SeedSource/ProcessorConfig`; impls inherit `ColumnGenerator*/SeedReader|FileSystemSeedReader/Processor`). Registered via `pyproject.toml` `[project.entry-points."data_designer.plugins"]`.
- `typer` CLI: `data-designer config list` (providers/models), config management, `data-designer plugin` (discover/list), MCP CLI config.
- **Coding-agent skill / `AGENTS.md`**: DD ships a Claude-Code-oriented skill — describe the dataset in NL and it designs schema/validation/generation. We already proxy this at `factory_routes.py:698-729` (`/designer/agent-context`).

### 2.10 Deployment: Library vs Microservice
Same `DataDesignerConfigBuilder` API both ways — **verified** that `config_builder` code transfers identically and only the execution interface changes (see §2.0). **Microservice** (NeMo Microservices PySDK 25.12.0/26.3.0, NGC Docker image, REST `preview`/`create`, integrates NIM/Customizer/Evaluator) → option to run generation on the **DGX Spark** as a shared service with zero pipeline rewrite.

### 2.11 Notable recipes / dev-notes / reference datasets to mine
- **Recipes:** `text-to-sql` + `nemotron-super-text-to-sql`, `multi-turn-chat`, `search-agent SFT`, `deep-research-trajectories` (MCP), `frontier-judge-qa-filter`. The distillation recipe is a copyable file: `docs/assets/recipes/trace_ingestion/agent_rollout_distillation.py` — exact classes `dd.AgentRolloutSeedSource` (`--format atif|claude_code|codex|hermes_agent`) → `LLMStructuredColumnConfig("sft_record", AgentRolloutFinetuningRecord{instruction,response,skill_tags})` → three `ExpressionColumnConfig` (`sft_instruction`/`sft_response`/`sft_skill_tags`) → `LLMJudgeColumnConfig("sft_quality_judge_result", scores=SFT_JUDGE_SCORES)` (5 `dd.Score`: `groundedness, standalone_task, response_quality, faithfulness, training_utility`, each 0–4).
- **Dev-notes:** **Nemotron-Personas** (multi-locale persona diversity), **prompt-sensitivity** (diverse preambles for robustness), **Retriever SDG plugin**, **adaptive concurrency / async-all-the-way-down**.
- **Reference dataset (verified):** **`nvidia/Nemotron-PII`** on HF — CC-BY-4.0, 100k+ records, persona-grounded (U.S.-Census-grounded), tagged `datadesigner` + `synthetic-data`, **explicitly built with Data Designer** (used to fine-tune GLiNER-PII). Directly relevant to our regex+LLM PII redaction layer and a template for a persona-grounded safety pipeline. *(Search HF for the `datadesigner` tag to find more.)*
- **Boundary (verified):** `nvidia/Nemotron-Agentic-v1` was **NOT** built with Data Designer (open-LLM generation + LLM-judge filtering) — don't assume every Nemotron agentic dataset is a DD recipe.

---

## 3. Gap analysis — have vs. available

| Capability | We have | v0.6.1 offers | Weakness closed | Priority |
|---|---|---|---|---|
| Trace seeding | hand-rolled extractor (`:451-503`) | `AgentRolloutSeedSource` (CLAUDE_CODE/HERMES) | maintenance burden | **P1** |
| Trace→SFT | bespoke `coding_agent_sft` | distillation recipe (`TraceDigest`/`FinetuningRecord`/judge) | quality, reuse | **P1** |
| Real tool exec | simulated in `tool_use_sft` | MCP tool columns | **#6** (tool-use) | **P2** |
| Code validation | judge reads only | `ValidationColumnConfig` + local/remote verifier | **#6** | **P2** |
| Dedup | post-hoc embeddings | `EmbeddingColumnConfig` in-DAG | **#9** | **P2** |
| Multi-stage gen | none (single `generate`) | `compose_workflow` chaining | curriculum | **P3** |
| NeMo export | hand-written JSONL (`:589-629`) | `SchemaTransformProcessorConfig` | brittleness | **P3** |
| Solution code | generic LLM-Text | `LLMCodeColumnConfig` | quality | **P3** |
| Per-column provider | implicit aliases | explicit `ModelConfig` routing | **#7** | **P3** |
| Diversity | Category samplers | Personas / diverse preambles / Subcategory | mode collapse | **P4** |
| Extensibility | inline pipelines | plugin (verifier/quality as plugins) | reuse | **P4** |
| Observability | job-level only | adaptive-concurrency metrics, token costs | **#10**, **#8** | **P4** |
| Deploy | embedded library | microservice on DGX | scale | **P5** |
| NL dataset design | agent-context proxy only | DD coding-agent skill end-to-end | UX | **P5** |

---

## 4. Phased implementation plan

> Sequenced so each phase is independently shippable and unblocks the next. Every phase keeps the **optional-dependency + graceful-fallback** contract (`DATA_DESIGNER_AVAILABLE`).

### Phase 0 — Foundation: version bump + API reconciliation *(small, unblocks all)*
1. `requirements.txt` (and `requirements-training.txt`): set the optional extra to `data-designer>=0.6.1` (note the new split packages `data-designer-config`/`data-designer-engine` are pulled transitively). Keep commented/optional but document the install line.
2. Verify imports against 0.6.1: confirm `data_designer.config` + `data_designer.interface.DataDesigner` are still the entry points after the config/engine split (read the repo `README.md`/`AGENTS.md`). Adjust if the namespace moved.
3. Extend feature detection (`data_designer.py:34-48`) to add: `HAS_CODE_COLUMN` (`LLMCodeColumnConfig`), `HAS_AGENT_ROLLOUT` (`AgentRolloutSeedSource`/`AgentRolloutFormat`), `HAS_MCP` (`ToolConfig`/`LocalStdioMCPProvider`), `HAS_WORKFLOW` (`compose_workflow`), `HAS_SCHEMA_TRANSFORM` (`SchemaTransformProcessorConfig`), `HAS_SEED_DATASET_COLUMN`.
4. Reconcile the in-flight `ChatCompletionInferenceParams` migration against the 0.6.1 `inference-parameters` page (it was written for 0.5.x). Run `tests/factory/test_data_designer.py` + `test_pipeline_builders.py` green.
5. **Free wins from the bump:** the **default async engine** (v0.6.0, adaptive per-provider concurrency), **run-resume**, and **reasoning-token usage tracking** come with the version — partial relief for weaknesses #8/#10 with no code. Keep `DATA_DESIGNER_ASYNC_ENGINE=0` documented as the escape hatch if a pipeline misbehaves on the async path during the transition.
6. **Exit:** `pip install data-designer>=0.6.1` works; all feature flags resolve; existing 5 pipelines still build & preview on the async engine.

#### ✅ Phase 0 — DONE (2026-06-17). What shipped + 0.6.1 API drift discovered

Installed **data-designer 0.6.1** locally (per-user site, Python 3.14; no downgrades — unrelated `browser-use`/`readme-renderer` pins warn but are upward bumps). **235 factory tests pass**, ruff+black clean. Kept DD **optional** (`bashgym[data-designer]` extra; graceful `DATA_DESIGNER_AVAILABLE` fallback) and **dual-use** (training-loop *and* standalone).

Empirically verified against the installed package (not just docs) and **reconciled real breaking drift** that docs alone didn't reveal:
- **Provider wiring moved**: `DataDesignerConfigBuilder` no longer takes `model_providers=` — providers now attach to `DataDesigner(model_providers=[...])` and `ModelConfig` binds via `provider=<name>`. Refactored into `build_model_providers()` + `_provider_name_for()`; the `designer` property now constructs `DataDesigner(model_providers=...)`. (`ModelConfig.provider=None` is deprecated → issue #589; all inline configs now bind a provider.)
- **Generation API**: `designer.generate(builder, num_rows=)` → `designer.create(builder, num_records=).load_dataset()`; `preview(...).dataset`; `validate()` returns `None` and raises on invalid (wrapper updated; column introspection via best-effort `_builder_column_names`).
- **Seed sources**: `dd.FileSeedSource` → `dd.LocalFileSeedSource(path=)`; `HuggingFaceSeedSource` now takes only `path/token/endpoint` (no `split`/`name`/`column_mapping`) → `from_dataset` materializes via the `datasets` lib → `DataFrameSeedSource` when remap/subset/non-default-split is requested.
- **Feature flags**: all new flags resolve **True** on 0.6.1 — `LLMCodeColumnConfig`, `EmbeddingColumnConfig`, `CustomColumnConfig`, `ImageColumnConfig`, `SeedDatasetColumnConfig`, `AgentRolloutSeedSource` (formats `CLAUDE_CODE/CODEX/HERMES_AGENT/PI_CODING_AGENT/ATIF`), `ToolConfig`+`LocalStdioMCPProvider`, `SchemaTransformProcessorConfig`, `compose_workflow`. **This retires the §2.5 feature-gate caveat for the standalone library** (Embedding/Custom *do* exist; gate was only a microservices-PySDK-docs gap).
- `ChatCompletionInferenceParams(temperature/top_p/max_tokens)` confirmed correct (the in-flight 0.5.x migration is right); import surface unchanged.

**Carried into Phase 2 (blocking for full generation):** Data Designer 0.6.x **removed the row-level `filter` processor** (only `DROP_COLUMNS`/`SCHEMA_TRANSFORM` remain). The 5 pipelines' quality gates were removed to allow construction (judge columns retained on rows); re-implement each as `ValidationColumnConfig(drop=True)` — needs a live generation run (API budget) to verify judge-column row shapes.

**Runtime note (Windows):** DD prints emoji/rich output; the cp1252 console raises `UnicodeEncodeError` on `validate()`/generation. Set `PYTHONUTF8=1` (or `PYTHONIOENCODING=utf-8`) for the backend/launcher — candidate one-line fix in `run_backend.py`/`dev.ps1`.

**Known behavior:** standalone `validate()`/`preview()` build the pipeline **without a seed source**, so seed-derived references (`seed_task`, …) report as missing — expected; `from_traces`/`from_dataset` attach the seed first. (Small future nicety: attach a representative sample seed for validate/preview.)

### Phase 1 — Native ingestion + trace distillation *(highest ROI)*
1. **Swap seed extraction for `AgentRolloutSeedSource`.** In `from_traces()` (`data_designer.py:143-174`), when `HAS_AGENT_ROLLOUT`, use `AgentRolloutSeedSource(format=CLAUDE_CODE/HERMES_AGENT, path=...)` instead of `_extract_seeds_from_traces` + `DataFrameSeedSource`. Keep the hand-rolled path as fallback for older DD. This deletes the brittle `:451-503` extractor from the hot path and yields richer fields (`tool_call_count`, `final_assistant_message`, `git_branch`, `is_sidechain`).
   - **Keep our importers** (`trace_capture/importers/`) for *classification* (gold/failed/quality) — they carry verification metadata DD's ingestion lacks. AgentRolloutSeedSource is for *synthesis seeding* only. Document the division.
2. **New pipeline `coding_agent_distill`** in `designer_pipelines/` implementing the distillation recipe: `AgentRolloutTraceDigest` → `AgentRolloutFinetuningRecord` → `LLMJudgeColumnConfig(SFT_JUDGE_SCORES)` → flatten. Register in `PIPELINES`. Default teacher configurable (`nvidia/nemotron-3-super-120b-a12b`, or our DGX Ollama).
3. Tests: extend `test_pipeline_builders.py` with the new pipeline; add an `AgentRolloutSeedSource` builder test (guarded by feature flag).
4. **Exit:** "Generate from traces" path uses native ingestion; `coding_agent_distill` available end-to-end and exported to NeMo.

#### ✅ Phase 1 — DONE (2026-06-17)

**243 factory tests pass**, ruff+black clean. Implemented against the installed 0.6.1 (recipe + kwargs verified empirically, not guessed).
- **Native ingestion via new `from_agent_rollouts(rollout_format, path=, num_records=, recursive=, file_pattern=)`** on `DataDesignerPipeline` — wraps `dd.AgentRolloutSeedSource` (formats `claude_code`/`codex`/`hermes_agent`/`pi_coding_agent`/`atif`; `atif` requires `path`). **Design refinement vs the original plan:** kept `from_traces` (it seeds from BashGym's *processed* gold-trace schema) and added `from_agent_rollouts` as a *parallel* path that ingests *raw* rollouts (`~/.claude/projects/*.jsonl`, Hermes, …) — they consume different data shapes, so this adds the capability without breaking the gold-trace flow. Importers still own classification.
- **New `coding_agent_distill` pipeline** (`designer_pipelines/coding_agent_distill.py`, registered in `PIPELINES`) faithful to NVIDIA's recipe: `LLMStructuredColumnConfig(trace_digest=AgentRolloutTraceDigest)` → `LLMStructuredColumnConfig(sft_record=AgentRolloutFinetuningRecord)` → `LLMJudgeColumnConfig(scores=SFT_JUDGE_SCORES, 5 dims 0-4)` → expression columns (`sft_instruction`/`sft_response`/`sft_skill_tags`/`trace_training_value`/`recommended_for_sft`). Teacher reuses the base text/code/judge aliases (point them at `nvidia/nemotron-3-super-120b-a12b` or a DGX model for real distillation).
- **Bonus fix found via review:** `from_traces`/`from_unstructured` still used the 0.5.x `DataFrameSeedSource(data=)` kwarg → corrected to `df=` (0.6.1); added a regression test exercising the real seed source.
- Tests: 6-pipeline construction smoke (incl. distill), `from_agent_rollouts` (resolver, atif-needs-path, unsupported-guard, seed-attach), distill internals (5 judge dims, model fields), from_traces real-seed regression.
- **Deferred to Phase 6:** API route (`/designer/*`) + `DataDesignerTab` wiring for the rollout seed-format selector and the distill pipeline card (Phase 1 delivers the backend/programmatic path; usable standalone today via `DataDesignerPipeline`).

### Phase 2 — Real validation & in-pipeline dedup *(closes #6, #9)*
1. **`ValidationColumnConfig` with a BashGym verifier hook.** Add a `LocalCallableValidatorParams(validation_function=run_sandbox_verifier)` (or `RemoteValidatorParams` → a new `/api/verify` endpoint wrapping `bashgym/judge/verifier.py`) so generated solutions are **executed** (pytest/`verify.sh`) and gated on real pass/fail, not just judged. Add to `coding_agent_sft` + `coding_agent_distill` behind a `validate_execution: bool` config flag.
2. **Cheap pre-gate:** add `CodeValidatorParams(code_lang=PYTHON)` Ruff lint as a fast filter before the expensive execution validator.
3. **`EmbeddingColumnConfig` in-DAG dedup** — **if `HAS_EMBEDDING_COLUMN` resolves true in the installed 0.6.1** (see §2.5 caveat — unconfirmed in the microservices line), replace the post-hoc `EmbeddingDeduplicator` step for DD-generated rows with an embedding column + filter processor (cosine ≥0.95, mirroring `dedup.py:29`). If absent, fall back to a `LocalCallable`/`Custom` column or keep post-hoc dedup. Saves generation/judge spend either way once moved before the judge.
4. **Exit:** DD output for coding pipelines carries an execution-verified flag; dedup happens before final judge.

#### ✅ Phase 2 — DONE (2026-06-17), reshaped by two live-verified 0.6.x findings

**257 factory tests pass**, ruff+black clean. Verified end-to-end against the live NVIDIA NIM API (capstone: full `coding_agent_sft` generated structured solution + expressions + judge + flag, then export-gated).

**Finding 1 — auth wiring was broken.** Our `${NVIDIA_API_KEY}` placeholder is sent *literally* by DD's default `PlaintextResolver` → auth failure. **Fixed:** `build_model_providers` now sets `api_key` = the env-var **name**, and the `designer` property passes `secret_resolver=CompositeResolver([EnvironmentResolver, PlaintextResolver])` (`build_secret_resolver`). Env-name resolves from the environment; an explicit raw key falls through to plaintext. *This unblocked all live generation.*

**Finding 2 — there is no in-pipeline row-filter.** `ValidationColumnConfig(drop=True)` drops the **column**, not invalid rows (verified: 10/10 rows survived). The 0.6.x-native pattern (per NVIDIA's own recipe) is **flag-then-filter**. **Implemented:** each base pipeline emits a boolean `passes_quality` `ExpressionColumnConfig`; `export_nemo(quality_flag_column="passes_quality", keep_only_passing=True)` applies the gate (auto-falls back to the distill pipeline's `recommended_for_sft`). This re-introduces the gates removed in Phase 0, the native way. Judge sub-scores are referenced **nested** (`quality_score.correctness.score`), not flat.
- **Latent expression bugs fixed (only surfaced live):** dpo `chosen`/`rejected` compared judge *dicts* (`judge_a.quality`) → fixed to `.quality.score`; distill `recommended_for_sft` used the recipe's flat `groundedness_score` → fixed to nested `sft_quality_judge_result.groundedness.score`.

**Model adaptability (per user feedback — no hardcoded model IDs; any open model from Ollama/Unsloth/HF/NIM):**
- Completed the **NIM gap** in the shared discovery: `detector.get_nim_models()` + wired into `get_available_models(include_cloud=True)` → discovery now spans Ollama/LM-Studio (local) + live NIM catalog + curated HF training/teacher.
- DD: `list_inference_models()` (cross-source catalog for UI/selection), `provider_model_ids(provider, endpoint)` (live `/v1/models` for any OpenAI-compatible endpoint), and `PipelineConfig.resolve_models()` — validates text/code/judge against the live catalog and **substitutes** stale/unserved IDs (verified: swapped the stale `qwen/qwen2.5-coder-32b-instruct` default → an available coder), preferring instruction/chat-tuned models.
- **Caveat (verified):** NIM's `/v1/models` lists some **non-invokable** models (`starcoder2-15b`, `deepseek-coder-6.7b-instruct` listed but fail chat health-checks). So listing ≠ usable; `resolve_models` picks from the listing and DD health-checks at generation. The user's real target — **DGX Ollama** — lists only served models, so it's reliable there. A future refinement could health-check candidates.

**Scope adjusted vs original Phase 2 (premised on a row-drop that 0.6.x lacks):**
- **In-DAG embedding dedup → deferred.** Dedup is cross-row; with no row-drop and `EmbeddingColumnConfig` only producing vectors, in-pipeline dedup isn't possible in 0.6.x. Post-hoc `EmbeddingDeduplicator` stays. (Weakness #9 is a 0.6.x limitation, not a wiring gap.)
- **Execution/sandbox-verifier validator → deferred to Phase 3.** Our pipelines emit tool-use *trajectories*, not standalone testable code, and validators annotate (don't drop). Real execution fits the MCP work (Phase 3).

**Also:** Windows UTF-8 fix shipped (`run_backend.py` sets `PYTHONUTF8=1`/`PYTHONIOENCODING` for the uvicorn child; `dev.ps1` sets it in the backend job) — DD's emoji output no longer crashes the cp1252 console.

**Deferred to Phase 6 (UI):** `/designer/*` + `DataDesignerTab` wiring for model selection (from `list_inference_models`), the quality-gate toggle, and the rollout/distill controls.

### Phase 3 — Real tool-use via MCP *(closes #6 for agentic data)*
1. **Wrap our sandbox as an MCP server** (`bashgym/mcp/sandbox_server.py`) exposing BASH/READ/WRITE/EDIT against `sandbox.py` with existing guardrails (`_is_dangerous_command`).
2. **Upgrade `tool_use_sft`** to grant `tool_alias` to its generation column via `LocalStdioMCPProvider` + `ToolConfig`, so trajectories execute real tools. Add safety: allowlist + per-row budget + timeout.
3. Optional new pipeline `deep_research_trajectories` modeled on the DD recipe (MCP + search tool) for browse/agent data.
4. **Exit:** tool-use trajectories contain real tool calls + observations from our sandbox.

#### ✅ Phase 3 — DONE (2026-06-17)

**269 factory tests pass** (+12 MCP), ruff+black clean; live `mcp_tool_use` run completed end-to-end against NIM.
- **Sandbox MCP server** (`bashgym/mcp/sandbox_server.py`, FastMCP over stdio): tools `bash/read_file/write_file/edit_file/grep/list_files`. Backend chosen by `BASHGYM_MCP_BACKEND` (`auto`|`docker`|`local`, per the user's choice): **DockerWorkspace** (reuses `SandboxManager`, network-off, dangerous-cmd guard) with a **guarded-local fallback** (path-confined temp workspace, dangerous-cmd guard, per-call timeout). Refactored the dangerous-command guard into a shared `is_dangerous_command` in `arena/sandbox.py` (now **case-insensitive** — fixes a latent miss on `-R` variants).
- **DD wiring:** `build_mcp_providers` (LocalStdioMCPProvider launching `python -m bashgym.mcp.sandbox_server` with a full env so the subprocess can import bashgym), `build_sandbox_tool_config` (allowlist + `max_tool_call_turns` + `timeout_sec`), `build_base_config(tool_configs=...)`, and the `designer` property passes `mcp_providers`. New `PipelineConfig` fields (`enable_tools`/`mcp_backend`/`mcp_tool_alias`/`mcp_max_tool_turns`/`mcp_tool_timeout_sec`); `__post_init__` sets `enable_tools` for tool pipelines so MCP attaches **order-independently**.
- **New `mcp_tool_use` pipeline** (registered only when `HAS_MCP`): generates a task, grants the agent column the sandbox `tool_alias` for real execution, judges, and flags `passes_quality`.
- **Live verification:** DD spawned the MCP subprocess, the tool health-check passed, providers registered, all columns generated, and the model emitted a `write_file` tool call. **Caveat:** the transcript showed tool-call-shaped JSON with a single request, i.e. llama-3.3-70b (via NIM/LiteLLM) emitted the call as *text* rather than DD driving a full multi-turn execution loop — real execution engagement depends on the model's native function-calling support (a model-selection matter; tool-calling-capable / DGX models would exercise the loop). Infra (server, guards, confinement, wiring, registration, health-check) is verified correct.

**Deferred to Phase 6 (UI):** `DataDesignerTab` controls for the tool backend + the `mcp_tool_use` pipeline.

### Phase 4 — Workflow chaining + processors + observability *(closes #8/#10 partially)*
1. **Replace `_write_nemo_jsonl` / `export_nemo`** with `SchemaTransformProcessorConfig` (chat/messages) + `DropColumnsProcessorConfig` (strip intermediates) as terminal processors. Keep our splitter for train/val until DD covers it.
2. **`compose_workflow` multi-stage** entry point `from_traces_chained()`: stage1 ingest+generate → stage2 validate → stage3 judge-filter → stage4 schema-transform. Surface as an optional "advanced/curriculum" mode.
3. **Observability:** capture DD's adaptive-concurrency / token-cost / per-column counts and stream them over the existing training/factory WebSocket; record in `DesignerJobResponse`.
4. **Exit:** generation reports per-column timings + token cost + filter counts; export is processor-driven.

#### ✅ Phase 4 — DONE (2026-06-17)

**273 factory tests pass** (+4), ruff+black clean; live workflow run verified.
- **Workflow chaining:** `DataDesignerPipeline.generate_chained(stages)` wraps `compose_workflow` → `add_stage(name, builder, num_records=, output_processors=, output=)` → `run().load_dataset()`. Enables multi-stage / curriculum generation and processor-driven stages. (Experimental in DD: linear topology, no stage-resume.)
- **Processor-based ChatML export:** `messages_schema_transform(user_col, assistant_col, system_prompt=)` returns a `SchemaTransformProcessorConfig` emitting an OpenAI-style `messages` column — the 0.6.x-native export replacing hand-assembled JSONL. **Live-verified:** a chained run produced `{"messages": [{system},{user},{assistant}]}` via the stage's `output_processor`.
- **Observability:** `GenerationStats` (records, filtered_out, stage names) on `pipeline.last_stats`; `export_nemo` already returns `filtered_out`. Token costs / per-column timings are emitted by DD's async engine to logs (the engine exposes `usage_stats` internally; not surfaced on the public `DatasetCreationResults`, so we don't couple to it).
- **Scope note:** the async engine (default since v0.6.0) already provides adaptive per-provider concurrency + run-resume + reasoning-token tracking (the throughput half of #8), gained on the version bump.

**Deferred to Phase 6 (UI):** stream `GenerationStats` + DD's usage logs over the factory WebSocket and surface a chained/curriculum builder in `DataDesignerTab`.

### Phase 5 — Plugins, provider routing, diversity *(reuse + quality)*
1. **BashGym DD plugin package** (`bashgym/factory/dd_plugin/`): (a) `SEED_READER` for our gold/failed trace layout (if richer than AgentRollout), (b) `PROCESSOR`/validator wrapping the verifier, (c) `COLUMN_GENERATOR` exposing the 7-metric `quality_calculator` score. Register via entry points so it works from the `data-designer` CLI too.
2. **Explicit per-column provider routing** (closes #7): give each `LLMTextColumnConfig`/`LLMCodeColumnConfig`/`LLMJudgeColumnConfig` an explicit `model_alias`→`ModelConfig(provider=...)`; wire **DGX Ollama** as an OpenAI-compatible `ModelProvider(endpoint="http://192.168.50.173:11434/v1")` for cheap local generation, NVIDIA NIM for teacher.
3. **Diversity:** adopt **Nemotron-Personas** + diverse-preamble sampling and `Subcategory` samplers in the prompt-generation columns to fight mode collapse.
4. **Exit:** our verifier/quality scorer usable as DD plugins; per-column provider control; persona-diversified prompts.

#### ✅ Phase 5 — DONE (2026-06-17)

**284 factory tests pass** (+11), ruff+black clean.
- **Per-column provider routing (closes #7):** confirmed the existing `ProviderSpec.models` + `_provider_name_for` already routes per alias (verified: text/code→ollama, judge→nvidia). Added ergonomic presets in `designer_pipelines`: **`ollama_provider_spec(models, endpoint=)`** (OpenAI-compatible Ollama / DGX Spark — no API key; endpoint auto-`/v1`; `OLLAMA_BASE_URL` default) and **`nim_provider_spec(models)`**. So a mixed config routes cheap columns to **DGX Ollama** and the judge to NIM/Claude — directly serving the deploy-on-DGX architecture.
- **Diversity:** `subcategory_sampler(name, category_column, values)` (SUBCATEGORY conditioned on a parent category) and `persona_sampler(with_synthetic_personas=True)` (Nemotron-Personas PERSON sampler) for prompt diversity / robustness.

- **BashGym Data Designer plugin (`bashgym/factory/dd_plugin/`):** a `SEED_READER` plugin (`GoldTraceSeedSource` config + `GoldTraceSeedReader` impl subclassing DD's `FileSystemSeedReader`) that exposes our *processed* gold-trace schema (`data/gold_traces/*.json`) as a first-class DD seed source — the complement to native `AgentRolloutSeedSource` (raw rollouts). Registered via `pyproject.toml` `[project.entry-points."data_designer.plugins"]`. **Verified end-to-end:** after `pip install -e . --no-deps`, DD's `PluginManager` discovers `bashgym_gold_trace` as a SEED_READER, and the reader extracts seed rows (task/tools/complexity/language/step_count, skipping prompt-less traces) so any DD config or the `data-designer` CLI can `seed_type: "bashgym_gold_trace"`.
- *Note:* a verifier-validator and 7-metric quality `COLUMN_GENERATOR` were considered but are semantically trace-oriented (our quality/verify operate on traces, not freshly-generated rows), so the seed-reader is the well-scoped, genuinely-useful plugin; the others fit better as future post-processing.

### Phase 6 — Frontend surfacing + optional microservice
1. **`DataDesignerTab` upgrades** (see §5).
2. **Optional DD microservice on DGX** for large runs (NGC Docker image), selectable as a backend alongside library mode — reuse the SSH/remote-trainer pattern.

#### ✅ Phase 6 — backend + frontend DONE (2026-06-17); microservice deferred

**Backend (`api/factory_routes.py`, no conflict with the parallel WIP), 284 factory tests pass:**
- `GET /designer/models` — adaptable inference-model catalog (`list_inference_models` + optional provider `/v1/models`), run via `asyncio.to_thread` so the sync discovery / blocking httpx don't deadlock the event loop. Verified: 121 models.
- `GET /designer/pipelines` — real **column-DAG introspection** (replaced `columns=[]`); now lists all 7 pipelines incl `coding_agent_distill`, `mcp_tool_use` with their columns.
- `DesignerCreateRequest` + `run_designer_job`: `seed_type:"agent_rollouts"`(+`seed_format`)→`from_agent_rollouts`, `keep_only_passing`→`export_nemo` gate, `mcp_backend`→tool pipeline.

**Frontend (`DataDesignerTab.tsx` + `api.ts`, both clean — not in the parallel WIP), tsc + eslint clean:**
Rebuilt the tab in the app's **Botanical Brutalism** (per `~/.claude/DESIGN_SYSTEM.md`): live **Text/Code/Judge model pickers** grouped by provider from `/designer/models` (with refresh + count), the selected pipeline's **column-DAG flow** (boxes + accent arrows), **Agent-Rollout** seed type with a **rollout-format** selector + path, a **Quality-Gate** press-in toggle, and an **MCP tool-backend** selector for `mcp_tool_use`. Header shows `N pipelines · M models`.

**Deferred:** DD **microservice on DGX** (NGC Docker, library↔microservice via the same `config_builder`); WebSocket streaming of `GenerationStats` + token-usage logs; live visual QA (`/qa`) of the rebuilt tab.

---

## 5. BashGym ↔ Data Designer interface upgrades

### Backend (`bashgym/api/factory_routes.py`, `schemas.py`)
- `/designer/pipelines`: stop returning `columns=[]` (`factory_routes.py:599`) — introspect the builder so the UI can render the real column DAG.
- `DesignerCreateRequest`: add `seed_format` (CLAUDE_CODE/HERMES/…), `validate_execution`, `enable_tools`/`tool_alias`, `provider_routing` (per-column), `workflow_chained`, `teacher_model`.
- New routes: `POST /api/verify` (verifier-as-validator endpoint for `RemoteValidatorParams`); `GET /designer/columns` (catalog of available `dd.*` column types from feature flags); stream observability over WS.
- Surface the DD **CLI** (`config list`, `plugin` discover) behind `/designer/cli/*` for diagnostics (extends the existing agent-context proxy).

### Frontend (`frontend/src/components/factory/DataDesignerTab.tsx`)
- **Seed source selector**: trace-format dropdown (Claude Code / Hermes / dir) backed by `AgentRolloutSeedSource`.
- **New pipeline cards**: `coding_agent_distill`, `deep_research_trajectories` (MCP).
- **Toggles**: "Execute & verify generated code" (validators), "Use real tools (MCP)", "Chain stages (curriculum)".
- **Per-column provider routing** UI (model/provider per LLM column) — fixes the implicit-routing footgun.
- **Live observability** panel: rows generated/filtered, per-column timings, token cost (reuse training WS).
- **Column-DAG preview** from the new introspection endpoint.
- **NL dataset design**: a "Describe your dataset" box that calls DD's coding-agent skill / agent-context to draft a config (already half-wired at `:698-729`).
- Update the "supported column types" callout in `FactoryDashboard.tsx:94` to the full v0.6.1 list.

---

## 6. Risks, constraints, decisions

- **Import surface after the 0.6.1 config/engine split** — verify `data_designer.config` / `data_designer.interface` paths before touching anything else (Phase 0 gate). *Read the repo, don't assume.*
- **Workflow chaining is experimental** (linear only, **no stage resume**) — treat as advanced/opt-in; don't make the default path depend on it.
- **Validators lint, not execute, by default** — real execution requires our local-callable/remote hook into the Docker sandbox. Sandbox the validator runs (reuse `_is_dangerous_command`, network-off) — generated code is untrusted.
- **MCP tool execution** runs real subprocesses during generation — enforce DD's allowlist/budget/timeout *and* our sandbox guardrails; never point it at a non-sandboxed shell.
- **AgentRolloutSeedSource vs our importers** — decision: keep importers for classification/quality, use the seed source only for synthesis seeding (don't lose verification metadata). Documented in Phase 1.
- **Provider keys / Ollama on DGX** — per-column routing to `192.168.50.173:11434/v1` needs the DGX reachable; fall back to NIM. Honors the "no on-the-fly env/dep changes without a plan" rule — this *is* the plan.
- **Teacher model** — recipe uses `nvidia/nemotron-3-super-120b-a12b`; confirm availability on NIM/our budget vs distilling with a local DGX model.
- **Two product lines drift** — the standalone library (`data-designer` 0.6.1) and the microservices PySDK (25.12.0/26.3.0) version independently and may not have identical column/feature sets (Embedding/Custom in point). Pin to the **library** line for Phases 0–5; only touch the microservices SDK in Phase 6, and re-verify feature parity there.
- **Async-engine default** — 0.6.x runs the async engine by default; if any existing pipeline regresses, the `DATA_DESIGNER_ASYNC_ENGINE=0` fallback exists for *one transitional release only* — so fix-forward, don't depend on it.
- **Embedding/Custom/Image columns unconfirmed** — feature-gate everything that depends on them; never hard-require.

### Open questions to resolve during implementation *(flagged by the research pass as under-verified)*
- Exact `data-designer config|plugin` **subcommand surface** in 0.6.x (only `--version`, `config list`, and "plugins stable" are confirmed) — enumerate from the installed CLI (`data-designer --help`).
- Precise **provider/endpoint** enumeration in 0.6.x (Ollama/NIM/NeMo-microservices/OpenAI-compatible) — inference is LiteLLM-based, so OpenAI-compatible is expected; confirm against `concepts/models/model-providers` + the installed package before wiring DGX Ollama.
- Concrete **Data Designer ↔ NeMo Gym / NeMo RL** integration path beyond agent-rollout seeding + Nemotron-PII provenance — not established; treat any tight coupling as research, not a given.

### Decisions needed before Phase 1
1. Replace the hand-rolled seed extractor entirely, or keep it as fallback only? (plan assumes **fallback only**).
2. Distillation teacher: NVIDIA NIM Nemotron-Super vs DGX-local model?
3. Pin `data-designer` as a real (non-optional) dependency now, or keep optional with graceful fallback? (plan keeps **optional**).

---

## 7. Sources
- NeMo Data Designer docs (latest): Columns, Seed Datasets, **Agent Rollout Ingestion**, Validators, **Tool Use & MCP**, Workflow Chaining, Processors, Plugins, Deployment Options, Recipes (Agent Rollout Trace Distillation, Text-to-SQL, Multi-Turn Chat, Deep Research Trajectories), Dev Notes (Nemotron-Personas, Prompt Sensitivity, Retriever SDG, Adaptive Concurrency) — `https://docs.nvidia.com/nemo/datadesigner/latest/llms.txt`
- PyPI `data-designer` **0.6.1** (Apache-2.0; Python 3.10–3.14; `data-designer-config`/`-engine`; `typer` CLI) — `https://pypi.org/pypi/data-designer/json`
- Repo `NVIDIA-NeMo/DataDesigner` (releases, `AGENTS.md`, `README.md`, issue #513 Pi rollout) — `https://github.com/NVIDIA-NeMo/DataDesigner`
- Release notes (verified, with PRs): `https://github.com/NVIDIA-NeMo/DataDesigner/releases/tag/v0.6.0` and `.../tag/v0.6.1`; Pi rollout issue/PR `#513`/`#514`
- Distillation recipe source: `https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/docs/assets/recipes/trace_ingestion/agent_rollout_distillation.py`
- NeMo Microservices PySDK 25.12.0 column-config reference (Validation/Judge/Score/samplers): `https://docs.nvidia.com/nemo/microservices/25.12.0/pysdk/reference/nemo_microservices/nemo_microservices.data_designer.config.column_configs.html`; 26.3.0 SDK resources / migration
- NeMo Microservices Data Designer Service — `https://docs.nvidia.com/nemo/microservices/latest/data-designer/index.html`
- Reference dataset: `https://huggingface.co/datasets/nvidia/Nemotron-PII` (+ blog `https://huggingface.co/blog/nvidia/nemotron-pii`); HF inference-providers integration `https://huggingface.co/docs/inference-providers/integrations/datadesigner`
- Internal: `tasks/status-review-2026-06-11.md` (20-weakness list), `tasks/platform-recap-2026-06-16.md`, memory `project_data_designer_upgrade.md`

> **Docs-host caveat:** the old `nvidia-nemo.github.io/DataDesigner/*` URLs now 404 — content migrated to `docs.nvidia.com/nemo/datadesigner/`. Findings above were verified against repo source at the **v0.6.1 tag** where live docs links had moved. A deep-research adversarial pass (103 agents) confirmed the version line, async-engine default, agent-rollout formats, distillation recipe, validators, judge/Score tooling, and Nemotron-PII provenance; it refuted an "11 column types incl. Image/Embedding/Custom" enumeration (hence the §2.5 feature-gate) and a detailed plugin-CLI lineage (hence the open question).
