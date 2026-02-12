# DataDesigner Integration Design

> Replace BashGym's augmentation layer with NVIDIA NeMo DataDesigner v0.5.0 for structured, validated, scalable synthetic data generation.

**Date**: 2026-02-11
**Status**: Proposed
**Scope**: Factory layer augmentation replacement + multi-source ingestion

---

## Problem

The current `DataFactory.augment_example()` approach has limitations:

1. **Single-prompt augmentation** — one LLM call per variation, no structural diversity
2. **No generation-time validation** — quality scoring happens post-hoc, not during generation
3. **Manual pipeline** — too many steps from traces to training-ready JSONL
4. **Limited sources** — only traces and security datasets; no easy HuggingFace or unstructured data ingestion
5. **No reproducibility** — pipeline configs are code, not declarative artifacts

## Solution

Integrate NVIDIA NeMo DataDesigner v0.5.0 as the synthetic data generation engine. DataDesigner provides column-based DAG execution, statistical samplers, LLM-as-Judge validation, CLI tools, and multi-source seed datasets.

---

## Architecture

### Pipeline Flow

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  BashGym Traces   │   │  HuggingFace /   │   │  Unstructured    │
│  (Gold Sessions)  │   │  CSV / Parquet   │   │  (PDFs, docs,    │
│                   │   │  / JSON files    │   │   code repos)    │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                      │                       │
    ExampleGenerator     DataDesigner              MCP Tool Use
    → seed prompts       SeedDataset               → extract content
         │                      │                  → structure it
         └──────────┬───────────┘                       │
                    ▼                                    │
           DataDesigner Pipeline  ◄──────────────────────┘
           (Columns + Samplers + Validators + Judge)
                    │
                    ▼
            SchemaTransform → NeMo JSONL
                    │
                    ▼
             train.jsonl + val.jsonl
```

### What Changes

| Component | Before | After |
|-----------|--------|-------|
| Augmentation | `DataFactory.augment_example()` — single LLM call | DataDesigner pipeline — column DAG with samplers, validators, judges |
| Diversity | LLM temperature only | Statistical samplers (category, uniform, gaussian) + LLM generation |
| Validation | Post-hoc `quality_calculator.py` | Generation-time LLM-Judge + code validators (Ruff) |
| Format output | Manual JSONL construction | `SchemaTransformProcessor` → NeMo messages format |
| Config | Python code in `DataFactoryConfig` | Declarative YAML pipeline configs |
| External data | Not supported | HuggingFace, CSV, Parquet, JSON seed datasets |
| Unstructured | Not supported | MCP tool calling for PDFs, docs, code |
| CLI | None | `data-designer preview`, `create`, `validate` |

### What Stays the Same

- **Trace import** (`ClaudeSessionImporter`) — unchanged
- **Trace processing** (`TraceProcessor`) — unchanged
- **Example generation** (`ExampleGenerator`) — unchanged, now feeds seed data to DataDesigner
- **NeMo client** (`NeMoClient`) — unchanged, still handles training job submission
- **Security ingester** (`SecurityIngester`) — unchanged
- **Quality calculator** — augmented with Judge results but not replaced
- **API routes structure** — extended, not replaced

---

## Module Design

### New Files

```
bashgym/factory/
├── data_designer.py              # DataDesigner pipeline orchestrator
├── designer_configs/             # YAML pipeline configurations
│   ├── coding_agent_sft.yaml     # SFT: task → solution → validate
│   ├── coding_agent_dpo.yaml     # DPO: good/bad pairs via judge scoring
│   ├── from_huggingface.yaml     # External dataset ingestion template
│   └── from_unstructured.yaml    # MCP-based unstructured ingestion
```

### Core Class: `DataDesignerPipeline`

```python
@dataclass
class PipelineConfig:
    """Configuration for a DataDesigner generation pipeline."""
    config_path: Optional[str] = None        # YAML/JSON config file
    config_url: Optional[str] = None         # Remote config URL
    provider: str = "nvidia"                 # Default LLM provider
    model_alias: str = "nvidia-text"         # Default model
    judge_model_alias: str = "nvidia-text"   # Model for LLM-as-Judge
    num_records: int = 100                   # Records to generate
    buffer_size: int = 100                   # Batch size
    max_parallel_requests: int = 4           # Concurrent LLM calls
    output_dir: Path = Path("data/designer_output")
    seed_source: Optional[str] = None        # HF dataset, file path, or trace dir
    seed_format: str = "auto"                # auto, csv, parquet, json, huggingface
    train_val_split: float = 0.9


class DataDesignerPipeline:
    """Bridge between BashGym and DataDesigner generation."""

    def __init__(self, config: PipelineConfig):
        self.designer = DataDesigner()
        self.config = config

    # --- Entry Points ---

    def from_traces(self, trace_dir: Path, num_records: int) -> Results:
        """Generate training data from gold traces as seed dataset.

        1. Load gold traces via ExampleGenerator
        2. Extract seed prompts and solutions
        3. Build DataDesigner pipeline with trace seeds
        4. Generate augmented examples with statistical diversity
        5. Validate via LLM-Judge + code validators
        """

    def from_dataset(self, source: str, num_records: int,
                     column_mapping: Optional[dict] = None) -> Results:
        """Generate from HuggingFace dataset or local file.

        source: HF dataset name ("bigcode/starcoderdata") or file path
        column_mapping: Map source columns to pipeline columns
        """

    def from_unstructured(self, path: Path, num_records: int) -> Results:
        """Generate from unstructured documents via MCP tool calling.

        Uses DataDesigner's MCP integration to read PDFs, code files,
        or documentation during generation. The LLM reads source material
        and synthesizes instruction-response pairs grounded in content.
        """

    def from_config(self, config_path: str, num_records: int) -> Results:
        """Generate from a custom YAML/JSON pipeline config."""

    # --- Operations ---

    def preview(self, num_records: int = 5) -> Results:
        """Quick preview of generated data."""

    def create(self, num_records: int) -> Results:
        """Full dataset generation."""

    def validate_config(self, config_path: str) -> dict:
        """Validate a pipeline config without running generation."""

    def export_nemo(self, results, output_dir: Path) -> dict:
        """Export results to NeMo train/val JSONL format.

        Uses SchemaTransformProcessor to reshape columns into:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

        Returns: {"train_path": ..., "val_path": ..., "train_count": ..., "val_count": ...}
        """

    def push_to_hub(self, results, repo_id: str) -> str:
        """Publish dataset to HuggingFace Hub with auto-generated card."""

    # --- Pipeline Builders ---

    def _build_sft_pipeline(self, seeds: list[dict]) -> ConfigBuilder:
        """Build SFT pipeline: seed → sample → generate → validate → format."""

    def _build_dpo_pipeline(self, seeds: list[dict]) -> ConfigBuilder:
        """Build DPO pipeline: generate chosen/rejected via judge scoring."""

    def _build_ingestion_pipeline(self, source: str) -> ConfigBuilder:
        """Build external dataset ingestion pipeline."""

    def _build_unstructured_pipeline(self, path: Path) -> ConfigBuilder:
        """Build MCP-based unstructured data pipeline."""
```

---

## Pipeline Configs

### SFT Pipeline (`coding_agent_sft.yaml`)

```yaml
# Coding agent SFT training data generation
# Seed: gold traces or external dataset
# Output: NeMo messages format JSONL

columns:
  # --- Sampling for diversity ---
  task_category:
    type: sampler
    sampler_type: category
    values: [bug_fix, feature, refactor, test, docs, config, debug, optimize]

  complexity:
    type: sampler
    sampler_type: category
    values: [simple, moderate, complex]

  language:
    type: sampler
    sampler_type: category
    values: [python, typescript, javascript, rust, go, bash]

  # --- Generation ---
  task_prompt:
    type: llm_text
    model_alias: "nvidia-text"
    prompt: |
      You are generating training data for a coding AI agent.

      Seed task: {{ seed_task }}
      Category: {{ task_category }}
      Complexity: {{ complexity }}
      Language: {{ language }}

      Generate a realistic, specific coding task prompt that a developer would give
      to an AI coding assistant. The task should be a {{ complexity }} {{ task_category }}
      task in {{ language }}.

      Requirements:
      - Be specific about files, functions, or components involved
      - Include enough context for the agent to understand the codebase
      - Match the style and scope of the seed task
      - Output only the task prompt, nothing else

  solution:
    type: llm_text
    model_alias: "nvidia-text"
    prompt: |
      You are a coding AI agent solving this task:

      {{ task_prompt }}

      Solve this task step by step. Show your work as a sequence of tool calls:
      - Read files to understand context
      - Edit files to make changes
      - Run commands to test
      - Verify the solution works

      Format each step as:
      [TOOL: tool_name] arguments
      [OUTPUT: result]

  # --- Validation ---
  quality_score:
    type: llm_judge
    model_alias: "nvidia-text"
    rubric:
      correctness:
        description: "Does the solution correctly address the task?"
        weight: 0.4
      completeness:
        description: "Are all parts of the task handled?"
        weight: 0.3
      code_quality:
        description: "Is the code clean, idiomatic, and well-structured?"
        weight: 0.2
      tool_usage:
        description: "Does the agent use appropriate tools in a logical sequence?"
        weight: 0.1

# Drop low-quality examples (score < 0.6)
processors:
  - type: schema_transform
    name: nemo_format
    template:
      messages:
        - role: system
          content: "You are a coding AI agent. Solve tasks step by step using available tools."
        - role: user
          content: "{{ task_prompt }}"
        - role: assistant
          content: "{{ solution }}"

run_config:
  buffer_size: 100
  max_parallel_requests: 4
  max_conversation_restarts: 3
  shutdown_error_rate: 0.3
```

### DPO Pipeline (`coding_agent_dpo.yaml`)

```yaml
# DPO preference pair generation
# Uses LLM-Judge to create chosen/rejected pairs

columns:
  task_category:
    type: sampler
    sampler_type: category
    values: [bug_fix, feature, refactor, test, debug]

  task_prompt:
    type: llm_text
    model_alias: "nvidia-text"
    prompt: |
      Seed: {{ seed_task }}
      Category: {{ task_category }}
      Generate a specific {{ task_category }} coding task.

  solution_a:
    type: llm_text
    model_alias: "nvidia-text"
    prompt: "Solve step by step with tool calls: {{ task_prompt }}"

  solution_b:
    type: llm_text
    model_alias: "nvidia-text"
    prompt: "Solve step by step with tool calls: {{ task_prompt }}"

  judge_a:
    type: llm_judge
    model_alias: "nvidia-text"
    rubric:
      correctness: { weight: 0.5 }
      completeness: { weight: 0.3 }
      efficiency: { weight: 0.2 }

  judge_b:
    type: llm_judge
    model_alias: "nvidia-text"
    rubric:
      correctness: { weight: 0.5 }
      completeness: { weight: 0.3 }
      efficiency: { weight: 0.2 }

  # Expression column to pick chosen/rejected
  chosen:
    type: expression
    template: "{% if judge_a > judge_b %}{{ solution_a }}{% else %}{{ solution_b }}{% endif %}"

  rejected:
    type: expression
    template: "{% if judge_a > judge_b %}{{ solution_b }}{% else %}{{ solution_a }}{% endif %}"

processors:
  - type: schema_transform
    name: dpo_format
    template:
      prompt: "{{ task_prompt }}"
      chosen: "{{ chosen }}"
      rejected: "{{ rejected }}"
```

---

## API Integration

### New Endpoints (`factory_routes.py`)

```python
# --- DataDesigner Endpoints ---

@router.post("/api/factory/designer/preview")
async def designer_preview(request: DesignerPreviewRequest):
    """Preview generated data with any pipeline config.

    Request: {config: str, num_records: int, seed_source: optional str}
    Response: {records: list[dict], columns: list[str]}
    """

@router.post("/api/factory/designer/create")
async def designer_create(request: DesignerCreateRequest):
    """Start full dataset generation job.

    Request: {config: str, num_records: int, seed_source: str,
              output_dir: str, provider: str}
    Response: {job_id: str, status: str}
    """

@router.get("/api/factory/designer/jobs/{job_id}")
async def designer_job_status(job_id: str):
    """Get generation job progress.

    Response: {job_id, status, progress_pct, records_generated,
               records_validated, errors, elapsed_time}
    """

@router.get("/api/factory/designer/configs")
async def list_designer_configs():
    """List available pipeline YAML configs.

    Response: {configs: [{name, description, columns, seed_type}]}
    """

@router.post("/api/factory/designer/from-hf")
async def designer_from_huggingface(request: HuggingFaceRequest):
    """Generate training data from a HuggingFace dataset.

    Request: {dataset: str, subset: optional str, split: str,
              num_records: int, column_mapping: dict}
    Response: {job_id: str, status: str}
    """

@router.post("/api/factory/designer/validate")
async def validate_designer_config(request: ValidateRequest):
    """Validate a pipeline config without running generation.

    Request: {config: str (YAML content or path)}
    Response: {valid: bool, errors: list[str], columns: list[str]}
    """

@router.post("/api/factory/designer/push-to-hub")
async def push_to_huggingface(request: PushToHubRequest):
    """Publish generated dataset to HuggingFace Hub.

    Request: {job_id: str, repo_id: str, private: bool}
    Response: {url: str, card_generated: bool}
    """
```

---

## CLI Integration

### New Subcommands (`main.py`)

```bash
# Preview with built-in config
python main.py designer preview --config coding_agent_sft
python main.py designer preview --config coding_agent_sft --num-records 10

# Full generation from traces
python main.py designer create --config coding_agent_sft \
  --seed-source data/gold_traces/ --num-records 5000

# Generate from HuggingFace dataset
python main.py designer create --config from_huggingface \
  --seed-source "bigcode/starcoderdata" --num-records 1000

# Generate from unstructured docs
python main.py designer create --config from_unstructured \
  --seed-source ./docs/ --num-records 500

# Custom YAML config
python main.py designer create --config path/to/custom.yaml --num-records 2000

# Validate config
python main.py designer validate --config path/to/custom.yaml

# Direct passthrough to data-designer CLI
python main.py designer raw preview my_config.yaml
python main.py designer raw create my_config.yaml --num-records 1000

# Export to HuggingFace Hub
python main.py designer push-to-hub --job-id <id> --repo "user/dataset-name"
```

---

## Configuration

### Provider Setup

DataDesigner providers configured via `data-designer config providers` CLI or environment variables:

```bash
# NVIDIA (default, free tier via build.nvidia.com)
export NVIDIA_API_KEY="nvapi-..."

# Anthropic (via custom provider - OpenAI-compatible)
# Requires anthropic-openai-proxy or litellm
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Self-hosted vLLM
# Configure via: data-designer config providers
```

### BashGym Settings Integration

```python
# settings.py additions
class DataDesignerSettings:
    default_provider: str = "nvidia"
    default_model_alias: str = "nvidia-text"
    judge_model_alias: str = "nvidia-text"
    default_num_records: int = 100
    default_buffer_size: int = 100
    max_parallel_requests: int = 4
    configs_dir: Path = Path("bashgym/factory/designer_configs")
    output_dir: Path = Path("data/designer_output")
```

---

## Dependencies

```
# requirements.txt addition
data-designer>=0.5.0
```

DataDesigner's dependencies are lightweight — it's an orchestration layer that calls external LLM APIs. No GPU requirements for the generation itself.

---

## Migration Path

1. **Phase 1**: Add `data-designer` dependency, create `data_designer.py` module and YAML configs
2. **Phase 2**: Add API endpoints and CLI subcommands
3. **Phase 3**: Wire up dashboard UI (DataDesigner tab on Training page)
4. **Phase 4**: Deprecate `DataFactory.augment_example()` and `_augment_with_*` methods
5. **Phase 5**: Remove deprecated methods in next major version

Existing `DataFactory` methods (`process_gold_trace`, `generate_dpo_pairs`, `save_training_batch`) continue working. Only the augmentation path changes.

---

## Success Criteria

- Generate 1000 diverse SFT examples from 50 gold traces in under 30 minutes
- LLM-Judge quality scores average above 0.7 for generated examples
- `data-designer preview` returns results in under 30 seconds
- HuggingFace dataset import works with any text/code dataset
- All existing tests continue passing
- New tests cover DataDesigner pipeline, API endpoints, and CLI commands
