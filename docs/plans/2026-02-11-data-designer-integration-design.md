# DataDesigner Integration & Orchestration Agent Design

> A unified design for (1) replacing BashGym's augmentation layer with NVIDIA NeMo DataDesigner v0.5.0 and (2) building an orchestration agent that supervises the platform, decomposes specs, and distributes work to parallel Claude Code sessions.

**Date**: 2026-02-11
**Status**: Proposed
**Scope**: Factory layer replacement + multi-source ingestion + orchestration agent

---

## Table of Contents

1. [Problems](#problems)
2. [Part 1: DataDesigner Integration](#part-1-datadesigner-integration)
3. [Part 2: Orchestration Agent](#part-2-orchestration-agent)
4. [Migration Path](#migration-path)
5. [Success Criteria](#success-criteria)

---

## Problems

### Data Pipeline

1. **Single-prompt augmentation** -- one LLM call per variation, no structural diversity
2. **No generation-time validation** -- quality scoring happens post-hoc, not during generation
3. **Manual pipeline** -- too many steps from traces to training-ready JSONL
4. **Limited sources** -- only traces and security datasets; no HuggingFace or unstructured data
5. **No reproducibility** -- pipeline configs are code, not declarative artifacts

### Platform Orchestration

6. **No task decomposition** -- users must manually break work into small tasks
7. **No parallel execution** -- one agent, one task at a time
8. **No spec-to-implementation flow** -- no way to hand a full spec and get coordinated work
9. **Training data from orchestration** -- multi-agent traces are high-value training signal that we don't capture

---

# Part 1: DataDesigner Integration

## Architecture

### Pipeline Flow

```
+-----------------+   +-----------------+   +-----------------+
| BashGym Traces  |   | HuggingFace /   |   | Unstructured    |
| (Gold Sessions) |   | CSV / Parquet   |   | (PDFs, docs,    |
|                 |   | / JSON files    |   |  code repos)    |
+-------+---------+   +-------+---------+   +-------+---------+
        |                      |                     |
   ExampleGenerator     DataDesigner            MCP Tool Use
   -> seed prompts      SeedDataset            -> extract content
        |                      |                     |
        +----------+-----------+                     |
                   v                                 |
          DataDesigner Pipeline  <--------------------+
          (Columns + Samplers + Validators + Judge)
                   |
                   v
           SchemaTransform -> NeMo JSONL
                   |
                   v
            train.jsonl + val.jsonl
```

### What Changes

| Component | Before | After |
|-----------|--------|-------|
| Augmentation | `DataFactory.augment_example()` -- single LLM call | DataDesigner `ConfigBuilder` -- column DAG with samplers, validators, judges |
| Diversity | LLM temperature only | Statistical samplers (category, gaussian, uniform, scipy) + LLM generation |
| Validation | Post-hoc `quality_calculator.py` | Generation-time LLM-Judge + code validators (Ruff/SQLFluff) + processor filtering |
| Code generation | `LLMTextColumnConfig` | `LLMCodeColumnConfig` with auto code extraction from markdown fences |
| Tool-use data | Not supported | `LLMStructuredColumnConfig` with Pydantic schemas for tool call JSON |
| Format output | Manual JSONL construction | `SchemaTransformProcessor` -> NeMo messages format |
| Config | Python code in `DataFactoryConfig` | `DataDesignerConfigBuilder` Python API (serializable to YAML/JSON) |
| External data | Not supported | `HuggingFaceSeedSource`, `FileSeedSource`, `DataFrameSeedSource` |
| Unstructured | Not supported | MCP tool calling for PDFs, docs, code |
| CLI | None | `data-designer preview`, `create`, `validate` |
| Publishing | Not supported | `results.push_to_hub()` to HuggingFace Hub |

### What Stays the Same

- **Trace import** (`ClaudeSessionImporter`) -- unchanged
- **Trace processing** (`TraceProcessor`) -- unchanged
- **Example generation** (`ExampleGenerator`) -- unchanged, now feeds seed data to DataDesigner
- **NeMo client** (`NeMoClient`) -- unchanged, still handles training job submission
- **Security ingester** (`SecurityIngester`) -- unchanged
- **Quality calculator** -- augmented with Judge results but not replaced
- **API routes structure** -- extended, not replaced

---

## Module Design

### New Files

```
bashgym/factory/
  data_designer.py           # DataDesigner pipeline orchestrator
  designer_pipelines/        # Pre-built pipeline builders (Python modules)
    __init__.py
    coding_agent_sft.py      # SFT: task -> solution -> validate
    coding_agent_dpo.py      # DPO: good/bad pairs via judge scoring
    tool_use_sft.py          # Tool-use training data with Pydantic schemas
    from_external.py         # External dataset ingestion
    from_unstructured.py     # MCP-based unstructured ingestion
```

### Core Class: `DataDesignerPipeline`

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner
from pydantic import BaseModel, Field

@dataclass
class PipelineConfig:
    """Configuration for a DataDesigner generation pipeline."""
    pipeline: str = "coding_agent_sft"       # Pipeline name or path to config
    provider: str = "nvidia"                 # Default LLM provider
    text_model: str = "nvidia-text"          # Model for text generation
    code_model: str = "nvidia-text"          # Model for code generation
    judge_model: str = "nvidia-text"         # Model for LLM-as-Judge
    num_records: int = 100                   # Records to generate
    buffer_size: int = 100                   # Batch size
    max_parallel_requests: int = 4           # Concurrent LLM calls
    output_dir: Path = Path("data/designer_output")
    seed_source: Optional[str] = None        # HF dataset, file path, or trace dir
    train_val_split: float = 0.9
    temperature_text: float = 0.85           # Creative text generation
    temperature_code: float = 0.2            # Deterministic code generation
    temperature_judge: float = 0.1           # Consistent evaluation


class DataDesignerPipeline:
    """Bridge between BashGym and DataDesigner generation."""

    def __init__(self, config: PipelineConfig):
        self.designer = DataDesigner()
        self.config = config

    # --- Entry Points ---

    def from_traces(self, trace_dir: Path, num_records: int) -> pd.DataFrame:
        """Generate training data from gold traces as seed dataset."""
        seeds = self._extract_seeds_from_traces(trace_dir)
        seed_df = pd.DataFrame(seeds)
        builder = self._get_pipeline_builder()
        builder.with_seed_dataset(dd.DataFrameSeedSource(data=seed_df))
        return self.designer.generate(builder, num_rows=num_records)

    def from_dataset(self, source: str, num_records: int,
                     column_mapping: Optional[dict] = None) -> pd.DataFrame:
        """Generate from HuggingFace dataset or local file."""
        builder = self._get_pipeline_builder()
        if source.startswith("datasets/") or "/" in source and not Path(source).exists():
            builder.with_seed_dataset(dd.HuggingFaceSeedSource(path=source))
        else:
            builder.with_seed_dataset(dd.FileSeedSource(path=source))
        return self.designer.generate(builder, num_rows=num_records)

    def from_unstructured(self, path: Path, num_records: int) -> pd.DataFrame:
        """Generate from unstructured documents via MCP tool calling."""

    def from_config(self, config_path: str, num_records: int) -> pd.DataFrame:
        """Generate from a custom config file (YAML/JSON/Python)."""

    # --- Operations ---

    def preview(self, num_records: int = 5) -> pd.DataFrame:
        """Quick preview of generated data."""
        builder = self._get_pipeline_builder()
        return self.designer.preview(builder, num_rows=num_records)

    def validate(self) -> dict:
        """Validate pipeline config without running generation."""
        builder = self._get_pipeline_builder()
        return self.designer.validate(builder)

    def export_nemo(self, df: pd.DataFrame, output_dir: Path) -> dict:
        """Export to NeMo train/val JSONL with schema transform."""

    def push_to_hub(self, df: pd.DataFrame, repo_id: str) -> str:
        """Publish dataset to HuggingFace Hub."""

    # --- Pipeline Resolution ---

    def _get_pipeline_builder(self) -> dd.DataDesignerConfigBuilder:
        """Resolve pipeline name to a ConfigBuilder."""
        from bashgym.factory.designer_pipelines import PIPELINES
        if self.config.pipeline in PIPELINES:
            return PIPELINES[self.config.pipeline](self.config)
        # Treat as file path
        return dd.DataDesignerConfigBuilder.from_config(
            json.load(open(self.config.pipeline))
        )
```

---

## Pipeline Implementations

### SFT Pipeline (`coding_agent_sft.py`)

Uses the actual DataDesigner Python API with proper column types, temperature tuning, and validation.

```python
import data_designer.config as dd
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Structured tool call for training data."""
    tool: str = Field(description="Tool name: read, edit, bash, write, glob, grep")
    arguments: str = Field(description="Tool arguments")
    reasoning: str = Field(description="Why this tool call is needed")


class AgentStep(BaseModel):
    """Single step in an agent solution."""
    thought: str = Field(description="Agent reasoning before action")
    tool_call: ToolCall = Field(description="The tool invocation")
    observation: str = Field(description="Expected tool output")


class AgentSolution(BaseModel):
    """Complete agent solution with tool-use trajectory."""
    plan: str = Field(description="High-level approach")
    steps: list[AgentStep] = Field(description="Ordered list of tool calls")
    summary: str = Field(description="What was accomplished")


def build_sft_pipeline(config) -> dd.DataDesignerConfigBuilder:
    """Build the coding agent SFT training data pipeline."""

    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="text-model",
                model="meta/llama-3.3-70b-instruct",
                inference_parameters=dd.InferenceParameters(
                    temperature=config.temperature_text,
                    top_p=0.99,
                    max_tokens=2048,
                ),
            ),
            dd.ModelConfig(
                alias="code-model",
                model="qwen/qwen2.5-coder-32b-instruct",
                inference_parameters=dd.InferenceParameters(
                    temperature=config.temperature_code,
                    top_p=0.95,
                    max_tokens=4096,
                ),
            ),
            dd.ModelConfig(
                alias="judge-model",
                model="meta/llama-3.3-70b-instruct",
                inference_parameters=dd.InferenceParameters(
                    temperature=config.temperature_judge,
                    max_tokens=1024,
                ),
            ),
        ],
        model_providers=[
            dd.ModelProvider(
                name="nvidia-nim",
                endpoint="https://integrate.api.nvidia.com/v1",
                provider_type="openai",
                api_key="${NVIDIA_API_KEY}",
            ),
        ],
    )

    # --- Sampling for statistical diversity ---

    builder.add_column(dd.SamplerColumnConfig(
        name="task_category",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["bug_fix", "feature", "refactor", "test", "docs",
                    "config", "debug", "optimize", "security_fix"],
            weights=[0.2, 0.25, 0.15, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="complexity",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["simple", "moderate", "complex"],
            weights=[0.3, 0.5, 0.2],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="language",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["python", "typescript", "javascript", "rust", "go", "bash"],
            weights=[0.35, 0.2, 0.15, 0.1, 0.1, 0.1],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="codebase_size",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["single_file", "small_project", "medium_project"],
            weights=[0.3, 0.5, 0.2],
        ),
    ))

    # --- Task prompt generation ---

    builder.add_column(dd.LLMTextColumnConfig(
        name="task_prompt",
        model_alias="text-model",
        prompt=(
            "You are generating training data for a coding AI agent.\n\n"
            "Seed task: {{ seed_task }}\n"
            "Category: {{ task_category }}\n"
            "Complexity: {{ complexity }}\n"
            "Language: {{ language }}\n"
            "Codebase size: {{ codebase_size }}\n\n"
            "Generate a realistic, specific coding task prompt that a developer "
            "would give to an AI coding assistant. The task should be a "
            "{{ complexity }} {{ task_category }} task in {{ language }}.\n\n"
            "Requirements:\n"
            "- Be specific about files, functions, or components involved\n"
            "- Include enough context for the agent to understand the codebase\n"
            "- Match the style and scope of the seed task\n"
            "- Output ONLY the task prompt, nothing else"
        ),
    ))

    # --- Solution generation as structured tool-use trajectory ---

    builder.add_column(dd.LLMStructuredColumnConfig(
        name="solution",
        model_alias="code-model",
        prompt=(
            "You are a coding AI agent. Solve this task step by step.\n\n"
            "Task: {{ task_prompt }}\n\n"
            "Show your complete solution as a sequence of tool calls. "
            "Available tools: read (read files), edit (modify files), "
            "bash (run commands), write (create files), glob (find files), "
            "grep (search content).\n\n"
            "Think carefully about each step. Provide your reasoning, "
            "the exact tool call, and the expected observation."
        ),
        output_format=AgentSolution,
    ))

    # --- Code extraction (if solution contains code blocks) ---

    builder.add_column(dd.ExpressionColumnConfig(
        name="solution_text",
        expr=(
            "Plan: {{ solution.plan }}\n\n"
            "{% for step in solution.steps %}"
            "Step {{ loop.index }}:\n"
            "Thought: {{ step.thought }}\n"
            "[{{ step.tool_call.tool }}] {{ step.tool_call.arguments }}\n"
            "Output: {{ step.observation }}\n\n"
            "{% endfor %}"
            "Summary: {{ solution.summary }}"
        ),
    ))

    # --- Quality validation ---

    builder.add_column(dd.LLMJudgeColumnConfig(
        name="quality_score",
        model_alias="judge-model",
        prompt=(
            "Evaluate this coding agent solution:\n\n"
            "Task: {{ task_prompt }}\n\n"
            "Solution:\n{{ solution_text }}"
        ),
        scores=[
            dd.Score(
                name="correctness",
                description="Does the solution correctly address the task?",
                options={
                    "5": "Perfect -- addresses every aspect correctly",
                    "4": "Minor issues that don't affect functionality",
                    "3": "Works but misses some requirements",
                    "2": "Significant logic errors",
                    "1": "Does not address the task",
                },
            ),
            dd.Score(
                name="tool_usage",
                description="Does the agent use appropriate tools in a logical sequence?",
                options={
                    "5": "Optimal tool usage -- reads before editing, tests after changes",
                    "4": "Good tool usage with minor inefficiencies",
                    "3": "Acceptable but could be improved",
                    "2": "Illogical tool sequence",
                    "1": "Wrong tools or missing critical steps",
                },
            ),
            dd.Score(
                name="completeness",
                description="Is the solution thorough and complete?",
                options={
                    "5": "Handles all requirements including edge cases",
                    "4": "Handles main requirements, minor gaps",
                    "3": "Partially complete",
                    "2": "Missing major parts",
                    "1": "Incomplete",
                },
            ),
        ],
    ))

    # --- Filter low-quality examples ---

    builder.add_processor(
        processor_type="filter",
        condition="quality_score.correctness >= 3 and quality_score.tool_usage >= 3",
    )

    # --- Drop intermediate columns from output ---

    # The solution structured object is kept for reference but solution_text
    # is the training-ready format

    return builder
```

### DPO Pipeline (`coding_agent_dpo.py`)

```python
def build_dpo_pipeline(config) -> dd.DataDesignerConfigBuilder:
    """Build DPO preference pair pipeline.

    Generates two solutions for each task, judges both,
    and uses expression columns to assign chosen/rejected.
    """

    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="text-model",
                model="meta/llama-3.3-70b-instruct",
                inference_parameters=dd.InferenceParameters(
                    temperature=0.85,
                    max_tokens=2048,
                ),
            ),
            dd.ModelConfig(
                alias="solution-model-a",
                model="meta/llama-3.3-70b-instruct",
                inference_parameters=dd.InferenceParameters(
                    temperature=0.9,   # Higher temp for more variation
                    max_tokens=4096,
                ),
            ),
            dd.ModelConfig(
                alias="solution-model-b",
                model="meta/llama-3.3-70b-instruct",
                inference_parameters=dd.InferenceParameters(
                    temperature=0.5,   # Lower temp for different style
                    max_tokens=4096,
                ),
            ),
            dd.ModelConfig(
                alias="judge-model",
                model="meta/llama-3.3-70b-instruct",
                inference_parameters=dd.InferenceParameters(
                    temperature=0.1,
                    max_tokens=1024,
                ),
            ),
        ],
        model_providers=[
            dd.ModelProvider(
                name="nvidia-nim",
                endpoint="https://integrate.api.nvidia.com/v1",
                provider_type="openai",
                api_key="${NVIDIA_API_KEY}",
            ),
        ],
    )

    # Samplers for diversity
    builder.add_column(dd.SamplerColumnConfig(
        name="task_category",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["bug_fix", "feature", "refactor", "test", "debug"],
        ),
    ))

    # Task prompt
    builder.add_column(dd.LLMTextColumnConfig(
        name="task_prompt",
        model_alias="text-model",
        prompt=(
            "Seed: {{ seed_task }}\nCategory: {{ task_category }}\n"
            "Generate a specific {{ task_category }} coding task."
        ),
    ))

    # Two independent solutions with different temperatures
    builder.add_column(dd.LLMTextColumnConfig(
        name="solution_a",
        model_alias="solution-model-a",
        prompt="Solve step by step with tool calls:\n\n{{ task_prompt }}",
    ))

    builder.add_column(dd.LLMTextColumnConfig(
        name="solution_b",
        model_alias="solution-model-b",
        prompt="Solve step by step with tool calls:\n\n{{ task_prompt }}",
    ))

    # Judge both solutions
    builder.add_column(dd.LLMJudgeColumnConfig(
        name="judge_a",
        model_alias="judge-model",
        prompt="Evaluate:\n\nTask: {{ task_prompt }}\n\nSolution:\n{{ solution_a }}",
        scores=[
            dd.Score(name="quality", description="Overall solution quality",
                     options={"5": "Excellent", "4": "Good", "3": "Acceptable",
                              "2": "Below average", "1": "Poor"}),
        ],
    ))

    builder.add_column(dd.LLMJudgeColumnConfig(
        name="judge_b",
        model_alias="judge-model",
        prompt="Evaluate:\n\nTask: {{ task_prompt }}\n\nSolution:\n{{ solution_b }}",
        scores=[
            dd.Score(name="quality", description="Overall solution quality",
                     options={"5": "Excellent", "4": "Good", "3": "Acceptable",
                              "2": "Below average", "1": "Poor"}),
        ],
    ))

    # Expression columns to pick chosen/rejected
    builder.add_column(dd.ExpressionColumnConfig(
        name="chosen",
        expr="{% if judge_a.quality >= judge_b.quality %}{{ solution_a }}{% else %}{{ solution_b }}{% endif %}",
    ))

    builder.add_column(dd.ExpressionColumnConfig(
        name="rejected",
        expr="{% if judge_a.quality >= judge_b.quality %}{{ solution_b }}{% else %}{{ solution_a }}{% endif %}",
    ))

    # Filter: only keep pairs where scores differ (meaningful preference signal)
    builder.add_processor(
        processor_type="filter",
        condition="judge_a.quality != judge_b.quality",
    )

    return builder
```

### Tool-Use SFT Pipeline (`tool_use_sft.py`)

Generates structured tool-call training data with Pydantic schema enforcement.

```python
from pydantic import BaseModel, Field
from typing import Literal


class ToolCallRequest(BaseModel):
    """A single tool call request from the agent."""
    name: Literal["read", "edit", "bash", "write", "glob", "grep"]
    arguments: dict = Field(description="Tool-specific arguments")


class ToolCallResponse(BaseModel):
    """Expected response from a tool call."""
    success: bool
    output: str = Field(description="Tool output or error message")


class ToolUseConversation(BaseModel):
    """A complete tool-use conversation turn."""
    user_request: str
    agent_thought: str
    tool_calls: list[ToolCallRequest]
    tool_responses: list[ToolCallResponse]
    agent_response: str


def build_tool_use_pipeline(config) -> dd.DataDesignerConfigBuilder:
    """Build pipeline for structured tool-use training data."""

    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="main-model",
                model="meta/llama-3.3-70b-instruct",
                inference_parameters=dd.InferenceParameters(
                    temperature=0.7,
                    max_tokens=4096,
                ),
            ),
        ],
        model_providers=[
            dd.ModelProvider(
                name="nvidia-nim",
                endpoint="https://integrate.api.nvidia.com/v1",
                provider_type="openai",
                api_key="${NVIDIA_API_KEY}",
            ),
        ],
    )

    builder.add_column(dd.SamplerColumnConfig(
        name="tool_focus",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["read_heavy", "edit_heavy", "bash_heavy", "mixed"],
            weights=[0.2, 0.3, 0.2, 0.3],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="num_tools",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["1", "2", "3", "4", "5"],
            weights=[0.1, 0.2, 0.3, 0.25, 0.15],
        ),
    ))

    builder.add_column(dd.LLMStructuredColumnConfig(
        name="conversation",
        model_alias="main-model",
        prompt=(
            "Generate a realistic coding AI agent conversation that uses "
            "{{ num_tools }} tool calls. The conversation should focus on "
            "{{ tool_focus }} operations.\n\n"
            "Seed context: {{ seed_task }}\n\n"
            "The user request should be a natural coding task. The agent should "
            "think about what to do, make tool calls, observe results, and "
            "provide a helpful response."
        ),
        output_format=ToolUseConversation,
    ))

    return builder
```

---

## API Integration

### New Endpoints (`factory_routes.py`)

```python
@router.post("/api/factory/designer/preview")
async def designer_preview(request: DesignerPreviewRequest):
    """Preview generated data with any pipeline config.
    Request: {pipeline: str, num_records: int, seed_source: optional str}
    Response: {records: list[dict], columns: list[str]}
    """

@router.post("/api/factory/designer/create")
async def designer_create(request: DesignerCreateRequest):
    """Start full dataset generation job (background).
    Request: {pipeline: str, num_records: int, seed_source: str,
              output_dir: str, provider: str}
    Response: {job_id: str, status: str}
    """

@router.get("/api/factory/designer/jobs/{job_id}")
async def designer_job_status(job_id: str):
    """Get generation job progress.
    Response: {job_id, status, progress_pct, records_generated,
               records_validated, errors, elapsed_time}
    """

@router.get("/api/factory/designer/pipelines")
async def list_pipelines():
    """List available pipeline builders.
    Response: {pipelines: [{name, description, columns, seed_type}]}
    """

@router.post("/api/factory/designer/from-hf")
async def designer_from_huggingface(request: HuggingFaceRequest):
    """Generate training data from a HuggingFace dataset.
    Request: {dataset: str, subset: str, split: str,
              num_records: int, column_mapping: dict}
    Response: {job_id: str, status: str}
    """

@router.post("/api/factory/designer/validate")
async def validate_config(request: ValidateRequest):
    """Validate a pipeline config without running.
    Request: {pipeline: str}
    Response: {valid: bool, errors: list[str], columns: list[str]}
    """

@router.post("/api/factory/designer/push-to-hub")
async def push_to_hub(request: PushToHubRequest):
    """Publish generated dataset to HuggingFace Hub.
    Request: {job_id: str, repo_id: str, private: bool}
    Response: {url: str, card_generated: bool}
    """
```

---

## CLI Integration

```bash
# Preview with built-in pipeline
python main.py designer preview --pipeline coding_agent_sft
python main.py designer preview --pipeline coding_agent_sft --num-records 10

# Full generation from traces
python main.py designer create --pipeline coding_agent_sft \
  --seed-source data/gold_traces/ --num-records 5000

# Generate from HuggingFace dataset
python main.py designer create --pipeline from_external \
  --seed-source "bigcode/starcoderdata" --num-records 1000

# Direct data-designer CLI passthrough
python main.py designer raw preview my_config.yaml
python main.py designer raw create my_config.yaml --num-records 1000
```

---

## Configuration

### Provider Setup

DataDesigner uses LiteLLM under the hood -- any OpenAI-compatible endpoint works.

```python
# In pipeline builders, providers are configured as:
dd.ModelProvider(
    name="nvidia-nim",
    endpoint="https://integrate.api.nvidia.com/v1",
    provider_type="openai",
    api_key="${NVIDIA_API_KEY}",       # Env var reference, never plaintext
)

# For Anthropic (via litellm):
dd.ModelProvider(
    name="anthropic",
    endpoint="https://api.anthropic.com/v1",
    provider_type="openai",            # litellm handles translation
    api_key="${ANTHROPIC_API_KEY}",
)

# For local vLLM/Ollama:
dd.ModelProvider(
    name="local",
    endpoint="http://localhost:11434/v1",
    provider_type="openai",
    api_key="not-needed",
)
```

### Temperature Strategy

| Task Type | Temperature | Rationale |
|-----------|-------------|-----------|
| Creative text (task prompts, reviews) | 0.8 -- 0.95 | Maximum diversity |
| Code generation (solutions) | 0.1 -- 0.3 | Deterministic correctness |
| LLM-as-Judge evaluation | 0.05 -- 0.15 | Consistent scoring |
| Structured JSON output | 0.3 -- 0.7 | Balance diversity and validity |
| DPO solution variants | 0.5 vs 0.9 | Intentional quality spread |

### Dependencies

```
# requirements.txt addition
data-designer>=0.5.0
```

DataDesigner is lightweight -- it's an orchestration layer using LiteLLM for LLM calls. No GPU requirements for generation.

---

# Part 2: Orchestration Agent

## Concept

An orchestration agent that supervises BashGym, accepts development specs from users, decomposes them into a task DAG, and distributes work to multiple parallel Claude Code sessions.

## Technical Foundation

### Claude Code SDK (`claude-agent-sdk`)

The Agent SDK provides programmatic access to the full Claude Code toolset:

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async for message in query(
    prompt="Fix the bug in auth.py",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
        max_turns=30,
        max_budget_usd=5.00,
        permission_mode="acceptEdits",
    ),
):
    if hasattr(message, "result"):
        print(message.result)
```

### Claude Code CLI Headless Mode

For subprocess-based parallel execution:

```bash
claude -p "Fix the bug" \
  --output-format json \
  --max-turns 30 \
  --max-budget-usd 5.00 \
  --allowedTools "Read,Edit,Bash,Glob,Grep" \
  --dangerously-skip-permissions
```

Returns structured JSON with `session_id` for resumption.

### Key Capabilities

| Capability | Mechanism |
|-----------|-----------|
| Parallel sessions | Each `claude -p` is independent subprocess |
| Session resumption | `--resume <session_id>` continues with full context |
| Budget control | `--max-budget-usd` per worker |
| Turn limits | `--max-turns` per worker |
| Structured output | `--output-format json` + `--json-schema` |
| Tool restrictions | `--allowedTools` / `--disallowedTools` |
| Streaming | `--output-format stream-json --verbose` |
| Custom prompts | `--system-prompt` / `--append-system-prompt` |

## Architecture

```
+-----------------------------------------------------------+
|                    USER SUBMITS SPEC                       |
|        "Build a REST API for task management"              |
+----------------------------+------------------------------+
                             |
+----------------------------v------------------------------+
|               ORCHESTRATOR AGENT (Opus 4.6)               |
|                                                            |
|  Phase 1: PLAN                                             |
|    - Analyze spec with Anthropic Messages API              |
|    - Decompose into Task DAG (nodes + dependencies)        |
|    - Present plan to user for approval                     |
|                                                            |
|  Phase 2: DISPATCH                                         |
|    - Create git worktree per independent task branch       |
|    - Spawn workers (claude -p) via asyncio subprocess      |
|    - Assign tasks respecting dependency order              |
|                                                            |
|  Phase 3: MONITOR                                          |
|    - Stream worker progress via WebSocket                  |
|    - Handle failures: retry with modified prompt           |
|    - Detect blockers, reassign if needed                   |
|                                                            |
|  Phase 4: SYNTHESIZE                                       |
|    - Collect results from all workers                      |
|    - Merge git worktrees (resolve conflicts if needed)     |
|    - Run verification tests                                |
|    - Feed all traces into training pipeline                |
+------+----------------+----------------+---------+---------+
       |                |                |         |
+------v------+  +------v------+  +------v------+  |
| Worker A    |  | Worker B    |  | Worker C    |  |
| (Sonnet)    |  | (Sonnet)    |  | (Sonnet)    |  |
|             |  |             |  |             |  |
| git worktree|  | git worktree|  | git worktree|  |
| feature/api |  | feature/db  |  | feature/test|  |
|             |  |             |  |             |  |
| Budget: $2  |  | Budget: $3  |  | Budget: $1  |  |
| Turns: 30   |  | Turns: 50   |  | Turns: 20   |  |
+------+------+  +------+------+  +------+------+  |
       |                |                |          |
       v                v                v          v
   Traces ----------> Factory Pipeline ---------> Training
```

## Module Structure

```
bashgym/orchestrator/
  __init__.py
  agent.py            # OrchestrationAgent -- supervisor brain
  task_dag.py         # TaskNode, TaskDAG, topological sort, dependency resolution
  dispatcher.py       # WorkerPool, worker spawning, lifecycle management
  worktree.py         # Git worktree create/delete/merge
  synthesizer.py      # Result aggregation, merge conflict resolution
  prompts.py          # System prompts for orchestrator and workers
  models.py           # Dataclasses: Spec, TaskNode, WorkerConfig, WorkerResult
```

## Core Classes

### `models.py` -- Data Model

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    CRITICAL = 1    # Blocks many others
    HIGH = 2        # Core functionality
    NORMAL = 3      # Standard features
    LOW = 4         # Nice-to-have


@dataclass
class OrchestratorSpec:
    """User-submitted development specification."""
    title: str
    description: str
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    repository: Optional[str] = None
    base_branch: str = "main"


@dataclass
class TaskNode:
    """Single task in the decomposition DAG."""
    id: str
    title: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)  # Task IDs
    files_touched: list[str] = field(default_factory=list)  # Expected files
    estimated_turns: int = 20
    budget_usd: float = 2.0
    worker_prompt: str = ""               # Specific prompt for the worker
    worker_id: Optional[str] = None       # Assigned worker session ID
    worktree_path: Optional[Path] = None
    result: Optional['WorkerResult'] = None


@dataclass
class WorkerConfig:
    """Configuration for a worker Claude Code session."""
    model: str = "sonnet"
    max_turns: int = 30
    max_budget_usd: float = 5.0
    allowed_tools: list[str] = field(default_factory=lambda: [
        "Read", "Edit", "Write", "Bash", "Glob", "Grep"
    ])
    system_prompt_append: str = ""
    worktree_path: Optional[Path] = None


@dataclass
class WorkerResult:
    """Result from a completed worker session."""
    task_id: str
    session_id: str
    success: bool
    output: str
    exit_code: int
    duration_seconds: float
    tokens_used: int
    cost_usd: float
    trace_path: Optional[Path] = None
    files_modified: list[str] = field(default_factory=list)
```

### `task_dag.py` -- Task Decomposition

```python
class TaskDAG:
    """Directed acyclic graph of tasks with dependency resolution."""

    def __init__(self):
        self.nodes: dict[str, TaskNode] = {}

    def add_task(self, task: TaskNode) -> None:
        """Add a task node to the DAG."""

    def get_ready_tasks(self) -> list[TaskNode]:
        """Get tasks whose dependencies are all completed.
        Returns tasks in priority order.
        """

    def topological_sort(self) -> list[TaskNode]:
        """Return tasks in valid execution order."""

    def mark_completed(self, task_id: str, result: WorkerResult) -> list[TaskNode]:
        """Mark task complete and return newly unblocked tasks."""

    def mark_failed(self, task_id: str, error: str) -> list[TaskNode]:
        """Mark task failed. Returns tasks that are now blocked."""

    def get_critical_path(self) -> list[TaskNode]:
        """Calculate the longest dependency chain (minimum total time)."""

    def detect_file_conflicts(self) -> list[tuple[str, str]]:
        """Find task pairs that touch the same files (potential merge conflicts)."""

    @classmethod
    async def from_spec(cls, spec: OrchestratorSpec, api_key: str) -> 'TaskDAG':
        """Use Opus to decompose a spec into a TaskDAG.
        Calls Anthropic Messages API with structured output.
        """
```

### `dispatcher.py` -- Worker Management

```python
class WorkerPool:
    """Manages parallel Claude Code worker sessions."""

    def __init__(self, max_workers: int = 5, default_config: WorkerConfig = None):
        self.max_workers = max_workers
        self.default_config = default_config or WorkerConfig()
        self.active_workers: dict[str, asyncio.subprocess.Process] = {}
        self.results: dict[str, WorkerResult] = {}

    async def spawn_worker(
        self,
        task: TaskNode,
        config: Optional[WorkerConfig] = None,
    ) -> str:
        """Spawn a new Claude Code worker for a task.

        Creates a subprocess running:
            claude -p "<prompt>" --output-format json
                --max-turns N --max-budget-usd N
                --allowedTools "..." --dangerously-skip-permissions

        Returns: worker session ID
        """

    async def wait_for_worker(self, worker_id: str, timeout: float = 600) -> WorkerResult:
        """Wait for a specific worker to complete."""

    async def wait_for_any(self, timeout: float = 600) -> WorkerResult:
        """Wait for any active worker to complete. Returns first result."""

    async def cancel_worker(self, worker_id: str) -> None:
        """Cancel a running worker."""

    async def get_worker_output_stream(self, worker_id: str) -> AsyncIterator[str]:
        """Stream worker output in real-time (for WebSocket forwarding)."""

    @property
    def available_slots(self) -> int:
        return self.max_workers - len(self.active_workers)
```

### `worktree.py` -- Git Worktree Management

```python
class WorktreeManager:
    """Manages git worktrees for task isolation."""

    def __init__(self, repo_path: Path, worktree_base: Path = None):
        self.repo_path = repo_path
        self.worktree_base = worktree_base or repo_path / ".worktrees"
        self.active_worktrees: dict[str, Path] = {}

    async def create(self, task_id: str, branch_name: str,
                     base_branch: str = "main") -> Path:
        """Create an isolated git worktree for a task.

        git worktree add <path> -b <branch_name> <base_branch>

        Returns: Path to the worktree directory
        """

    async def merge(self, task_id: str, target_branch: str = "main") -> MergeResult:
        """Merge a task's worktree branch back into target.

        Returns MergeResult with conflicts list if any.
        """

    async def cleanup(self, task_id: str) -> None:
        """Remove a worktree and its branch.

        git worktree remove <path>
        git branch -d <branch_name>
        """

    async def cleanup_all(self) -> None:
        """Remove all active worktrees."""
```

### `agent.py` -- Orchestration Agent

```python
class OrchestrationAgent:
    """The supervisor that decomposes specs and coordinates workers."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-6",
        max_workers: int = 5,
        repo_path: Optional[Path] = None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.pool = WorkerPool(max_workers=max_workers)
        self.worktrees = WorktreeManager(repo_path) if repo_path else None
        self.dag: Optional[TaskDAG] = None
        self.ws_manager = None  # Set by API layer

    async def submit_spec(self, spec: OrchestratorSpec) -> TaskDAG:
        """Phase 1: Decompose spec into TaskDAG using Opus.

        Returns the DAG for user approval before execution.
        """
        self.dag = await TaskDAG.from_spec(spec, self.client)
        return self.dag

    async def execute(self, dag: Optional[TaskDAG] = None) -> list[WorkerResult]:
        """Phase 2-4: Execute approved DAG.

        1. Get ready tasks (no unmet dependencies)
        2. Create worktrees for each
        3. Spawn workers up to max_workers
        4. As workers complete, mark done and spawn newly unblocked tasks
        5. On failure, retry with modified prompt (up to 2x)
        6. Collect all results
        7. Merge worktrees
        8. Feed traces to training pipeline
        """
        dag = dag or self.dag
        results = []

        while not dag.is_complete():
            # Get tasks ready to run
            ready = dag.get_ready_tasks()

            # Spawn workers for ready tasks (up to available slots)
            for task in ready[:self.pool.available_slots]:
                if self.worktrees:
                    task.worktree_path = await self.worktrees.create(
                        task.id, f"task/{task.id}"
                    )
                worker_id = await self.pool.spawn_worker(task)
                dag.nodes[task.id].status = TaskStatus.RUNNING
                dag.nodes[task.id].worker_id = worker_id

                # Broadcast status
                if self.ws_manager:
                    await self.ws_manager.broadcast_to_topic(
                        "orchestration",
                        {"type": "task_started", "task_id": task.id}
                    )

            # Wait for any worker to finish
            result = await self.pool.wait_for_any()
            results.append(result)

            if result.success:
                newly_unblocked = dag.mark_completed(result.task_id, result)
                # Broadcast completion + newly available tasks
            else:
                # Retry logic: modify prompt, re-spawn
                task = dag.nodes[result.task_id]
                if task.retry_count < 2:
                    task.retry_count += 1
                    task.worker_prompt = self._modify_prompt_for_retry(
                        task, result
                    )
                    task.status = TaskStatus.PENDING
                else:
                    blocked = dag.mark_failed(result.task_id, result.output)

        # Merge all worktrees
        if self.worktrees:
            for task_id in dag.completed_tasks():
                await self.worktrees.merge(task_id)
            await self.worktrees.cleanup_all()

        # Feed traces to training pipeline
        await self._ingest_traces(results)

        return results

    async def _ingest_traces(self, results: list[WorkerResult]) -> None:
        """Feed orchestration traces into the Factory pipeline.

        Multi-agent traces are high-value training signal:
        - Task decomposition patterns
        - Tool-use sequences
        - Error recovery strategies
        - Coordination patterns
        """
```

## API Integration

### New Endpoints

```python
@router.post("/api/orchestrate/submit")
async def submit_spec(request: SpecRequest):
    """Submit a development spec for decomposition.
    Returns the TaskDAG for user approval.

    Request: {title, description, constraints, acceptance_criteria, repository}
    Response: {job_id, dag: {tasks: [...], dependencies: [...]}}
    """

@router.post("/api/orchestrate/{job_id}/approve")
async def approve_plan(job_id: str):
    """Approve the decomposed plan and start execution.
    Response: {status: "executing", workers_spawned: int}
    """

@router.get("/api/orchestrate/{job_id}/status")
async def get_status(job_id: str):
    """Get orchestration job status.
    Response: {status, tasks: [{id, title, status, worker_id, progress}],
               total_cost, elapsed_time}
    """

@router.post("/api/orchestrate/{job_id}/task/{task_id}/retry")
async def retry_task(job_id: str, task_id: str):
    """Retry a failed task with modified prompt."""

@router.delete("/api/orchestrate/{job_id}")
async def cancel_job(job_id: str):
    """Cancel all workers and clean up worktrees."""

@router.websocket("/ws/orchestrate/{job_id}")
async def orchestrate_ws(websocket: WebSocket, job_id: str):
    """Real-time updates for orchestration progress.
    Streams: task_started, task_completed, task_failed,
             worker_output, merge_result, traces_ingested
    """
```

## The Ouroboros Loop

The orchestrator completes BashGym's self-improvement flywheel:

```
User spec
    |
    v
Orchestrator decomposes (Opus)
    |
    v
Workers execute (Sonnet, generating traces)
    |
    v
Traces -> Factory pipeline -> Training data
    |
    v
Training data + DataDesigner augmentation -> Fine-tune student models
    |
    v
Student models -> Replace Sonnet workers for simple tasks (via ModelRouter)
    |
    v
Better student -> Handles more tasks -> More traces -> Better training
    |
    v
Eventually: Student model handles orchestration too
```

## Model Selection Strategy

| Role | Model | Rationale |
|------|-------|-----------|
| Orchestrator planning | Opus 4.6 | Highest reasoning for decomposition |
| Worker execution | Sonnet 4.5 | Best cost/quality for implementation |
| Quick validation | Haiku 4.5 | Fast/cheap for checking results |
| Merge conflict resolution | Sonnet 4.5 | Needs code understanding |
| DataDesigner text generation | Configurable | User's choice via settings |
| DataDesigner code generation | Configurable | User's choice via settings |

## Risk Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Windows subprocess issues | High | Use `asyncio.create_subprocess_exec` with timeouts; test on Windows CI |
| API rate limiting | High | Exponential backoff in dispatcher; limit to 3-5 concurrent workers |
| Token cost explosion | High | `--max-budget-usd` per worker; Sonnet for workers; total job budget cap |
| Git merge conflicts | Medium | Worktrees isolate branches; file-level task decomposition; dedicated merge step |
| Worker runaway | Medium | `--max-turns` cap; subprocess timeout; heartbeat monitoring |
| Task decomposition quality | Critical | Opus for planning; user approves plan before execution |

---

# Migration Path

## Phase 1: DataDesigner Foundation (Week 1-2)

- Add `data-designer>=0.5.0` dependency
- Create `bashgym/factory/data_designer.py` with `DataDesignerPipeline`
- Implement `coding_agent_sft.py` and `coding_agent_dpo.py` pipeline builders
- Add API endpoints for preview/create/validate
- Add CLI subcommands

## Phase 2: DataDesigner Polish (Week 3)

- Implement `tool_use_sft.py` and `from_external.py` pipeline builders
- Add HuggingFace import and push-to-hub
- Wire up dashboard UI (DataDesigner tab)
- Add unstructured data path with MCP

## Phase 3: Orchestrator Foundation (Week 4-5)

- Build `task_dag.py` with spec decomposition via Anthropic Messages API
- Build `worktree.py` for git worktree management
- Build `dispatcher.py` using `asyncio.create_subprocess_exec` for Claude CLI workers
- Add `POST /api/orchestrate/submit` and approval flow

## Phase 4: Orchestrator Coordination (Week 6-7)

- Implement dependency-aware scheduling (topological sort)
- Add retry logic with prompt modification
- Build `synthesizer.py` for result merging
- Add WebSocket streaming for real-time progress
- Implement budget tracking

## Phase 5: Close the Loop (Week 8)

- Connect orchestrator traces to Factory pipeline
- Train student models on orchestration patterns
- Implement confidence-based routing in orchestrator
- Deprecate `DataFactory.augment_example()` and `_augment_with_*` methods

---

# Success Criteria

## DataDesigner

- Generate 1000 diverse SFT examples from 50 gold traces in under 30 minutes
- LLM-Judge quality scores average above 0.7 (correctness >= 3, tool_usage >= 3)
- `data-designer preview` returns results in under 30 seconds
- HuggingFace dataset import works with any text/code dataset
- Code validation (Ruff) passes for 90%+ of generated Python examples
- All existing tests continue passing

## Orchestrator

- Decompose a 500-word spec into a valid TaskDAG in under 60 seconds
- Execute 5 parallel workers without subprocess hangs on Windows
- Complete a 3-feature spec with 5 workers in under 15 minutes
- All worker traces successfully ingested into training pipeline
- Git worktree merge succeeds without manual intervention for 80%+ of tasks
- Total cost for a medium spec stays under $10 USD
