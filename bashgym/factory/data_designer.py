"""
DataDesigner Pipeline Integration

Bridges BashGym's training data pipeline with NVIDIA NeMo DataDesigner v0.6.1+.
Provides entry points for generating training data from traces, external datasets,
and unstructured documents through DataDesigner's column DAG execution engine.

Module 3: Data Synthesis (The "Factory") - DataDesigner Integration
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# DataDesigner is an optional dependency
try:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    DATA_DESIGNER_AVAILABLE = True
except ImportError:
    DATA_DESIGNER_AVAILABLE = False

# Feature detection for Data Designer capabilities.
# Every flag is hasattr-guarded so the module imports cleanly whether or not the
# optional `data-designer` package is installed, and degrades gracefully across
# the v0.5.x -> v0.6.1 surface (and the separately-versioned NeMo Microservices
# PySDK line, which may not expose every column type).
HAS_STRUCTURED_COLUMN = False
HAS_JUDGE_COLUMN = False
HAS_VALIDATION_COLUMN = False
HAS_EMBEDDING_COLUMN = False
HAS_CUSTOM_COLUMN = False
HAS_EXPRESSION_COLUMN = False
# v0.6.1 capabilities
HAS_CODE_COLUMN = False
HAS_SEED_DATASET_COLUMN = False
HAS_AGENT_ROLLOUT = False
HAS_MCP = False
HAS_WORKFLOW = False
HAS_SCHEMA_TRANSFORM = False

if DATA_DESIGNER_AVAILABLE:
    HAS_STRUCTURED_COLUMN = hasattr(dd, "LLMStructuredColumnConfig")
    HAS_JUDGE_COLUMN = hasattr(dd, "LLMJudgeColumnConfig")
    HAS_VALIDATION_COLUMN = hasattr(dd, "ValidationColumnConfig")
    HAS_EMBEDDING_COLUMN = hasattr(dd, "EmbeddingColumnConfig")
    HAS_CUSTOM_COLUMN = hasattr(dd, "CustomColumnConfig")
    HAS_EXPRESSION_COLUMN = hasattr(dd, "ExpressionColumnConfig")
    # v0.6.1: code-generation column, seed-dataset column, native agent-rollout
    # ingestion, in-pipeline MCP tool use, workflow chaining, schema-transform
    # processor (chat/messages export).
    HAS_CODE_COLUMN = hasattr(dd, "LLMCodeColumnConfig")
    HAS_SEED_DATASET_COLUMN = hasattr(dd, "SeedDatasetColumnConfig")
    HAS_AGENT_ROLLOUT = hasattr(dd, "AgentRolloutSeedSource") and hasattr(dd, "AgentRolloutFormat")
    HAS_MCP = hasattr(dd, "ToolConfig") and hasattr(dd, "LocalStdioMCPProvider")
    HAS_WORKFLOW = hasattr(DataDesigner, "compose_workflow")
    HAS_SCHEMA_TRANSFORM = hasattr(dd, "SchemaTransformProcessorConfig")

logger = logging.getLogger(__name__)

# Pipelines that require the sandbox MCP tool server (real tool execution).
_TOOL_PIPELINES = {"mcp_tool_use"}


def _is_truthy(value: Any) -> bool:
    """Coerce a Data Designer flag cell (bool, or a rendered string) to a bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


_PROVIDER_KEY_ENV = {
    "nvidia": "NVIDIA_API_KEY",
    "nvidia-nim": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_CODE_MODEL_RE = re.compile(r"cod(e|er)|deepseek|starcoder|granite-.*code", re.IGNORECASE)
# Instruction/chat-tuned models support the chat-completions API; base/completion
# models (e.g. starcoder2-15b without "-instruct") fail chat health checks.
_CHAT_MODEL_RE = re.compile(r"instruct|chat|nemotron", re.IGNORECASE)


def _looks_like_code_model(model_id: str) -> bool:
    return bool(_CODE_MODEL_RE.search(model_id or ""))


def _looks_like_chat_model(model_id: str) -> bool:
    return bool(_CHAT_MODEL_RE.search(model_id or ""))


def provider_model_ids(provider: str, endpoint: str) -> list[str]:
    """Bare model IDs served by an OpenAI-compatible provider endpoint.

    Works for any ``/v1/models`` endpoint (NVIDIA NIM, Ollama, vLLM, OpenAI,
    OpenRouter, ...) so model selection adapts to whatever is actually served.
    Best-effort: returns [] on any error (offline, auth, unknown provider).
    """
    import httpx

    headers = {}
    key_env = _PROVIDER_KEY_ENV.get(provider)
    if key_env and os.environ.get(key_env):
        headers["Authorization"] = f"Bearer {os.environ[key_env]}"
    try:
        resp = httpx.get(f"{endpoint.rstrip('/')}/models", headers=headers, timeout=10.0)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", []) if m.get("id")]
    except Exception as e:  # noqa: BLE001 - discovery is best-effort
        logger.debug("provider_model_ids(%s) failed: %s", provider, e)
        return []


def list_inference_models(code_only: bool = False) -> list[dict[str, Any]]:
    """Discover inference models across configured open-model sources.

    Adaptable: aggregates local Ollama / LM Studio models and the live NVIDIA NIM
    catalog via ``bashgym.providers`` discovery. Models served from HuggingFace /
    Unsloth weights appear once served (NIM, or pulled into Ollama). Intended for
    UI/CLI selection. Best-effort: returns [] if discovery is unavailable. Must be
    called from a synchronous context (it drives the async discovery internally).
    """
    try:
        from bashgym.providers.detector import get_available_models_sync

        cats = get_available_models_sync(code_only=code_only)
    except Exception as e:  # noqa: BLE001
        logger.debug("list_inference_models discovery failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cat in ("inference", "local"):
        for m in cats.get(cat, []):
            d = m.to_dict() if hasattr(m, "to_dict") else dict(m)
            if d.get("id") and d["id"] not in seen:
                seen.add(d["id"])
                out.append(d)
    return out


@dataclass
class ProviderSpec:
    """Configuration for a single LLM provider in multi-provider pipelines."""

    name: str  # e.g., "nvidia", "anthropic"
    endpoint: str  # API endpoint URL
    api_key: str | None = None  # API key (or env var name)
    models: list[str] = field(default_factory=list)  # Model aliases this provider serves


@dataclass
class PipelineConfig:
    """Configuration for a DataDesigner generation pipeline."""

    # Pipeline selection
    pipeline: str = "coding_agent_sft"

    # LLM provider settings
    provider: str = "nvidia"
    provider_endpoint: str = "https://integrate.api.nvidia.com/v1"
    provider_api_key: str | None = None

    # Multi-provider support (overrides single provider when set)
    providers: list[ProviderSpec] = field(default_factory=list)

    # Model aliases
    text_model: str = "meta/llama-3.3-70b-instruct"
    code_model: str = "deepseek-ai/deepseek-v4-flash"
    judge_model: str = "meta/llama-3.3-70b-instruct"

    # Generation settings
    num_records: int = 100
    buffer_size: int = 100
    max_parallel_requests: int = 4

    # Output
    output_dir: Path = field(default_factory=lambda: Path("data/designer_output"))
    train_val_split: float = 0.9

    # Temperature strategy
    temperature_text: float = 0.85
    temperature_code: float = 0.2
    temperature_judge: float = 0.1

    # Seed source
    seed_source: str | None = None

    # MCP tool-use (Phase 3) — real tool execution during generation
    enable_tools: bool = False
    mcp_tool_alias: str = "sandbox"
    mcp_backend: str = "auto"  # auto | docker | local
    mcp_max_tool_turns: int = 8
    mcp_tool_timeout_sec: int = 120

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if not self.provider_api_key:
            self.provider_api_key = os.environ.get("NVIDIA_API_KEY")
        # Tool-use pipelines require the sandbox MCP server; flag it at construction
        # so the DataDesigner instance attaches the MCP provider regardless of the
        # order in which the builder and the .designer property are accessed.
        if self.pipeline in _TOOL_PIPELINES:
            self.enable_tools = True

    def resolve_models(self) -> "PipelineConfig":
        """Adapt text/code/judge models to the configured provider's live catalog.

        Substitutes an available model when a configured one isn't served (so the
        defaults are preferences, not hardcoded requirements). Works against any
        OpenAI-compatible provider (NVIDIA NIM, Ollama/DGX, vLLM, ...). Best-effort:
        no-ops if discovery returns nothing (offline / unknown provider). Returns
        self for chaining; mutate then build/generate.
        """
        available = provider_model_ids(self.provider, self.provider_endpoint)
        if not available:
            return self
        available_set = set(available)
        # Prefer instruction/chat-tuned models (base/completion models fail chat
        # health checks); for code, prefer code+chat, else any chat model.
        chat = [m for m in available if _looks_like_chat_model(m)]
        general_pool = chat or available
        code_pool = [m for m in chat if _looks_like_code_model(m)] or general_pool

        def pick(preferred: str, pool: list[str]) -> str:
            return preferred if preferred in available_set else pool[0]

        self.text_model = pick(self.text_model, general_pool)
        self.code_model = pick(self.code_model, code_pool)
        self.judge_model = pick(self.judge_model, general_pool)
        return self


@dataclass
class GenerationStats:
    """Lightweight observability for a generation run.

    Token costs / per-column timings are emitted to DD's logs by the async
    engine; these are the robustly-accessible counts (records, filtered rows,
    and the stage names for chained workflows).
    """

    records: int = 0
    filtered_out: int = 0
    stages: list[str] = field(default_factory=list)


class DataDesignerPipeline:
    """
    Bridge between BashGym and NVIDIA NeMo DataDesigner v0.6.1+.

    Provides three entry points for generating training data:
    1. from_traces() - Seed from gold execution traces
    2. from_dataset() - Seed from HuggingFace or local datasets
    3. from_unstructured() - Seed from raw documents/code repos

    All entry points feed into DataDesigner's column DAG engine which handles:
    - Statistical sampling for diversity
    - LLM-based generation (text, code, structured)
    - LLM-as-Judge validation
    - Processor filtering for quality control
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._designer = None
        self._builder = None
        self.last_stats: GenerationStats | None = None

    @property
    def designer(self):
        """Lazy-init DataDesigner instance.

        As of Data Designer 0.6.x, model providers are attached to the
        DataDesigner instance (``DataDesigner(model_providers=...)``) rather than
        to the config builder.
        """
        if self._designer is None:
            if not DATA_DESIGNER_AVAILABLE:
                raise ImportError(
                    "data-designer>=0.6.1 is required. Install with: pip install data-designer"
                )
            from bashgym.factory.designer_pipelines import (
                build_mcp_providers,
                build_model_providers,
                build_secret_resolver,
            )

            self._designer = DataDesigner(
                model_providers=build_model_providers(self.config),
                mcp_providers=build_mcp_providers(self.config) or None,
                secret_resolver=build_secret_resolver(),
            )
        return self._designer

    # =========================================================================
    # Entry Points
    # =========================================================================

    def from_traces(
        self,
        trace_dir: Path,
        num_records: int | None = None,
    ) -> "pd.DataFrame":
        """Generate training data using gold traces as seed dataset.

        Extracts task prompts and tool-use patterns from traces,
        then uses DataDesigner to generate diverse variations.

        Args:
            trace_dir: Directory containing gold trace JSON files
            num_records: Number of records to generate (overrides config)

        Returns:
            DataFrame with generated training data
        """
        seeds = self._extract_seeds_from_traces(trace_dir)
        if not seeds:
            raise ValueError(f"No valid traces found in {trace_dir}")

        logger.info(f"Extracted {len(seeds)} seed tasks from traces")

        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataDesigner integration")

        seed_df = pd.DataFrame(seeds)
        builder = self._get_pipeline_builder()
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))

        num = num_records or self.config.num_records
        return self.designer.create(builder, num_records=num).load_dataset()

    def from_agent_rollouts(
        self,
        rollout_format: str = "claude_code",
        path: str | Path | None = None,
        num_records: int | None = None,
        recursive: bool = True,
        file_pattern: str | None = None,
    ) -> "pd.DataFrame":
        """Generate training data seeded directly from native agent rollouts.

        Uses Data Designer's ``AgentRolloutSeedSource`` (v0.6.x) to ingest raw
        rollout artifacts straight from disk -- no hand-rolled trace parsing.
        Supported formats: ``claude_code`` (~/.claude/projects), ``codex``
        (~/.codex/sessions), ``hermes_agent`` (~/.hermes/sessions),
        ``pi_coding_agent`` (~/.pi/agent/sessions), ``atif`` (requires ``path``).

        This complements ``from_traces`` (which seeds from BashGym's processed
        gold-trace schema). Pair it with the ``coding_agent_distill`` pipeline to
        distill rollouts into standalone SFT examples.

        Args:
            rollout_format: Rollout format name (see above).
            path: Optional directory override (required for the ``atif`` format).
            num_records: Number of records to generate (overrides config).
            recursive: Recurse into subdirectories when scanning for rollouts.
            file_pattern: Optional glob override for rollout files.

        Returns:
            DataFrame with generated training data.
        """
        if not HAS_AGENT_ROLLOUT:
            raise RuntimeError(
                "AgentRolloutSeedSource requires data-designer>=0.6.x (HAS_AGENT_ROLLOUT is False)."
            )
        if path is None and rollout_format.strip().lower() == "atif":
            raise ValueError("path is required for the 'atif' rollout format")

        builder = self._get_pipeline_builder()
        seed_kwargs: dict[str, Any] = {
            "format": self._resolve_rollout_format(rollout_format),
            "recursive": recursive,
        }
        if path is not None:
            seed_kwargs["path"] = str(path)
        if file_pattern is not None:
            seed_kwargs["file_pattern"] = file_pattern
        builder.with_seed_dataset(dd.AgentRolloutSeedSource(**seed_kwargs))

        num = num_records or self.config.num_records
        return self.designer.create(builder, num_records=num).load_dataset()

    @staticmethod
    def _resolve_rollout_format(rollout_format: str) -> "dd.AgentRolloutFormat":
        """Map a format name (e.g. ``claude_code``) to the dd.AgentRolloutFormat enum."""
        key = rollout_format.strip().upper()
        try:
            return getattr(dd.AgentRolloutFormat, key)
        except AttributeError:
            valid = [m for m in dir(dd.AgentRolloutFormat) if m.isupper()]
            raise ValueError(f"Unknown rollout format '{rollout_format}'. Valid: {valid}") from None

    def from_dataset(
        self,
        source: str,
        num_records: int | None = None,
        column_mapping: dict[str, str] | None = None,
        subset: str | None = None,
        split: str = "train",
    ) -> "pd.DataFrame":
        """Generate training data from a HuggingFace dataset or local file.

        Args:
            source: HuggingFace dataset ID (e.g. "bigcode/starcoderdata")
                   or local file path (CSV, Parquet, JSON, JSONL)
            num_records: Number of records to generate
            column_mapping: Map source columns to expected seed columns
                           e.g. {"instruction": "seed_task", "input": "seed_context"}
            subset: HuggingFace dataset subset/config name
            split: Dataset split to use (default: "train")

        Returns:
            DataFrame with generated training data
        """
        builder = self._get_pipeline_builder()

        source_path = Path(source)
        needs_materialize = bool(column_mapping) or bool(subset) or split != "train"

        if source_path.exists():
            if column_mapping:
                # Remap columns by materializing to a DataFrame seed source.
                seed_df = self._read_tabular(source_path).rename(columns=column_mapping)
                builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))
            else:
                # Data Designer 0.6.x: local files use LocalFileSeedSource(path=...).
                builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(source_path)))
        elif needs_materialize:
            # 0.6.x HuggingFaceSeedSource accepts only path/token/endpoint, so load the
            # requested subset/split (and apply any column remap) via the datasets lib.
            from datasets import load_dataset

            seed_df = load_dataset(source, name=subset, split=split).to_pandas()
            if column_mapping:
                seed_df = seed_df.rename(columns=column_mapping)
            builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))
        else:
            builder.with_seed_dataset(
                dd.HuggingFaceSeedSource(path=source, token=os.environ.get("HF_TOKEN"))
            )

        num = num_records or self.config.num_records
        return self.designer.create(builder, num_records=num).load_dataset()

    def prepare_source(
        self,
        source_id: str,
        *,
        goal: str = "sft",
        output_dir: str | Path | None = None,
        input_path: str | Path | None = None,
        limit: int | None = None,
        allow_eval_only: bool = False,
        override_reason: str | None = None,
    ) -> dict[str, Any]:
        """Prepare source and dataset cards for a curated public source.

        This is the Data Designer bridge for the BashGym Source Library. It
        writes source provenance before any generation or download work starts.
        """

        from bashgym.sources import get_source, prepare_source_artifacts, prepare_source_manifest

        card = get_source(source_id)
        target_dir = Path(output_dir) if output_dir else self.config.output_dir
        manifest = prepare_source_manifest(
            card,
            goal=goal,
            output_dir=target_dir,
            allow_eval_only=allow_eval_only,
            override_reason=override_reason,
        )
        if not manifest["use_verdict"]["ok"]:
            codes = ", ".join(manifest["use_verdict"]["blocking_codes"])
            raise ValueError(f"source {source_id!r} cannot be used for {goal!r}: {codes}")

        dataset_card = {
            "schema_version": "bashgym.dataset_card.v1",
            "source_id": card.id,
            "source_name": card.name,
            "goal": goal,
            "adapter": card.adapter,
            "artifact_types": [artifact.value for artifact in card.artifact_types],
            "training_eligible": card.training_eligible,
            "eval_only": card.eval_only,
            "license": card.license,
            "split_policy": card.split_policy,
            "decontam_notes": card.decontam_notes,
            "source_manifest_path": manifest.get("manifest_path"),
            "quality_notes": card.source_quality_notes,
        }
        artifact_report = None
        if input_path is not None:
            artifact_report = prepare_source_artifacts(
                card,
                goal=goal,
                input_path=input_path,
                output_dir=target_dir,
                allow_eval_only=allow_eval_only,
                override_reason=override_reason,
                limit=limit,
            )
            if not artifact_report["ok"]:
                raise ValueError(
                    f"source {source_id!r} adapter failed for {goal!r}: "
                    f"{', '.join(artifact_report['errors'])}"
                )
            dataset_card["artifact_report_path"] = artifact_report.get("report_path")
            dataset_card["artifacts"] = artifact_report.get("artifacts", [])
        target_dir.mkdir(parents=True, exist_ok=True)
        dataset_card_path = target_dir / "dataset_card.json"
        dataset_card_path.write_text(
            json.dumps(dataset_card, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "source_manifest": manifest,
            "dataset_card": dataset_card,
            "dataset_card_path": str(dataset_card_path),
            "artifact_report": artifact_report,
        }

    def from_source(
        self,
        source_id: str,
        *,
        goal: str = "sft",
        num_records: int | None = None,
        output_dir: str | Path | None = None,
        input_path: str | Path | None = None,
        allow_eval_only: bool = False,
        override_reason: str | None = None,
    ) -> "pd.DataFrame":
        """Generate data from a curated source card.

        The first source integration path uses Hugging Face-backed source cards
        or an explicit ``PipelineConfig.seed_source`` override, then delegates to
        ``from_dataset`` after writing source/dataset cards.
        """

        from bashgym.sources import get_source

        prepared = self.prepare_source(
            source_id,
            goal=goal,
            output_dir=output_dir,
            input_path=input_path,
            limit=num_records,
            allow_eval_only=allow_eval_only,
            override_reason=override_reason,
        )
        card = get_source(source_id)
        seed_source = (
            str(input_path)
            if input_path is not None
            else card.huggingface_id or self.config.seed_source
        )
        if not seed_source:
            raise ValueError(
                f"source {source_id!r} does not define a Hugging Face dataset. "
                "Set PipelineConfig.seed_source to a local seed file for this adapter."
            )
        if output_dir:
            self.config.output_dir = Path(output_dir)
        logger.info(
            "Prepared source %s for %s with dataset card %s",
            source_id,
            goal,
            prepared["dataset_card_path"],
        )
        return self.from_dataset(seed_source, num_records=num_records)

    @staticmethod
    def _read_tabular(path: Path) -> "pd.DataFrame":
        """Read a local CSV/Parquet/JSON/JSONL file into a DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required to read local seed files")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in (".jsonl", ".ndjson"):
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            return pd.read_json(path)
        return pd.read_csv(path)

    def from_unstructured(
        self,
        path: Path,
        num_records: int | None = None,
    ) -> "pd.DataFrame":
        """Generate training data from unstructured documents.

        Processes PDFs, markdown docs, code repositories etc. into seed data,
        then generates training examples via the DataDesigner pipeline.

        Args:
            path: Path to file or directory of unstructured documents
            num_records: Number of records to generate

        Returns:
            DataFrame with generated training data
        """
        seeds = self._extract_seeds_from_unstructured(path)
        if not seeds:
            raise ValueError(f"No extractable content found at {path}")

        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataDesigner integration")

        seed_df = pd.DataFrame(seeds)
        builder = self._get_pipeline_builder()
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))

        num = num_records or self.config.num_records
        return self.designer.create(builder, num_records=num).load_dataset()

    def from_config(
        self,
        config_path: str,
        num_records: int | None = None,
    ) -> "pd.DataFrame":
        """Generate from a custom DataDesigner config file.

        Supports YAML, JSON, or Python config files.

        Args:
            config_path: Path to config file
            num_records: Number of records to generate

        Returns:
            DataFrame with generated training data
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_file.suffix in (".yaml", ".yml"):
            import yaml

            with open(config_file) as f:
                raw = yaml.safe_load(f)
        elif config_file.suffix == ".json":
            with open(config_file) as f:
                raw = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")

        builder = dd.DataDesignerConfigBuilder.from_config(raw)
        num = num_records or self.config.num_records
        return self.designer.create(builder, num_records=num).load_dataset()

    def generate_chained(
        self,
        stages: list[dict],
        workflow_name: str = "bashgym-chain",
    ) -> "pd.DataFrame":
        """Run a multi-stage Data Designer workflow and return the final dataset.

        Each ``stage`` is a dict::

            {"name": str, "builder": DataDesignerConfigBuilder,
             "num_records": int | None, "output_processors": list | None,
             "output": str}

        Stages run sequentially (each stage's output threads into the next);
        ``output_processors`` (e.g. ``messages_schema_transform``) reshape a
        stage's output — the 0.6.x-native, processor-driven path for
        curriculum/multi-stage generation and ChatML export.

        Experimental in Data Designer 0.6.x: linear topology only, no stage-resume.
        Populates ``self.last_stats``.
        """
        if not stages:
            raise ValueError("generate_chained requires at least one stage")
        workflow = self.designer.compose_workflow(name=workflow_name)
        for st in stages:
            workflow.add_stage(
                st["name"],
                st["builder"],
                num_records=st.get("num_records"),
                output_processors=st.get("output_processors"),
                output=st.get("output", "final"),
            )
        df = workflow.run().load_dataset()
        self.last_stats = GenerationStats(records=len(df), stages=[s["name"] for s in stages])
        return df

    # =========================================================================
    # Operations
    # =========================================================================

    def preview(self, num_records: int = 5) -> "pd.DataFrame":
        """Quick preview of generated data without full pipeline run.

        Args:
            num_records: Number of preview records (default 5)

        Returns:
            Small DataFrame for inspection
        """
        builder = self._get_pipeline_builder()
        return self.designer.preview(builder, num_records=num_records).dataset

    def validate(self) -> dict[str, Any]:
        """Validate pipeline configuration without running generation.

        As of Data Designer 0.6.x, ``DataDesigner.validate()`` returns ``None`` and
        raises on an invalid config, so success means "did not raise".

        Returns:
            Dict with validation results: {valid, errors, columns, dag_order}
        """
        builder = self._get_pipeline_builder()
        columns = self._builder_column_names(builder)
        try:
            self.designer.validate(builder)
            return {"valid": True, "errors": [], "columns": columns, "dag_order": columns}
        except Exception as e:
            return {"valid": False, "errors": [str(e)], "columns": columns}

    @staticmethod
    def _builder_column_names(builder) -> list[str]:
        """Best-effort column-name introspection across Data Designer versions.

        The 0.6.x ``DataDesignerConfigBuilder`` no longer exposes a public
        ``columns`` attribute, so fall back through known accessors.
        """
        for attr in ("columns", "column_names"):
            cols = getattr(builder, attr, None)
            if cols:
                try:
                    return [getattr(c, "name", c) for c in cols]
                except TypeError:
                    pass
        private = getattr(builder, "_column_configs", None)
        if isinstance(private, dict):
            return list(private.keys())
        return []

    def export_nemo(
        self,
        df: "pd.DataFrame",
        output_dir: Path | None = None,
        quality_flag_column: str | None = "passes_quality",
        keep_only_passing: bool = True,
    ) -> dict[str, Any]:
        """Export generated data to NeMo train/val JSONL format.

        Splits data according to train_val_split and converts to NeMo-compatible
        messages format. When ``keep_only_passing`` is set and a quality-flag
        column is present (``passes_quality``, or the distill pipeline's
        ``recommended_for_sft``), low-quality rows are dropped first. This is
        BashGym's quality gate: Data Designer 0.6.x has no in-pipeline row-filter
        (``ValidationColumnConfig(drop=True)`` drops the *column*, not rows), so
        pipelines emit a boolean flag column and the gate is applied here.

        Args:
            df: DataFrame from generation
            output_dir: Output directory (defaults to config.output_dir)
            quality_flag_column: Boolean column to gate on (None disables the gate).
            keep_only_passing: Drop rows whose quality flag is falsy.

        Returns:
            Dict with {train_path, val_path, train_count, val_count, filtered_out}
        """
        out = output_dir or self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Quality gate: drop rows whose flag column is falsy (0.6.x flag-then-filter).
        filtered_out = 0
        flag = quality_flag_column
        if keep_only_passing:
            if (flag is None or flag not in df.columns) and "recommended_for_sft" in df.columns:
                flag = "recommended_for_sft"
            if flag and flag in df.columns:
                before = len(df)
                df = df[df[flag].map(_is_truthy)]
                filtered_out = before - len(df)
                logger.info(f"Quality gate ({flag}): kept {len(df)}/{before}")

        # Split
        split_idx = int(len(df) * self.config.train_val_split)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        train_path = out / "train.jsonl"
        val_path = out / "val.jsonl"

        # Convert to NeMo messages format
        self._write_nemo_jsonl(train_df, train_path)
        self._write_nemo_jsonl(val_df, val_path)

        result = {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_count": len(train_df),
            "val_count": len(val_df),
            "filtered_out": filtered_out,
        }
        logger.info(
            f"Exported {result['train_count']} train + "
            f"{result['val_count']} val examples to {out} "
            f"(quality-filtered out {filtered_out})"
        )
        return result

    def push_to_hub(
        self,
        df: "pd.DataFrame",
        repo_id: str,
        private: bool = True,
    ) -> str:
        """Publish generated dataset to HuggingFace Hub.

        Args:
            df: DataFrame to publish
            repo_id: HuggingFace repo ID (e.g. "username/dataset-name")
            private: Whether to create a private dataset

        Returns:
            URL of the published dataset
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for push_to_hub. "
                "Install with: pip install huggingface_hub"
            )

        import tempfile

        tmp_path = Path(tempfile.gettempdir()) / "bashgym_dataset.parquet"
        df.to_parquet(tmp_path)
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo="data/train.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
        tmp_path.unlink(missing_ok=True)

        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Published dataset to {url}")
        return url

    # =========================================================================
    # Pipeline Resolution
    # =========================================================================

    def _get_pipeline_builder(self) -> "dd.DataDesignerConfigBuilder":
        """Resolve pipeline name to a ConfigBuilder instance.

        Looks up the pipeline name in the registered pipeline builders,
        or treats it as a file path to a config.
        """
        if not DATA_DESIGNER_AVAILABLE:
            raise ImportError(
                "data-designer>=0.6.1 is required. Install with: pip install data-designer"
            )

        from bashgym.factory.designer_pipelines import PIPELINES

        if self.config.pipeline in PIPELINES:
            builder_fn = PIPELINES[self.config.pipeline]
            return builder_fn(self.config)

        # Try as file path
        config_path = Path(self.config.pipeline)
        if config_path.exists():
            if config_path.suffix == ".json":
                with open(config_path) as f:
                    return dd.DataDesignerConfigBuilder.from_config(json.load(f))
            elif config_path.suffix in (".yaml", ".yml"):
                import yaml

                with open(config_path) as f:
                    return dd.DataDesignerConfigBuilder.from_config(yaml.safe_load(f))

        available = list(PIPELINES.keys())
        raise ValueError(f"Unknown pipeline '{self.config.pipeline}'. " f"Available: {available}")

    # =========================================================================
    # Seed Extraction
    # =========================================================================

    def _extract_seeds_from_traces(self, trace_dir: Path) -> list[dict[str, Any]]:
        """Extract seed task prompts from gold traces.

        Each trace yields one seed record with:
        - seed_task: The user's initial prompt
        - seed_tools: List of tools used
        - seed_complexity: Inferred complexity (step count)
        - seed_language: Detected primary language
        """
        seeds = []
        trace_dir = Path(trace_dir)

        for trace_file in trace_dir.glob("*.json"):
            try:
                with open(trace_file, encoding="utf-8", errors="replace") as f:
                    trace_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(trace_data, dict):
                continue

            metadata = trace_data.get("metadata", {})
            prompt = metadata.get("user_initial_prompt", "")
            if not prompt:
                continue

            trace_steps = trace_data.get("trace", [])
            tools_used = list({s.get("tool_name", "unknown") for s in trace_steps})

            # Infer complexity from step count
            step_count = len(trace_steps)
            if step_count <= 5:
                complexity = "simple"
            elif step_count <= 15:
                complexity = "moderate"
            else:
                complexity = "complex"

            # Detect primary language from file extensions
            language = self._detect_language(trace_steps)

            seeds.append(
                {
                    "seed_task": prompt,
                    "seed_tools": ", ".join(tools_used),
                    "seed_complexity": complexity,
                    "seed_language": language,
                    "seed_step_count": step_count,
                }
            )

        return seeds

    def _extract_seeds_from_unstructured(self, path: Path) -> list[dict[str, Any]]:
        """Extract seed data from unstructured documents.

        Supports:
        - .py, .ts, .js, .rs, .go files -> code-based seeds
        - .md, .txt files -> documentation-based seeds
        - Directories -> recursively process all supported files
        """
        seeds = []
        path = Path(path)

        if path.is_file():
            files = [path]
        elif path.is_dir():
            supported = {".py", ".ts", ".js", ".rs", ".go", ".md", ".txt", ".json"}
            files = [f for f in path.rglob("*") if f.suffix in supported]
        else:
            return seeds

        for file_path in files[:500]:  # Cap at 500 files
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue

            if not content.strip():
                continue

            # Truncate very long files
            if len(content) > 10000:
                content = content[:10000] + "\n... (truncated)"

            ext = file_path.suffix.lstrip(".")
            lang_map = {
                "py": "python",
                "ts": "typescript",
                "js": "javascript",
                "rs": "rust",
                "go": "go",
                "md": "markdown",
                "txt": "text",
            }

            seeds.append(
                {
                    "seed_task": f"Work with {file_path.name}",
                    "seed_context": content,
                    "seed_language": lang_map.get(ext, ext),
                    "seed_file_type": ext,
                }
            )

        return seeds

    def _detect_language(self, trace_steps: list[dict[str, Any]]) -> str:
        """Detect primary programming language from trace steps."""
        ext_counts: dict[str, int] = {}
        ext_to_lang = {
            ".py": "python",
            ".ts": "typescript",
            ".js": "javascript",
            ".rs": "rust",
            ".go": "go",
            ".sh": "bash",
            ".rb": "ruby",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
        }

        for step in trace_steps:
            command = step.get("command", "")
            for ext, lang in ext_to_lang.items():
                if ext in command:
                    ext_counts[lang] = ext_counts.get(lang, 0) + 1

        if ext_counts:
            return max(ext_counts, key=ext_counts.get)
        return "python"  # Default

    # =========================================================================
    # NeMo Export
    # =========================================================================

    def _write_nemo_jsonl(self, df: "pd.DataFrame", output_path: Path) -> None:
        """Write DataFrame to NeMo-compatible messages JSONL.

        Expected DataFrame columns:
        - task_prompt: User instruction
        - solution_text or solution: Assistant response
        - quality_score (optional): Judge scores for metadata

        Falls back to using all string columns if expected columns missing.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                # Find the task prompt column
                task = ""
                for col in ["task_prompt", "prompt", "instruction", "user_message"]:
                    if col in row and row[col]:
                        task = str(row[col])
                        break

                # Find the response column
                response = ""
                for col in ["solution_text", "solution", "response", "assistant_response"]:
                    if col in row and row[col]:
                        val = row[col]
                        if isinstance(val, dict):
                            response = json.dumps(val)
                        else:
                            response = str(val)
                        break

                if not task or not response:
                    continue

                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task},
                        {"role": "assistant", "content": response},
                    ]
                }
                f.write(json.dumps(record) + "\n")


SYSTEM_PROMPT = """You are an expert software development agent. You execute tasks by running bash commands, reading files, and making edits. You think step-by-step and verify your work.

When given a task:
1. Analyze the requirements
2. Plan your approach
3. Execute commands to accomplish the task
4. Verify the results

You have access to these tools:
- Bash: Execute shell commands
- Read: Read file contents
- Write: Write to files
- Edit: Make targeted edits to files

Always explain your reasoning before executing commands."""
