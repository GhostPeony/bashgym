"""
DataDesigner Pipeline Integration

Bridges BashGym's training data pipeline with NVIDIA NeMo DataDesigner v0.5.0.
Provides entry points for generating training data from traces, external datasets,
and unstructured documents through DataDesigner's column DAG execution engine.

Module 3: Data Synthesis (The "Factory") - DataDesigner Integration
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING

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

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a DataDesigner generation pipeline."""

    # Pipeline selection
    pipeline: str = "coding_agent_sft"

    # LLM provider settings
    provider: str = "nvidia"
    provider_endpoint: str = "https://integrate.api.nvidia.com/v1"
    provider_api_key: Optional[str] = None

    # Model aliases
    text_model: str = "meta/llama-3.3-70b-instruct"
    code_model: str = "qwen/qwen2.5-coder-32b-instruct"
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
    seed_source: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if not self.provider_api_key:
            self.provider_api_key = os.environ.get("NVIDIA_API_KEY")


class DataDesignerPipeline:
    """
    Bridge between BashGym and NVIDIA NeMo DataDesigner v0.5.0.

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

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._designer = None
        self._builder = None

    @property
    def designer(self):
        """Lazy-init DataDesigner instance."""
        if self._designer is None:
            if not DATA_DESIGNER_AVAILABLE:
                raise ImportError(
                    "data-designer>=0.5.0 is required. "
                    "Install with: pip install data-designer"
                )
            self._designer = DataDesigner()
        return self._designer

    # =========================================================================
    # Entry Points
    # =========================================================================

    def from_traces(
        self,
        trace_dir: Path,
        num_records: Optional[int] = None,
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
        builder.with_seed_dataset(dd.DataFrameSeedSource(data=seed_df))

        num = num_records or self.config.num_records
        return self.designer.generate(builder, num_rows=num)

    def from_dataset(
        self,
        source: str,
        num_records: Optional[int] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        subset: Optional[str] = None,
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
        if source_path.exists():
            # Local file
            builder.with_seed_dataset(dd.FileSeedSource(
                path=str(source_path),
                column_mapping=column_mapping,
            ))
        else:
            # Treat as HuggingFace dataset
            seed_kwargs = {"path": source, "split": split}
            if subset:
                seed_kwargs["name"] = subset
            if column_mapping:
                seed_kwargs["column_mapping"] = column_mapping
            builder.with_seed_dataset(dd.HuggingFaceSeedSource(**seed_kwargs))

        num = num_records or self.config.num_records
        return self.designer.generate(builder, num_rows=num)

    def from_unstructured(
        self,
        path: Path,
        num_records: Optional[int] = None,
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
        builder.with_seed_dataset(dd.DataFrameSeedSource(data=seed_df))

        num = num_records or self.config.num_records
        return self.designer.generate(builder, num_rows=num)

    def from_config(
        self,
        config_path: str,
        num_records: Optional[int] = None,
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
        return self.designer.generate(builder, num_rows=num)

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
        return self.designer.preview(builder, num_rows=num_records)

    def validate(self) -> Dict[str, Any]:
        """Validate pipeline configuration without running generation.

        Returns:
            Dict with validation results: {valid, errors, columns, dag}
        """
        builder = self._get_pipeline_builder()
        try:
            result = self.designer.validate(builder)
            return {
                "valid": True,
                "errors": [],
                "columns": [c.name for c in builder.columns],
                "dag_order": result.get("execution_order", []),
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "columns": [c.name for c in builder.columns],
            }

    def export_nemo(
        self,
        df: "pd.DataFrame",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Export generated data to NeMo train/val JSONL format.

        Splits data according to train_val_split and converts to
        NeMo-compatible messages format.

        Args:
            df: DataFrame from generation
            output_dir: Output directory (defaults to config.output_dir)

        Returns:
            Dict with {train_path, val_path, train_count, val_count}
        """
        out = output_dir or self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

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
        }
        logger.info(
            f"Exported {result['train_count']} train + "
            f"{result['val_count']} val examples to {out}"
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
                "data-designer>=0.5.0 is required. "
                "Install with: pip install data-designer"
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
        raise ValueError(
            f"Unknown pipeline '{self.config.pipeline}'. "
            f"Available: {available}"
        )

    # =========================================================================
    # Seed Extraction
    # =========================================================================

    def _extract_seeds_from_traces(self, trace_dir: Path) -> List[Dict[str, Any]]:
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
                with open(trace_file, "r", encoding="utf-8", errors="replace") as f:
                    trace_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if not isinstance(trace_data, dict):
                continue

            metadata = trace_data.get("metadata", {})
            prompt = metadata.get("user_initial_prompt", "")
            if not prompt:
                continue

            trace_steps = trace_data.get("trace", [])
            tools_used = list({
                s.get("tool_name", "unknown") for s in trace_steps
            })

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

            seeds.append({
                "seed_task": prompt,
                "seed_tools": ", ".join(tools_used),
                "seed_complexity": complexity,
                "seed_language": language,
                "seed_step_count": step_count,
            })

        return seeds

    def _extract_seeds_from_unstructured(self, path: Path) -> List[Dict[str, Any]]:
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
            except (IOError, UnicodeDecodeError):
                continue

            if not content.strip():
                continue

            # Truncate very long files
            if len(content) > 10000:
                content = content[:10000] + "\n... (truncated)"

            ext = file_path.suffix.lstrip(".")
            lang_map = {
                "py": "python", "ts": "typescript", "js": "javascript",
                "rs": "rust", "go": "go", "md": "markdown", "txt": "text",
            }

            seeds.append({
                "seed_task": f"Work with {file_path.name}",
                "seed_context": content,
                "seed_language": lang_map.get(ext, ext),
                "seed_file_type": ext,
            })

        return seeds

    def _detect_language(self, trace_steps: List[Dict[str, Any]]) -> str:
        """Detect primary programming language from trace steps."""
        ext_counts: Dict[str, int] = {}
        ext_to_lang = {
            ".py": "python", ".ts": "typescript", ".js": "javascript",
            ".rs": "rust", ".go": "go", ".sh": "bash", ".rb": "ruby",
            ".java": "java", ".cpp": "cpp", ".c": "c",
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
