"""Tests for Data Designer pipeline integration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from bashgym.factory.data_designer import (
    HAS_CUSTOM_COLUMN,
    HAS_EMBEDDING_COLUMN,
    HAS_EXPRESSION_COLUMN,
    HAS_JUDGE_COLUMN,
    HAS_STRUCTURED_COLUMN,
    HAS_VALIDATION_COLUMN,
    SYSTEM_PROMPT,
    DataDesignerPipeline,
    PipelineConfig,
    ProviderSpec,
)

# =========================================================================
# ProviderSpec
# =========================================================================


class TestProviderSpec:
    def test_default_values(self):
        spec = ProviderSpec(name="nvidia", endpoint="https://example.com")
        assert spec.name == "nvidia"
        assert spec.api_key is None
        assert spec.models == []

    def test_with_models(self):
        spec = ProviderSpec(
            name="anthropic",
            endpoint="https://api.anthropic.com",
            models=["judge-model"],
        )
        assert spec.models == ["judge-model"]

    def test_with_api_key(self):
        spec = ProviderSpec(
            name="nvidia",
            endpoint="https://nim.example.com",
            api_key="test-key-123",
        )
        assert spec.api_key == "test-key-123"

    def test_multiple_models(self):
        spec = ProviderSpec(
            name="nvidia",
            endpoint="https://nim.example.com",
            models=["code-model", "text-model", "judge-model"],
        )
        assert len(spec.models) == 3
        assert "code-model" in spec.models


# =========================================================================
# PipelineConfig
# =========================================================================


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.pipeline == "coding_agent_sft"
        assert config.provider == "nvidia"
        assert config.providers == []
        assert config.num_records == 100
        assert config.train_val_split == 0.9

    def test_multi_provider_config(self):
        providers = [
            ProviderSpec(
                name="nvidia",
                endpoint="https://nim.example.com",
                models=["code-model"],
            ),
            ProviderSpec(
                name="anthropic",
                endpoint="https://api.anthropic.com",
                models=["judge-model"],
            ),
        ]
        config = PipelineConfig(providers=providers)
        assert len(config.providers) == 2
        assert config.providers[0].name == "nvidia"
        assert config.providers[1].name == "anthropic"

    def test_output_dir_string_to_path(self):
        config = PipelineConfig(output_dir="some/path")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("some/path")

    def test_output_dir_path_stays_path(self):
        config = PipelineConfig(output_dir=Path("another/path"))
        assert isinstance(config.output_dir, Path)

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = PipelineConfig()
            assert config.provider_api_key == "test-key"

    def test_api_key_explicit_overrides_env(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "env-key"}):
            config = PipelineConfig(provider_api_key="explicit-key")
            assert config.provider_api_key == "explicit-key"

    def test_api_key_none_when_env_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            config = PipelineConfig()
            assert config.provider_api_key is None

    def test_temperature_defaults(self):
        config = PipelineConfig()
        assert config.temperature_text == 0.85
        assert config.temperature_code == 0.2
        assert config.temperature_judge == 0.1

    def test_custom_models(self):
        config = PipelineConfig(
            text_model="custom/text-model",
            code_model="custom/code-model",
            judge_model="custom/judge-model",
        )
        assert config.text_model == "custom/text-model"
        assert config.code_model == "custom/code-model"
        assert config.judge_model == "custom/judge-model"

    def test_buffer_and_parallel_defaults(self):
        config = PipelineConfig()
        assert config.buffer_size == 100
        assert config.max_parallel_requests == 4


# =========================================================================
# Feature Detection
# =========================================================================


class TestFeatureDetection:
    def test_feature_flags_exist(self):
        """Feature flags should be defined regardless of DD availability."""
        from bashgym.factory import data_designer

        assert hasattr(data_designer, "HAS_STRUCTURED_COLUMN")
        assert hasattr(data_designer, "HAS_JUDGE_COLUMN")
        assert hasattr(data_designer, "HAS_VALIDATION_COLUMN")
        assert hasattr(data_designer, "HAS_EMBEDDING_COLUMN")
        assert hasattr(data_designer, "HAS_CUSTOM_COLUMN")
        assert hasattr(data_designer, "HAS_EXPRESSION_COLUMN")

    def test_feature_flags_are_bool(self):
        """All feature flags should be booleans."""
        assert isinstance(HAS_STRUCTURED_COLUMN, bool)
        assert isinstance(HAS_JUDGE_COLUMN, bool)
        assert isinstance(HAS_VALIDATION_COLUMN, bool)
        assert isinstance(HAS_EMBEDDING_COLUMN, bool)
        assert isinstance(HAS_CUSTOM_COLUMN, bool)
        assert isinstance(HAS_EXPRESSION_COLUMN, bool)


# =========================================================================
# DataDesignerPipeline.__init__
# =========================================================================


class TestDataDesignerPipelineInit:
    def test_default_config(self):
        pipeline = DataDesignerPipeline()
        assert pipeline.config is not None
        assert pipeline.config.pipeline == "coding_agent_sft"
        assert pipeline._designer is None

    def test_custom_config(self):
        config = PipelineConfig(pipeline="tool_use_sft", num_records=50)
        pipeline = DataDesignerPipeline(config=config)
        assert pipeline.config.pipeline == "tool_use_sft"
        assert pipeline.config.num_records == 50

    def test_designer_property_raises_without_dd(self):
        """Accessing designer when data-designer is not installed should raise."""
        pipeline = DataDesignerPipeline()
        with patch("bashgym.factory.data_designer.DATA_DESIGNER_AVAILABLE", False):
            with pytest.raises(ImportError, match="data-designer"):
                _ = pipeline.designer


# =========================================================================
# _extract_seeds_from_traces
# =========================================================================


class TestExtractSeedsFromTraces:
    def test_extracts_seeds_from_valid_traces(self):
        pipeline = DataDesignerPipeline()

        trace_data = {
            "metadata": {"user_initial_prompt": "Fix the login bug"},
            "trace": [
                {"tool_name": "Read", "command": "cat src/auth.py"},
                {"tool_name": "Edit", "command": "edit src/auth.py"},
                {"tool_name": "Bash", "command": "pytest tests/test_auth.py"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "trace_001.json"
            trace_file.write_text(json.dumps(trace_data), encoding="utf-8")

            seeds = pipeline._extract_seeds_from_traces(Path(tmpdir))

        assert len(seeds) == 1
        assert seeds[0]["seed_task"] == "Fix the login bug"
        assert seeds[0]["seed_complexity"] == "simple"  # 3 steps
        assert seeds[0]["seed_step_count"] == 3
        assert "python" in seeds[0]["seed_language"]

    def test_skips_traces_without_prompt(self):
        pipeline = DataDesignerPipeline()

        trace_data = {
            "metadata": {},
            "trace": [{"tool_name": "Bash", "command": "ls"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "trace_no_prompt.json"
            trace_file.write_text(json.dumps(trace_data), encoding="utf-8")

            seeds = pipeline._extract_seeds_from_traces(Path(tmpdir))

        assert len(seeds) == 0

    def test_skips_invalid_json_files(self):
        pipeline = DataDesignerPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.json"
            bad_file.write_text("not valid json {{", encoding="utf-8")

            seeds = pipeline._extract_seeds_from_traces(Path(tmpdir))

        assert len(seeds) == 0

    def test_skips_non_dict_json(self):
        pipeline = DataDesignerPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            list_file = Path(tmpdir) / "list.json"
            list_file.write_text("[1, 2, 3]", encoding="utf-8")

            seeds = pipeline._extract_seeds_from_traces(Path(tmpdir))

        assert len(seeds) == 0

    def test_complexity_levels(self):
        """Test complexity classification by step count."""
        pipeline = DataDesignerPipeline()

        # Simple: <= 5 steps
        simple_trace = {
            "metadata": {"user_initial_prompt": "simple task"},
            "trace": [{"tool_name": "Bash", "command": "ls"}] * 3,
        }
        # Moderate: 6-15 steps
        moderate_trace = {
            "metadata": {"user_initial_prompt": "moderate task"},
            "trace": [{"tool_name": "Bash", "command": "ls"}] * 10,
        }
        # Complex: > 15 steps
        complex_trace = {
            "metadata": {"user_initial_prompt": "complex task"},
            "trace": [{"tool_name": "Bash", "command": "ls"}] * 20,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (trace, expected) in enumerate(
                [
                    (simple_trace, "simple"),
                    (moderate_trace, "moderate"),
                    (complex_trace, "complex"),
                ]
            ):
                path = Path(tmpdir) / f"trace_{i}.json"
                path.write_text(json.dumps(trace), encoding="utf-8")

            seeds = pipeline._extract_seeds_from_traces(Path(tmpdir))

        seeds_by_task = {s["seed_task"]: s for s in seeds}
        assert seeds_by_task["simple task"]["seed_complexity"] == "simple"
        assert seeds_by_task["moderate task"]["seed_complexity"] == "moderate"
        assert seeds_by_task["complex task"]["seed_complexity"] == "complex"

    def test_empty_directory(self):
        pipeline = DataDesignerPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            seeds = pipeline._extract_seeds_from_traces(Path(tmpdir))

        assert len(seeds) == 0

    def test_tools_deduplication(self):
        """Tool names should be deduplicated."""
        pipeline = DataDesignerPipeline()

        trace_data = {
            "metadata": {"user_initial_prompt": "do stuff"},
            "trace": [
                {"tool_name": "Read", "command": "cat file.py"},
                {"tool_name": "Read", "command": "cat other.py"},
                {"tool_name": "Edit", "command": "edit file.py"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_text(json.dumps(trace_data), encoding="utf-8")

            seeds = pipeline._extract_seeds_from_traces(Path(tmpdir))

        tools = seeds[0]["seed_tools"].split(", ")
        assert len(tools) == len(set(tools))


# =========================================================================
# _detect_language
# =========================================================================


class TestDetectLanguage:
    def test_python_detection(self):
        pipeline = DataDesignerPipeline()
        steps = [
            {"command": "cat src/main.py"},
            {"command": "edit src/utils.py"},
            {"command": "pytest tests/test_main.py"},
        ]
        assert pipeline._detect_language(steps) == "python"

    def test_typescript_detection(self):
        pipeline = DataDesignerPipeline()
        steps = [
            {"command": "cat src/index.ts"},
            {"command": "edit src/App.ts"},
        ]
        assert pipeline._detect_language(steps) == "typescript"

    def test_rust_detection(self):
        pipeline = DataDesignerPipeline()
        steps = [
            {"command": "cat src/main.rs"},
            {"command": "cargo build"},
        ]
        assert pipeline._detect_language(steps) == "rust"

    def test_defaults_to_python_when_no_extensions(self):
        pipeline = DataDesignerPipeline()
        steps = [
            {"command": "ls"},
            {"command": "echo hello"},
        ]
        assert pipeline._detect_language(steps) == "python"

    def test_empty_steps(self):
        pipeline = DataDesignerPipeline()
        assert pipeline._detect_language([]) == "python"

    def test_mixed_languages_picks_most_common(self):
        pipeline = DataDesignerPipeline()
        steps = [
            {"command": "cat src/main.py"},
            {"command": "cat src/utils.py"},
            {"command": "cat src/index.ts"},
        ]
        assert pipeline._detect_language(steps) == "python"

    def test_steps_without_command_key(self):
        """Steps missing the 'command' key should not crash."""
        pipeline = DataDesignerPipeline()
        steps = [
            {"tool_name": "Read"},
            {"observation": "file contents"},
        ]
        assert pipeline._detect_language(steps) == "python"


# =========================================================================
# _write_nemo_jsonl
# =========================================================================


class TestWriteNemoJsonl:
    def test_writes_valid_jsonl(self):
        """Should write proper NeMo messages JSONL format."""
        pd = pytest.importorskip("pandas")

        pipeline = DataDesignerPipeline()
        df = pd.DataFrame(
            {
                "task_prompt": ["Fix the bug", "Add a feature"],
                "solution_text": ["I fixed it", "I added it"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"
            pipeline._write_nemo_jsonl(df, output_path)

            lines = output_path.read_text(encoding="utf-8").strip().split("\n")

        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "messages" in record
        assert len(record["messages"]) == 3
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][0]["content"] == SYSTEM_PROMPT
        assert record["messages"][1]["role"] == "user"
        assert record["messages"][1]["content"] == "Fix the bug"
        assert record["messages"][2]["role"] == "assistant"
        assert record["messages"][2]["content"] == "I fixed it"

    def test_skips_rows_without_task_or_response(self):
        """Rows missing task or response columns should be skipped."""
        pd = pytest.importorskip("pandas")

        pipeline = DataDesignerPipeline()
        df = pd.DataFrame(
            {
                "task_prompt": ["Fix the bug", "", "Add feature"],
                "solution_text": ["I fixed it", "Some response", ""],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"
            pipeline._write_nemo_jsonl(df, output_path)

            content = output_path.read_text(encoding="utf-8").strip()
            lines = content.split("\n") if content else []

        assert len(lines) == 1

    def test_column_fallback_order(self):
        """Should try multiple column names in priority order."""
        pd = pytest.importorskip("pandas")

        pipeline = DataDesignerPipeline()
        # Using 'prompt' and 'response' as fallback column names
        df = pd.DataFrame(
            {
                "prompt": ["User question"],
                "response": ["Assistant answer"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"
            pipeline._write_nemo_jsonl(df, output_path)

            lines = output_path.read_text(encoding="utf-8").strip().split("\n")

        record = json.loads(lines[0])
        assert record["messages"][1]["content"] == "User question"
        assert record["messages"][2]["content"] == "Assistant answer"

    def test_dict_response_serialized_to_json(self):
        """Dict responses should be JSON-serialized."""
        pd = pytest.importorskip("pandas")

        pipeline = DataDesignerPipeline()
        df = pd.DataFrame(
            {
                "task_prompt": ["Solve this"],
                "solution": [{"plan": "do stuff", "steps": []}],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"
            pipeline._write_nemo_jsonl(df, output_path)

            lines = output_path.read_text(encoding="utf-8").strip().split("\n")

        record = json.loads(lines[0])
        # The response should be a JSON string of the dict
        response_content = record["messages"][2]["content"]
        parsed = json.loads(response_content)
        assert parsed["plan"] == "do stuff"


# =========================================================================
# from_config
# =========================================================================


class TestFromConfig:
    @patch("bashgym.factory.data_designer.DATA_DESIGNER_AVAILABLE", True)
    def test_from_json_config(self):
        """Should load and generate from a JSON config file."""
        pipeline = DataDesignerPipeline()

        mock_builder = MagicMock()
        mock_df = MagicMock()

        config_data = {"columns": [], "models": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config_data), encoding="utf-8")

            with patch("bashgym.factory.data_designer.dd", create=True) as mock_dd:
                mock_dd.DataDesignerConfigBuilder.from_config.return_value = mock_builder

                with patch.object(
                    DataDesignerPipeline,
                    "designer",
                    new_callable=PropertyMock,
                ) as mock_designer_prop:
                    mock_designer = MagicMock()
                    mock_designer.generate.return_value = mock_df
                    mock_designer_prop.return_value = mock_designer

                    result = pipeline.from_config(str(config_path))

        assert result == mock_df
        mock_dd.DataDesignerConfigBuilder.from_config.assert_called_once_with(config_data)

    def test_from_config_missing_file_raises(self):
        pipeline = DataDesignerPipeline()
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            pipeline.from_config("/nonexistent/config.json")

    def test_from_config_unsupported_format_raises(self):
        pipeline = DataDesignerPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text("", encoding="utf-8")

            with pytest.raises(ValueError, match="Unsupported config format"):
                pipeline.from_config(str(config_path))


# =========================================================================
# validate
# =========================================================================


class TestValidate:
    @patch("bashgym.factory.data_designer.DATA_DESIGNER_AVAILABLE", True)
    def test_validate_success(self):
        pipeline = DataDesignerPipeline()

        mock_builder = MagicMock()
        col1 = MagicMock()
        col1.name = "task_prompt"
        col2 = MagicMock()
        col2.name = "solution"
        mock_builder.columns = [col1, col2]

        with patch.object(pipeline, "_get_pipeline_builder", return_value=mock_builder):
            with patch.object(
                DataDesignerPipeline,
                "designer",
                new_callable=PropertyMock,
            ) as mock_designer_prop:
                mock_designer = MagicMock()
                mock_designer.validate.return_value = {
                    "execution_order": ["task_prompt", "solution"]
                }
                mock_designer_prop.return_value = mock_designer

                result = pipeline.validate()

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["columns"] == ["task_prompt", "solution"]
        assert result["dag_order"] == ["task_prompt", "solution"]

    @patch("bashgym.factory.data_designer.DATA_DESIGNER_AVAILABLE", True)
    def test_validate_failure(self):
        pipeline = DataDesignerPipeline()

        mock_builder = MagicMock()
        col1 = MagicMock()
        col1.name = "broken_col"
        mock_builder.columns = [col1]

        with patch.object(pipeline, "_get_pipeline_builder", return_value=mock_builder):
            with patch.object(
                DataDesignerPipeline,
                "designer",
                new_callable=PropertyMock,
            ) as mock_designer_prop:
                mock_designer = MagicMock()
                mock_designer.validate.side_effect = ValueError("DAG cycle detected")
                mock_designer_prop.return_value = mock_designer

                result = pipeline.validate()

        assert result["valid"] is False
        assert "DAG cycle detected" in result["errors"][0]
        assert result["columns"] == ["broken_col"]


# =========================================================================
# export_nemo
# =========================================================================


class TestExportNemo:
    def test_export_splits_correctly(self):
        """Should split data according to train_val_split."""
        pd = pytest.importorskip("pandas")

        config = PipelineConfig(train_val_split=0.8)
        pipeline = DataDesignerPipeline(config=config)

        df = pd.DataFrame(
            {
                "task_prompt": [f"Task {i}" for i in range(10)],
                "solution_text": [f"Solution {i}" for i in range(10)],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.export_nemo(df, output_dir=Path(tmpdir))

        assert result["train_count"] == 8
        assert result["val_count"] == 2
        assert "train_path" in result
        assert "val_path" in result

    def test_export_creates_output_dir(self):
        pd = pytest.importorskip("pandas")

        pipeline = DataDesignerPipeline()
        df = pd.DataFrame({"task_prompt": ["Task"], "solution_text": ["Solution"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "deep" / "nested" / "output"
            pipeline.export_nemo(df, output_dir=output_dir)
            assert output_dir.exists()

    def test_export_uses_config_output_dir_as_default(self):
        pd = pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "designer_output"
            config = PipelineConfig(output_dir=out_dir)
            pipeline = DataDesignerPipeline(config=config)

            df = pd.DataFrame({"task_prompt": ["Task"], "solution_text": ["Solution"]})
            result = pipeline.export_nemo(df)
            assert result["train_path"].startswith(str(out_dir))


# =========================================================================
# _get_pipeline_builder
# =========================================================================


class TestGetPipelineBuilder:
    @patch("bashgym.factory.data_designer.DATA_DESIGNER_AVAILABLE", False)
    def test_raises_when_dd_unavailable(self):
        pipeline = DataDesignerPipeline()
        with pytest.raises(ImportError, match="data-designer"):
            pipeline._get_pipeline_builder()

    @patch("bashgym.factory.data_designer.DATA_DESIGNER_AVAILABLE", True)
    def test_unknown_pipeline_raises(self):
        pipeline = DataDesignerPipeline(PipelineConfig(pipeline="nonexistent_pipeline"))
        with patch("bashgym.factory.designer_pipelines.PIPELINES", {}):
            with pytest.raises(ValueError, match="Unknown pipeline"):
                pipeline._get_pipeline_builder()

    @patch("bashgym.factory.data_designer.DATA_DESIGNER_AVAILABLE", True)
    def test_known_pipeline_calls_builder(self):
        mock_builder = MagicMock()
        mock_builder_fn = MagicMock(return_value=mock_builder)

        pipeline = DataDesignerPipeline(PipelineConfig(pipeline="test_pipeline"))
        with patch(
            "bashgym.factory.designer_pipelines.PIPELINES",
            {"test_pipeline": mock_builder_fn},
        ):
            result = pipeline._get_pipeline_builder()

        assert result == mock_builder
        mock_builder_fn.assert_called_once_with(pipeline.config)


# =========================================================================
# SYSTEM_PROMPT
# =========================================================================


class TestSystemPrompt:
    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_tools(self):
        assert "Bash" in SYSTEM_PROMPT
        assert "Read" in SYSTEM_PROMPT
        assert "Edit" in SYSTEM_PROMPT
