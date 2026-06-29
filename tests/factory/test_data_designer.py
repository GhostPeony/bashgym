"""Tests for Data Designer pipeline integration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from bashgym.factory.data_designer import (
    DATA_DESIGNER_AVAILABLE,
    SYSTEM_PROMPT,
    DataDesignerPipeline,
    PipelineConfig,
    ProviderSpec,
    _is_truthy,
    _looks_like_chat_model,
    _looks_like_code_model,
    list_inference_models,
    provider_model_ids,
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
    # Original (v0.5.x) flags plus the v0.6.1-capability flags.
    _FLAGS = (
        "HAS_STRUCTURED_COLUMN",
        "HAS_JUDGE_COLUMN",
        "HAS_VALIDATION_COLUMN",
        "HAS_EMBEDDING_COLUMN",
        "HAS_CUSTOM_COLUMN",
        "HAS_EXPRESSION_COLUMN",
        "HAS_CODE_COLUMN",
        "HAS_SEED_DATASET_COLUMN",
        "HAS_AGENT_ROLLOUT",
        "HAS_MCP",
        "HAS_WORKFLOW",
        "HAS_SCHEMA_TRANSFORM",
    )

    def test_feature_flags_exist(self):
        """Feature flags should be defined regardless of DD availability."""
        from bashgym.factory import data_designer

        for flag in self._FLAGS:
            assert hasattr(data_designer, flag), f"missing feature flag {flag}"

    def test_feature_flags_are_bool(self):
        """All feature flags should be booleans."""
        from bashgym.factory import data_designer

        for flag in self._FLAGS:
            assert isinstance(getattr(data_designer, flag), bool), f"{flag} is not bool"

    @pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
    def test_flags_true_when_installed(self):
        """With data-designer 0.6.1 installed, every capability flag resolves True."""
        for flag in self._FLAGS:
            from bashgym.factory import data_designer

            assert getattr(data_designer, flag) is True, f"{flag} should be True on 0.6.1"


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

    def test_prepare_source_writes_source_and_dataset_cards(self, tmp_path):
        pipeline = DataDesignerPipeline(PipelineConfig(output_dir=tmp_path))

        prepared = pipeline.prepare_source("helpsteer2", goal="reward_model")

        assert prepared["source_manifest"]["source"]["id"] == "helpsteer2"
        assert prepared["dataset_card"]["schema_version"] == "bashgym.dataset_card.v1"
        assert prepared["dataset_card"]["source_id"] == "helpsteer2"
        assert tmp_path.joinpath("source_manifest.json").exists()
        assert tmp_path.joinpath("dataset_card.json").exists()

    def test_prepare_source_blocks_eval_only_training_use(self, tmp_path):
        pipeline = DataDesignerPipeline(PipelineConfig(output_dir=tmp_path))

        with pytest.raises(ValueError, match="eval_only_source_for_training"):
            pipeline.prepare_source("harbor_terminal_bench", goal="sft")

    def test_prepare_source_can_convert_local_input_artifacts(self, tmp_path):
        source_path = tmp_path / "source.jsonl"
        source_path.write_text(
            json.dumps(
                {
                    "id": "uf-1",
                    "prompt": "Fix a failing test.",
                    "chosen": "Run pytest and patch the failing function.",
                    "rejected": "Claim success without running tests.",
                    "metadata": {
                        "quality_score": 0.9,
                        "label_source": "fixture",
                        "decontamination_status": "checked",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        pipeline = DataDesignerPipeline(PipelineConfig(output_dir=tmp_path / "out"))

        prepared = pipeline.prepare_source(
            "ultrafeedback_binarized",
            goal="dpo",
            input_path=source_path,
        )

        assert prepared["artifact_report"]["ok"] is True
        assert prepared["dataset_card"]["artifacts"][0]["artifact_type"] == "dpo_pairs"
        assert tmp_path.joinpath("out", "dpo_pairs.jsonl").exists()

    def test_prepare_source_can_fetch_then_convert_artifacts(self, tmp_path, monkeypatch):
        def fake_fetch(card, *, output_dir, split, subset=None, revision=None, limit=None):
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            records_path = output_path / "source_records.jsonl"
            records_path.write_text(
                json.dumps(
                    {
                        "id": "uf-1",
                        "prompt": "Fix a failing test.",
                        "chosen": "Run pytest and patch the failing function.",
                        "rejected": "Claim success without running tests.",
                        "metadata": {"decontamination_status": "checked"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return {
                "schema_version": "bashgym.source_fetch.v1",
                "ok": True,
                "source_id": card.id,
                "source_name": card.name,
                "huggingface_id": card.huggingface_id,
                "split": split,
                "subset": subset,
                "revision": revision,
                "limit": limit,
                "output_dir": str(output_path),
                "records_path": str(records_path),
                "report_path": str(output_path / "source_fetch_report.json"),
                "record_count": 1,
                "truncated": False,
                "warnings": [],
                "errors": [],
            }

        monkeypatch.setattr("bashgym.sources.fetch_source_records", fake_fetch)
        pipeline = DataDesignerPipeline(PipelineConfig(output_dir=tmp_path / "out"))

        prepared = pipeline.prepare_source(
            "ultrafeedback_binarized",
            goal="dpo",
            fetch=True,
            limit=1,
        )

        assert prepared["fetch_report"]["schema_version"] == "bashgym.source_fetch.v1"
        assert prepared["dataset_card"]["source_records_path"].endswith("source_records.jsonl")
        assert prepared["artifact_report"]["ok"] is True
        assert tmp_path.joinpath("out", "dpo_pairs.jsonl").exists()


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
                    mock_designer.create.return_value.load_dataset.return_value = mock_df
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
# Live construction against the installed data-designer (0.6.1)
# =========================================================================


@pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
class TestRealPipelineConstruction:
    """Build real config DAGs against the installed Data Designer (no model calls).

    Exercises the 0.6.x API surface: provider wiring lives on the DataDesigner
    instance, ModelConfig binds via provider name, and every pipeline's column
    DAG constructs.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "coding_agent_sft",
            "coding_agent_dpo",
            "coding_agent_distill",
            "tool_use_sft",
            "from_external",
            "from_source",
            "from_unstructured",
        ],
    )
    def test_pipeline_builds(self, name):
        from bashgym.factory.designer_pipelines import PIPELINES

        builder = PIPELINES[name](PipelineConfig(pipeline=name))
        assert builder is not None

    def test_build_model_providers_single(self):
        from bashgym.factory.designer_pipelines import build_model_providers

        providers = build_model_providers(PipelineConfig(provider="nvidia"))
        assert len(providers) == 1
        assert providers[0].name == "nvidia"

    def test_build_model_providers_multi(self):
        from bashgym.factory.designer_pipelines import build_model_providers

        config = PipelineConfig(
            providers=[
                ProviderSpec(name="nvidia", endpoint="https://nim.example.com"),
                ProviderSpec(name="anthropic", endpoint="https://api.anthropic.com"),
            ]
        )
        providers = build_model_providers(config)
        assert {p.name for p in providers} == {"nvidia", "anthropic"}

    def test_designer_instantiates_with_providers(self):
        pipeline = DataDesignerPipeline(PipelineConfig(pipeline="coding_agent_sft"))
        # Accessing .designer builds DataDesigner(model_providers=...).
        assert pipeline.designer is not None


# =========================================================================
# from_agent_rollouts (native AgentRolloutSeedSource ingestion)
# =========================================================================


class TestFromAgentRollouts:
    def test_raises_when_agent_rollout_unsupported(self):
        """Without AgentRolloutSeedSource support, the call raises clearly."""
        pipeline = DataDesignerPipeline()
        with patch("bashgym.factory.data_designer.HAS_AGENT_ROLLOUT", False):
            with pytest.raises(RuntimeError, match="AgentRolloutSeedSource"):
                pipeline.from_agent_rollouts()

    @pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
    def test_resolve_rollout_format(self):
        import data_designer.config as dd

        pipeline = DataDesignerPipeline()
        assert pipeline._resolve_rollout_format("claude_code") == dd.AgentRolloutFormat.CLAUDE_CODE
        assert (
            pipeline._resolve_rollout_format("HERMES_AGENT") == dd.AgentRolloutFormat.HERMES_AGENT
        )
        with pytest.raises(ValueError, match="Unknown rollout format"):
            pipeline._resolve_rollout_format("not_a_format")

    @pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
    def test_atif_requires_path(self):
        pipeline = DataDesignerPipeline(PipelineConfig(pipeline="coding_agent_distill"))
        with pytest.raises(ValueError, match="path is required"):
            pipeline.from_agent_rollouts(rollout_format="atif")

    @pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
    def test_attaches_seed_and_creates(self):
        pipeline = DataDesignerPipeline(PipelineConfig(pipeline="coding_agent_distill"))
        mock_builder = MagicMock()
        mock_df = MagicMock()
        with patch.object(pipeline, "_get_pipeline_builder", return_value=mock_builder):
            with patch.object(
                DataDesignerPipeline, "designer", new_callable=PropertyMock
            ) as mock_designer_prop:
                mock_designer = MagicMock()
                mock_designer.create.return_value.load_dataset.return_value = mock_df
                mock_designer_prop.return_value = mock_designer
                result = pipeline.from_agent_rollouts(rollout_format="claude_code", num_records=7)

        assert result is mock_df
        mock_builder.with_seed_dataset.assert_called_once()
        _, kwargs = mock_designer.create.call_args
        assert kwargs.get("num_records") == 7


# =========================================================================
# coding_agent_distill pipeline internals
# =========================================================================


@pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
class TestDistillPipeline:
    def test_judge_scores_five_dimensions(self):
        from bashgym.factory.designer_pipelines.coding_agent_distill import _sft_judge_scores

        scores = _sft_judge_scores()
        assert len(scores) == 5
        assert {s.name for s in scores} == {
            "groundedness",
            "standalone_task",
            "response_quality",
            "faithfulness",
            "training_utility",
        }

    def test_models_have_expected_fields(self):
        from bashgym.factory.designer_pipelines.coding_agent_distill import (
            AgentRolloutFinetuningRecord,
            AgentRolloutTraceDigest,
        )

        assert "user_goal" in AgentRolloutTraceDigest.model_fields
        assert "training_value" in AgentRolloutTraceDigest.model_fields
        assert "instruction" in AgentRolloutFinetuningRecord.model_fields
        assert "response" in AgentRolloutFinetuningRecord.model_fields


# =========================================================================
# from_traces seed attach (real DataFrameSeedSource against 0.6.1)
# =========================================================================


@pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
class TestFromTracesSeed:
    def test_attaches_real_dataframe_seed(self, tmp_path):
        """Exercises real dd.DataFrameSeedSource(df=...) — guards the 0.6.1 kwarg."""
        import data_designer.config as dd

        trace = {
            "metadata": {"user_initial_prompt": "Fix the failing test"},
            "trace": [{"tool_name": "bash", "command": "pytest tests/"}],
        }
        (tmp_path / "t.json").write_text(json.dumps(trace), encoding="utf-8")

        pipeline = DataDesignerPipeline()
        mock_builder = MagicMock()
        mock_df = MagicMock()
        with patch.object(pipeline, "_get_pipeline_builder", return_value=mock_builder):
            with patch.object(
                DataDesignerPipeline, "designer", new_callable=PropertyMock
            ) as mock_designer_prop:
                mock_designer = MagicMock()
                mock_designer.create.return_value.load_dataset.return_value = mock_df
                mock_designer_prop.return_value = mock_designer
                result = pipeline.from_traces(tmp_path, num_records=3)

        assert result is mock_df
        seed_arg = mock_builder.with_seed_dataset.call_args[0][0]
        assert isinstance(seed_arg, dd.DataFrameSeedSource)


# =========================================================================
# Quality-gate export filtering + model adaptability
# =========================================================================


class TestIsTruthy:
    def test_bools(self):
        assert _is_truthy(True) is True
        assert _is_truthy(False) is False

    def test_strings(self):
        assert _is_truthy("true") is True
        assert _is_truthy("True") is True
        assert _is_truthy("1") is True
        assert _is_truthy("yes") is True
        assert _is_truthy("false") is False
        assert _is_truthy("") is False

    def test_other(self):
        assert _is_truthy(1) is True
        assert _is_truthy(0) is False
        assert _is_truthy(None) is False


class TestModelHelpers:
    def test_looks_like_code_model(self):
        assert _looks_like_code_model("deepseek-ai/deepseek-coder-6.7b-instruct")
        assert _looks_like_code_model("bigcode/starcoder2-15b")
        assert _looks_like_code_model("mistralai/codestral-22b-instruct-v0.1")
        assert not _looks_like_code_model("meta/llama-3.3-70b-instruct")

    def test_looks_like_chat_model(self):
        assert _looks_like_chat_model("meta/llama-3.3-70b-instruct")
        assert _looks_like_chat_model("nvidia/llama-3.1-nemotron-70b-instruct")
        assert not _looks_like_chat_model("bigcode/starcoder2-15b")


class TestExportNemoQualityGate:
    def _df(self, pd, flag_col="passes_quality", flags=(True, False, True)):
        return pd.DataFrame(
            {
                "task_prompt": [f"task {i}" for i in range(len(flags))],
                "solution_text": [f"sol {i}" for i in range(len(flags))],
                flag_col: list(flags),
            }
        )

    def test_filters_failing_rows(self):
        pd = pytest.importorskip("pandas")
        pipeline = DataDesignerPipeline()
        with tempfile.TemporaryDirectory() as tmp:
            res = pipeline.export_nemo(self._df(pd), output_dir=Path(tmp))
        assert res["filtered_out"] == 1
        assert res["train_count"] + res["val_count"] == 2

    def test_no_filter_when_disabled(self):
        pd = pytest.importorskip("pandas")
        pipeline = DataDesignerPipeline()
        with tempfile.TemporaryDirectory() as tmp:
            res = pipeline.export_nemo(self._df(pd), output_dir=Path(tmp), keep_only_passing=False)
        assert res["filtered_out"] == 0
        assert res["train_count"] + res["val_count"] == 3

    def test_no_flag_column_keeps_all(self):
        pd = pytest.importorskip("pandas")
        pipeline = DataDesignerPipeline()
        df = pd.DataFrame({"task_prompt": ["a", "b"], "solution_text": ["x", "y"]})
        with tempfile.TemporaryDirectory() as tmp:
            res = pipeline.export_nemo(df, output_dir=Path(tmp))
        assert res["filtered_out"] == 0

    def test_recommended_for_sft_fallback(self):
        pd = pytest.importorskip("pandas")
        pipeline = DataDesignerPipeline()
        df = self._df(pd, flag_col="recommended_for_sft", flags=(True, False, False))
        with tempfile.TemporaryDirectory() as tmp:
            res = pipeline.export_nemo(df, output_dir=Path(tmp))  # default flag absent
        assert res["filtered_out"] == 2


class TestResolveModels:
    def test_substitutes_unavailable(self):
        available = [
            "meta/llama-3.3-70b-instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
        ]
        with patch("bashgym.factory.data_designer.provider_model_ids", return_value=available):
            cfg = PipelineConfig(
                text_model="meta/llama-3.3-70b-instruct",
                code_model="qwen/qwen2.5-coder-32b-instruct",  # not served
                judge_model="meta/llama-3.3-70b-instruct",
            ).resolve_models()
        assert cfg.text_model == "meta/llama-3.3-70b-instruct"  # available -> kept
        assert cfg.code_model == "deepseek-ai/deepseek-coder-6.7b-instruct"  # swapped
        assert cfg.judge_model == "meta/llama-3.3-70b-instruct"

    def test_noop_when_discovery_empty(self):
        with patch("bashgym.factory.data_designer.provider_model_ids", return_value=[]):
            cfg = PipelineConfig(code_model="qwen/qwen2.5-coder-32b-instruct").resolve_models()
        assert cfg.code_model == "qwen/qwen2.5-coder-32b-instruct"  # unchanged


class TestProviderModelIds:
    def test_returns_empty_on_error(self):
        with patch("httpx.get", side_effect=RuntimeError("boom")):
            assert provider_model_ids("nvidia", "https://example.com/v1") == []


class TestListInferenceModels:
    def test_aggregates_discovery(self):
        fake = {
            "inference": [{"id": "nim/meta/llama-3.3-70b-instruct"}],
            "local": [{"id": "ollama/qwen2.5"}],
        }
        with patch("bashgym.providers.detector.get_available_models_sync", return_value=fake):
            models = list_inference_models()
        ids = {m["id"] for m in models}
        assert ids == {"nim/meta/llama-3.3-70b-instruct", "ollama/qwen2.5"}

    def test_returns_empty_on_failure(self):
        with patch(
            "bashgym.providers.detector.get_available_models_sync",
            side_effect=RuntimeError("down"),
        ):
            assert list_inference_models() == []


# =========================================================================
# Phase 4: workflow chaining + schema-transform export
# =========================================================================


class TestMessagesSchemaTransform:
    @pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
    def test_template_with_system(self):
        from bashgym.factory.designer_pipelines import messages_schema_transform

        proc = messages_schema_transform("task_prompt", "solution_text", system_prompt="sys")
        msgs = proc.template["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[0]["content"] == "sys"
        assert "{{ task_prompt }}" in msgs[1]["content"]
        assert "{{ solution_text }}" in msgs[2]["content"]

    @pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
    def test_template_without_system(self):
        from bashgym.factory.designer_pipelines import messages_schema_transform

        proc = messages_schema_transform("u", "a")
        assert [m["role"] for m in proc.template["messages"]] == ["user", "assistant"]


class TestGenerateChained:
    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="at least one stage"):
            DataDesignerPipeline().generate_chained([])

    @pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
    def test_runs_workflow_and_records_stats(self):
        pipeline = DataDesignerPipeline()
        mock_df = MagicMock()
        mock_df.__len__.return_value = 3
        with patch.object(
            DataDesignerPipeline, "designer", new_callable=PropertyMock
        ) as mock_designer_prop:
            designer = MagicMock()
            workflow = MagicMock()
            designer.compose_workflow.return_value = workflow
            workflow.run.return_value.load_dataset.return_value = mock_df
            mock_designer_prop.return_value = designer

            df = pipeline.generate_chained(
                [
                    {"name": "gen", "builder": MagicMock(), "num_records": 5},
                    {"name": "to_msgs", "builder": MagicMock(), "output": "processor:x"},
                ]
            )

        assert df is mock_df
        assert workflow.add_stage.call_count == 2
        assert pipeline.last_stats.records == 3
        assert pipeline.last_stats.stages == ["gen", "to_msgs"]


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
