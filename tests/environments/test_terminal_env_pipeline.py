"""Tests for the terminal environment Data Designer pipeline registration."""

import json

import pytest

from bashgym.environments.builder import materialize_environment
from bashgym.environments.loader import environment_from_record
from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig
from bashgym.factory.designer_pipelines import PIPELINES
from bashgym.factory.designer_pipelines.terminal_env_generation import (
    EnvironmentFileDraft,
    TerminalEnvironmentDraft,
)


def _generated_terminal_row() -> dict:
    draft = TerminalEnvironmentDraft(
        summary="Repair a CSV summarizer and verify its aggregate output.",
        files=[
            EnvironmentFileDraft(
                path="summarize.py",
                content="raise NotImplementedError\n",
                purpose="Implementation the agent must repair",
            ),
            EnvironmentFileDraft(
                path="tests/test_summary.py",
                content="def test_summary():\n    assert True\n",
                purpose="Executable success verifier",
            ),
        ],
        verifier_path="tests/test_summary.py",
        verifier_command="pytest -q tests/test_summary.py",
        setup_commands=["python -m pip install pytest"],
        expected_solution_shape="Implement the missing CSV aggregation.",
    )
    return {
        "id": "designer_terminal_001",
        "task_prompt": "Repair summarize.py so the CSV totals are correct.",
        "domain": "data_processing",
        "skill_type": "python",
        "verifier_kind": "exact_success",
        "environment_draft": draft.model_dump(),
        "passes_quality": True,
    }


def test_terminal_env_generation_pipeline_registered():
    assert "terminal_env_generation" in PIPELINES
    assert PIPELINES["terminal_env_generation"].__name__ == "build_terminal_env_pipeline"


def test_terminal_environment_draft_schema_fields():
    assert "files" in TerminalEnvironmentDraft.model_fields
    assert "verifier_command" in TerminalEnvironmentDraft.model_fields
    assert "setup_commands" in TerminalEnvironmentDraft.model_fields
    assert "path" in EnvironmentFileDraft.model_fields
    assert "content" in EnvironmentFileDraft.model_fields


def test_generated_nested_draft_round_trips_to_materializable_environment(tmp_path):
    spec = environment_from_record(_generated_terminal_row(), source="data_designer")

    assert spec.validation_errors() == []
    assert spec.instruction == "Repair summarize.py so the CSV totals are correct."
    assert spec.files == {
        "summarize.py": "raise NotImplementedError\n",
        "tests/test_summary.py": "def test_summary():\n    assert True\n",
    }
    assert spec.verifier.command == "pytest -q tests/test_summary.py"
    assert spec.verifier.path == "tests/test_summary.py"
    assert spec.build.setup_commands == ["python -m pip install pytest"]

    materialized = materialize_environment(spec, tmp_path)

    assert (materialized.path / "summarize.py").read_text(encoding="utf-8") == (
        "raise NotImplementedError\n"
    )
    assert (materialized.path / "tests" / "test_summary.py").is_file()


def test_nemo_export_writes_generated_terminal_environment_row(tmp_path):
    pd = pytest.importorskip("pandas")
    pipeline = DataDesignerPipeline(PipelineConfig(train_val_split=1.0))

    result = pipeline.export_nemo(pd.DataFrame([_generated_terminal_row()]), output_dir=tmp_path)

    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result["train_count"] == 1
    assert result["val_count"] == 0
    assert len(records) == 1
    assert records[0]["messages"][1]["content"] == (
        "Repair summarize.py so the CSV totals are correct."
    )
    assert json.loads(records[0]["messages"][2]["content"])["verifier_command"] == (
        "pytest -q tests/test_summary.py"
    )
