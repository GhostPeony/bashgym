"""Tests for the terminal environment Data Designer pipeline registration."""

from bashgym.factory.designer_pipelines import PIPELINES
from bashgym.factory.designer_pipelines.terminal_env_generation import (
    EnvironmentFileDraft,
    TerminalEnvironmentDraft,
)


def test_terminal_env_generation_pipeline_registered():
    assert "terminal_env_generation" in PIPELINES
    assert PIPELINES["terminal_env_generation"].__name__ == "build_terminal_env_pipeline"


def test_terminal_environment_draft_schema_fields():
    assert "files" in TerminalEnvironmentDraft.model_fields
    assert "verifier_command" in TerminalEnvironmentDraft.model_fields
    assert "setup_commands" in TerminalEnvironmentDraft.model_fields
    assert "path" in EnvironmentFileDraft.model_fields
    assert "content" in EnvironmentFileDraft.model_fields
