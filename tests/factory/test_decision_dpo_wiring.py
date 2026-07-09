"""Tests for decision-level (step-level FAILURE->SUCCESS) DPO mining and its
wiring into DataFactory.process_trace_directory.
"""

from unittest import mock

import pytest

from bashgym.factory.data_factory import DataFactory, DataFactoryConfig
from bashgym.factory.trace_processor import ProcessedTrace, TraceProcessor, TraceQualityMetrics


def _failure_then_success_steps():
    """Normalized steps that deterministically yield a FAILURE decision (step 0,
    explicit success=False + error output) followed by a SUCCESS decision (step 1,
    a different command with a git-commit within lookahead)."""
    return [
        {
            "tool": "Bash",
            "command": "cat config.yaml",
            "output": "cat: config.yaml: No such file or directory",
            "success": False,
            "exit_code": 1,
            "cognitive": {
                "thinking": (
                    "Let me try reading the config with cat. I could also use a "
                    "different approach if that fails."
                )
            },
            "metadata": {},
        },
        {
            "tool": "Bash",
            "command": "ls -la",
            "output": "config.yml settings.json",
            "success": True,
            "exit_code": 0,
            "cognitive": {
                "thinking": "That didn't work. Let me list the directory to find the right filename."
            },
            "metadata": {},
        },
        {
            "tool": "Bash",
            "command": "git commit -m 'fix config path'",
            "output": "1 file changed",
            "success": True,
            "exit_code": 0,
            "cognitive": {},
            "metadata": {},
        },
    ]


def _processed(steps, tmp_path):
    return ProcessedTrace(
        trace_id="t1",
        original_path=tmp_path / "t1.json",
        task_prompt="Fix the config path",
        normalized_steps=steps,
        quality_metrics=TraceQualityMetrics(),
        metadata={},
    )


def _factory(tmp_path, **cfg):
    return DataFactory(DataFactoryConfig(output_dir=str(tmp_path / "out"), **cfg))


class TestDecisionDpoForTrace:
    def test_mines_failure_to_success_pair(self, tmp_path):
        factory = _factory(tmp_path)
        processed = _processed(_failure_then_success_steps(), tmp_path)
        pairs = factory._decision_dpo_for_trace(processed)
        assert len(pairs) >= 1
        pair = pairs[0]
        assert pair.metadata.get("decision_level") is True
        # rejected = the failed action; chosen = the recovery action
        assert "cat config.yaml" in pair.rejected
        assert "ls -la" in pair.chosen
        assert pair.prompt  # carries the system prompt + task
        assert pair.metadata["pair_generation_method"] == "decision_level_failure_recovery"
        assert pair.metadata["prompt_hash"]
        assert pair.metadata["chosen_trace_id"] == "t1"
        assert pair.metadata["rejected_trace_id"] == "t1"

    def test_no_decisions_returns_empty(self, tmp_path):
        factory = _factory(tmp_path)
        # A single clean step triggers no FAILURE->SUCCESS sequence
        processed = _processed(
            [{"tool": "Bash", "command": "ls", "output": "ok", "success": True, "exit_code": 0}],
            tmp_path,
        )
        assert factory._decision_dpo_for_trace(processed) == []

    def test_empty_steps_returns_empty(self, tmp_path):
        factory = _factory(tmp_path)
        assert factory._decision_dpo_for_trace(_processed([], tmp_path)) == []


class TestProcessTraceDirectoryWiring:
    @pytest.mark.asyncio
    async def test_includes_decision_dpo(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        (gold / "g1.json").write_text(
            "{}"
        )  # process_gold_trace -> None; decision path uses the mock
        factory = _factory(tmp_path)
        processed = _processed(_failure_then_success_steps(), tmp_path)
        with mock.patch.object(TraceProcessor, "process_trace", return_value=processed):
            _train, dpo = await factory.process_trace_directory(gold)
        decision_dpo = [d for d in dpo if d.metadata.get("decision_level")]
        assert len(decision_dpo) >= 1

    @pytest.mark.asyncio
    async def test_flag_disables_decision_dpo(self, tmp_path):
        gold = tmp_path / "gold"
        gold.mkdir()
        (gold / "g1.json").write_text("{}")
        factory = _factory(tmp_path, generate_decision_dpo=False)
        processed = _processed(_failure_then_success_steps(), tmp_path)
        with mock.patch.object(TraceProcessor, "process_trace", return_value=processed) as m:
            _train, dpo = await factory.process_trace_directory(gold)
        assert [d for d in dpo if d.metadata.get("decision_level")] == []
        m.assert_not_called()  # the disabled block never instantiates the processor

    @pytest.mark.asyncio
    async def test_one_bad_trace_does_not_break_batch(self, tmp_path):
        """A trace that raises during processing is skipped, not fatal."""
        gold = tmp_path / "gold"
        gold.mkdir()
        (gold / "g1.json").write_text("{}")
        factory = _factory(tmp_path)
        with mock.patch.object(TraceProcessor, "process_trace", side_effect=RuntimeError("boom")):
            train, dpo = await factory.process_trace_directory(gold)
        # No crash; just no decision pairs from the broken trace
        assert [d for d in dpo if d.metadata.get("decision_level")] == []
