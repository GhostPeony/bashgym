"""
Tests for Loss-Targeted Mining in TraceResearcher.

Verifies LossTargetedMiner class and new search params without
requiring actual GPU/model loading (all model calls are mocked).
"""

from unittest.mock import MagicMock, patch

from bashgym.gym.trace_researcher import (
    DATA_SEARCH_SPACE,
    DataPipelineConfig,
    LossTargetedMiner,
    TraceResearchConfig,
)


class TestDataSearchSpaceNewParams:
    """Verify the new search params exist."""

    def test_loss_bias_weight_in_search_space(self):
        assert "loss_bias_weight" in DATA_SEARCH_SPACE
        spec = DATA_SEARCH_SPACE["loss_bias_weight"]
        assert spec["type"] == "float"
        assert spec["min"] == 0.0
        assert spec["max"] == 2.0


class TestDataPipelineConfigNewFields:
    """Verify the new fields on DataPipelineConfig."""

    def test_default_values(self):
        config = DataPipelineConfig()
        assert config.loss_bias_weight == 0.0
        assert config.focus_repos == []
        assert config.focus_tool_patterns == []

    def test_set_focus_repos(self):
        config = DataPipelineConfig(focus_repos=["bashgym", "ghostwork"])
        assert config.focus_repos == ["bashgym", "ghostwork"]


class TestTraceResearchConfigNewFields:
    """Verify the new loss mining config fields."""

    def test_defaults(self):
        config = TraceResearchConfig()
        assert config.loss_mining_enabled is False
        assert config.loss_mining_adapter_path == ""
        assert config.loss_mining_top_k == 20

    def test_enable_mining(self):
        config = TraceResearchConfig(
            loss_mining_enabled=True,
            loss_mining_model="unsloth/gemma-4-E4B-it",
            loss_mining_adapter_path="/path/to/adapter",
            loss_mining_top_k=10,
        )
        assert config.loss_mining_enabled is True
        assert config.loss_mining_top_k == 10


class TestLossTargetedMiner:
    """Test LossTargetedMiner with mocked model."""

    def _make_example(self, repo_name: str, tool_names: list[str]) -> dict:
        """Create a minimal training example dict."""
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": f"Fix bug in {repo_name}"},
        ]
        for tool in tool_names:
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": tool, "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "content": "ok"})
        return {
            "messages": messages,
            "metadata": {"repo_name": repo_name},
        }

    @patch("bashgym.gym.trace_researcher.LossTargetedMiner._ensure_loaded")
    def test_score_examples(self, mock_load):
        """Verify examples are scored and sorted by loss descending."""
        # Mock model that returns loss = len(messages)
        mock_models = {
            "ft_loss": lambda msgs: float(len(msgs)),
            "has_adapter": False,
        }
        mock_load.return_value = mock_models

        miner = LossTargetedMiner(top_k=5)
        miner._models = mock_models  # bypass lazy load

        examples = [
            self._make_example("repo-a", ["Bash"]),            # 4 messages -> loss 4.0
            self._make_example("repo-b", ["Bash", "Read"]),    # 6 messages -> loss 6.0
            self._make_example("repo-c", ["Edit"]),            # 4 messages -> loss 4.0
        ]

        scored = miner.score_examples(examples)

        assert len(scored) == 3
        # Highest loss first
        assert scored[0][1] == 6.0
        assert scored[0][0]["metadata"]["repo_name"] == "repo-b"

    @patch("bashgym.gym.trace_researcher.LossTargetedMiner._ensure_loaded")
    def test_extract_signatures(self, mock_load):
        """Verify repo names and tool patterns are extracted from top-k."""
        miner = LossTargetedMiner(top_k=2)

        scored = [
            (self._make_example("bashgym", ["Bash", "Edit"]), 5.0),
            (self._make_example("ghostwork", ["Read"]), 4.5),
            (self._make_example("other-repo", ["Glob"]), 1.0),  # not in top-k
        ]

        repos, tools = miner.extract_signatures(scored)

        assert "bashgym" in repos
        assert "ghostwork" in repos
        assert "other-repo" not in repos
        assert "Bash" in tools
        assert "Edit" in tools
        assert "Read" in tools
        assert "Glob" not in tools  # outside top_k

    def test_empty_examples(self):
        """Score empty list returns empty."""
        miner = LossTargetedMiner()
        miner._models = {"ft_loss": lambda msgs: 1.0, "has_adapter": False}
        assert miner.score_examples([]) == []

    def test_examples_without_messages_skipped(self):
        """Examples without 'messages' key are silently skipped."""
        miner = LossTargetedMiner()
        miner._models = {"ft_loss": lambda msgs: 1.0, "has_adapter": False}
        scored = miner.score_examples([{"no_messages": True}])
        assert scored == []
