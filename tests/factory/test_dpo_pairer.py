"""
Tests for DPO Failure Pairing.

All embedding API calls are mocked — no NIM API key needed.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from bashgym.factory.dpo_pairer import (
    _extract_prompt,
    _extract_repo_name,
    _serialize_trace_response,
    pair_failures_for_dpo,
)


def _write_trace(directory: Path, name: str, trace: dict) -> Path:
    """Write a trace JSON file to a directory."""
    path = directory / name
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


def _make_trace(repo_name: str, prompt: str, tool_steps: list[str], success: bool) -> dict:
    """Create a minimal trace dict."""
    steps = [
        {"tool_name": tool, "command": f"{tool.lower()} cmd", "output": "ok", "success": success}
        for tool in tool_steps
    ]
    return {
        "metadata": {
            "primary_repo": {"name": repo_name},
            "user_initial_prompt": prompt,
        },
        "trace": steps,
    }


class TestHelpers:
    """Test helper functions."""

    def test_extract_repo_name(self):
        trace = {"metadata": {"primary_repo": {"name": "bashgym"}}}
        assert _extract_repo_name(trace) == "bashgym"

    def test_extract_repo_name_missing(self):
        assert _extract_repo_name({}) == ""
        assert _extract_repo_name({"metadata": {}}) == ""

    def test_extract_prompt_from_metadata(self):
        trace = {"metadata": {"user_initial_prompt": "Fix the bug"}}
        assert _extract_prompt(trace) == "Fix the bug"

    def test_extract_prompt_from_messages(self):
        trace = {"messages": [{"role": "user", "content": "Add tests"}]}
        assert _extract_prompt(trace) == "Add tests"

    def test_extract_prompt_fallback(self):
        trace = {"trace": [{"command": "git status"}]}
        assert "git status" in _extract_prompt(trace)

    def test_serialize_trace_response_messages(self):
        trace = {
            "messages": [
                {"role": "assistant", "content": "Let me check."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "Bash", "arguments": '{"cmd": "ls"}'}}],
                },
            ]
        }
        result = _serialize_trace_response(trace)
        assert "Let me check" in result
        assert "Bash" in result

    def test_serialize_trace_response_steps(self):
        trace = {"trace": [{"tool_name": "Read", "command": "/path/file.py"}]}
        result = _serialize_trace_response(trace)
        assert "Read" in result
        assert "/path/file.py" in result


class TestPairFailuresForDpo:
    """Test the main pairing function with mocked embeddings."""

    @patch("bashgym.factory.dpo_pairer.EmbeddingDeduplicator")
    def test_basic_pairing(self, MockDedup):
        """Verify that gold and failed traces are paired by similarity."""
        # Mock the deduplicator
        mock_instance = MockDedup.return_value
        # Gold embeddings: 2 traces, each a simple vector
        # Failed embeddings: 1 trace
        # We want failed[0] to be most similar to gold[1]
        mock_instance.compute_embeddings.side_effect = [
            [[1.0, 0.0], [0.0, 1.0]],  # gold embeddings
            [[0.1, 0.9]],               # failed embeddings (similar to gold[1])
        ]
        mock_instance._cosine_similarity.side_effect = lambda a, b: sum(
            x * y for x, y in zip(a, b)
        ) / (sum(x**2 for x in a) ** 0.5 * sum(x**2 for x in b) ** 0.5 + 1e-9)

        with tempfile.TemporaryDirectory() as tmpdir:
            gold_dir = Path(tmpdir) / "gold"
            failed_dir = Path(tmpdir) / "failed"
            gold_dir.mkdir()
            failed_dir.mkdir()

            _write_trace(gold_dir, "g1.json", _make_trace("repo-a", "Fix A", ["Bash"], True))
            _write_trace(gold_dir, "g2.json", _make_trace("repo-b", "Fix B", ["Read"], True))
            _write_trace(failed_dir, "f1.json", _make_trace("repo-b", "Fix B v2", ["Read"], False))

            pairs = pair_failures_for_dpo(gold_dir, failed_dir, similarity_threshold=0.5)

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.prompt  # has a prompt
        assert pair.chosen  # gold response
        assert pair.rejected  # failed response
        assert pair.metadata["similarity"] > 0

    @patch("bashgym.factory.dpo_pairer.EmbeddingDeduplicator")
    def test_below_threshold_no_pair(self, MockDedup):
        """If similarity is below threshold, no pair is produced."""
        mock_instance = MockDedup.return_value
        mock_instance.compute_embeddings.side_effect = [
            [[1.0, 0.0]],   # gold
            [[0.0, 1.0]],   # failed (orthogonal = 0 similarity)
        ]
        mock_instance._cosine_similarity.return_value = 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            gold_dir = Path(tmpdir) / "gold"
            failed_dir = Path(tmpdir) / "failed"
            gold_dir.mkdir()
            failed_dir.mkdir()

            _write_trace(gold_dir, "g1.json", _make_trace("repo-a", "task", ["Bash"], True))
            _write_trace(failed_dir, "f1.json", _make_trace("repo-x", "task", ["Edit"], False))

            pairs = pair_failures_for_dpo(gold_dir, failed_dir, similarity_threshold=0.8)

        assert len(pairs) == 0

    @patch("bashgym.factory.dpo_pairer.EmbeddingDeduplicator")
    def test_api_unavailable_returns_empty(self, MockDedup):
        """When NIM API is unavailable, return empty list gracefully."""
        mock_instance = MockDedup.return_value
        mock_instance.compute_embeddings.side_effect = RuntimeError("NIM API unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            gold_dir = Path(tmpdir) / "gold"
            failed_dir = Path(tmpdir) / "failed"
            gold_dir.mkdir()
            failed_dir.mkdir()

            _write_trace(gold_dir, "g1.json", _make_trace("r", "t", ["Bash"], True))
            _write_trace(failed_dir, "f1.json", _make_trace("r", "t", ["Bash"], False))

            pairs = pair_failures_for_dpo(gold_dir, failed_dir)

        assert pairs == []

    def test_empty_dirs(self):
        """Empty directories return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gold_dir = Path(tmpdir) / "gold"
            failed_dir = Path(tmpdir) / "failed"
            gold_dir.mkdir()
            failed_dir.mkdir()

            pairs = pair_failures_for_dpo(gold_dir, failed_dir)

        assert pairs == []

    @patch("bashgym.factory.dpo_pairer.EmbeddingDeduplicator")
    def test_max_pairs_limit(self, MockDedup):
        """Respects max_pairs limit."""
        mock_instance = MockDedup.return_value
        # 3 gold, 3 failed, all similar
        mock_instance.compute_embeddings.side_effect = [
            [[1.0, 0.0]] * 3,  # gold
            [[0.9, 0.1]] * 3,  # failed
        ]
        mock_instance._cosine_similarity.return_value = 0.99

        with tempfile.TemporaryDirectory() as tmpdir:
            gold_dir = Path(tmpdir) / "gold"
            failed_dir = Path(tmpdir) / "failed"
            gold_dir.mkdir()
            failed_dir.mkdir()

            for i in range(3):
                _write_trace(gold_dir, f"g{i}.json", _make_trace("r", "t", ["Bash"], True))
                _write_trace(failed_dir, f"f{i}.json", _make_trace("r", "t", ["Bash"], False))

            pairs = pair_failures_for_dpo(gold_dir, failed_dir, max_pairs=2)

        assert len(pairs) <= 2
