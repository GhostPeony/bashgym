"""Unit tests for bashgym.research.scoring. No network, all inputs mocked."""
from datetime import datetime, timedelta

import pytest

from bashgym.research.scoring import DatasetMetadata, score_dataset


def _fresh_meta(**overrides) -> DatasetMetadata:
    """Build a 'nominal good' DatasetMetadata, override specific fields per test."""
    base = DatasetMetadata(
        repo_id="test-org/test-dataset",
        tags=["code-generation", "task_categories:text-generation"],
        license="apache-2.0",
        num_rows=5_000,
        download_size_bytes=50_000_000,
        features={"messages": "list<struct>"},
        last_modified=datetime.now() - timedelta(days=30),
        downloads=5_000,
        gated=False,
        description="A high quality code generation dataset.",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestHardFilters:
    def test_rejects_gated_dataset(self):
        meta = _fresh_meta(gated=True)
        result = score_dataset(meta)
        assert result.rejected is True
        assert result.rejection_reason is not None
        assert "gated" in result.rejection_reason.lower()

    def test_rejects_non_commercial_license(self):
        meta = _fresh_meta(license="cc-by-nc-4.0")
        result = score_dataset(meta)
        assert result.rejected is True
        assert "license" in result.rejection_reason.lower()

    def test_rejects_too_small(self):
        meta = _fresh_meta(num_rows=5)
        result = score_dataset(meta)
        assert result.rejected is True
        assert "small" in result.rejection_reason.lower() or "size" in result.rejection_reason.lower()

    def test_rejects_too_large(self):
        meta = _fresh_meta(num_rows=5_000_000)
        result = score_dataset(meta)
        assert result.rejected is True
        assert "large" in result.rejection_reason.lower() or "size" in result.rejection_reason.lower()


class TestTaskMatchDimension:
    def test_high_score_for_direct_tag_match(self):
        meta = _fresh_meta(tags=["code-generation", "agentic", "tool-use"])
        result = score_dataset(meta)
        assert result.rejected is False
        task_lines = [r for r in result.reasons if "task" in r.lower()]
        assert len(task_lines) >= 1
        assert any("+" in r for r in task_lines)

    def test_low_score_for_no_code_tags(self):
        meta = _fresh_meta(tags=["translation", "multilingual"], description="A translation corpus.")
        result = score_dataset(meta)
        assert result.rejected is False
        task_lines = [r for r in result.reasons if "task" in r.lower()]
        assert len(task_lines) >= 1
        # The task-match contribution should be 0.00 (no signal)
        assert any("+0.00" in r for r in task_lines)


class TestLicenseDimension:
    def test_apache_2_full_score(self):
        meta = _fresh_meta(license="apache-2.0")
        result = score_dataset(meta)
        assert any("license" in r and "apache" in r.lower() for r in result.reasons)

    def test_cc_by_sa_warned_not_rejected(self):
        meta = _fresh_meta(license="cc-by-sa-4.0")
        result = score_dataset(meta)
        assert result.rejected is False
        assert any("cc-by-sa" in w.lower() for w in result.warnings)


class TestSizeDimension:
    def test_ideal_size_full_score(self):
        meta = _fresh_meta(num_rows=4_200)
        result = score_dataset(meta)
        assert any("size" in r.lower() and "ideal" in r.lower() for r in result.reasons)

    def test_oversize_downweight(self):
        meta = _fresh_meta(num_rows=800_000)  # large but not hard-rejected
        result = score_dataset(meta)
        assert result.rejected is False
        assert any("size" in r.lower() for r in result.reasons)


class TestFreshnessDimension:
    def test_recent_full_score(self):
        meta = _fresh_meta(last_modified=datetime.now() - timedelta(days=30))
        result = score_dataset(meta)
        assert any("fresh" in r.lower() or "updated" in r.lower() for r in result.reasons)

    def test_stale_downweight(self):
        meta = _fresh_meta(last_modified=datetime.now() - timedelta(days=900))
        result = score_dataset(meta)
        assert result.rejected is False


class TestSchemaInference:
    def test_sft_messages_schema(self):
        meta = _fresh_meta(features={"messages": "list<struct>"})
        result = score_dataset(meta)
        assert result.bashgym_format == "sft"

    def test_dpo_chosen_rejected_schema(self):
        meta = _fresh_meta(features={"prompt": "string", "chosen": "string", "rejected": "string"})
        result = score_dataset(meta)
        assert result.bashgym_format == "dpo"

    def test_grpo_prompt_plus_tests(self):
        meta = _fresh_meta(features={"prompt": "string", "tests": "string"})
        result = score_dataset(meta)
        assert result.bashgym_format == "grpo"

    def test_instruction_output_maps_to_sft(self):
        meta = _fresh_meta(features={"instruction": "string", "output": "string"})
        result = score_dataset(meta)
        assert result.bashgym_format == "sft"

    def test_no_match_returns_none_format(self):
        meta = _fresh_meta(features={"foo": "string", "bar": "int"})
        result = score_dataset(meta)
        assert result.bashgym_format is None


class TestHeuristicFormatInference:
    """Fallback schema detection via PROMPT_LIKE_COLS + RESPONSE_LIKE_COLS."""

    def test_mbpp_text_plus_code_maps_to_grpo(self):
        """mbpp has text (prompt) + code (response) + test_list → GRPO-compatible."""
        meta = _fresh_meta(features={"task_id": "string", "text": "string", "code": "string", "test_list": "list"})
        result = score_dataset(meta)
        assert result.bashgym_format == "grpo"
        # Heuristic matches get lower schema credit than exact matches.
        assert any("heuristic" in r.lower() for r in result.reasons)

    def test_simple_text_plus_code_maps_to_sft(self):
        """Without tests, text+code is a plain SFT pair."""
        meta = _fresh_meta(features={"text": "string", "code": "string"})
        result = score_dataset(meta)
        assert result.bashgym_format == "sft"
        assert any("heuristic" in r.lower() for r in result.reasons)

    def test_swe_bench_problem_statement_plus_patch_maps_to_sft(self):
        meta = _fresh_meta(features={
            "instance_id": "string",
            "problem_statement": "string",
            "patch": "string",
            "FAIL_TO_PASS": "list",
        })
        result = score_dataset(meta)
        # problem_statement (prompt-like) + patch (response-like) + FAIL_TO_PASS (test-like)
        # → GRPO takes priority over SFT via the has_test_like branch
        assert result.bashgym_format == "grpo"

    def test_humaneval_pro_raw_problem_raw_solution(self):
        meta = _fresh_meta(features={
            "id": "string",
            "raw_problem": "string",
            "raw_solution": "string",
            "test_code": "string",
        })
        result = score_dataset(meta)
        # raw_problem + test_code → grpo
        assert result.bashgym_format == "grpo"

    def test_exact_schema_beats_heuristic(self):
        """When exact and heuristic patterns both match, exact wins."""
        meta = _fresh_meta(features={"prompt": "string", "completion": "string", "text": "string", "code": "string"})
        result = score_dataset(meta)
        assert result.bashgym_format == "sft"
        # Should be exact match (full schema credit), not heuristic.
        assert any("exact" in r.lower() for r in result.reasons)

    def test_prompt_only_does_not_match(self):
        """Single prompt-like column without a response-like column should not match."""
        meta = _fresh_meta(features={"text": "string"})
        result = score_dataset(meta)
        assert result.bashgym_format is None

    def test_response_only_does_not_match(self):
        """Single response-like column without a prompt-like column should not match."""
        meta = _fresh_meta(features={"code": "string"})
        result = score_dataset(meta)
        assert result.bashgym_format is None


class TestOverallScoreBounds:
    def test_score_in_0_to_10_range(self):
        meta = _fresh_meta()
        result = score_dataset(meta)
        assert 0.0 <= result.score <= 10.0

    def test_nominal_good_dataset_scores_above_5(self):
        """apache-2.0 + code tag + ideal size + recent + messages schema + popular → high score."""
        meta = _fresh_meta(downloads=50_000)
        result = score_dataset(meta)
        assert result.score >= 5.0, f"expected >=5.0, got {result.score}: {result.reasons}"


class TestDownloadCommand:
    def test_command_references_repo_id(self):
        meta = _fresh_meta()
        result = score_dataset(meta)
        assert meta.repo_id in result.download_command

    def test_command_is_valid_python_expression(self):
        import ast
        meta = _fresh_meta()
        result = score_dataset(meta)
        ast.parse(result.download_command, mode="exec")

    def test_command_references_DataDesignerPipeline(self):
        meta = _fresh_meta()
        result = score_dataset(meta)
        assert "DataDesignerPipeline" in result.download_command
        assert "from_dataset" in result.download_command
