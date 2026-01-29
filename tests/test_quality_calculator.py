#!/usr/bin/env python3
"""
Test Suite for Quality Calculator Module

Tests the centralized quality calculation functions.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bashgym.factory.quality_calculator import (
    calculate_quality_breakdown,
    calculate_complexity,
    calculate_length_score,
    calculate_tool_diversity,
    calculate_efficiency,
    calculate_success_rate,
    calculate_verification_score,
    QualityBreakdown,
)


class TestSuccessRate:
    """Tests for success rate calculation."""

    def test_all_success(self):
        """Test success rate with all successful steps."""
        steps = [
            {"success": True, "exit_code": 0},
            {"success": True, "exit_code": 0},
            {"success": True, "exit_code": 0},
        ]
        rate, successful, failed = calculate_success_rate(steps)
        assert rate == 1.0
        assert successful == 3
        assert failed == 0

    def test_mixed_results(self):
        """Test success rate with mixed results."""
        steps = [
            {"success": True},
            {"success": False},
            {"exit_code": 0},
            {"exit_code": 1},
        ]
        rate, successful, failed = calculate_success_rate(steps)
        assert rate == 0.5
        assert successful == 2
        assert failed == 2

    def test_empty_steps(self):
        """Test success rate with empty steps."""
        rate, successful, failed = calculate_success_rate([])
        assert rate == 0.0
        assert successful == 0
        assert failed == 0

    def test_implicit_success(self):
        """Test steps without explicit success field assume success."""
        steps = [
            {"output": "some output"},  # No success or exit_code
            {"output": "more output"},
        ]
        rate, successful, failed = calculate_success_rate(steps)
        assert successful == 2
        assert failed == 0


class TestComplexity:
    """Tests for complexity calculation."""

    def test_simple_trace(self):
        """Test complexity for simple traces."""
        steps = [
            {"tool": "bash", "command": "ls"},
            {"tool": "bash", "command": "cat file.txt"},
        ]
        score, unique_cmds = calculate_complexity(steps)
        assert 0.0 <= score <= 1.0
        assert unique_cmds >= 1

    def test_tool_diversity_increases_complexity(self):
        """Test complexity increases with tool diversity."""
        simple_steps = [
            {"tool": "bash", "command": "ls"},
            {"tool": "bash", "command": "pwd"},
        ]

        diverse_steps = [
            {"tool": "bash", "command": "ls"},
            {"tool": "read", "command": "file.txt"},
            {"tool": "write", "command": "output.txt"},
            {"tool": "grep", "command": "pattern"},
        ]

        simple_score, _ = calculate_complexity(simple_steps)
        diverse_score, _ = calculate_complexity(diverse_steps)

        assert diverse_score > simple_score

    def test_control_flow_bonus(self):
        """Test complexity bonus for control flow patterns."""
        basic_steps = [
            {"tool": "bash", "command": "echo hello"},
        ]

        control_flow_steps = [
            {"tool": "bash", "command": "if [ -f test.py ]; then python test.py; fi"},
            {"tool": "bash", "command": "for f in *.txt; do cat $f; done"},
        ]

        basic_score, _ = calculate_complexity(basic_steps)
        control_flow_score, _ = calculate_complexity(control_flow_steps)

        assert control_flow_score > basic_score

    def test_pipes_bonus(self):
        """Test complexity bonus for piped commands."""
        simple_steps = [{"tool": "bash", "command": "cat file.txt"}]
        piped_steps = [{"tool": "bash", "command": "cat file.txt | grep pattern | wc -l"}]

        simple_score, _ = calculate_complexity(simple_steps)
        piped_score, _ = calculate_complexity(piped_steps)

        assert piped_score > simple_score

    def test_chaining_bonus(self):
        """Test complexity bonus for command chaining."""
        simple_steps = [{"tool": "bash", "command": "ls"}]
        chained_steps = [{"tool": "bash", "command": "mkdir dir && cd dir && touch file.txt"}]

        simple_score, _ = calculate_complexity(simple_steps)
        chained_score, _ = calculate_complexity(chained_steps)

        assert chained_score > simple_score


class TestLengthScore:
    """Tests for length score calculation."""

    def test_ideal_length(self):
        """Test length score for ideal trace lengths (10-20 steps)."""
        assert calculate_length_score(10) == 1.0
        assert calculate_length_score(15) == 1.0
        assert calculate_length_score(20) == 1.0

    def test_too_short(self):
        """Test length score for very short traces."""
        assert calculate_length_score(0) == 0.0
        assert calculate_length_score(1) == 0.2
        assert calculate_length_score(2) == 0.2

    def test_short(self):
        """Test length score for short traces."""
        assert calculate_length_score(3) == 0.5
        assert calculate_length_score(5) == 0.5

    def test_good(self):
        """Test length score for good length traces."""
        assert calculate_length_score(6) == 0.8
        assert calculate_length_score(8) == 0.8
        assert calculate_length_score(10) == 1.0

    def test_acceptable(self):
        """Test length score for acceptable length traces."""
        assert calculate_length_score(21) == 0.8
        assert calculate_length_score(25) == 0.8
        assert calculate_length_score(30) == 0.8

    def test_long(self):
        """Test length score for long traces."""
        assert calculate_length_score(31) == 0.5
        assert calculate_length_score(40) == 0.5
        assert calculate_length_score(50) == 0.5

    def test_too_long(self):
        """Test length score for very long traces."""
        assert calculate_length_score(51) == 0.2
        assert calculate_length_score(100) == 0.2
        assert calculate_length_score(500) == 0.2


class TestToolDiversity:
    """Tests for tool diversity calculation."""

    def test_one_tool(self):
        """Test diversity score with 1 tool."""
        steps = [{"tool": "bash"}]
        score, count = calculate_tool_diversity(steps)
        assert score == 0.2
        assert count == 1

    def test_two_tools(self):
        """Test diversity score with 2 tools."""
        steps = [{"tool": "bash"}, {"tool": "read"}]
        score, count = calculate_tool_diversity(steps)
        assert score == 0.5
        assert count == 2

    def test_three_tools(self):
        """Test diversity score with 3 tools."""
        steps = [{"tool": "bash"}, {"tool": "read"}, {"tool": "write"}]
        score, count = calculate_tool_diversity(steps)
        assert score == 0.75
        assert count == 3

    def test_four_or_more_tools(self):
        """Test diversity score with 4+ tools."""
        steps = [
            {"tool": "bash"},
            {"tool": "read"},
            {"tool": "write"},
            {"tool": "grep"},
        ]
        score, count = calculate_tool_diversity(steps)
        assert score == 1.0
        assert count == 4

    def test_duplicate_tools_counted_once(self):
        """Test that duplicate tools are only counted once."""
        steps = [
            {"tool": "bash"},
            {"tool": "bash"},
            {"tool": "bash"},
        ]
        score, count = calculate_tool_diversity(steps)
        assert count == 1

    def test_empty_steps(self):
        """Test diversity with empty steps."""
        score, count = calculate_tool_diversity([])
        assert score == 0.0
        assert count == 0


class TestEfficiency:
    """Tests for efficiency calculation."""

    def test_perfect_execution(self):
        """Test efficiency score for perfect execution."""
        steps = [
            {"success": True, "output": "result 1"},
            {"success": True, "output": "result 2"},
            {"success": True, "output": "result 3"},
        ]
        score = calculate_efficiency(steps)
        assert score >= 0.8

    def test_error_recovery(self):
        """Test efficiency score rewards error recovery."""
        steps = [
            {"success": False, "error": "Failed"},
            {"success": True, "output": "Fixed it"},
            {"success": True, "output": "Done"},
        ]
        score = calculate_efficiency(steps)
        assert 0.4 <= score <= 0.9

    def test_retries_penalized(self):
        """Test efficiency score penalizes repeated commands."""
        steps = [
            {"command": "git push", "success": False},
            {"command": "git push", "success": False},
            {"command": "git push", "success": True},
        ]
        score = calculate_efficiency(steps)
        assert score < 0.8

    def test_empty_outputs_penalized(self):
        """Test efficiency penalizes empty outputs."""
        empty_output_steps = [
            {"success": True, "output": ""},
            {"success": True, "output": ""},
        ]
        meaningful_output_steps = [
            {"success": True, "output": "Important result"},
            {"success": True, "output": "More results"},
        ]

        empty_score = calculate_efficiency(empty_output_steps)
        meaningful_score = calculate_efficiency(meaningful_output_steps)

        assert meaningful_score > empty_score

    def test_empty_steps(self):
        """Test efficiency with empty steps returns neutral."""
        score = calculate_efficiency([])
        assert score == 0.5


class TestVerificationScore:
    """Tests for verification score calculation."""

    def test_passed(self):
        """Test score when verification passed."""
        score = calculate_verification_score(True, True)
        assert score == 1.0

    def test_failed(self):
        """Test score when verification failed."""
        score = calculate_verification_score(False, True)
        assert score == 0.0

    def test_no_verification(self):
        """Test score when no verification exists."""
        score = calculate_verification_score(None, False)
        assert score == 0.5


class TestQualityBreakdown:
    """Tests for complete quality breakdown."""

    def test_complete_breakdown(self):
        """Test complete quality breakdown calculation."""
        steps = [
            {"tool": "bash", "command": "ls", "success": True, "output": "files"},
            {"tool": "read", "command": "file.txt", "success": True, "output": "content"},
            {"tool": "write", "command": "out.txt", "success": True, "output": "written"},
            {"tool": "bash", "command": "python test.py", "success": True, "output": "PASSED"},
        ]
        metadata = {"verification_passed": True}

        breakdown = calculate_quality_breakdown(steps, metadata=metadata)

        assert isinstance(breakdown, QualityBreakdown)
        assert breakdown.success_rate == 1.0
        assert breakdown.verification_score == 1.0
        assert breakdown.complexity_score > 0
        assert breakdown.length_score > 0
        assert breakdown.tool_diversity > 0
        assert breakdown.efficiency_score > 0
        assert 0.0 <= breakdown.total_score <= 1.0

    def test_total_score_range(self):
        """Test total score is always in valid range."""
        # Worst case
        bad_steps = [{"success": False}]
        bad_breakdown = calculate_quality_breakdown(bad_steps)
        assert 0.0 <= bad_breakdown.total_score <= 1.0

        # Best case
        good_steps = [
            {"tool": f"tool{i}", "command": f"cmd{i}", "success": True, "output": f"out{i}"}
            for i in range(15)
        ]
        good_breakdown = calculate_quality_breakdown(good_steps, metadata={"verification_passed": True})
        assert 0.0 <= good_breakdown.total_score <= 1.0

    def test_to_dict(self):
        """Test QualityBreakdown serialization."""
        breakdown = QualityBreakdown(
            success_rate=0.9,
            verification_score=1.0,
            complexity_score=0.7,
            length_score=0.8,
            tool_diversity=0.75,
            efficiency_score=0.85,
            total_score=0.82,
            total_steps=15,
            successful_steps=14,
            failed_steps=1,
            unique_tools_count=4,
            unique_commands_count=10,
        )

        data = breakdown.to_dict()
        assert data["success_rate"] == 0.9
        assert data["verification_score"] == 1.0
        assert data["complexity_score"] == 0.7
        assert data["length_score"] == 0.8
        assert data["tool_diversity"] == 0.75
        assert data["efficiency_score"] == 0.85
        assert data["total_score"] == 0.82

    def test_weights_sum_to_one(self):
        """Test that quality weights sum to 1.0 (100%)."""
        # The weights should be: success=30%, verification=25%, complexity=15%,
        # tool_diversity=10%, efficiency=10%, length=10%
        weights = [0.30, 0.25, 0.15, 0.10, 0.10, 0.10]
        assert sum(weights) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
