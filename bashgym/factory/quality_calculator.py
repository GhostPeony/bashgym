"""
Quality Calculator Module

Centralized quality calculation for trace analysis.
Used by both API routes and TraceProcessor for consistent metrics.

Module: Factory Layer - Data Quality Assessment
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


# Command patterns to identify for complexity scoring
COMMAND_PATTERNS = {
    r'cd\s+[\w/.-]+': 'cd <path>',
    r'cat\s+[\w/.-]+': 'cat <file>',
    r'ls\s+-?\w*\s*[\w/.-]*': 'ls <options> <path>',
    r'grep\s+.*': 'grep <pattern> <file>',
    r'sed\s+.*': 'sed <expression> <file>',
    r'pip\s+install\s+.*': 'pip install <packages>',
    r'python\s+[\w/.-]+': 'python <script>',
    r'git\s+\w+.*': 'git <command>',
    r'npm\s+\w+.*': 'npm <command>',
    r'yarn\s+\w+.*': 'yarn <command>',
    r'docker\s+\w+.*': 'docker <command>',
    r'make\s+\w*': 'make <target>',
}


@dataclass
class QualityBreakdown:
    """Complete quality breakdown for a trace."""

    success_rate: float       # 0-1, % of successful steps
    verification_score: float # 0-1, verification passed/attempted
    complexity_score: float   # 0-1, based on tool diversity + command patterns
    length_score: float       # 0-1, bell curve around ideal length
    tool_diversity: float     # 0-1, unique tools used
    efficiency_score: float   # 0-1, output quality vs errors
    total_score: float        # Weighted combination

    # Verification flags (derived from metadata, not from score)
    has_verification: bool = False
    verification_passed_flag: Optional[bool] = None  # True/False/None

    # Raw metrics for debugging/display
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    unique_tools_count: int = 0
    unique_commands_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "success_rate": round(self.success_rate, 3),
            "verification_score": round(self.verification_score, 3),
            "complexity_score": round(self.complexity_score, 3),
            "length_score": round(self.length_score, 3),
            "tool_diversity": round(self.tool_diversity, 3),
            "efficiency_score": round(self.efficiency_score, 3),
            "total_score": round(self.total_score, 3),
        }


def calculate_success_rate(steps: List[Dict[str, Any]]) -> tuple[float, int, int]:
    """
    Calculate success rate from steps.

    Returns:
        Tuple of (success_rate, successful_count, failed_count)
    """
    if not steps:
        return 0.0, 0, 0

    successful = 0
    failed = 0

    for step in steps:
        # Check various success indicators
        exit_code = step.get("exit_code")
        success = step.get("success")
        error = step.get("error")
        is_error = step.get("is_error")

        if success is True or exit_code == 0:
            successful += 1
        elif success is False or (exit_code is not None and exit_code != 0) or is_error:
            failed += 1
        elif error is not None:
            failed += 1
        # else: no explicit indicator, don't count (avoids inflating success rate)

    # Calculate rate from steps with known outcomes only
    counted = successful + failed
    rate = successful / counted if counted > 0 else 0.0

    return rate, successful, failed


def calculate_verification_score(
    verification_passed: Optional[bool],
    has_verification: bool
) -> float:
    """
    Calculate verification score.

    Args:
        verification_passed: True if tests passed, False if failed, None if unknown
        has_verification: Whether verification metadata exists

    Returns:
        Score from 0-1
    """
    if verification_passed is True:
        return 1.0
    elif has_verification and verification_passed is False:
        return 0.0
    elif has_verification:
        # Has verification info but unclear result
        return 0.3
    else:
        # No verification info - neutral score
        return 0.5


def calculate_complexity(steps: List[Dict[str, Any]]) -> tuple[float, int]:
    """
    Calculate complexity score based on tool diversity, command patterns, and control flow.

    Full analysis includes:
    - Unique tools used (normalized by max expected ~6)
    - Unique command patterns
    - Control flow patterns (conditionals, loops)
    - Piping and chaining

    Args:
        steps: List of trace steps

    Returns:
        Tuple of (complexity_score 0-1, unique_commands_count)
    """
    if not steps:
        return 0.0, 0

    score = 0.0

    # Collect tools and commands
    tools = set()
    commands = []

    for step in steps:
        tool = step.get("tool_name") or step.get("tool") or step.get("type") or "unknown"
        tools.add(tool.lower())

        command = step.get("command") or step.get("input") or ""
        if isinstance(command, str):
            commands.append(command)

    # Tool diversity contribution (max ~9 points for 6 tools)
    score += len(tools) * 1.5

    # Command pattern diversity
    unique_patterns = set()
    for cmd in commands:
        matched = False
        for pattern, normalized in COMMAND_PATTERNS.items():
            if re.match(pattern, cmd):
                unique_patterns.add(normalized)
                matched = True
                break
        if not matched and cmd:
            # Use first 50 chars as pattern identifier
            unique_patterns.add(cmd[:50])

    score += len(unique_patterns) * 0.5

    # Control flow complexity bonuses
    all_commands = " ".join(commands)

    # Conditionals
    if "if " in all_commands or "then" in all_commands or "else" in all_commands:
        score += 2.0

    # Loops
    if "for " in all_commands or "while " in all_commands or "do " in all_commands:
        score += 2.0

    # Pipes
    if "|" in all_commands:
        score += 1.0

    # Boolean operators / command chaining
    if "&&" in all_commands or "||" in all_commands:
        score += 1.0

    # Normalize to 0-1 (max expected ~12 points)
    normalized = min(score / 12.0, 1.0)

    return normalized, len(unique_patterns)


def calculate_length_score(total_steps: int) -> float:
    """
    Calculate length score using bell curve centered on ideal length.

    Scoring:
    - 0 steps: 0.0 (empty)
    - 1-2 steps: 0.2 (too short, likely incomplete)
    - 3-5 steps: 0.5 (short)
    - 6-9 steps: 0.8 (good)
    - 10-20 steps: 1.0 (ideal)
    - 21-30 steps: 0.8 (acceptable)
    - 31-50 steps: 0.5 (getting long)
    - 51+ steps: 0.2 (too long, likely inefficient)

    Args:
        total_steps: Number of steps in trace

    Returns:
        Score from 0-1
    """
    if total_steps <= 0:
        return 0.0
    elif total_steps <= 2:
        return 0.2
    elif total_steps <= 5:
        return 0.5
    elif total_steps < 10:
        return 0.8
    elif total_steps <= 20:
        return 1.0
    elif total_steps <= 30:
        return 0.8
    elif total_steps <= 50:
        return 0.5
    else:
        return 0.2


def calculate_tool_diversity(steps: List[Dict[str, Any]]) -> tuple[float, int]:
    """
    Calculate tool diversity score based on unique tools used.

    Scoring:
    - 1 tool: 0.2
    - 2 tools: 0.5
    - 3 tools: 0.75
    - 4+ tools: 1.0

    Args:
        steps: List of trace steps

    Returns:
        Tuple of (diversity_score 0-1, unique_tools_count)
    """
    if not steps:
        return 0.0, 0

    tools = set()
    for step in steps:
        tool = step.get("tool_name") or step.get("tool") or step.get("type")
        if tool:
            tools.add(tool.lower())

    count = len(tools)

    if count <= 0:
        return 0.0, 0
    elif count == 1:
        return 0.2, count
    elif count == 2:
        return 0.5, count
    elif count == 3:
        return 0.75, count
    else:
        return 1.0, count


def calculate_efficiency(steps: List[Dict[str, Any]]) -> float:
    """
    Calculate efficiency score based on work quality.

    Components:
    1. Error recovery (40%): failure->success patterns show good recovery
    2. Output meaningfulness (40%): non-empty, non-error outputs
    3. Retry penalty (20%): penalize repeated identical commands

    Args:
        steps: List of trace steps

    Returns:
        Score from 0-1
    """
    if not steps:
        return 0.5  # Neutral for empty

    # 1. Error recovery score (40%)
    recovery_score = 1.0
    failures_recovered = 0
    total_failures = 0

    for i, step in enumerate(steps):
        is_failure = (
            step.get("success") is False or
            (step.get("exit_code") is not None and step.get("exit_code") != 0) or
            step.get("is_error") is True or
            step.get("error") is not None
        )

        if is_failure:
            total_failures += 1
            # Check if next step exists and succeeded
            if i + 1 < len(steps):
                next_step = steps[i + 1]
                next_success = (
                    next_step.get("success") is True or
                    next_step.get("exit_code") == 0 or
                    (next_step.get("success") is None and next_step.get("exit_code") is None)
                )
                if next_success:
                    failures_recovered += 1

    if total_failures > 0:
        recovery_score = failures_recovered / total_failures

    # 2. Output meaningfulness score (40%)
    meaningful_outputs = 0
    for step in steps:
        output = step.get("output") or step.get("result") or ""
        if isinstance(output, str) and len(output.strip()) > 0:
            # Check it's not just an error message
            output_lower = output.lower()
            is_error_output = any(
                err in output_lower
                for err in ["error:", "exception:", "traceback", "failed:", "fatal:"]
            )
            if not is_error_output:
                meaningful_outputs += 1

    meaningful_score = meaningful_outputs / len(steps) if steps else 0

    # 3. Retry penalty score (20%)
    commands = []
    for step in steps:
        cmd = step.get("command") or step.get("input") or ""
        if isinstance(cmd, str):
            commands.append(cmd)

    unique_commands = len(set(commands))
    total_commands = len(commands)
    retry_score = unique_commands / total_commands if total_commands > 0 else 1.0

    # Combine with weights
    efficiency = (
        recovery_score * 0.4 +
        meaningful_score * 0.4 +
        retry_score * 0.2
    )

    return min(max(efficiency, 0.0), 1.0)


def calculate_quality_breakdown(
    steps: List[Dict[str, Any]],
    verification_passed: Optional[bool] = None,
    has_verification: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> QualityBreakdown:
    """
    Calculate complete quality breakdown for a trace.

    Args:
        steps: List of trace steps
        verification_passed: Whether verification tests passed
        has_verification: Whether verification info exists
        metadata: Optional trace metadata (can extract verification info)

    Returns:
        QualityBreakdown with all metrics
    """
    # Extract verification from metadata if not provided
    if metadata:
        if verification_passed is None:
            verification_passed = metadata.get("verification_passed")
        if not has_verification:
            has_verification = "verification_passed" in metadata

    # Calculate individual metrics
    success_rate, successful, failed = calculate_success_rate(steps)
    verification_score = calculate_verification_score(verification_passed, has_verification)
    complexity_score, unique_commands = calculate_complexity(steps)
    length_score = calculate_length_score(len(steps))
    tool_diversity, unique_tools = calculate_tool_diversity(steps)
    efficiency_score = calculate_efficiency(steps)

    # Calculate weighted total
    # Weights: success=30%, verification=25%, complexity=15%,
    #          tool_diversity=10%, efficiency=10%, length=10%
    total_score = (
        success_rate * 0.30 +
        verification_score * 0.25 +
        complexity_score * 0.15 +
        tool_diversity * 0.10 +
        efficiency_score * 0.10 +
        length_score * 0.10
    )

    # Guard against NaN
    total_score = total_score if total_score == total_score else 0.0
    total_score = min(max(total_score, 0.0), 1.0)

    return QualityBreakdown(
        success_rate=success_rate,
        verification_score=verification_score,
        complexity_score=complexity_score,
        length_score=length_score,
        tool_diversity=tool_diversity,
        efficiency_score=efficiency_score,
        total_score=total_score,
        has_verification=has_verification,
        verification_passed_flag=verification_passed,
        total_steps=len(steps),
        successful_steps=successful,
        failed_steps=failed,
        unique_tools_count=unique_tools,
        unique_commands_count=unique_commands,
    )


def calculate_quality_from_trace(trace_data: Dict[str, Any]) -> QualityBreakdown:
    """
    Calculate quality breakdown from a complete trace dictionary.

    Convenience function that extracts steps and metadata from trace structure.

    Args:
        trace_data: Complete trace dictionary with 'trace' and 'metadata' keys

    Returns:
        QualityBreakdown with all metrics
    """
    steps = trace_data.get("trace", [])
    metadata = trace_data.get("metadata", {})

    return calculate_quality_breakdown(
        steps=steps,
        metadata=metadata
    )
