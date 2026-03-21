"""
Example Generator for The Factory Layer

Converts raw coding sessions (traces) into individual training examples.
Sessions are entire coding interactions that may contain multiple logical tasks.
This module segments sessions and converts them to NeMo-compatible training format.

Module 3: Data Synthesis (The "Factory")
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .data_factory import (
    TOOL_SCHEMAS,
    TrainingExample,
    build_tool_call_messages,
)


class SegmentationType(Enum):
    """Strategies for segmenting sessions into examples."""

    TIME_GAP = "time_gap"  # Segment at time gaps > threshold
    GIT_COMMIT = "git_commit"  # Segment at git commits
    DIRECTORY_CHANGE = "directory_change"  # Segment when working directory changes
    FILE_SCOPE = "file_scope"  # Segment when file scope changes significantly
    COMBINED = "combined"  # Use all heuristics


@dataclass
class TraceStep:
    """A single step from a trace session."""

    index: int
    tool_name: str
    command: str
    output: str
    success: bool | None
    timestamp: str | None = None
    working_dir: str | None = None
    metadata: dict[str, Any] | None = None
    span_id: str | None = None  # Causal span: groups reasoning + action

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "TraceStep":
        """Create TraceStep from dictionary."""
        return cls(
            index=index,
            tool_name=data.get("tool_name", data.get("tool", "unknown")),
            command=data.get("command", data.get("input", "")),
            output=data.get("output", data.get("result", ""))[:2000],  # Truncate
            success=data.get(
                "success", data.get("exit_code") == 0 if "exit_code" in data else None
            ),
            timestamp=data.get("timestamp"),
            working_dir=data.get("cwd", data.get("working_directory")),
            metadata=data.get("metadata"),
            span_id=data.get("span_id"),
        )


@dataclass
class TaskSegment:
    """A segment of steps that represent a coherent task."""

    steps: list[TraceStep]
    inferred_prompt: str
    start_index: int
    end_index: int
    confidence: float = 0.5  # Confidence in the segmentation
    repo_name: str | None = None
    repo_path: str | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class ExampleGeneratorConfig:
    """Configuration for the Example Generator."""

    # Segmentation settings
    segmentation_strategy: SegmentationType = SegmentationType.COMBINED
    time_gap_minutes: int = 5  # Time gap threshold for segmentation
    min_steps_per_segment: int = 2
    max_steps_per_segment: int = 50

    # Prompt inference settings
    infer_from_commands: bool = True
    infer_from_file_patterns: bool = True

    # Quality filtering
    min_success_rate: float = 0.5
    require_file_operations: bool = False

    # Cognitive trace settings (AgentTrace-inspired)
    include_cognitive: bool = True  # Include thinking/reasoning in training examples

    # Output settings
    output_dir: str = "data/training_examples"


class ExampleGenerator:
    """
    Converts coding sessions into training examples.

    A session is an entire coding interaction containing multiple tool calls.
    This class segments sessions into logical tasks and converts each to
    a training example in NeMo-compatible format.
    """

    SYSTEM_PROMPT = """You are an expert software development agent. You execute tasks by running bash commands, reading files, and making edits. You think step-by-step and verify your work.

When given a task:
1. Analyze the requirements
2. Plan your approach
3. Execute commands to accomplish the task
4. Verify the results

You have access to these tools:
- Bash: Execute shell commands
- Read: Read file contents
- Write: Write to files
- Edit: Make targeted edits to files

Always explain your reasoning before executing commands."""

    # Patterns that suggest task boundaries
    TASK_BOUNDARY_PATTERNS = [
        "git commit",
        "git push",
        "pytest",
        "npm test",
        "make test",
        "cargo test",
    ]

    # Patterns for inferring task types
    TASK_PATTERNS = {
        "fix": ["fix", "bug", "error", "issue", "patch"],
        "add": ["add", "create", "implement", "new", "feature"],
        "update": ["update", "modify", "change", "refactor"],
        "test": ["test", "spec", "pytest", "jest", "unittest"],
        "docs": ["doc", "readme", "comment", "documentation"],
        "config": ["config", "setup", "install", "configure"],
    }

    def __init__(self, config: ExampleGeneratorConfig | None = None):
        """Initialize the Example Generator."""
        self.config = config or ExampleGeneratorConfig()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_session(self, session_path: Path) -> tuple[list[TraceStep], dict[str, Any]]:
        """
        Load a session trace file.

        Args:
            session_path: Path to the session JSON file

        Returns:
            Tuple of (steps, metadata)
        """
        with open(session_path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        # Handle both formats: {"trace": [...], "metadata": {...}} or [...]
        if isinstance(data, list):
            raw_steps = data
            metadata = {}
        else:
            raw_steps = data.get("trace", data.get("steps", []))
            metadata = data.get("metadata", {})

        steps = [TraceStep.from_dict(s, i) for i, s in enumerate(raw_steps)]
        return steps, metadata

    def segment_session(
        self, steps: list[TraceStep], metadata: dict[str, Any]
    ) -> list[TaskSegment]:
        """
        Segment a session into logical task boundaries.

        Uses multiple heuristics to identify where one task ends and another begins:
        - Time gaps between tool calls
        - Git commits (task completion markers)
        - Major directory changes
        - File scope changes

        Args:
            steps: List of trace steps
            metadata: Session metadata

        Returns:
            List of TaskSegment objects
        """
        if not steps:
            return []

        # Extract repo info from session metadata
        primary_repo = metadata.get("primary_repo", {})
        repo_name = primary_repo.get("name") if isinstance(primary_repo, dict) else None
        repo_path = primary_repo.get("path") if isinstance(primary_repo, dict) else None

        if len(steps) <= self.config.max_steps_per_segment:
            # Session is small enough to be a single example
            prompt = self._infer_task_prompt(steps, metadata)
            return [
                TaskSegment(
                    steps=steps,
                    inferred_prompt=prompt,
                    start_index=0,
                    end_index=len(steps) - 1,
                    confidence=0.8,
                    repo_name=repo_name,
                    repo_path=repo_path,
                )
            ]

        # Find segment boundaries
        boundaries = self._find_segment_boundaries(steps)

        # Create segments
        segments = []
        start_idx = 0

        for end_idx in boundaries:
            segment_steps = steps[start_idx : end_idx + 1]

            # Skip segments that are too small
            if len(segment_steps) < self.config.min_steps_per_segment:
                continue

            # Truncate segments that are too large
            if len(segment_steps) > self.config.max_steps_per_segment:
                segment_steps = segment_steps[: self.config.max_steps_per_segment]

            prompt = self._infer_task_prompt(segment_steps, metadata)

            segments.append(
                TaskSegment(
                    steps=segment_steps,
                    inferred_prompt=prompt,
                    start_index=start_idx,
                    end_index=min(end_idx, start_idx + self.config.max_steps_per_segment - 1),
                    confidence=0.6,
                    repo_name=repo_name,
                    repo_path=repo_path,
                )
            )

            start_idx = end_idx + 1

        # Handle remaining steps
        if start_idx < len(steps):
            remaining = steps[start_idx:]
            if len(remaining) >= self.config.min_steps_per_segment:
                prompt = self._infer_task_prompt(remaining, metadata)
                segments.append(
                    TaskSegment(
                        steps=remaining,
                        inferred_prompt=prompt,
                        start_index=start_idx,
                        end_index=len(steps) - 1,
                        confidence=0.5,
                        repo_name=repo_name,
                        repo_path=repo_path,
                    )
                )

        return segments

    def assign_spans(self, steps: list[TraceStep]) -> None:
        """Assign causal span IDs to groups of related steps.

        A span groups a reasoning chain (thinking → action). New spans start when:
        - A new thinking/cognitive block appears (new reasoning chain)
        - There's no cognitive data linking to the previous step

        This enables span-based segmentation: span boundaries are natural
        task boundaries because each span represents one think-then-act cycle.
        """
        current_span = uuid.uuid4().hex[:12]

        for i, step in enumerate(steps):
            meta = step.metadata or {}
            cognitive = meta.get("cognitive") or {}
            has_thinking = bool(cognitive.get("thinking") or meta.get("thinking_content"))

            if i > 0 and has_thinking:
                # New thinking block = new reasoning chain = new span
                current_span = uuid.uuid4().hex[:12]

            step.span_id = current_span

    def _find_segment_boundaries(self, steps: list[TraceStep]) -> list[int]:
        """Find indices where segments should end.

        Uses both traditional heuristics and causal span boundaries:
        - Git commits (strong signal)
        - Time gaps > threshold
        - Directory changes
        - Cognitive span boundaries (new reasoning chain starts)
        """
        # Assign spans first so we can use them for boundary detection
        self.assign_spans(steps)

        boundaries = []

        for i, step in enumerate(steps):
            if i == 0:
                continue

            is_boundary = False

            # Check for git commits (strong boundary signal)
            if self._is_commit_step(step):
                is_boundary = True

            # Check for time gaps
            elif self._has_time_gap(steps[i - 1], step):
                is_boundary = True

            # Check for directory changes
            elif self._has_directory_change(steps[i - 1], step):
                is_boundary = True

            # Check for cognitive span boundary (new reasoning chain)
            elif self._has_cognitive_boundary(steps[i - 1], step):
                is_boundary = True

            if is_boundary:
                boundaries.append(i - 1)  # End segment at previous step

        return boundaries

    def _is_commit_step(self, step: TraceStep) -> bool:
        """Check if step is a git commit."""
        cmd = step.command.lower()
        return "git commit" in cmd or "git push" in cmd

    def _has_time_gap(self, prev_step: TraceStep, curr_step: TraceStep) -> bool:
        """Check if there's a significant time gap between steps."""
        if not prev_step.timestamp or not curr_step.timestamp:
            return False

        try:
            prev_time = datetime.fromisoformat(prev_step.timestamp.replace("Z", "+00:00"))
            curr_time = datetime.fromisoformat(curr_step.timestamp.replace("Z", "+00:00"))
            gap_minutes = (curr_time - prev_time).total_seconds() / 60
            return gap_minutes > self.config.time_gap_minutes
        except (ValueError, TypeError):
            return False

    def _has_directory_change(self, prev_step: TraceStep, curr_step: TraceStep) -> bool:
        """Check if working directory changed significantly."""
        if not prev_step.working_dir or not curr_step.working_dir:
            return False

        prev_parts = Path(prev_step.working_dir).parts
        curr_parts = Path(curr_step.working_dir).parts

        # Check if we moved to a completely different project
        if len(prev_parts) >= 2 and len(curr_parts) >= 2:
            # Compare top-level directories
            if prev_parts[:2] != curr_parts[:2]:
                return True

        return False

    def _has_cognitive_boundary(self, prev_step: TraceStep, curr_step: TraceStep) -> bool:
        """Check if a new reasoning chain starts at curr_step.

        A cognitive boundary occurs when:
        - The current step has a new span_id (assigned by assign_spans)
        - AND there's been a meaningful gap in reasoning (not just consecutive
          tool calls in the same chain)

        This is weaker than git commit or time gap boundaries — it only triggers
        when the span actually changed AND there are enough steps to warrant
        splitting.
        """
        if not prev_step.span_id or not curr_step.span_id:
            return False
        return prev_step.span_id != curr_step.span_id

    def _infer_task_prompt(self, steps: list[TraceStep], metadata: dict[str, Any]) -> str:
        """
        Infer a task prompt from the steps.

        Priority:
        1. Use metadata.user_initial_prompt if available
        2. Analyze command patterns to infer task type
        3. Generate generic description based on files touched
        """
        # Check for explicit prompt in metadata
        if metadata.get("user_initial_prompt"):
            return metadata["user_initial_prompt"]

        # Analyze commands to infer task
        commands = [s.command.lower() for s in steps]
        all_commands = " ".join(commands)

        # Detect task type
        task_type = "work on"
        for type_name, patterns in self.TASK_PATTERNS.items():
            if any(p in all_commands for p in patterns):
                task_type = type_name
                break

        # Extract file targets
        files_touched = set()
        for step in steps:
            if step.tool_name.lower() in ("read", "write", "edit"):
                # Extract file path from command
                parts = step.command.split()
                if parts:
                    file_path = parts[-1] if len(parts) > 1 else parts[0]
                    if "/" in file_path or "\\" in file_path or "." in file_path:
                        files_touched.add(Path(file_path).name)

        # Build prompt
        if files_touched:
            file_list = ", ".join(list(files_touched)[:3])
            if len(files_touched) > 3:
                file_list += f" and {len(files_touched) - 3} more files"
            return f"{task_type.title()} {file_list}"

        # Fallback to tool summary
        tool_counts = {}
        for step in steps:
            tool = step.tool_name.lower()
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        tool_summary = ", ".join(
            [
                f"{count} {tool}"
                for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:3]
            ]
        )
        return f"Coding task: {tool_summary}"

    def segment_to_example(self, segment: TaskSegment) -> TrainingExample:
        """
        Convert a TaskSegment to a TrainingExample.

        Produces both structured multi-turn messages (with tool_calls and tool
        roles) and a legacy markdown response for backward compatibility.

        When include_cognitive is enabled (default), injects thinking/reasoning
        content into assistant messages before tool_calls, teaching the model
        to reason-then-act.

        Args:
            segment: The segment to convert

        Returns:
            TrainingExample in NeMo-compatible format with structured messages
        """
        # Build structured messages from steps, preserving metadata for cognitive extraction
        raw_steps = []
        per_step_success = []
        for step in segment.steps:
            step_dict = {
                "tool_name": step.tool_name,
                "command": step.command,
                "output": step.output,
                "success": step.success,
            }
            # Preserve metadata so build_tool_call_messages can extract cognitive data
            if hasattr(step, "metadata") and step.metadata:
                step_dict["metadata"] = step.metadata
            raw_steps.append(step_dict)
            per_step_success.append(bool(step.success) if step.success is not None else True)

        structured_messages = build_tool_call_messages(
            raw_steps,
            self.SYSTEM_PROMPT,
            segment.inferred_prompt,
            include_cognitive=self.config.include_cognitive,
        )

        # Also build legacy markdown response for backward compat
        response_parts = []
        for i, step in enumerate(segment.steps, 1):
            tool_name = step.tool_name.lower()

            if tool_name == "bash":
                response_parts.append(f"**Step {i}: Execute command**")
                response_parts.append(f"```bash\n{step.command}\n```")
                if step.output and len(step.output.strip()) > 0:
                    output = step.output[:500]
                    response_parts.append(f"Output:\n```\n{output}\n```")
            elif tool_name in ("read", "write", "edit"):
                response_parts.append(f"**Step {i}: {tool_name.title()} file**")
                response_parts.append(f"```\n{step.command}\n```")
            elif tool_name in ("glob", "grep"):
                response_parts.append(f"**Step {i}: Search files**")
                response_parts.append(f"```\n{step.command}\n```")
            else:
                response_parts.append(f"**Step {i}: {tool_name}**")
                response_parts.append(f"```\n{step.command}\n```")

            response_parts.append("")  # Empty line between steps

        assistant_response = "\n".join(response_parts)

        # Generate unique ID
        example_id = hashlib.sha256(
            f"{segment.inferred_prompt}{assistant_response}".encode()
        ).hexdigest()[:16]

        # Calculate success rate for metadata
        successes = sum(1 for s in segment.steps if s.success is True)
        total = len(segment.steps)
        success_rate = successes / total if total > 0 else 0.0

        return TrainingExample(
            example_id=example_id,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=segment.inferred_prompt,
            assistant_response=assistant_response,
            messages=structured_messages,
            tools=TOOL_SCHEMAS,
            metadata={
                "step_count": segment.step_count,
                "start_index": segment.start_index,
                "end_index": segment.end_index,
                "segmentation_confidence": segment.confidence,
                "success_rate": success_rate,
                "per_step_success": per_step_success,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "repo_name": segment.repo_name,
                "repo_path": segment.repo_path,
            },
        )

    def generate_examples(self, session_path: Path) -> list[TrainingExample]:
        """
        Generate training examples from a session file.

        This is the main entry point for converting a session to examples.

        Args:
            session_path: Path to the session JSON file

        Returns:
            List of TrainingExample objects
        """
        try:
            steps, metadata = self.load_session(session_path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading session {session_path}: {e}")
            return []

        if not steps:
            return []

        # Segment the session
        segments = self.segment_session(steps, metadata)

        # Convert segments to examples
        examples = []
        for segment in segments:
            # Filter by quality
            successes = sum(1 for s in segment.steps if s.success is True)
            success_rate = successes / len(segment.steps) if segment.steps else 0

            if success_rate < self.config.min_success_rate:
                continue

            example = self.segment_to_example(segment)
            examples.append(example)

        return examples

    def process_directory(
        self, trace_dir: Path, output_path: Path | None = None
    ) -> tuple[list[TrainingExample], dict[str, Any]]:
        """
        Process all sessions in a directory.

        Args:
            trace_dir: Directory containing session files
            output_path: Optional path to save examples as JSONL

        Returns:
            Tuple of (examples, statistics)
        """
        all_examples = []
        stats = {
            "sessions_processed": 0,
            "sessions_skipped": 0,
            "examples_generated": 0,
            "total_steps": 0,
        }

        trace_files = list(Path(trace_dir).glob("*.json"))
        print(f"Processing {len(trace_files)} sessions from {trace_dir}...")

        for trace_path in trace_files:
            examples = self.generate_examples(trace_path)

            if examples:
                all_examples.extend(examples)
                stats["sessions_processed"] += 1
                stats["examples_generated"] += len(examples)
                stats["total_steps"] += sum(e.metadata.get("step_count", 0) for e in examples)
            else:
                stats["sessions_skipped"] += 1

        print(f"Generated {len(all_examples)} examples from {stats['sessions_processed']} sessions")

        # Save if output path provided
        if output_path and all_examples:
            self.save_examples(all_examples, output_path)

        return all_examples, stats

    def save_examples(self, examples: list[TrainingExample], output_path: Path) -> Path:
        """
        Save examples to a JSONL file (NeMo-compatible format).

        Uses structured multi-turn messages with tool_calls when available,
        falling back to the legacy 3-message format otherwise.

        Args:
            examples: List of TrainingExample objects
            output_path: Path to save the JSONL file

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in examples:
                # Use structured format (to_dict prefers messages when available)
                record = example.to_dict()
                f.write(json.dumps(record) + "\n")

        print(f"Saved {len(examples)} examples to {output_path}")
        return output_path

    def export_for_nemo(
        self,
        examples: list[TrainingExample],
        output_dir: Path,
        train_split: float = 0.9,
        repo_filter: list[str] | None = None,
    ) -> dict[str, Path]:
        """
        Export examples in NeMo training format with train/val split.

        Args:
            examples: List of training examples
            output_dir: Directory to save files
            train_split: Fraction for training (rest goes to validation)
            repo_filter: Optional list of repo names to include. When set,
                only examples whose metadata.repo_name is in this list are exported.

        Returns:
            Dict with paths to train and val files
        """
        import random

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Apply repo filter if specified
        if repo_filter:
            filter_set = set(repo_filter)
            examples = [e for e in examples if e.metadata.get("repo_name") in filter_set]

        # Shuffle and split
        shuffled = examples.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_split)
        train_examples = shuffled[:split_idx]
        val_examples = shuffled[split_idx:]

        # Save files
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        train_path = output_dir / f"train_{timestamp}.jsonl"
        val_path = output_dir / f"val_{timestamp}.jsonl"

        self.save_examples(train_examples, train_path)
        self.save_examples(val_examples, val_path)

        return {
            "train": train_path,
            "validation": val_path,
            "train_count": len(train_examples),
            "val_count": len(val_examples),
        }


def main():
    """Example usage of the Example Generator."""
    generator = ExampleGenerator()

    # Process pending traces
    trace_dir = Path("data/traces")
    if trace_dir.exists():
        examples, stats = generator.process_directory(
            trace_dir, output_path=Path("data/training_examples/generated.jsonl")
        )

        print("\nStatistics:")
        print(f"  Sessions processed: {stats['sessions_processed']}")
        print(f"  Sessions skipped: {stats['sessions_skipped']}")
        print(f"  Examples generated: {stats['examples_generated']}")
        print(f"  Total steps: {stats['total_steps']}")

        if examples:
            export_result = generator.export_for_nemo(examples, Path("data/training_batches"))
            print("\nExported for NeMo training:")
            print(f"  Train: {export_result['train']} ({export_result['train_count']} examples)")
            print(f"  Val: {export_result['validation']} ({export_result['val_count']} examples)")
    else:
        print("No traces directory found")


if __name__ == "__main__":
    main()
