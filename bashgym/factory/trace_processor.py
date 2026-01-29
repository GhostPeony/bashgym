"""
Trace Processor for The Factory Layer

Processes raw execution traces into structured formats suitable for
training. Handles trace normalization, filtering, and quality scoring.

Extended with Safe Synthesizer integration for privacy-preserving processing.

Module 3: Data Synthesis (The "Factory")
"""

import os
import json
import re
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Set, TYPE_CHECKING
from datetime import datetime, timezone
from collections import defaultdict
import hashlib

if TYPE_CHECKING:
    from .safe_synthesizer import SafeSynthesizer

from .quality_calculator import calculate_quality_breakdown


@dataclass
class TraceQualityMetrics:
    """Quality metrics for a trace.

    Updated to include 6 metrics with balanced weights:
    - success_rate: 30%
    - verification: 25%
    - complexity: 15%
    - tool_diversity: 10%
    - efficiency: 10%
    - length: 10%
    """

    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    unique_commands: int = 0
    avg_output_length: float = 0.0
    has_verification: bool = False
    verification_passed: bool = False
    complexity_score: float = 0.0
    tool_diversity: float = 0.0
    efficiency_score: float = 0.0
    length_score: float = 0.0
    verification_score: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps

    @property
    def quality_score(self) -> float:
        """
        Calculate overall quality score (0-1).

        Updated weights:
        - Success rate (30%)
        - Verification (25%)
        - Complexity (15%)
        - Tool diversity (10%)
        - Efficiency (10%)
        - Length (10%)
        """
        score = (
            self.success_rate * 0.30 +
            self.verification_score * 0.25 +
            self.complexity_score * 0.15 +
            self.tool_diversity * 0.10 +
            self.efficiency_score * 0.10 +
            self.length_score * 0.10
        )

        # Guard against NaN
        if score != score:
            score = 0.0

        return min(max(score, 0.0), 1.0)


@dataclass
class ProcessedTrace:
    """A processed and normalized trace."""
    
    trace_id: str
    original_path: Path
    task_prompt: str
    normalized_steps: List[Dict[str, Any]]
    quality_metrics: TraceQualityMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "task_prompt": self.task_prompt,
            "steps": self.normalized_steps,
            "quality": {
                "score": self.quality_metrics.quality_score,
                "success_rate": self.quality_metrics.success_rate,
                "total_steps": self.quality_metrics.total_steps,
                "verification_passed": self.quality_metrics.verification_passed
            },
            "metadata": self.metadata
        }


class TraceProcessor:
    """
    Processes raw execution traces for training data generation.

    Features:
    - Trace normalization and cleaning
    - Quality scoring and filtering
    - Deduplication
    - Command pattern extraction
    - (Optional) Safe Synthesizer integration for privacy-preserving PII handling
    """

    # Common command patterns to normalize
    COMMAND_PATTERNS = {
        r'cd\s+[\w/.-]+': 'cd <path>',
        r'cat\s+[\w/.-]+': 'cat <file>',
        r'ls\s+-?\w*\s*[\w/.-]*': 'ls <options> <path>',
        r'grep\s+.*': 'grep <pattern> <file>',
        r'sed\s+.*': 'sed <expression> <file>',
        r'pip\s+install\s+.*': 'pip install <packages>',
        r'python\s+[\w/.-]+': 'python <script>',
        r'git\s+\w+.*': 'git <command>',
    }

    # Sensitive patterns to redact (fallback when Safe Synthesizer not available)
    SENSITIVE_PATTERNS = [
        r'(?i)(api[_-]?key|token|secret|password|auth)\s*[=:]\s*["\']?[\w-]+["\']?',
        r'(?i)bearer\s+[\w.-]+',
        r'sk-[a-zA-Z0-9]+',  # OpenAI-style keys
        r'nvapi-[a-zA-Z0-9-]+',  # NVIDIA API keys
    ]

    def __init__(
        self,
        min_quality_score: float = 0.3,
        max_output_length: int = 5000,
        deduplicate: bool = True,
        safe_synthesizer: Optional["SafeSynthesizer"] = None
    ):
        """Initialize the trace processor.

        Args:
            min_quality_score: Minimum quality score for traces
            max_output_length: Maximum output length before truncation
            deduplicate: Whether to deduplicate traces
            safe_synthesizer: Optional SafeSynthesizer for advanced PII handling
        """
        self.min_quality_score = min_quality_score
        self.max_output_length = max_output_length
        self.deduplicate = deduplicate
        self.seen_hashes: Set[str] = set()
        self.safe_synthesizer = safe_synthesizer
    
    def process_trace(self, trace_path: Path) -> Optional[ProcessedTrace]:
        """
        Process a single trace file.
        
        Args:
            trace_path: Path to the trace JSON file
            
        Returns:
            ProcessedTrace or None if trace doesn't meet quality threshold
        """
        try:
            with open(trace_path, 'r') as f:
                raw_trace = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading trace {trace_path}: {e}")
            return None
        
        # Extract components
        metadata = raw_trace.get("metadata", {})
        raw_steps = raw_trace.get("trace", [])
        
        # Get task prompt
        task_prompt = metadata.get("user_initial_prompt", "")
        if not task_prompt:
            return None
        
        # Normalize steps
        normalized_steps = self._normalize_steps(raw_steps)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality(raw_trace, normalized_steps)
        
        # Check quality threshold
        if quality_metrics.quality_score < self.min_quality_score:
            return None
        
        # Generate trace ID
        trace_id = self._generate_trace_id(task_prompt, normalized_steps)
        
        # Check for duplicates
        if self.deduplicate and trace_id in self.seen_hashes:
            return None
        self.seen_hashes.add(trace_id)
        
        # Build processed metadata, preserving bashbros extensions if present
        processed_metadata = {
            "original_metadata": metadata,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Check for bashbros source and preserve extensions
        source_tool = metadata.get("source_tool", "")
        if source_tool == "bashbros":
            processed_metadata["source_tool"] = "bashbros"
            # Preserve bashbros-specific extensions
            if "bashbros_extensions" in raw_trace:
                processed_metadata["bashbros_extensions"] = raw_trace["bashbros_extensions"]

        return ProcessedTrace(
            trace_id=trace_id,
            original_path=trace_path,
            task_prompt=task_prompt,
            normalized_steps=normalized_steps,
            quality_metrics=quality_metrics,
            metadata=processed_metadata
        )
    
    def _normalize_steps(self, raw_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize trace steps.
        
        - Redact sensitive information
        - Truncate long outputs
        - Standardize field names
        """
        normalized = []
        
        for step in raw_steps:
            # Extract fields with fallbacks
            tool_name = step.get("tool_name", step.get("tool", "unknown"))
            command = step.get("command", step.get("input", ""))
            output = step.get("output", step.get("result", ""))
            success = step.get("success", step.get("exit_code") == 0 if "exit_code" in step else None)
            
            # Redact sensitive information
            command = self._redact_sensitive(str(command))
            output = self._redact_sensitive(str(output))
            
            # Truncate long outputs
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... [truncated]"
            
            normalized.append({
                "tool": tool_name.lower(),
                "command": command,
                "output": output,
                "success": success
            })
        
        return normalized
    
    def _redact_sensitive(self, text: str) -> str:
        """Redact sensitive information from text using regex patterns."""
        for pattern in self.SENSITIVE_PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text)
        return text

    async def _redact_sensitive_with_synthesizer(self, text: str) -> str:
        """
        Redact sensitive information using Safe Synthesizer.

        Provides more comprehensive PII detection than regex patterns.
        Falls back to regex if Safe Synthesizer is not available.
        """
        if not self.safe_synthesizer:
            return self._redact_sensitive(text)

        try:
            detections = self.safe_synthesizer.detect_pii(text)
            if detections:
                processed_text, _ = self.safe_synthesizer.replace_pii(text, detections)
                return processed_text
            return text
        except Exception:
            # Fallback to regex on any error
            return self._redact_sensitive(text)

    async def process_trace_async(self, trace_path: Path) -> Optional[ProcessedTrace]:
        """
        Process a single trace file with async Safe Synthesizer support.

        Args:
            trace_path: Path to the trace JSON file

        Returns:
            ProcessedTrace or None if trace doesn't meet quality threshold
        """
        try:
            with open(trace_path, 'r') as f:
                raw_trace = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading trace {trace_path}: {e}")
            return None

        # Extract components
        metadata = raw_trace.get("metadata", {})
        raw_steps = raw_trace.get("trace", [])

        # Get task prompt
        task_prompt = metadata.get("user_initial_prompt", "")
        if not task_prompt:
            return None

        # Normalize steps with async PII handling
        normalized_steps = await self._normalize_steps_async(raw_steps)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality(raw_trace, normalized_steps)

        # Check quality threshold
        if quality_metrics.quality_score < self.min_quality_score:
            return None

        # Generate trace ID
        trace_id = self._generate_trace_id(task_prompt, normalized_steps)

        # Check for duplicates
        if self.deduplicate and trace_id in self.seen_hashes:
            return None
        self.seen_hashes.add(trace_id)

        # Build processed metadata, preserving bashbros extensions if present
        processed_metadata = {
            "original_metadata": metadata,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "privacy_processed": self.safe_synthesizer is not None,
        }

        # Check for bashbros source and preserve extensions
        source_tool = metadata.get("source_tool", "")
        if source_tool == "bashbros":
            processed_metadata["source_tool"] = "bashbros"
            # Preserve bashbros-specific extensions
            if "bashbros_extensions" in raw_trace:
                processed_metadata["bashbros_extensions"] = raw_trace["bashbros_extensions"]

        return ProcessedTrace(
            trace_id=trace_id,
            original_path=trace_path,
            task_prompt=task_prompt,
            normalized_steps=normalized_steps,
            quality_metrics=quality_metrics,
            metadata=processed_metadata
        )

    async def _normalize_steps_async(
        self,
        raw_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize trace steps with async Safe Synthesizer support.
        """
        normalized = []

        for step in raw_steps:
            # Extract fields with fallbacks
            tool_name = step.get("tool_name", step.get("tool", "unknown"))
            command = step.get("command", step.get("input", ""))
            output = step.get("output", step.get("result", ""))
            success = step.get("success", step.get("exit_code") == 0 if "exit_code" in step else None)

            # Redact sensitive information
            if self.safe_synthesizer:
                command = await self._redact_sensitive_with_synthesizer(str(command))
                output = await self._redact_sensitive_with_synthesizer(str(output))
            else:
                command = self._redact_sensitive(str(command))
                output = self._redact_sensitive(str(output))

            # Truncate long outputs
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... [truncated]"

            normalized.append({
                "tool": tool_name.lower(),
                "command": command,
                "output": output,
                "success": success
            })

        return normalized

    async def process_directory_async(
        self,
        trace_dir: Path,
        output_path: Optional[Path] = None
    ) -> List[ProcessedTrace]:
        """
        Process all traces in a directory with async support.

        Args:
            trace_dir: Directory containing trace files
            output_path: Optional path to save processed traces

        Returns:
            List of processed traces
        """
        processed = []
        trace_files = list(Path(trace_dir).glob("*.json"))

        print(f"Processing {len(trace_files)} traces from {trace_dir}...")

        for trace_path in trace_files:
            result = await self.process_trace_async(trace_path)
            if result:
                processed.append(result)

        print(f"Processed {len(processed)} traces (filtered {len(trace_files) - len(processed)})")

        # Save if output path provided
        if output_path and processed:
            self._save_processed(processed, output_path)

        return processed
    
    def _calculate_quality(
        self,
        raw_trace: Dict[str, Any],
        normalized_steps: List[Dict[str, Any]]
    ) -> TraceQualityMetrics:
        """Calculate quality metrics for a trace using the centralized calculator."""
        metadata = raw_trace.get("metadata", {})

        # Use centralized quality calculator
        quality = calculate_quality_breakdown(steps=normalized_steps, metadata=metadata)

        # Calculate average output length (not in centralized calc)
        outputs = [s.get("output", "") for s in normalized_steps]
        avg_output_length = sum(len(o) for o in outputs) / len(outputs) if outputs else 0.0

        # Populate TraceQualityMetrics from breakdown
        metrics = TraceQualityMetrics(
            total_steps=quality.total_steps,
            successful_steps=quality.successful_steps,
            failed_steps=quality.failed_steps,
            unique_commands=quality.unique_commands_count,
            avg_output_length=avg_output_length,
            has_verification="verification_passed" in metadata,
            verification_passed=metadata.get("verification_passed", False),
            complexity_score=quality.complexity_score,
            tool_diversity=quality.tool_diversity,
            efficiency_score=quality.efficiency_score,
            length_score=quality.length_score,
            verification_score=quality.verification_score,
        )

        return metrics

    def _calculate_complexity(self, steps: List[Dict[str, Any]]) -> float:
        """
        Calculate complexity score for a trace.

        Delegates to centralized quality calculator for consistency.
        Returns normalized score (0-1).
        """
        from .quality_calculator import calculate_complexity
        score, _ = calculate_complexity(steps)
        return score
    
    def _generate_trace_id(
        self,
        task_prompt: str,
        steps: List[Dict[str, Any]]
    ) -> str:
        """Generate a unique ID for a trace based on content."""
        # Create a canonical representation
        canonical = {
            "task": task_prompt.strip().lower(),
            "commands": [s.get("command", "")[:100] for s in steps]
        }
        
        content = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def process_directory(
        self,
        trace_dir: Path,
        output_path: Optional[Path] = None
    ) -> List[ProcessedTrace]:
        """
        Process all traces in a directory.
        
        Args:
            trace_dir: Directory containing trace files
            output_path: Optional path to save processed traces
            
        Returns:
            List of processed traces
        """
        processed = []
        trace_files = list(Path(trace_dir).glob("*.json"))
        
        print(f"Processing {len(trace_files)} traces from {trace_dir}...")
        
        for trace_path in trace_files:
            result = self.process_trace(trace_path)
            if result:
                processed.append(result)
        
        print(f"Processed {len(processed)} traces (filtered {len(trace_files) - len(processed)})")
        
        # Save if output path provided
        if output_path and processed:
            self._save_processed(processed, output_path)
        
        return processed
    
    def _save_processed(
        self,
        traces: List[ProcessedTrace],
        output_path: Path
    ) -> None:
        """Save processed traces to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        
        print(f"Saved {len(traces)} processed traces to {output_path}")
    
    def extract_command_patterns(
        self,
        traces: List[ProcessedTrace]
    ) -> Dict[str, int]:
        """
        Extract common command patterns from traces.
        
        Useful for understanding what commands the agent uses most.
        """
        pattern_counts = defaultdict(int)
        
        for trace in traces:
            for step in trace.normalized_steps:
                cmd = step.get("command", "")
                
                # Try to match known patterns
                matched = False
                for pattern, normalized in self.COMMAND_PATTERNS.items():
                    if re.match(pattern, cmd):
                        pattern_counts[normalized] += 1
                        matched = True
                        break
                
                if not matched:
                    # Extract command name
                    parts = cmd.split()
                    if parts:
                        pattern_counts[parts[0]] += 1
        
        return dict(sorted(pattern_counts.items(), key=lambda x: -x[1]))
    
    def generate_statistics(
        self,
        traces: List[ProcessedTrace]
    ) -> Dict[str, Any]:
        """Generate statistics about processed traces."""
        if not traces:
            return {"error": "No traces to analyze"}
        
        quality_scores = [t.quality_metrics.quality_score for t in traces]
        step_counts = [t.quality_metrics.total_steps for t in traces]
        success_rates = [t.quality_metrics.success_rate for t in traces]
        
        return {
            "total_traces": len(traces),
            "quality_scores": {
                "mean": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores)
            },
            "step_counts": {
                "mean": sum(step_counts) / len(step_counts),
                "min": min(step_counts),
                "max": max(step_counts)
            },
            "success_rates": {
                "mean": sum(success_rates) / len(success_rates),
                "min": min(success_rates),
                "max": max(success_rates)
            },
            "verification_passed": sum(
                1 for t in traces if t.quality_metrics.verification_passed
            ),
            "command_patterns": self.extract_command_patterns(traces)
        }


def main():
    """Example usage of the Trace Processor."""
    processor = TraceProcessor(
        min_quality_score=0.3,
        deduplicate=True
    )
    
    # Process gold traces
    gold_dir = Path("data/gold_traces")
    if gold_dir.exists():
        processed = processor.process_directory(
            gold_dir,
            output_path=Path("data/processed_traces/gold_processed.jsonl")
        )
        
        # Generate statistics
        stats = processor.generate_statistics(processed)
        print("\nTrace Statistics:")
        print(json.dumps(stats, indent=2))
    else:
        print("No gold traces directory found")


if __name__ == "__main__":
    main()
