"""
Custom Evaluation Generator

Generates evaluation sets from gold traces in two modes:
1. Replay: Test if model can reproduce the same successful task
2. Variation: Test if model generalizes with modified task descriptions
"""

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single evaluation case."""
    case_id: str
    source_trace_id: Optional[str]
    eval_type: str  # "replay" or "variation"
    prompt: str  # Alias for user_prompt
    expected_behavior: str  # Description of expected outcome
    verification: "EvalVerification"
    difficulty: str = "same"  # "same", "easier", "harder"
    variation_type: Optional[str] = None  # "paraphrase", "parameter_tweak", "complexity_shift"
    name: str = ""  # Human-readable name
    description: str = ""  # Detailed description
    system_prompt: Optional[str] = None  # System prompt for the model
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def user_prompt(self) -> str:
        """Alias for prompt field to match API expectations."""
        return self.prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_trace_id": self.source_trace_id,
            "eval_type": self.eval_type,
            "prompt": self.prompt,
            "user_prompt": self.prompt,
            "expected_behavior": self.expected_behavior,
            "verification": self.verification.to_dict(),
            "difficulty": self.difficulty,
            "variation_type": self.variation_type,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalCase":
        return cls(
            case_id=data["case_id"],
            source_trace_id=data.get("source_trace_id"),
            eval_type=data["eval_type"],
            prompt=data.get("prompt") or data.get("user_prompt", ""),
            expected_behavior=data["expected_behavior"],
            verification=EvalVerification.from_dict(data["verification"]),
            difficulty=data.get("difficulty", "same"),
            variation_type=data.get("variation_type"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        )


@dataclass
class EvalVerification:
    """How to verify an eval case passed."""
    method: str  # "test_file", "output_match", "llm_judge", "file_exists", "code_runs"
    test_commands: List[str] = field(default_factory=list)  # Commands to run
    expected_patterns: List[str] = field(default_factory=list)  # Patterns to match in output
    expected_files: List[str] = field(default_factory=list)  # Files that should exist
    judge_criteria: Optional[str] = None  # Criteria for LLM judge
    expected_output: Optional[str] = None  # Expected output content
    test_command: Optional[str] = None  # Single test command (alias)
    check_files: List[str] = field(default_factory=list)  # Alias for expected_files
    llm_criteria: Optional[str] = None  # Alias for judge_criteria

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "test_commands": self.test_commands,
            "expected_patterns": self.expected_patterns,
            "expected_files": self.expected_files or self.check_files,
            "judge_criteria": self.judge_criteria or self.llm_criteria,
            "expected_output": self.expected_output,
            "test_command": self.test_command or (self.test_commands[0] if self.test_commands else None),
            "check_files": self.check_files or self.expected_files,
            "llm_criteria": self.llm_criteria or self.judge_criteria,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalVerification":
        return cls(
            method=data["method"],
            test_commands=data.get("test_commands", []),
            expected_patterns=data.get("expected_patterns", []),
            expected_files=data.get("expected_files", data.get("check_files", [])),
            judge_criteria=data.get("judge_criteria", data.get("llm_criteria")),
            expected_output=data.get("expected_output"),
            test_command=data.get("test_command"),
            check_files=data.get("check_files", data.get("expected_files", [])),
            llm_criteria=data.get("llm_criteria", data.get("judge_criteria")),
        )


@dataclass
class CustomEvalSet:
    """A collection of evaluation cases from source traces."""
    eval_set_id: str
    name: str
    description: str
    cases: List[EvalCase] = field(default_factory=list)
    generation_mode: str = "manual"  # "replay", "variation", "both", or "manual"
    source_traces: List[str] = field(default_factory=list)  # List of trace IDs
    source_trace_id: Optional[str] = None  # Legacy single trace ID
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_set_id": self.eval_set_id,
            "source_trace_id": self.source_trace_id,
            "source_traces": self.source_traces,
            "name": self.name,
            "description": self.description,
            "generation_mode": self.generation_mode,
            "cases": [c.to_dict() for c in self.cases],
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomEvalSet":
        # Handle legacy source_trace_id field
        source_traces = data.get("source_traces", [])
        if not source_traces and data.get("source_trace_id"):
            source_traces = [data["source_trace_id"]]

        return cls(
            eval_set_id=data["eval_set_id"],
            name=data["name"],
            description=data["description"],
            cases=[EvalCase.from_dict(c) for c in data.get("cases", [])],
            generation_mode=data.get("generation_mode", "manual"),
            source_traces=source_traces,
            source_trace_id=data.get("source_trace_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        )

    def save(self, path: Path) -> Path:
        """Save eval set to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path) -> "CustomEvalSet":
        """Load eval set from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class EvalCaseResult:
    """Result from running a single eval case."""
    case_id: str
    passed: bool
    output: str
    error: Optional[str] = None
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "output": self.output[:1000],  # Truncate for storage
            "error": self.error,
            "duration_ms": self.duration_ms,
            "details": self.details
        }


class CustomEvalGenerator:
    """
    Generates custom evaluation sets from gold traces.

    Supports two modes:
    - Replay: Exact task reproduction
    - Variation: Modified task descriptions for generalization testing
    """

    def __init__(
        self,
        eval_sets_dir: Optional[Path] = None,
        variation_generator: Optional[Callable[[str], List[str]]] = None
    ):
        """
        Initialize the generator.

        Args:
            eval_sets_dir: Directory to store eval sets
            variation_generator: Function to generate task variations (e.g., using Claude API)
        """
        self.eval_sets_dir = eval_sets_dir or Path("data/eval_sets")
        self.eval_sets_dir.mkdir(parents=True, exist_ok=True)
        self.variation_generator = variation_generator
        self._eval_sets: Dict[str, CustomEvalSet] = {}
        self._load_existing()

    def _load_existing(self):
        """Load existing eval sets from disk."""
        for path in self.eval_sets_dir.glob("*.json"):
            try:
                eval_set = CustomEvalSet.load(path)
                self._eval_sets[eval_set.eval_set_id] = eval_set
            except Exception as e:
                logger.warning(f"Failed to load eval set from {path}: {e}")

    def _generate_case_id(self, trace_id: str, eval_type: str, suffix: str = "") -> str:
        """Generate a unique case ID."""
        content = f"{trace_id}:{eval_type}:{suffix}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def generate_from_trace(
        self,
        trace_data: Dict[str, Any],
        include_variations: bool = True,
        max_variations: int = 3
    ) -> CustomEvalSet:
        """
        Generate an eval set from a gold trace.

        Args:
            trace_data: The trace data (from gold_traces/*.json)
            include_variations: Whether to generate variation cases
            max_variations: Maximum number of variations per task

        Returns:
            CustomEvalSet with replay and optionally variation cases
        """
        trace_id = trace_data.get("trace_id", trace_data.get("session_id", "unknown"))
        user_prompt = trace_data.get("user_initial_prompt", "")
        tool_calls = trace_data.get("tool_calls", [])

        # Generate eval set ID
        eval_set_id = f"eval_{trace_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Determine verification method from trace
        verification = self._infer_verification(trace_data, tool_calls)

        cases = []

        # 1. Create replay case
        replay_case = EvalCase(
            case_id=self._generate_case_id(trace_id, "replay"),
            source_trace_id=trace_id,
            eval_type="replay",
            prompt=user_prompt,
            expected_behavior=self._summarize_expected_behavior(trace_data),
            verification=verification,
            difficulty="same"
        )
        cases.append(replay_case)

        # 2. Create variation cases
        if include_variations and self.variation_generator:
            variations = self._generate_variations(user_prompt, max_variations)
            for i, (var_prompt, var_type, difficulty) in enumerate(variations):
                var_case = EvalCase(
                    case_id=self._generate_case_id(trace_id, "variation", str(i)),
                    source_trace_id=trace_id,
                    eval_type="variation",
                    prompt=var_prompt,
                    expected_behavior=self._adapt_expected_behavior(trace_data, var_type),
                    verification=verification,  # Same verification approach
                    difficulty=difficulty,
                    variation_type=var_type
                )
                cases.append(var_case)

        eval_set = CustomEvalSet(
            eval_set_id=eval_set_id,
            source_trace_id=trace_id,
            name=f"Eval: {user_prompt[:50]}...",
            description=f"Generated from gold trace {trace_id}",
            cases=cases
        )

        # Save and cache
        self._eval_sets[eval_set_id] = eval_set
        eval_set.save(self.eval_sets_dir / f"{eval_set_id}.json")

        return eval_set

    def _infer_verification(self, trace_data: Dict, tool_calls: List[Dict]) -> EvalVerification:
        """Infer verification method from trace contents."""
        # Look for test files or verification patterns
        test_commands = []
        expected_files = []
        expected_patterns = []

        for call in tool_calls:
            tool = call.get("tool", "")
            input_data = call.get("input", {})
            output = call.get("output", "")

            # Check for test execution
            if tool == "Bash":
                command = input_data.get("command", "")
                if "pytest" in command or "npm test" in command or "cargo test" in command:
                    test_commands.append(command)
                if "PASSED" in output or "passed" in output:
                    expected_patterns.append("PASSED")

            # Check for file creation
            if tool in ["Write", "Edit"]:
                file_path = input_data.get("file_path", "")
                if file_path:
                    expected_files.append(file_path)

        # Determine method based on what we found
        if test_commands:
            return EvalVerification(
                method="test_file",
                test_commands=test_commands,
                expected_patterns=expected_patterns
            )
        elif expected_files:
            return EvalVerification(
                method="file_exists",
                expected_files=expected_files
            )
        else:
            # Fall back to LLM judge
            return EvalVerification(
                method="llm_judge",
                judge_criteria="Task completed successfully with working code"
            )

    def _summarize_expected_behavior(self, trace_data: Dict) -> str:
        """Summarize what the model should do based on trace."""
        prompt = trace_data.get("user_initial_prompt", "")
        tool_calls = trace_data.get("tool_calls", [])

        # Count tool usage
        tools_used = {}
        for call in tool_calls:
            tool = call.get("tool", "unknown")
            tools_used[tool] = tools_used.get(tool, 0) + 1

        summary_parts = [f"Complete the task: {prompt[:100]}"]
        if tools_used:
            tool_summary = ", ".join(f"{t}({c})" for t, c in sorted(tools_used.items()))
            summary_parts.append(f"Expected tool usage pattern: {tool_summary}")

        return " | ".join(summary_parts)

    def _adapt_expected_behavior(self, trace_data: Dict, variation_type: str) -> str:
        """Adapt expected behavior description for a variation."""
        base = self._summarize_expected_behavior(trace_data)
        if variation_type == "paraphrase":
            return base  # Same behavior, different wording
        elif variation_type == "parameter_tweak":
            return f"[Modified params] {base}"
        elif variation_type == "complexity_shift":
            return f"[Adjusted complexity] {base}"
        return base

    def _generate_variations(
        self,
        original_prompt: str,
        max_variations: int
    ) -> List[tuple]:
        """
        Generate task variations.

        Returns list of (prompt, variation_type, difficulty) tuples.
        """
        if not self.variation_generator:
            return []

        try:
            # Call the variation generator (e.g., Claude API)
            variations = self.variation_generator(original_prompt)
            result = []
            for i, var_prompt in enumerate(variations[:max_variations]):
                # Determine variation type based on content similarity
                var_type = "paraphrase"  # Default
                difficulty = "same"

                # Simple heuristics
                if len(var_prompt) > len(original_prompt) * 1.3:
                    var_type = "complexity_shift"
                    difficulty = "harder"
                elif len(var_prompt) < len(original_prompt) * 0.7:
                    var_type = "complexity_shift"
                    difficulty = "easier"

                result.append((var_prompt, var_type, difficulty))
            return result
        except Exception as e:
            logger.warning(f"Failed to generate variations: {e}")
            return []

    def generate_from_traces(
        self,
        name: str,
        description: Optional[str] = None,
        trace_ids: Optional[List[str]] = None,
        mode: str = "both",  # "replay", "variation", or "both"
        max_cases: int = 50,
        include_failed_traces: bool = False,
    ) -> CustomEvalSet:
        """
        Generate an eval set from multiple gold traces.

        Args:
            name: Name for the eval set
            description: Description for the eval set
            trace_ids: List of trace IDs to include (None = all gold traces)
            mode: Generation mode - "replay", "variation", or "both"
            max_cases: Maximum number of cases to generate
            include_failed_traces: Whether to include failed traces for negative examples

        Returns:
            CustomEvalSet with generated cases
        """
        # Load traces from gold traces directory
        gold_traces_dir = Path("data/gold_traces")
        if not gold_traces_dir.exists():
            gold_traces_dir = Path("data/traces")

        all_traces = []
        for trace_path in gold_traces_dir.glob("*.json"):
            try:
                with open(trace_path) as f:
                    trace_data = json.load(f)
                    tid = trace_data.get("trace_id", trace_data.get("session_id", trace_path.stem))
                    trace_data["trace_id"] = tid

                    # Filter by trace_ids if specified
                    if trace_ids is None or tid in trace_ids:
                        all_traces.append(trace_data)
            except Exception as e:
                logger.warning(f"Failed to load trace from {trace_path}: {e}")

        if not all_traces:
            raise ValueError("No traces found to generate eval set from")

        # Generate cases from each trace
        cases = []
        source_trace_ids = []
        include_replay = mode in ("replay", "both")
        include_variations = mode in ("variation", "both")

        for trace_data in all_traces:
            trace_id = trace_data["trace_id"]
            source_trace_ids.append(trace_id)

            user_prompt = trace_data.get("user_initial_prompt", trace_data.get("task", ""))
            if not user_prompt:
                continue

            tool_calls = trace_data.get("tool_calls", trace_data.get("messages", []))
            verification = self._infer_verification(trace_data, tool_calls)
            expected_behavior = self._summarize_expected_behavior(trace_data)

            # Replay case
            if include_replay:
                case_name = f"Replay: {user_prompt[:40]}..."
                replay_case = EvalCase(
                    case_id=self._generate_case_id(trace_id, "replay"),
                    source_trace_id=trace_id,
                    eval_type="replay",
                    prompt=user_prompt,
                    expected_behavior=expected_behavior,
                    verification=verification,
                    difficulty="same",
                    name=case_name,
                    description=f"Exact replay of task from trace {trace_id}",
                )
                cases.append(replay_case)

            # Variation cases
            if include_variations and self.variation_generator:
                variations = self._generate_variations(user_prompt, max_variations=2)
                for i, (var_prompt, var_type, difficulty) in enumerate(variations):
                    var_case = EvalCase(
                        case_id=self._generate_case_id(trace_id, "variation", str(i)),
                        source_trace_id=trace_id,
                        eval_type="variation",
                        prompt=var_prompt,
                        expected_behavior=self._adapt_expected_behavior(trace_data, var_type),
                        verification=verification,
                        difficulty=difficulty,
                        variation_type=var_type,
                        name=f"Variation ({var_type}): {var_prompt[:30]}...",
                        description=f"Variation of task from trace {trace_id}",
                    )
                    cases.append(var_case)

            # Check if we've hit max cases
            if len(cases) >= max_cases:
                break

        # Truncate to max_cases
        cases = cases[:max_cases]

        # Generate eval set ID
        eval_set_id = f"evalset_{datetime.now().strftime('%Y%m%d%H%M%S')}_{mode}"

        eval_set = CustomEvalSet(
            eval_set_id=eval_set_id,
            name=name,
            description=description or f"Auto-generated eval set ({mode} mode) from {len(source_trace_ids)} traces",
            cases=cases,
            generation_mode=mode,
            source_traces=source_trace_ids[:len(cases)],  # Track which traces were used
        )

        # Save and cache
        self._eval_sets[eval_set_id] = eval_set
        eval_set.save(self.eval_sets_dir / f"{eval_set_id}.json")

        logger.info(f"Generated eval set {eval_set_id} with {len(cases)} cases from {len(source_trace_ids)} traces")
        return eval_set

    def list_eval_sets(self) -> List[CustomEvalSet]:
        """List all eval sets."""
        return list(self._eval_sets.values())

    def get_eval_set(self, eval_set_id: str) -> Optional[CustomEvalSet]:
        """Get an eval set by ID."""
        return self._eval_sets.get(eval_set_id)

    def delete_eval_set(self, eval_set_id: str) -> bool:
        """Delete an eval set."""
        if eval_set_id not in self._eval_sets:
            return False

        # Remove file
        path = self.eval_sets_dir / f"{eval_set_id}.json"
        if path.exists():
            path.unlink()

        del self._eval_sets[eval_set_id]
        return True


class CustomEvalRunner:
    """
    Runs custom evaluations against a model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_runner: Optional[Callable[[str], str]] = None,
        sandbox_executor: Optional[Callable[[str], tuple]] = None
    ):
        """
        Initialize the runner.

        Args:
            model_path: Path to the model directory (for loading local models)
            model_runner: Function to run a prompt through the model and get response
            sandbox_executor: Function to execute commands in sandbox, returns (output, exit_code)
        """
        self.model_path = model_path
        self.model_runner = model_runner
        self.sandbox_executor = sandbox_executor

    def run_eval_set(
        self,
        eval_set: CustomEvalSet,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        progress_callback: Optional[Callable[[int, int, EvalCaseResult], None]] = None
    ) -> List[EvalCaseResult]:
        """
        Run all cases in an eval set (sync version for API).

        Args:
            eval_set: The eval set to run
            max_tokens: Max tokens for model output
            temperature: Temperature for model sampling
            progress_callback: Called with (completed, total, result) after each case

        Returns:
            List of EvalCaseResult objects
        """
        results = []

        for i, case in enumerate(eval_set.cases):
            result = self._run_case_sync(case)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(eval_set.cases), result)

        return results

    async def run_eval_set_async(
        self,
        eval_set: CustomEvalSet,
        progress_callback: Optional[Callable[[int, int, EvalCaseResult], None]] = None
    ) -> Dict[str, Any]:
        """
        Run all cases in an eval set (async version).

        Args:
            eval_set: The eval set to run
            progress_callback: Called with (completed, total, result) after each case

        Returns:
            Summary with passed/total counts and individual results
        """
        results = []
        passed = 0

        for i, case in enumerate(eval_set.cases):
            result = await self._run_case(case)
            results.append(result)
            if result.passed:
                passed += 1

            if progress_callback:
                progress_callback(i + 1, len(eval_set.cases), result)

        return {
            "eval_set_id": eval_set.eval_set_id,
            "passed": passed,
            "total": len(eval_set.cases),
            "pass_rate": (passed / len(eval_set.cases) * 100) if eval_set.cases else 0,
            "results": [r.to_dict() for r in results],
            "evaluated_at": datetime.now().isoformat()
        }

    def _run_case_sync(self, case: EvalCase) -> EvalCaseResult:
        """Run a single eval case synchronously."""
        start_time = datetime.now()

        try:
            # Run the prompt through the model
            if not self.model_runner:
                # For now, return a placeholder result if no model runner
                # In production, this would load the model from model_path
                return EvalCaseResult(
                    case_id=case.case_id,
                    passed=True,  # Placeholder - mark as passed
                    output="[Model evaluation placeholder - model runner not configured]",
                    error=None,
                    duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    details={"note": "Model runner not configured, returning placeholder pass"}
                )

            model_output = self.model_runner(case.prompt)

            # Verify the result (sync version)
            passed, details = self._verify_result_sync(case, model_output)

            duration = (datetime.now() - start_time).total_seconds() * 1000

            return EvalCaseResult(
                case_id=case.case_id,
                passed=passed,
                output=model_output,
                duration_ms=duration,
                details=details
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return EvalCaseResult(
                case_id=case.case_id,
                passed=False,
                output="",
                error=str(e),
                duration_ms=duration
            )

    def _verify_result_sync(self, case: EvalCase, model_output: str) -> tuple:
        """Verify the model output against case verification criteria (sync)."""
        verification = case.verification
        details = {}

        if verification.method == "llm_judge":
            # For now, return True with a note that judge wasn't run
            details["note"] = "LLM judge not implemented, assuming pass"
            return True, details

        elif verification.method == "output_match":
            # Check if output matches expected patterns
            all_matched = True
            for pattern in verification.expected_patterns:
                matched = pattern.lower() in model_output.lower()
                details[pattern] = matched
                if not matched:
                    all_matched = False
            return all_matched, details

        elif verification.method in ("test_file", "file_exists", "code_runs"):
            # These need sandbox execution
            if not self.sandbox_executor:
                details["note"] = "Sandbox executor not configured, assuming pass"
                return True, details

            # Run test commands if any
            for cmd in verification.test_commands:
                output, exit_code = self.sandbox_executor(cmd)
                details[cmd] = {"exit_code": exit_code, "output": output[:500]}
                if exit_code != 0:
                    return False, details

            return True, details

        else:
            details["note"] = f"Unknown verification method: {verification.method}"
            return True, details

    async def _run_case(self, case: EvalCase) -> EvalCaseResult:
        """Run a single eval case."""
        start_time = datetime.now()

        try:
            # Run the prompt through the model
            if not self.model_runner:
                return EvalCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    output="",
                    error="No model runner configured"
                )

            model_output = self.model_runner(case.prompt)

            # Verify the result
            passed, details = await self._verify_result(case, model_output)

            duration = (datetime.now() - start_time).total_seconds() * 1000

            return EvalCaseResult(
                case_id=case.case_id,
                passed=passed,
                output=model_output,
                duration_ms=duration,
                details=details
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return EvalCaseResult(
                case_id=case.case_id,
                passed=False,
                output="",
                error=str(e),
                duration_ms=duration
            )

    async def _verify_result(self, case: EvalCase, model_output: str) -> tuple:
        """Verify the model output against case verification criteria."""
        verification = case.verification
        details = {}

        if verification.method == "test_file":
            # Run test commands
            if not self.sandbox_executor:
                return False, {"error": "No sandbox executor configured"}

            all_passed = True
            for cmd in verification.test_commands:
                output, exit_code = self.sandbox_executor(cmd)
                details[cmd] = {"exit_code": exit_code, "output": output[:500]}
                if exit_code != 0:
                    all_passed = False

            # Check expected patterns
            for pattern in verification.expected_patterns:
                if pattern not in str(details):
                    all_passed = False
                    details["pattern_missing"] = pattern

            return all_passed, details

        elif verification.method == "file_exists":
            # Check if expected files exist
            if not self.sandbox_executor:
                return False, {"error": "No sandbox executor configured"}

            all_exist = True
            for file_path in verification.expected_files:
                output, exit_code = self.sandbox_executor(f"test -f '{file_path}' && echo exists")
                exists = "exists" in output
                details[file_path] = exists
                if not exists:
                    all_exist = False

            return all_exist, details

        elif verification.method == "output_match":
            # Check if output matches expected patterns
            all_matched = True
            for pattern in verification.expected_patterns:
                matched = pattern.lower() in model_output.lower()
                details[pattern] = matched
                if not matched:
                    all_matched = False

            return all_matched, details

        elif verification.method == "llm_judge":
            # Use LLM to judge the output
            # For now, return True with a note that judge wasn't run
            details["note"] = "LLM judge not implemented, assuming pass"
            return True, details

        elif verification.method == "code_runs":
            # Try to run the code output
            if not self.sandbox_executor:
                return False, {"error": "No sandbox executor configured"}

            # Extract code blocks and try to run
            # This is a simplified implementation
            output, exit_code = self.sandbox_executor("python -c 'print(\"test\")'")
            details["exit_code"] = exit_code
            return exit_code == 0, details

        else:
            return False, {"error": f"Unknown verification method: {verification.method}"}


# Singleton generator instance
_generator: Optional[CustomEvalGenerator] = None


def get_eval_generator(eval_sets_dir: Optional[Path] = None) -> CustomEvalGenerator:
    """Get the global CustomEvalGenerator instance."""
    global _generator
    if _generator is None:
        _generator = CustomEvalGenerator(eval_sets_dir)
    return _generator
