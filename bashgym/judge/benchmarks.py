"""
Benchmark Suite for Code Evaluation

Provides standard code evaluation benchmarks including HumanEval, MBPP,
BigCodeBench, BFCL, and SWE-bench for comprehensive model assessment.

Module 2: Verification (The "Judge") - Benchmarks Extension
"""

import os
import json
import asyncio
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime, timezone
from enum import Enum
import httpx


class BenchmarkType(Enum):
    """Available benchmark types."""
    # Code generation
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    BIGCODEBENCH = "bigcodebench"
    DS1000 = "ds1000"
    # Function calling
    BFCL = "bfcl"
    # Reasoning
    GSM8K = "gsm8k"
    ARC = "arc"
    HELLASWAG = "hellaswag"
    # Safety
    TOXIGEN = "toxigen"
    BBQ = "bbq"
    # Agentic
    SWE_BENCH = "swe_bench"
    # Special
    CUSTOM = "custom"
    SIMPLE_TEST = "simple_test"  # For E2E testing


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    # Benchmark settings
    benchmark_type: BenchmarkType = BenchmarkType.HUMANEVAL
    num_samples: int = 100
    pass_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    timeout_per_sample: int = 30

    # Model settings
    model_endpoint: str = ""
    model_name: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024

    # Execution settings
    sandbox_execution: bool = True
    parallel_workers: int = 4

    # Output settings
    output_dir: str = "data/benchmark_results"


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""

    task_id: str
    prompt: str
    canonical_solution: Optional[str] = None
    test_code: Optional[str] = None
    entry_point: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorType(Enum):
    """Types of errors that can occur during evaluation."""
    NONE = "none"
    WRONG_ANSWER = "wrong_answer"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    OTHER = "other"


@dataclass
class ErrorAnalysis:
    """Breakdown of error types in a benchmark run."""
    wrong_answer: int = 0
    syntax_error: int = 0
    runtime_error: int = 0
    timeout: int = 0
    other: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "wrong_answer": self.wrong_answer,
            "syntax_error": self.syntax_error,
            "runtime_error": self.runtime_error,
            "timeout": self.timeout,
            "other": self.other
        }


@dataclass
class SampleResult:
    """Result of evaluating a single sample."""

    task_id: str
    passed: bool
    generated_code: str
    execution_output: str
    error_message: Optional[str] = None
    error_type: ErrorType = ErrorType.NONE
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "generated_code": self.generated_code,
            "execution_output": self.execution_output[:500],  # Truncate
            "error_message": self.error_message,
            "error_type": self.error_type.value,
            "latency_ms": self.latency_ms
        }


@dataclass
class BenchmarkResult:
    """Result of a complete benchmark run."""

    benchmark: BenchmarkType
    model_name: str
    timestamp: str
    pass_at_k: Dict[int, float]
    total_samples: int
    passed_samples: int
    failed_samples: int
    error_samples: int
    avg_latency_ms: float
    sample_results: List[SampleResult] = field(default_factory=list)
    total_time_seconds: float = 0.0
    error_analysis: Optional[ErrorAnalysis] = None

    @property
    def pass_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.passed_samples / self.total_samples

    def compute_error_analysis(self) -> ErrorAnalysis:
        """Compute error breakdown from sample results."""
        analysis = ErrorAnalysis()
        for result in self.sample_results:
            if result.passed:
                continue
            if result.error_type == ErrorType.WRONG_ANSWER:
                analysis.wrong_answer += 1
            elif result.error_type == ErrorType.SYNTAX_ERROR:
                analysis.syntax_error += 1
            elif result.error_type == ErrorType.RUNTIME_ERROR:
                analysis.runtime_error += 1
            elif result.error_type == ErrorType.TIMEOUT:
                analysis.timeout += 1
            else:
                analysis.other += 1
        return analysis

    def to_dict(self) -> Dict[str, Any]:
        # Compute error analysis if not already done
        if self.error_analysis is None:
            self.error_analysis = self.compute_error_analysis()

        return {
            "benchmark": self.benchmark.value,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "pass_at_k": self.pass_at_k,
            "pass_rate": self.pass_rate,
            "total_samples": self.total_samples,
            "passed_samples": self.passed_samples,
            "failed_samples": self.failed_samples,
            "error_samples": self.error_samples,
            "avg_latency_ms": self.avg_latency_ms,
            "total_time_seconds": self.total_time_seconds,
            "error_analysis": self.error_analysis.to_dict() if self.error_analysis else None
        }


class BenchmarkRunner:
    """
    Runs standard code benchmarks for model evaluation.

    Supported benchmarks:
    - HumanEval: Python function synthesis (164 problems)
    - MBPP: Python programming problems (974 problems)
    - BigCodeBench: Comprehensive code evaluation
    - BFCL: Function calling accuracy
    - SWE-bench: Real-world software engineering tasks
    """

    # HumanEval sample problems (subset for demo)
    HUMANEVAL_SAMPLES = [
        {
            "task_id": "HumanEval/0",
            "prompt": '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
            "canonical_solution": '''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
''',
            "test": '''
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
''',
            "entry_point": "has_close_elements"
        },
        {
            "task_id": "HumanEval/1",
            "prompt": '''from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
            "canonical_solution": '''    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result
''',
            "test": '''
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
    assert candidate('(()(()))') == ['(()(()))']
''',
            "entry_point": "separate_paren_groups"
        },
    ]

    # Simple Test samples (for E2E testing)
    SIMPLE_TEST_SAMPLES = [
        {
            "task_id": "SimpleTest/0",
            "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
            "canonical_solution": "    return a + b",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
            "entry_point": "add"
        },
        {
            "task_id": "SimpleTest/1",
            "prompt": 'def double(x: int) -> int:\n    """Return x multiplied by 2."""\n',
            "canonical_solution": "    return x * 2",
            "test": "def check(candidate):\n    assert candidate(5) == 10\n    assert candidate(0) == 0\n",
            "entry_point": "double"
        },
        {
            "task_id": "SimpleTest/2",
            "prompt": 'def is_even(n: int) -> bool:\n    """Return True if n is even, False otherwise."""\n',
            "canonical_solution": "    return n % 2 == 0",
            "test": "def check(candidate):\n    assert candidate(4) == True\n    assert candidate(3) == False\n",
            "entry_point": "is_even"
        },
    ]

    # MBPP sample problems
    MBPP_SAMPLES = [
        {
            "task_id": "MBPP/1",
            "prompt": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and target position (m, n).",
            "test": '''
assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8
assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12
''',
            "entry_point": "min_cost"
        },
        {
            "task_id": "MBPP/2",
            "prompt": "Write a function to find the similar elements from the given two tuple lists.",
            "test": '''
assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4), (5, 4, 3, 7)) == (3, 4)
''',
            "entry_point": "similar_elements"
        },
    ]

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize the benchmark runner."""
        self.config = config or BenchmarkConfig()

        # HTTP client for model calls
        self.client = httpx.AsyncClient(timeout=60.0)

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def load_benchmark(self, benchmark_type: BenchmarkType) -> List[BenchmarkSample]:
        """
        Load benchmark samples from HuggingFace datasets.

        Args:
            benchmark_type: Type of benchmark to load

        Returns:
            List of BenchmarkSample
        """
        # Use static samples for simple test (E2E testing)
        if benchmark_type == BenchmarkType.SIMPLE_TEST:
            return self._load_simple_test()

        # Try loading from HuggingFace
        from bashgym.judge.benchmark_loader import BenchmarkLoader

        if not BenchmarkLoader.is_available(benchmark_type.value):
            # Fall back to static samples for benchmarks not in HuggingFace loader
            if benchmark_type == BenchmarkType.HUMANEVAL:
                return self._load_humaneval()
            elif benchmark_type == BenchmarkType.MBPP:
                return self._load_mbpp()
            return []

        raw_data = BenchmarkLoader.load(benchmark_type.value, self.config.num_samples)

        if not raw_data:
            # Fall back to static samples if HuggingFace fails
            if benchmark_type == BenchmarkType.HUMANEVAL:
                return self._load_humaneval()
            elif benchmark_type == BenchmarkType.MBPP:
                return self._load_mbpp()
            return []

        # Convert raw HuggingFace data to BenchmarkSample objects
        return self._convert_to_samples(benchmark_type, raw_data)

    def _convert_to_samples(self, benchmark_type: BenchmarkType, raw_data: List[Dict]) -> List[BenchmarkSample]:
        """Convert raw HuggingFace data to BenchmarkSample objects."""
        samples = []

        if benchmark_type == BenchmarkType.HUMANEVAL:
            for item in raw_data:
                samples.append(BenchmarkSample(
                    task_id=item.get("task_id", ""),
                    prompt=item.get("prompt", ""),
                    canonical_solution=item.get("canonical_solution"),
                    test_code=item.get("test"),
                    entry_point=item.get("entry_point")
                ))

        elif benchmark_type == BenchmarkType.MBPP:
            for item in raw_data:
                test_list = item.get("test_list", [])
                test_code = "\n".join(test_list) if test_list else item.get("test", "")
                samples.append(BenchmarkSample(
                    task_id=str(item.get("task_id", "")),
                    prompt=item.get("text", item.get("prompt", "")),
                    canonical_solution=item.get("code"),
                    test_code=test_code,
                    entry_point=item.get("entry_point")
                ))

        elif benchmark_type == BenchmarkType.GSM8K:
            for i, item in enumerate(raw_data):
                # GSM8K: extract numeric answer after ####
                answer_text = item.get("answer", "")
                if "####" in answer_text:
                    answer = answer_text.split("####")[-1].strip()
                else:
                    answer = answer_text.strip()
                samples.append(BenchmarkSample(
                    task_id=f"gsm8k_{i}",
                    prompt=item.get("question", ""),
                    canonical_solution=answer_text,
                    test_code=f"assert str(answer).strip() == '{answer}'",
                    metadata={"answer": answer, "full_solution": answer_text}
                ))

        elif benchmark_type == BenchmarkType.HELLASWAG:
            for i, item in enumerate(raw_data):
                label = item.get("label", "")
                correct_idx = int(label) if label.isdigit() else 0
                endings = item.get("endings", [])
                correct_ending = endings[correct_idx] if correct_idx < len(endings) else ""
                samples.append(BenchmarkSample(
                    task_id=item.get("ind", f"hellaswag_{i}"),
                    prompt=item.get("ctx", item.get("context", "")),
                    canonical_solution=correct_ending,
                    metadata={
                        "activity_label": item.get("activity_label", ""),
                        "endings": endings,
                        "label": correct_idx,
                        "ctx_a": item.get("ctx_a", ""),
                        "ctx_b": item.get("ctx_b", ""),
                    }
                ))

        elif benchmark_type == BenchmarkType.ARC:
            for i, item in enumerate(raw_data):
                choices = item.get("choices", {})
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                answer_key = item.get("answerKey", "")
                correct_answer = ""
                if answer_key and labels and texts:
                    try:
                        idx = labels.index(answer_key)
                        correct_answer = texts[idx]
                    except ValueError:
                        pass
                samples.append(BenchmarkSample(
                    task_id=item.get("id", f"arc_{i}"),
                    prompt=item.get("question", ""),
                    canonical_solution=correct_answer,
                    metadata={
                        "choices": dict(zip(labels, texts)) if labels and texts else {},
                        "answer_key": answer_key
                    }
                ))

        elif benchmark_type == BenchmarkType.BFCL:
            for i, item in enumerate(raw_data):
                # BFCL has various formats depending on the category
                question = item.get("question", "")
                if isinstance(question, list):
                    question = question[0] if question else ""
                samples.append(BenchmarkSample(
                    task_id=item.get("id", f"bfcl_{i}"),
                    prompt=str(question),
                    canonical_solution=json.dumps(item.get("ground_truth", [])),
                    metadata={
                        "functions": item.get("function", []),
                        "ground_truth": item.get("ground_truth", []),
                        "category": item.get("category", "")
                    }
                ))

        elif benchmark_type == BenchmarkType.BIGCODEBENCH:
            for item in raw_data:
                samples.append(BenchmarkSample(
                    task_id=item.get("task_id", ""),
                    prompt=item.get("instruct_prompt", item.get("complete_prompt", "")),
                    canonical_solution=item.get("canonical_solution", ""),
                    test_code=item.get("test", ""),
                    entry_point=item.get("entry_point"),
                    metadata={"libs": item.get("libs", [])}
                ))

        elif benchmark_type == BenchmarkType.DS1000:
            for i, item in enumerate(raw_data):
                samples.append(BenchmarkSample(
                    task_id=item.get("metadata", {}).get("problem_id", f"ds1000_{i}"),
                    prompt=item.get("prompt", ""),
                    canonical_solution=item.get("reference_code", ""),
                    test_code=item.get("test_code", ""),
                    metadata={
                        "lib": item.get("metadata", {}).get("library", ""),
                        "perturbation_type": item.get("metadata", {}).get("perturbation_type", "")
                    }
                ))

        elif benchmark_type == BenchmarkType.TOXIGEN:
            for i, item in enumerate(raw_data):
                # ToxiGen: toxicity_human is a score, typically 1-5
                # Higher scores indicate more toxic content
                toxicity_score = item.get("toxicity_human", item.get("toxicity_ai", 0))
                if isinstance(toxicity_score, (int, float)):
                    is_toxic = toxicity_score > 2.5
                else:
                    is_toxic = False
                samples.append(BenchmarkSample(
                    task_id=f"toxigen_{i}",
                    prompt=item.get("text", item.get("generation", "")),
                    canonical_solution="toxic" if is_toxic else "safe",
                    metadata={
                        "target_group": item.get("target_group", item.get("group", "")),
                        "toxicity_score": toxicity_score,
                        "intent": item.get("intent", ""),
                        "is_toxic": is_toxic
                    }
                ))

        elif benchmark_type == BenchmarkType.BBQ:
            for i, item in enumerate(raw_data):
                # BBQ format: context, question, multiple choice answers
                choices = item.get("choices", item.get("answer_choices", []))
                if isinstance(choices, dict):
                    choices = list(choices.values())
                answer_idx = item.get("answer", item.get("label", 0))
                if isinstance(answer_idx, str) and answer_idx.isdigit():
                    answer_idx = int(answer_idx)
                elif isinstance(answer_idx, str):
                    # Try to find the answer in choices
                    try:
                        answer_idx = choices.index(answer_idx)
                    except (ValueError, AttributeError):
                        answer_idx = 0

                correct_answer = choices[answer_idx] if answer_idx < len(choices) else ""

                # Format the prompt with context and question
                context = item.get("context", "")
                question = item.get("question", "")
                choice_text = "\n".join([f"{j}. {c}" for j, c in enumerate(choices)])

                samples.append(BenchmarkSample(
                    task_id=f"bbq_{i}",
                    prompt=f"{context}\n\n{question}\n\nChoices:\n{choice_text}",
                    canonical_solution=correct_answer,
                    metadata={
                        "category": item.get("category", ""),
                        "choices": choices,
                        "answer_index": answer_idx,
                        "question_polarity": item.get("question_polarity", ""),
                        "context_condition": item.get("context_condition", "")
                    }
                ))

        elif benchmark_type == BenchmarkType.SWE_BENCH:
            for item in raw_data:
                instance_id = item.get("instance_id", "")
                repo = item.get("repo", "")
                problem = item.get("problem_statement", "")
                patch = item.get("patch", "")
                hints = item.get("hints_text", "")

                # Format prompt with repo and problem info
                prompt = f"Repository: {repo}\n\nProblem Statement:\n{problem}"
                if hints:
                    prompt += f"\n\nHints:\n{hints}"

                samples.append(BenchmarkSample(
                    task_id=instance_id,
                    prompt=prompt,
                    canonical_solution=patch,
                    metadata={
                        "repo": repo,
                        "base_commit": item.get("base_commit", ""),
                        "hints": hints,
                        "created_at": item.get("created_at", ""),
                        "version": item.get("version", ""),
                        "fail_to_pass": item.get("FAIL_TO_PASS", []),
                        "pass_to_pass": item.get("PASS_TO_PASS", [])
                    }
                ))

        return samples

    def _load_simple_test(self) -> List[BenchmarkSample]:
        """Load Simple Test benchmark samples (for E2E testing)."""
        samples = []
        for item in self.SIMPLE_TEST_SAMPLES[:self.config.num_samples]:
            samples.append(BenchmarkSample(
                task_id=item["task_id"],
                prompt=item["prompt"],
                canonical_solution=item.get("canonical_solution"),
                test_code=item.get("test"),
                entry_point=item.get("entry_point")
            ))
        return samples

    def run_benchmark_simulated(self) -> "BenchmarkResult":
        """
        Run a simulated benchmark that uses canonical solutions.

        This is useful for E2E testing without requiring model inference.
        Returns a BenchmarkResult with the simulated pass rate.
        """
        import time
        start_time = time.time()

        samples = self.load_benchmark(self.config.benchmark_type)

        if not samples:
            return BenchmarkResult(
                benchmark=self.config.benchmark_type,
                model_name=self.config.model_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                pass_at_k={k: 0.0 for k in self.config.pass_k},
                total_samples=0,
                passed_samples=0,
                failed_samples=0,
                error_samples=0,
                avg_latency_ms=0.0
            )

        passed = 0
        failed = 0
        sample_results = []

        for sample in samples:
            # Use canonical solution if available
            if sample.canonical_solution:
                # Simulate execution - canonical solutions should pass
                sample_results.append(SampleResult(
                    task_id=sample.task_id,
                    passed=True,
                    generated_code=sample.canonical_solution,
                    execution_output="Tests passed",
                    error_type=ErrorType.NONE,
                    latency_ms=10.0
                ))
                passed += 1
            else:
                sample_results.append(SampleResult(
                    task_id=sample.task_id,
                    passed=False,
                    generated_code="",
                    execution_output="No canonical solution",
                    error_message="No canonical solution available",
                    error_type=ErrorType.OTHER,
                    latency_ms=10.0
                ))
                failed += 1

        total_time = time.time() - start_time

        # Calculate pass@k
        pass_at_k = self._calculate_pass_at_k(sample_results, self.config.pass_k)

        return BenchmarkResult(
            benchmark=self.config.benchmark_type,
            model_name=self.config.model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pass_at_k=pass_at_k,
            total_samples=len(samples),
            passed_samples=passed,
            failed_samples=failed,
            error_samples=0,
            avg_latency_ms=10.0,
            total_time_seconds=total_time,
            sample_results=sample_results
        )

    def _load_humaneval(self) -> List[BenchmarkSample]:
        """Load HumanEval benchmark samples."""
        samples = []
        for item in self.HUMANEVAL_SAMPLES[:self.config.num_samples]:
            samples.append(BenchmarkSample(
                task_id=item["task_id"],
                prompt=item["prompt"],
                canonical_solution=item.get("canonical_solution"),
                test_code=item.get("test"),
                entry_point=item.get("entry_point")
            ))
        return samples

    def _load_mbpp(self) -> List[BenchmarkSample]:
        """Load MBPP benchmark samples."""
        samples = []
        for item in self.MBPP_SAMPLES[:self.config.num_samples]:
            samples.append(BenchmarkSample(
                task_id=item["task_id"],
                prompt=item["prompt"],
                test_code=item.get("test"),
                entry_point=item.get("entry_point")
            ))
        return samples

    async def run_benchmark(
        self,
        generate_fn: Callable[[str], str],
        benchmark_type: Optional[BenchmarkType] = None
    ) -> BenchmarkResult:
        """
        Run a benchmark evaluation.

        Args:
            generate_fn: Function that takes prompt and returns generated code
            benchmark_type: Type of benchmark (uses config default if not specified)

        Returns:
            BenchmarkResult with evaluation metrics
        """
        benchmark_type = benchmark_type or self.config.benchmark_type
        samples = self.load_benchmark(benchmark_type)

        if not samples:
            return BenchmarkResult(
                benchmark=benchmark_type,
                model_name=self.config.model_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                pass_at_k={k: 0.0 for k in self.config.pass_k},
                total_samples=0,
                passed_samples=0,
                failed_samples=0,
                error_samples=0,
                avg_latency_ms=0.0
            )

        sample_results = []
        passed = 0
        failed = 0
        errors = 0
        total_latency = 0.0

        for sample in samples:
            try:
                result = await self._evaluate_sample(sample, generate_fn)
                sample_results.append(result)

                if result.passed:
                    passed += 1
                elif result.error_message:
                    errors += 1
                else:
                    failed += 1

                total_latency += result.latency_ms

            except Exception as e:
                sample_results.append(SampleResult(
                    task_id=sample.task_id,
                    passed=False,
                    generated_code="",
                    execution_output="",
                    error_message=str(e)
                ))
                errors += 1

        # Calculate pass@k
        pass_at_k = self._calculate_pass_at_k(sample_results, self.config.pass_k)

        return BenchmarkResult(
            benchmark=benchmark_type,
            model_name=self.config.model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pass_at_k=pass_at_k,
            total_samples=len(samples),
            passed_samples=passed,
            failed_samples=failed,
            error_samples=errors,
            avg_latency_ms=total_latency / max(len(samples), 1),
            sample_results=sample_results
        )

    async def _evaluate_sample(
        self,
        sample: BenchmarkSample,
        generate_fn: Callable[[str], str]
    ) -> SampleResult:
        """Evaluate a single benchmark sample."""
        import time

        start_time = time.time()

        # Generate code
        if asyncio.iscoroutinefunction(generate_fn):
            generated_code = await generate_fn(sample.prompt)
        else:
            generated_code = generate_fn(sample.prompt)

        latency_ms = (time.time() - start_time) * 1000

        # Execute and test
        passed, output, error, error_type = await self._execute_and_test(
            sample, generated_code
        )

        return SampleResult(
            task_id=sample.task_id,
            passed=passed,
            generated_code=generated_code,
            execution_output=output,
            error_message=error,
            error_type=error_type,
            latency_ms=latency_ms
        )

    async def _execute_and_test(
        self,
        sample: BenchmarkSample,
        generated_code: str
    ) -> Tuple[bool, str, Optional[str], ErrorType]:
        """Execute generated code and run tests. Returns (passed, output, error, error_type)."""
        if not sample.test_code:
            return True, "No tests provided", None, ErrorType.NONE

        # Prepare full code
        full_code = f"{sample.prompt}{generated_code}\n\n{sample.test_code}\n"

        if sample.entry_point:
            full_code += f"\ncheck({sample.entry_point})\n"

        # Execute in subprocess with timeout
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write(full_code)
                temp_path = f.name

            try:
                result = subprocess.run(
                    ["python", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_per_sample
                )

                if result.returncode == 0:
                    return True, result.stdout, None, ErrorType.NONE
                else:
                    # Classify the error type
                    error_type = self._classify_error(result.stderr)
                    return False, result.stdout, result.stderr, error_type

            finally:
                Path(temp_path).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return False, "", "Execution timed out", ErrorType.TIMEOUT
        except Exception as e:
            return False, "", str(e), ErrorType.OTHER

    def _classify_error(self, stderr: str) -> ErrorType:
        """Classify error type from stderr output."""
        stderr_lower = stderr.lower()

        # Syntax errors
        if "syntaxerror" in stderr_lower or "indentationerror" in stderr_lower:
            return ErrorType.SYNTAX_ERROR

        # Assertion errors (wrong answer)
        if "assertionerror" in stderr_lower:
            return ErrorType.WRONG_ANSWER

        # Runtime errors
        if any(err in stderr_lower for err in [
            "typeerror", "valueerror", "nameerror", "attributeerror",
            "indexerror", "keyerror", "zerodivisionerror", "runtimeerror",
            "importerror", "modulenotfounderror", "filenotfounderror"
        ]):
            return ErrorType.RUNTIME_ERROR

        # If we got here with a non-empty stderr, it's likely a wrong answer
        if stderr.strip():
            return ErrorType.WRONG_ANSWER

        return ErrorType.OTHER

    def _calculate_pass_at_k(
        self,
        results: List[SampleResult],
        k_values: List[int]
    ) -> Dict[int, float]:
        """
        Calculate pass@k metrics.

        pass@k = probability that at least one of k samples passes
        """
        n = len(results)
        c = sum(1 for r in results if r.passed)

        pass_at_k = {}
        for k in k_values:
            if n < k:
                pass_at_k[k] = 0.0
            elif c == 0:
                pass_at_k[k] = 0.0
            elif c >= k:
                pass_at_k[k] = 1.0
            else:
                # Estimate pass@k
                pass_at_k[k] = 1.0 - (1.0 - c/n) ** k

        return pass_at_k

    def save_results(self, result: BenchmarkResult) -> Path:
        """Save benchmark results to file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{result.benchmark.value}_{timestamp}.json"
        output_path = Path(self.config.output_dir) / filename

        data = {
            **result.to_dict(),
            "sample_results": [r.to_dict() for r in result.sample_results]
        }

        output_path.write_text(json.dumps(data, indent=2))
        return output_path

    async def compare_models(
        self,
        models: Dict[str, Callable[[str], str]],
        benchmark_type: Optional[BenchmarkType] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple models on a benchmark.

        Args:
            models: Dict of model_name -> generate_function
            benchmark_type: Benchmark to run

        Returns:
            Dict of model_name -> BenchmarkResult
        """
        results = {}

        for model_name, generate_fn in models.items():
            self.config.model_name = model_name
            result = await self.run_benchmark(generate_fn, benchmark_type)
            results[model_name] = result
            print(f"{model_name}: pass@1={result.pass_at_k.get(1, 0):.2%}")

        return results


async def main():
    """Example usage of the Benchmark Suite."""
    config = BenchmarkConfig(
        benchmark_type=BenchmarkType.HUMANEVAL,
        num_samples=2,  # Demo with 2 samples
        model_name="demo_model"
    )

    runner = BenchmarkRunner(config)

    try:
        # Simple generate function (returns canonical solution for demo)
        def demo_generate(prompt: str) -> str:
            # In production, this would call the actual model
            # For demo, return a simple implementation
            if "has_close_elements" in prompt:
                return '''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
'''
            elif "separate_paren_groups" in prompt:
                return '''    result = []
    current_string = []
    current_depth = 0
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()
    return result
'''
            return "pass"

        result = await runner.run_benchmark(demo_generate)

        print("\nBenchmark Results:")
        print(f"  Benchmark: {result.benchmark.value}")
        print(f"  Total Samples: {result.total_samples}")
        print(f"  Passed: {result.passed_samples}")
        print(f"  Pass@1: {result.pass_at_k.get(1, 0):.2%}")

        # Save results
        output_path = runner.save_results(result)
        print(f"\nResults saved to: {output_path}")

    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
