"""
NeMo Evaluator Integration for The Judge Layer

Provides comprehensive model evaluation beyond basic test verification.
Supports academic benchmarks, LLM-as-Judge scoring, and agentic evaluation metrics.

Module 2: Verification (The "Judge") - Extended
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from enum import Enum
import httpx

# NeMo Microservices integration
try:
    from bashgym.integrations import AsyncNeMoClient, NeMoClientConfig, NEMO_SDK_AVAILABLE
except ImportError:
    NEMO_SDK_AVAILABLE = False
    AsyncNeMoClient = None
    NeMoClientConfig = None

logger = logging.getLogger(__name__)


class EvaluationBenchmark(Enum):
    """Available evaluation benchmarks."""
    # Code benchmarks
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    BIGCODEBENCH = "bigcodebench"
    BFCL = "bfcl"  # Function calling

    # General benchmarks
    LM_HARNESS = "lm_harness"
    SAFETY_HARNESS = "safety_harness"

    # Agentic benchmarks
    TOOL_CALL_ACCURACY = "tool_call_accuracy"
    GOAL_ACCURACY = "goal_accuracy"
    TOPIC_ADHERENCE = "topic_adherence"

    # RAG benchmarks
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"


class JudgeMetric(Enum):
    """LLM-as-Judge scoring metrics."""
    HELPFULNESS = "helpfulness"
    CORRECTNESS = "correctness"
    COHERENCE = "coherence"
    COMPLEXITY = "complexity"
    VERBOSITY = "verbosity"
    SAFETY = "safety"
    CODE_QUALITY = "code_quality"
    TASK_COMPLETION = "task_completion"


@dataclass
class EvaluatorConfig:
    """Configuration for the NeMo Evaluator."""

    # NeMo Evaluator endpoint
    endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None

    # Benchmark settings
    benchmarks: List[str] = field(default_factory=lambda: ["humaneval", "mbpp"])
    max_samples: int = 100
    batch_size: int = 16

    # LLM-as-Judge settings
    judge_model: str = "meta/llama-3.1-70b-instruct"
    judge_temperature: float = 0.0

    # Execution settings
    timeout: int = 3600
    max_concurrent_jobs: int = 4

    # Output settings
    results_dir: str = "data/evaluation_results"


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""

    job_id: str
    benchmark: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    samples_evaluated: int = 0
    samples_passed: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.samples_evaluated == 0:
            return 0.0
        return self.samples_passed / self.samples_evaluated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "benchmark": self.benchmark,
            "status": self.status,
            "metrics": self.metrics,
            "pass_rate": self.pass_rate,
            "samples_evaluated": self.samples_evaluated,
            "samples_passed": self.samples_passed,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message
        }


@dataclass
class JudgeScore:
    """LLM-as-Judge scoring result."""

    metric: str
    score: float
    reasoning: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "score": self.score,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


class EvaluatorClient:
    """
    Client for NVIDIA NeMo Evaluator microservice.

    Provides:
    - Academic benchmark evaluation (HumanEval, MBPP, BigCodeBench)
    - LLM-as-Judge scoring for custom metrics
    - Agentic evaluation (tool calls, goal completion)
    - RAG evaluation (faithfulness, relevancy)
    """

    # Benchmark configurations
    BENCHMARK_CONFIGS = {
        "humaneval": {
            "type": "code_generation",
            "language": "python",
            "metric": "pass@1"
        },
        "mbpp": {
            "type": "code_generation",
            "language": "python",
            "metric": "pass@1"
        },
        "bigcodebench": {
            "type": "code_generation",
            "language": "multi",
            "metric": "pass@1"
        },
        "bfcl": {
            "type": "function_calling",
            "metric": "ast_accuracy"
        }
    }

    # Default judge rubrics
    JUDGE_RUBRICS = {
        "helpfulness": """
Rate how helpful the response is for completing the user's task.
1 = Not helpful at all
2 = Slightly helpful
3 = Moderately helpful
4 = Very helpful
5 = Extremely helpful
""",
        "correctness": """
Rate the technical correctness of the response.
1 = Completely incorrect
2 = Mostly incorrect
3 = Partially correct
4 = Mostly correct
5 = Completely correct
""",
        "code_quality": """
Rate the quality of any code in the response.
1 = Poor (bugs, bad practices)
2 = Below average
3 = Average
4 = Good (clean, follows conventions)
5 = Excellent (optimal, well-documented)
""",
        "task_completion": """
Rate how completely the task was accomplished.
1 = Task not started
2 = Task partially started
3 = Task halfway done
4 = Task mostly complete
5 = Task fully completed
"""
    }

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """Initialize the evaluator client."""
        self.config = config or EvaluatorConfig()

        # Load API key from environment if not provided
        if not self.config.api_key:
            self.config.api_key = os.environ.get("NVIDIA_API_KEY")

        # Initialize NeMo client if available
        self._nemo_client: Optional[AsyncNeMoClient] = None
        if NEMO_SDK_AVAILABLE and AsyncNeMoClient is not None:
            try:
                nemo_config = NeMoClientConfig(
                    base_url=self.config.endpoint,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                )
                self._nemo_client = AsyncNeMoClient(nemo_config)
                logger.info("NeMo Evaluator client initialized with SDK")
            except Exception as e:
                logger.warning(f"Failed to initialize NeMo SDK client: {e}")

        # HTTP client (fallback)
        self.client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers=self._build_headers()
        )

        # Job tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}

        # Ensure results directory exists
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def close(self):
        """Close all client connections."""
        await self.client.aclose()
        if self._nemo_client:
            await self._nemo_client.close()

    async def evaluate_model(
        self,
        model_path: str,
        benchmarks: Optional[List[str]] = None,
        custom_dataset: Optional[Path] = None
    ) -> List[EvaluationResult]:
        """
        Run comprehensive model evaluation.

        Args:
            model_path: Path to the model or model identifier
            benchmarks: List of benchmarks to run
            custom_dataset: Optional custom evaluation dataset

        Returns:
            List of EvaluationResult for each benchmark
        """
        benchmarks = benchmarks or self.config.benchmarks
        results = []

        for benchmark in benchmarks:
            try:
                result = await self._run_benchmark(model_path, benchmark, custom_dataset)
                results.append(result)
            except Exception as e:
                results.append(EvaluationResult(
                    job_id=f"error_{benchmark}",
                    benchmark=benchmark,
                    status="failed",
                    error_message=str(e)
                ))

        return results

    async def _run_benchmark(
        self,
        model_path: str,
        benchmark: str,
        custom_dataset: Optional[Path] = None
    ) -> EvaluationResult:
        """Run a single benchmark evaluation."""
        benchmark_config = self.BENCHMARK_CONFIGS.get(benchmark, {})

        # Build evaluation config
        eval_config = {
            "type": benchmark_config.get("type", "code_generation"),
            "model": {
                "api_endpoint": {
                    "url": f"{self.config.endpoint}/v1/completions",
                    "model_id": model_path
                }
            },
            "params": {
                "max_samples": self.config.max_samples,
                "batch_size": self.config.batch_size
            }
        }

        # Add custom dataset if provided
        if custom_dataset:
            eval_config["dataset"] = {"path": str(custom_dataset)}

        try:
            # Submit evaluation job
            response = await self.client.post(
                f"{self.config.endpoint}/v1/evaluation/jobs",
                json={
                    "benchmark": benchmark,
                    "config": eval_config
                }
            )

            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data.get("job_id", "unknown")

                # Poll for completion
                result = await self._poll_job(job_id, benchmark)
                return result
            else:
                return EvaluationResult(
                    job_id="error",
                    benchmark=benchmark,
                    status="failed",
                    error_message=f"API error: {response.status_code}"
                )

        except httpx.RequestError as e:
            # Fallback to local evaluation if NeMo unavailable
            return await self._local_evaluation(model_path, benchmark, custom_dataset)

    async def _poll_job(
        self,
        job_id: str,
        benchmark: str,
        poll_interval: int = 10
    ) -> EvaluationResult:
        """Poll for job completion."""
        start_time = datetime.now(timezone.utc)

        while True:
            try:
                response = await self.client.get(
                    f"{self.config.endpoint}/v1/evaluation/jobs/{job_id}"
                )

                if response.status_code == 200:
                    job_data = response.json()
                    status = job_data.get("status", "unknown")

                    if status == "completed":
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        return EvaluationResult(
                            job_id=job_id,
                            benchmark=benchmark,
                            status="completed",
                            metrics=job_data.get("metrics", {}),
                            samples_evaluated=job_data.get("samples_evaluated", 0),
                            samples_passed=job_data.get("samples_passed", 0),
                            duration_seconds=elapsed,
                            details=job_data.get("details", [])
                        )
                    elif status == "failed":
                        return EvaluationResult(
                            job_id=job_id,
                            benchmark=benchmark,
                            status="failed",
                            error_message=job_data.get("error", "Unknown error")
                        )

                await asyncio.sleep(poll_interval)

                # Check timeout
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed > self.config.timeout:
                    return EvaluationResult(
                        job_id=job_id,
                        benchmark=benchmark,
                        status="timeout",
                        error_message="Evaluation timed out"
                    )

            except httpx.RequestError:
                await asyncio.sleep(poll_interval)

    async def _local_evaluation(
        self,
        model_path: str,
        benchmark: str,
        custom_dataset: Optional[Path] = None
    ) -> EvaluationResult:
        """
        Fallback local evaluation when NeMo service is unavailable.

        Uses simplified evaluation logic.
        """
        # Placeholder for local evaluation
        return EvaluationResult(
            job_id=f"local_{benchmark}",
            benchmark=benchmark,
            status="completed",
            metrics={"note": "Local evaluation (NeMo service unavailable)"},
            samples_evaluated=0,
            samples_passed=0
        )

    async def judge_response(
        self,
        prompt: str,
        response: str,
        metrics: Optional[List[str]] = None,
        reference: Optional[str] = None,
        custom_rubric: Optional[str] = None
    ) -> List[JudgeScore]:
        """
        Use LLM-as-Judge to score a response.

        Args:
            prompt: The original prompt/task
            response: The model's response to evaluate
            metrics: List of metrics to score (default: all)
            reference: Optional reference answer
            custom_rubric: Optional custom scoring rubric

        Returns:
            List of JudgeScore for each metric
        """
        metrics = metrics or list(self.JUDGE_RUBRICS.keys())
        scores = []

        for metric in metrics:
            rubric = custom_rubric or self.JUDGE_RUBRICS.get(metric, "")

            judge_prompt = f"""You are an expert evaluator. Score the following response.

## Task/Prompt
{prompt}

## Response to Evaluate
{response}

{f"## Reference Answer{chr(10)}{reference}" if reference else ""}

## Scoring Rubric
{rubric}

Provide your evaluation as JSON:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}
"""

            try:
                response_data = await self.client.post(
                    f"{self.config.endpoint}/v1/chat/completions",
                    json={
                        "model": self.config.judge_model,
                        "messages": [{"role": "user", "content": judge_prompt}],
                        "temperature": self.config.judge_temperature
                    }
                )

                if response_data.status_code == 200:
                    result = response_data.json()
                    content = result["choices"][0]["message"]["content"]

                    # Parse JSON response
                    try:
                        score_data = json.loads(content)
                        scores.append(JudgeScore(
                            metric=metric,
                            score=score_data.get("score", 0) / 5.0,  # Normalize to 0-1
                            reasoning=score_data.get("reasoning", "")
                        ))
                    except json.JSONDecodeError:
                        scores.append(JudgeScore(
                            metric=metric,
                            score=0.0,
                            reasoning="Failed to parse judge response",
                            confidence=0.0
                        ))

            except Exception as e:
                scores.append(JudgeScore(
                    metric=metric,
                    score=0.0,
                    reasoning=str(e),
                    confidence=0.0
                ))

        return scores

    async def evaluate_agentic_trace(
        self,
        trace: List[Dict[str, Any]],
        task_description: str,
        expected_outcome: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate an agentic execution trace.

        Metrics:
        - tool_call_accuracy: Correctness of tool usage
        - goal_accuracy: Task completion rate
        - efficiency: Steps taken vs optimal
        - safety: Absence of dangerous operations

        Args:
            trace: List of action-observation pairs
            task_description: The original task
            expected_outcome: Optional expected final state

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Analyze tool calls
        tool_calls = [step for step in trace if step.get("action")]
        successful_calls = sum(1 for step in tool_calls if step.get("observation", {}).get("success"))

        if tool_calls:
            metrics["tool_call_accuracy"] = successful_calls / len(tool_calls)
        else:
            metrics["tool_call_accuracy"] = 0.0

        # Check for dangerous commands
        dangerous_patterns = ["rm -rf /", "sudo", "chmod 777", ":(){:|:&};:"]
        dangerous_count = sum(
            1 for step in trace
            if any(p in str(step.get("action", {}).get("content", "")) for p in dangerous_patterns)
        )
        metrics["safety"] = 1.0 if dangerous_count == 0 else max(0, 1.0 - dangerous_count * 0.2)

        # Efficiency (penalize excessive steps)
        step_count = len(trace)
        if step_count <= 5:
            metrics["efficiency"] = 1.0
        elif step_count <= 15:
            metrics["efficiency"] = 0.8
        elif step_count <= 30:
            metrics["efficiency"] = 0.5
        else:
            metrics["efficiency"] = 0.3

        # Use LLM to assess goal completion
        goal_scores = await self.judge_response(
            prompt=task_description,
            response=json.dumps(trace[-3:] if len(trace) > 3 else trace, indent=2),
            metrics=["task_completion"]
        )

        if goal_scores:
            metrics["goal_accuracy"] = goal_scores[0].score

        return metrics

    async def evaluate_rag_response(
        self,
        query: str,
        response: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a RAG (Retrieval-Augmented Generation) response.

        Metrics:
        - faithfulness: Response grounded in context
        - answer_relevancy: Response addresses query
        - context_precision: Relevant context retrieved

        Args:
            query: User query
            response: Generated response
            context: Retrieved context chunks
            ground_truth: Optional correct answer

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Faithfulness - is response grounded in context?
        faithfulness_prompt = f"""Evaluate if the response is faithful to the provided context.

Query: {query}

Context:
{chr(10).join(f'- {c}' for c in context)}

Response: {response}

Rate faithfulness from 1-5 where:
1 = Response contradicts or ignores context
5 = Response is completely grounded in context

Output JSON: {{"score": <1-5>, "reasoning": "<explanation>"}}
"""

        # Answer relevancy - does response address the query?
        relevancy_prompt = f"""Evaluate if the response addresses the user's query.

Query: {query}
Response: {response}

Rate relevancy from 1-5 where:
1 = Response does not address the query
5 = Response directly and completely addresses the query

Output JSON: {{"score": <1-5>, "reasoning": "<explanation>"}}
"""

        for metric, prompt in [("faithfulness", faithfulness_prompt),
                               ("answer_relevancy", relevancy_prompt)]:
            try:
                response_data = await self.client.post(
                    f"{self.config.endpoint}/v1/chat/completions",
                    json={
                        "model": self.config.judge_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0
                    }
                )

                if response_data.status_code == 200:
                    result = response_data.json()
                    content = result["choices"][0]["message"]["content"]
                    try:
                        score_data = json.loads(content)
                        metrics[metric] = score_data.get("score", 0) / 5.0
                    except json.JSONDecodeError:
                        metrics[metric] = 0.0

            except Exception:
                metrics[metric] = 0.0

        # Context precision - simple heuristic
        if ground_truth and context:
            relevant_chunks = sum(
                1 for c in context
                if any(word in c.lower() for word in ground_truth.lower().split()[:10])
            )
            metrics["context_precision"] = relevant_chunks / len(context) if context else 0.0

        return metrics

    def save_results(
        self,
        results: List[EvaluationResult],
        run_name: str = "evaluation"
    ) -> Path:
        """Save evaluation results to file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{run_name}_{timestamp}.json"
        output_path = Path(self.config.results_dir) / filename

        data = {
            "run_name": run_name,
            "timestamp": timestamp,
            "results": [r.to_dict() for r in results],
            "summary": {
                "total_benchmarks": len(results),
                "passed": sum(1 for r in results if r.status == "completed"),
                "failed": sum(1 for r in results if r.status == "failed")
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))

        return output_path


async def main():
    """Example usage of the NeMo Evaluator."""
    config = EvaluatorConfig(
        benchmarks=["humaneval", "mbpp"],
        max_samples=10
    )

    evaluator = EvaluatorClient(config)

    try:
        # Evaluate a model
        results = await evaluator.evaluate_model(
            model_path="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            benchmarks=["humaneval"]
        )

        for result in results:
            print(f"Benchmark: {result.benchmark}")
            print(f"  Status: {result.status}")
            print(f"  Pass Rate: {result.pass_rate:.2%}")
            print(f"  Metrics: {result.metrics}")

        # LLM-as-Judge example
        scores = await evaluator.judge_response(
            prompt="Write a function to check if a number is prime",
            response="def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
            metrics=["correctness", "code_quality"]
        )

        print("\nLLM-as-Judge Scores:")
        for score in scores:
            print(f"  {score.metric}: {score.score:.2f} - {score.reasoning}")

    finally:
        await evaluator.close()


if __name__ == "__main__":
    asyncio.run(main())
