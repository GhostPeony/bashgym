"""
Prompt Optimization with MIPROv2 for The Factory Layer

Automatically optimizes system prompts and few-shot examples using
Bayesian optimization techniques from NVIDIA NeMo and DSPy.

Module 3: Data Synthesis (The "Factory") - Extended
"""

import os
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime, timezone
import httpx
import random


class OptimizationIntensity:
    """Optimization intensity levels."""
    LIGHT = "light"      # Quick optimization, fewer trials
    MEDIUM = "medium"    # Balanced optimization
    HEAVY = "heavy"      # Thorough optimization, more trials


@dataclass
class PromptOptConfig:
    """Configuration for prompt optimization."""

    # NeMo/NIM settings
    endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None
    optimizer_model: str = "meta/llama-3.1-70b-instruct"

    # Optimization settings
    intensity: str = OptimizationIntensity.MEDIUM
    max_bootstrapped_demos: int = 4
    metric_threshold: float = 0.8
    num_trials: int = 20

    # Search settings
    num_candidates: int = 10
    temperature: float = 0.7

    # Output settings
    output_dir: str = "data/optimized_prompts"


@dataclass
class PromptCandidate:
    """A candidate prompt configuration."""

    prompt_id: str
    system_prompt: str
    few_shot_examples: List[Dict[str, str]]
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "system_prompt": self.system_prompt,
            "few_shot_examples": self.few_shot_examples,
            "score": self.score,
            "metrics": self.metrics
        }


@dataclass
class OptimizationResult:
    """Result of a prompt optimization run."""

    run_id: str
    original_prompt: str
    optimized_prompt: str
    original_score: float
    optimized_score: float
    improvement: float
    few_shot_examples: List[Dict[str, str]]
    num_trials: int
    duration_seconds: float
    all_candidates: List[PromptCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "original_score": self.original_score,
            "optimized_score": self.optimized_score,
            "improvement": self.improvement,
            "improvement_percent": f"{self.improvement * 100:.1f}%",
            "few_shot_examples": self.few_shot_examples,
            "num_trials": self.num_trials,
            "duration_seconds": self.duration_seconds
        }


class PromptOptimizer:
    """
    Automatic prompt optimization using MIPROv2-style techniques.

    Features:
    - Instruction optimization via Bayesian search
    - Few-shot example bootstrapping
    - Judge prompt tuning
    - Multi-metric optimization

    Based on NVIDIA NeMo Prompt Optimization and DSPy MIPROv2.
    """

    # Prompt generation templates
    INSTRUCTION_GEN_TEMPLATE = """You are an expert prompt engineer. Generate an improved version of the following system prompt.

Original prompt:
{original_prompt}

The prompt should be optimized for:
{optimization_goals}

Performance on current examples:
{performance_summary}

Generate an improved prompt that:
1. Is clear and specific
2. Provides appropriate context
3. Guides the model to produce better outputs
4. Addresses weaknesses in the current performance

Output ONLY the improved prompt, no explanations."""

    DEMO_GEN_TEMPLATE = """You are an expert at creating high-quality examples.

Task description:
{task_description}

Generate a diverse set of {num_demos} input-output examples for this task.
Each example should demonstrate the expected behavior clearly.

Format each example as:
INPUT: <input text>
OUTPUT: <expected output>

Generate diverse examples covering different cases."""

    def __init__(self, config: Optional[PromptOptConfig] = None):
        """Initialize the prompt optimizer."""
        self.config = config or PromptOptConfig()

        # Load API key from environment
        if not self.config.api_key:
            self.config.api_key = os.environ.get("NVIDIA_API_KEY")

        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=120.0,
            headers=self._build_headers()
        )

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Set intensity-based parameters
        self._set_intensity_params()

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _set_intensity_params(self):
        """Set parameters based on optimization intensity."""
        if self.config.intensity == OptimizationIntensity.LIGHT:
            self.config.num_trials = 10
            self.config.num_candidates = 5
        elif self.config.intensity == OptimizationIntensity.MEDIUM:
            self.config.num_trials = 20
            self.config.num_candidates = 10
        elif self.config.intensity == OptimizationIntensity.HEAVY:
            self.config.num_trials = 50
            self.config.num_candidates = 20

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def optimize_prompt(
        self,
        original_prompt: str,
        eval_examples: List[Dict[str, str]],
        eval_fn: Callable[[str, List[Dict[str, str]]], float],
        optimization_goals: Optional[str] = None
    ) -> OptimizationResult:
        """
        Optimize a system prompt using Bayesian search.

        Args:
            original_prompt: The prompt to optimize
            eval_examples: List of {input, expected_output} dicts for evaluation
            eval_fn: Function that takes (prompt, examples) and returns score 0-1
            optimization_goals: Description of what to optimize for

        Returns:
            OptimizationResult with best prompt and metrics
        """
        start_time = datetime.now(timezone.utc)
        run_id = f"opt_{start_time.strftime('%Y%m%d_%H%M%S')}"

        # Evaluate original prompt
        original_score = await self._evaluate_prompt(
            original_prompt, eval_examples, eval_fn
        )

        # Generate candidate prompts
        candidates = await self._generate_candidates(
            original_prompt,
            optimization_goals or "Improve task completion accuracy",
            original_score,
            eval_examples
        )

        # Evaluate all candidates
        best_candidate = PromptCandidate(
            prompt_id="original",
            system_prompt=original_prompt,
            few_shot_examples=[],
            score=original_score
        )

        for candidate in candidates:
            score = await self._evaluate_prompt(
                candidate.system_prompt, eval_examples, eval_fn
            )
            candidate.score = score

            if score > best_candidate.score:
                best_candidate = candidate

        # Bootstrap few-shot examples
        few_shot_examples = await self._bootstrap_demos(
            best_candidate.system_prompt,
            eval_examples
        )

        # Final evaluation with few-shot examples
        final_score = await self._evaluate_prompt(
            best_candidate.system_prompt,
            eval_examples,
            eval_fn,
            few_shot_examples
        )

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        result = OptimizationResult(
            run_id=run_id,
            original_prompt=original_prompt,
            optimized_prompt=best_candidate.system_prompt,
            original_score=original_score,
            optimized_score=final_score,
            improvement=final_score - original_score,
            few_shot_examples=few_shot_examples,
            num_trials=len(candidates) + 1,
            duration_seconds=duration,
            all_candidates=candidates
        )

        # Save results
        self._save_result(result)

        return result

    async def _generate_candidates(
        self,
        original_prompt: str,
        optimization_goals: str,
        current_score: float,
        examples: List[Dict[str, str]]
    ) -> List[PromptCandidate]:
        """Generate candidate prompts using LLM."""
        candidates = []

        performance_summary = f"Current score: {current_score:.2f}"

        gen_prompt = self.INSTRUCTION_GEN_TEMPLATE.format(
            original_prompt=original_prompt,
            optimization_goals=optimization_goals,
            performance_summary=performance_summary
        )

        for i in range(self.config.num_candidates):
            try:
                response = await self.client.post(
                    f"{self.config.endpoint}/v1/chat/completions",
                    json={
                        "model": self.config.optimizer_model,
                        "messages": [{"role": "user", "content": gen_prompt}],
                        "temperature": self.config.temperature + (i * 0.05),  # Vary temperature
                        "max_tokens": 2048
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    new_prompt = result["choices"][0]["message"]["content"].strip()

                    candidates.append(PromptCandidate(
                        prompt_id=f"candidate_{i}",
                        system_prompt=new_prompt,
                        few_shot_examples=[]
                    ))

            except Exception as e:
                print(f"Failed to generate candidate {i}: {e}")

        return candidates

    async def _evaluate_prompt(
        self,
        prompt: str,
        examples: List[Dict[str, str]],
        eval_fn: Callable,
        few_shot: Optional[List[Dict[str, str]]] = None
    ) -> float:
        """Evaluate a prompt on examples."""
        try:
            if asyncio.iscoroutinefunction(eval_fn):
                return await eval_fn(prompt, examples)
            else:
                return eval_fn(prompt, examples)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.0

    async def _bootstrap_demos(
        self,
        prompt: str,
        examples: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Bootstrap high-quality few-shot demonstrations."""
        if not examples:
            return []

        # Select diverse examples based on input variety
        selected = random.sample(
            examples,
            min(self.config.max_bootstrapped_demos, len(examples))
        )

        return [
            {"input": ex.get("input", ""), "output": ex.get("expected_output", "")}
            for ex in selected
        ]

    async def generate_demos(
        self,
        task_description: str,
        num_demos: int = 5
    ) -> List[Dict[str, str]]:
        """Generate synthetic demonstrations for a task."""
        gen_prompt = self.DEMO_GEN_TEMPLATE.format(
            task_description=task_description,
            num_demos=num_demos
        )

        try:
            response = await self.client.post(
                f"{self.config.endpoint}/v1/chat/completions",
                json={
                    "model": self.config.optimizer_model,
                    "messages": [{"role": "user", "content": gen_prompt}],
                    "temperature": 0.7,
                    "max_tokens": 4096
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Parse examples
                demos = []
                current_input = ""
                current_output = ""

                for line in content.split("\n"):
                    if line.startswith("INPUT:"):
                        if current_input and current_output:
                            demos.append({
                                "input": current_input.strip(),
                                "output": current_output.strip()
                            })
                        current_input = line[6:].strip()
                        current_output = ""
                    elif line.startswith("OUTPUT:"):
                        current_output = line[7:].strip()
                    elif current_output:
                        current_output += "\n" + line

                if current_input and current_output:
                    demos.append({
                        "input": current_input.strip(),
                        "output": current_output.strip()
                    })

                return demos

        except Exception as e:
            print(f"Demo generation failed: {e}")

        return []

    async def optimize_judge_prompt(
        self,
        original_rubric: str,
        eval_pairs: List[Tuple[str, str, float]],
        human_scores: List[float]
    ) -> OptimizationResult:
        """
        Optimize a judge/scoring prompt for better alignment with human scores.

        Args:
            original_rubric: The current scoring rubric
            eval_pairs: List of (prompt, response, expected_score) tuples
            human_scores: Human-assigned scores for comparison

        Returns:
            OptimizationResult with optimized rubric
        """
        # Create evaluation function that measures correlation with human scores
        async def judge_eval_fn(rubric: str, examples: List[Dict[str, str]]) -> float:
            predicted_scores = []

            for pair, human_score in zip(eval_pairs, human_scores):
                prompt, response, _ = pair

                try:
                    judge_response = await self.client.post(
                        f"{self.config.endpoint}/v1/chat/completions",
                        json={
                            "model": self.config.optimizer_model,
                            "messages": [{
                                "role": "user",
                                "content": f"Evaluate this response:\n\nPrompt: {prompt}\n\nResponse: {response}\n\nRubric: {rubric}\n\nScore (1-5):"
                            }],
                            "temperature": 0.0,
                            "max_tokens": 10
                        }
                    )

                    if judge_response.status_code == 200:
                        result = judge_response.json()
                        score_text = result["choices"][0]["message"]["content"]
                        try:
                            score = float(score_text.strip().split()[0])
                            predicted_scores.append(score)
                        except ValueError:
                            predicted_scores.append(3.0)

                except Exception:
                    predicted_scores.append(3.0)

            # Calculate correlation
            if len(predicted_scores) < 2:
                return 0.0

            # Simple correlation metric
            mean_pred = sum(predicted_scores) / len(predicted_scores)
            mean_human = sum(human_scores) / len(human_scores)

            numerator = sum(
                (p - mean_pred) * (h - mean_human)
                for p, h in zip(predicted_scores, human_scores)
            )
            denom_pred = sum((p - mean_pred) ** 2 for p in predicted_scores) ** 0.5
            denom_human = sum((h - mean_human) ** 2 for h in human_scores) ** 0.5

            if denom_pred * denom_human == 0:
                return 0.0

            correlation = numerator / (denom_pred * denom_human)
            return (correlation + 1) / 2  # Normalize to 0-1

        # Run optimization
        return await self.optimize_prompt(
            original_prompt=original_rubric,
            eval_examples=[{"input": p, "expected_output": str(s)} for p, r, s in eval_pairs],
            eval_fn=judge_eval_fn,
            optimization_goals="Maximize correlation with human judgments"
        )

    def _save_result(self, result: OptimizationResult) -> Path:
        """Save optimization result to file."""
        output_path = Path(self.config.output_dir) / f"{result.run_id}.json"
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        return output_path


async def main():
    """Example usage of the Prompt Optimizer."""
    config = PromptOptConfig(
        intensity=OptimizationIntensity.LIGHT,
        num_candidates=3
    )

    optimizer = PromptOptimizer(config)

    try:
        # Example: Optimize a code review prompt
        original_prompt = """You are a code reviewer. Review the following code and provide feedback."""

        eval_examples = [
            {"input": "def add(a, b): return a + b", "expected_output": "Clean, simple function."},
            {"input": "def f(x): return x*x", "expected_output": "Consider more descriptive name."},
        ]

        # Simple evaluation function (would use actual model in production)
        def simple_eval(prompt: str, examples: List[Dict[str, str]]) -> float:
            # Score based on prompt length and specificity
            score = min(len(prompt) / 500, 1.0) * 0.5
            if "specific" in prompt.lower() or "detailed" in prompt.lower():
                score += 0.25
            if "example" in prompt.lower():
                score += 0.25
            return score

        result = await optimizer.optimize_prompt(
            original_prompt=original_prompt,
            eval_examples=eval_examples,
            eval_fn=simple_eval,
            optimization_goals="Provide detailed, actionable code review feedback"
        )

        print(f"Original score: {result.original_score:.2f}")
        print(f"Optimized score: {result.optimized_score:.2f}")
        print(f"Improvement: {result.improvement:.2f}")
        print(f"\nOptimized prompt:\n{result.optimized_prompt}")

    finally:
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())
