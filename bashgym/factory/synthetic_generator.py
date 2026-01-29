# bashgym/factory/synthetic_generator.py
"""Synthetic task generation from extracted patterns."""

import json
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import httpx

if TYPE_CHECKING:
    from bashgym.factory.pattern_extractor import TracePatterns


@dataclass
class SyntheticTask:
    """A synthetically generated task for training data.

    Attributes:
        task_id: Unique identifier for the task
        prompt: The task prompt/instruction
        target_files: Files expected to be modified
        task_type: Category of task (feature, bugfix, refactor, spec)
        expected_tools: Tools expected to be used (Read, Edit, Bash, etc.)
        source_pattern_id: ID of the pattern that seeded this task
        repo: Repository name this task is for
        metadata: Additional task metadata
    """
    task_id: str
    prompt: str
    target_files: List[str]
    task_type: str
    expected_tools: List[str]
    source_pattern_id: str
    repo: str
    metadata: Dict = field(default_factory=dict)


class GenerationStrategy(str, Enum):
    """Strategy for generating synthetic tasks.

    - TRACE_SEEDED: Generate tasks based on patterns extracted from traces
    - AUGMENTED: Use LLM to augment/vary existing trace patterns
    - SCHEMA_DRIVEN: Generate tasks from predefined schemas
    """
    TRACE_SEEDED = "trace_seeded"
    AUGMENTED = "augmented"
    SCHEMA_DRIVEN = "schema_driven"


@dataclass
class GenerationPreset:
    """Preset configuration for synthetic task generation.

    Attributes:
        label: Display name for the preset
        description: Explanation of when to use this preset
        target_examples: Number of examples to generate (None for custom)
        multiplier: Multiplier for trace count (None means auto-calculate)
    """
    label: str
    description: str
    target_examples: Optional[int]
    multiplier: Optional[int] = None  # None means auto-calculate


# Research-based presets for different training scenarios
PRESETS: Dict[str, GenerationPreset] = {
    "quick_test": GenerationPreset(
        label="Quick Test",
        description="Fast iteration, minimal generation",
        target_examples=100,
    ),
    "balanced": GenerationPreset(
        label="Balanced (Recommended)",
        description="Good quality/time tradeoff for LoRA",
        target_examples=500,
    ),
    "production": GenerationPreset(
        label="Production",
        description="Full dataset for best results",
        target_examples=2000,
    ),
    "custom": GenerationPreset(
        label="Custom",
        description="Set your own target",
        target_examples=None,
    ),
}


class SyntheticGenerator:
    """Generates synthetic tasks from extracted patterns.

    This class is responsible for:
    - Calculating multipliers needed to reach target example counts
    - Building generation prompts from patterns
    - Calling LLMs to generate synthetic tasks
    - Batch generation with progress tracking
    """

    def __init__(self):
        """Initialize the SyntheticGenerator with default presets."""
        self.presets = PRESETS

    def calculate_multiplier(self, seed_count: int, target_examples: int) -> int:
        """Calculate multiplier needed to reach target examples.

        The multiplier indicates how many synthetic tasks should be generated
        from each seed pattern to reach the target example count.

        Args:
            seed_count: Number of seed patterns/traces available
            target_examples: Desired number of training examples

        Returns:
            Integer multiplier. Returns 0 if seed_count is 0.
            Minimum return value is 1 (except for 0 seeds).
        """
        if seed_count == 0:
            return 0

        multiplier = math.ceil(target_examples / seed_count)
        return max(1, multiplier)

    def _build_generation_prompt(
        self,
        patterns: "TracePatterns",
        task_type: str,
        seed_prompts: List[str],
        target_files: Optional[List[str]] = None
    ) -> str:
        """Build prompt for LLM to generate synthetic tasks.

        Constructs a prompt that provides context about the repository,
        task type, and examples to guide the LLM in generating realistic
        synthetic coding tasks.

        Args:
            patterns: Extracted patterns from traces (repo name, frameworks, etc.)
            task_type: Type of task to generate (feature, bugfix, refactor, spec)
            seed_prompts: Example prompts from real traces to use as reference
            target_files: Optional list of specific files to target

        Returns:
            A formatted prompt string ready for LLM consumption.
        """
        frameworks = ", ".join(patterns.framework_hints) if patterns.framework_hints else "general"
        paths = ", ".join(patterns.common_paths[:5]) if patterns.common_paths else "various"

        examples = "\n".join(f'- "{p}"' for p in seed_prompts[:5])

        # Determine primary language
        primary_language = "python"
        if patterns.languages:
            primary_language = max(patterns.languages, key=patterns.languages.get)

        prompt = f"""You are generating coding tasks for the {patterns.repo_name} codebase.

Task type: {task_type}
Common directories: {paths}
Frameworks: {frameworks}
Primary language: {primary_language}

Example real tasks from this repo:
{examples}

Generate a new, realistic {task_type} task that:
- Matches the style and complexity of the examples
- Is specific and actionable
- Would require modifying 1-3 files
- Uses realistic file names for this project

Return ONLY the task prompt, nothing else. No quotes, no explanation."""

        return prompt

    async def _call_llm(self, prompt: str, provider: str = "nim") -> str:
        """Call LLM provider to generate text.

        Routes the request to the appropriate provider (NIM or Anthropic).

        Args:
            prompt: The prompt to send to the LLM
            provider: Which provider to use ('nim' or 'anthropic')

        Returns:
            Generated text from the LLM

        Raises:
            ValueError: If provider is not recognized
        """
        if provider == "nim":
            return await self._call_nim(prompt)
        elif provider == "anthropic":
            return await self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _call_nim(self, prompt: str) -> str:
        """Call NVIDIA NIM API.

        Makes an HTTP request to the NIM API endpoint for text generation.

        Args:
            prompt: The prompt to send to NIM

        Returns:
            Generated text from the NIM model
        """
        endpoint = os.getenv("NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
        model = os.getenv("NIM_MODEL", "qwen/qwen2.5-coder-32b-instruct")
        api_key = os.getenv("NVIDIA_API_KEY", "")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.8
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API.

        Uses the Anthropic SDK to make a request to Claude.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            Generated text from Claude
        """
        import anthropic
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    async def generate_task(
        self,
        patterns: "TracePatterns",
        task_type: str,
        seed_prompts: List[str],
        provider: str = "nim"
    ) -> SyntheticTask:
        """Generate a single synthetic task.

        Uses the LLM to generate a new task based on patterns extracted
        from real traces.

        Args:
            patterns: Extracted patterns from traces (repo name, frameworks, etc.)
            task_type: Type of task to generate (feature, bugfix, refactor, spec)
            seed_prompts: Example prompts from real traces to use as reference
            provider: LLM provider to use ('nim' or 'anthropic')

        Returns:
            A SyntheticTask with generated prompt and metadata
        """
        prompt = self._build_generation_prompt(patterns, task_type, seed_prompts)
        generated_prompt = await self._call_llm(prompt, provider)

        return SyntheticTask(
            task_id=f"synth_{uuid.uuid4().hex[:8]}",
            prompt=generated_prompt,
            target_files=[],
            task_type=task_type,
            expected_tools=["Read", "Edit"],
            source_pattern_id="",
            repo=patterns.repo_name
        )

    async def generate_batch(
        self,
        patterns: "TracePatterns",
        seed_prompts: List[str],
        count: int,
        provider: str = "nim",
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[SyntheticTask]:
        """Generate a batch of synthetic tasks.

        Generates multiple synthetic tasks by sampling task types from the
        patterns distribution and calling generate_task repeatedly with
        progress reporting.

        Args:
            patterns: Extracted patterns from traces (repo name, frameworks, etc.)
            seed_prompts: Example prompts from real traces to use as reference
            count: Number of tasks to generate
            provider: LLM provider to use ('nim' or 'anthropic')
            on_progress: Optional callback called with (current, total) after each task

        Returns:
            List of successfully generated SyntheticTask objects.
            Failed generations are skipped (logged but not included).
        """
        tasks = []
        task_types = list(patterns.task_types.keys()) if patterns.task_types else ["feature"]
        weights = list(patterns.task_types.values()) if patterns.task_types else [1.0]

        for i in range(count):
            # Sample task type based on distribution
            task_type = random.choices(task_types, weights=weights, k=1)[0]

            try:
                task = await self.generate_task(
                    patterns=patterns,
                    task_type=task_type,
                    seed_prompts=seed_prompts,
                    provider=provider
                )
                tasks.append(task)
            except Exception as e:
                # Log error but continue
                print(f"Failed to generate task {i}: {e}")

            if on_progress:
                on_progress(i + 1, count)

        return tasks

    async def generate_from_schema(
        self,
        repo_schema: Dict,
        count: int = 10,
        provider: str = "nim"
    ) -> List[SyntheticTask]:
        """Generate tasks from repo structure schema (schema-driven strategy).

        This is an alternative generation strategy that creates tasks from
        a repository structure schema without needing existing traces.

        Args:
            repo_schema: Dictionary containing repo info:
                - name: Repository name
                - structure: Dict mapping directories to file lists
                - frameworks: List of framework names
            count: Number of tasks to generate
            provider: LLM provider to use ('nim' or 'anthropic')

        Returns:
            List of generated SyntheticTask objects
        """
        tasks = []

        repo_name = repo_schema.get("name", "unknown")
        structure = repo_schema.get("structure", {})
        frameworks = repo_schema.get("frameworks", [])

        # Build structure description
        structure_desc = "\n".join(
            f"- {dir}: {', '.join(files)}"
            for dir, files in structure.items()
        )

        for i in range(count):
            schema_prompt = f"""Generate a coding task for a project with this structure:

Project: {repo_name}
Frameworks: {', '.join(frameworks)}
Structure:
{structure_desc}

Generate a realistic development task that would involve 1-3 of these files.
Return ONLY the task prompt, nothing else."""

            try:
                task_prompt = await self._call_llm(schema_prompt, provider)
                tasks.append(SyntheticTask(
                    task_id=f"schema_{uuid.uuid4().hex[:8]}",
                    prompt=task_prompt,
                    target_files=[],
                    task_type="feature",
                    expected_tools=["Read", "Edit", "Bash"],
                    source_pattern_id="schema",
                    repo=repo_name,
                    metadata={"strategy": "schema_driven"}
                ))
            except Exception as e:
                print(f"Schema generation failed: {e}")

        return tasks

    async def generate_augmented(
        self,
        seed_examples: List[Dict],
        variations_per_seed: int = 3,
        provider: str = "nim"
    ) -> List[SyntheticTask]:
        """Generate variations of existing prompts (augmented strategy).

        This strategy creates new tasks by augmenting existing seed examples.
        For each seed, multiple variations are generated that maintain the
        same complexity and intent but with different specific details.

        Args:
            seed_examples: List of seed examples with at least 'prompt' key.
                Optional keys: 'id', 'repo', 'response'
            variations_per_seed: Number of variations to create per seed (default 3)
            provider: LLM provider to use ('nim' or 'anthropic')

        Returns:
            List of SyntheticTask objects with augmented prompts
        """
        tasks = []

        for seed in seed_examples:
            prompt = seed.get("prompt", "")

            for i in range(variations_per_seed):
                augment_prompt = f"""Rewrite this coding task to create a variation:
Original: "{prompt}"

Create a similar but different task that:
- Has the same complexity level
- Targets similar types of files
- Uses different specific details

Return ONLY the rewritten task, nothing else."""

                try:
                    variation = await self._call_llm(augment_prompt, provider)
                    tasks.append(SyntheticTask(
                        task_id=f"aug_{uuid.uuid4().hex[:8]}",
                        prompt=variation,
                        target_files=[],
                        task_type="feature",
                        expected_tools=["Read", "Edit"],
                        source_pattern_id=seed.get("id", ""),
                        repo=seed.get("repo", ""),
                        metadata={"strategy": "augmented", "seed_prompt": prompt}
                    ))
                except Exception as e:
                    print(f"Augmentation failed: {e}")

        return tasks

    def export_to_nemo(
        self,
        tasks: List[SyntheticTask],
        output_dir: Path,
        train_ratio: float = 0.9,
        system_prompt: str = "You are a skilled coding assistant."
    ) -> Dict:
        """Export synthetic tasks to NeMo-compatible JSONL format.

        Creates train.jsonl, val.jsonl, and metadata.json files in the
        output directory. Tasks are shuffled and split according to train_ratio.

        Args:
            tasks: List of SyntheticTask objects to export
            output_dir: Directory to write output files
            train_ratio: Fraction of tasks for training (default 0.9)
            system_prompt: System prompt to include in each example

        Returns:
            Metadata dictionary with generation info
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle and split
        shuffled = tasks.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_tasks = shuffled[:split_idx]
        val_tasks = shuffled[split_idx:]

        # Convert to NeMo format
        def to_nemo(task: SyntheticTask) -> Dict:
            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task.prompt},
                    {"role": "assistant", "content": f"I'll help you {task.prompt.lower()}..."}
                ],
                "metadata": {
                    "synthetic": True,
                    "strategy": task.metadata.get("strategy", "trace_seeded"),
                    "task_type": task.task_type,
                    "repo": task.repo,
                    "task_id": task.task_id,
                    "generated_at": datetime.now().isoformat()
                }
            }

        # Write train.jsonl
        with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
            for task in train_tasks:
                f.write(json.dumps(to_nemo(task)) + "\n")

        # Write val.jsonl
        with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
            for task in val_tasks:
                f.write(json.dumps(to_nemo(task)) + "\n")

        # Write metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_examples": len(tasks),
            "train_examples": len(train_tasks),
            "val_examples": len(val_tasks),
            "train_ratio": train_ratio,
            "strategy": "trace_seeded"
        }

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata
