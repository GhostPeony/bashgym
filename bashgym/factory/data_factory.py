"""
Data Factory for The Factory Layer

Integrates with NVIDIA NeMo Data Designer to synthesize training data
from golden execution traces. Handles trace ingestion, prompt generation,
and training data formatting.

Extended with Schema Builder integration for rich synthetic data generation.

Module 3: Data Synthesis (The "Factory")
"""

import os
import json
import httpx
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum
import hashlib

if TYPE_CHECKING:
    from .schema_builder import DataDesignerClient, DataSchema


class SynthesisStrategy(Enum):
    """Data synthesis strategies."""
    DIRECT = "direct"           # Direct trace to training example
    AUGMENTED = "augmented"     # Augment with variations
    DISTILLED = "distilled"     # Distill to minimal steps
    CONTRASTIVE = "contrastive" # Generate positive/negative pairs (DPO)


class AugmentationProvider(Enum):
    """LLM provider for data augmentation."""
    NIM = "nim"           # NVIDIA NIM (qwen/qwen2.5-coder-32b-instruct)
    ANTHROPIC = "anthropic"  # Anthropic Claude (higher quality, higher cost)


@dataclass
class DataFactoryConfig:
    """Configuration for the Data Factory."""

    # NeMo Data Designer settings
    nemo_endpoint: str = "http://localhost:8000"
    nemo_api_key: Optional[str] = None

    # NVIDIA NIM settings (for LLM-based synthesis)
    nim_endpoint: str = "https://integrate.api.nvidia.com/v1"
    nim_api_key: Optional[str] = None
    nim_model: str = "qwen/qwen2.5-coder-32b-instruct"

    # Anthropic settings (for higher quality augmentation)
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-5-20250929"  # Claude Sonnet 4.5 - best balance of quality/cost

    # Augmentation provider choice
    augmentation_provider: AugmentationProvider = AugmentationProvider.ANTHROPIC

    # Synthesis settings
    strategy: SynthesisStrategy = SynthesisStrategy.AUGMENTED
    augmentation_factor: int = 3  # Generate N variations per trace
    max_sequence_length: int = 8192

    # Output settings
    output_dir: str = "data/training_batches"
    batch_size: int = 100

    # Quality settings
    min_trace_steps: int = 2
    max_trace_steps: int = 50
    require_successful_verification: bool = True


@dataclass
class TrainingExample:
    """A single training example."""

    example_id: str
    system_prompt: str
    user_prompt: str
    assistant_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.example_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
                {"role": "assistant", "content": self.assistant_response}
            ],
            "metadata": self.metadata
        }

    def to_chatml(self) -> str:
        """Convert to ChatML format."""
        return f"""<|im_start|>system
{self.system_prompt}<|im_end|>
<|im_start|>user
{self.user_prompt}<|im_end|>
<|im_start|>assistant
{self.assistant_response}<|im_end|>"""


@dataclass
class DPOExample:
    """A DPO (Direct Preference Optimization) training example."""

    example_id: str
    prompt: str
    chosen: str      # Preferred response (from gold trace)
    rejected: str    # Rejected response (from failed trace)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to DPO format (NVIDIA-compatible).

        Uses chosen_response/rejected_response field names as required by
        NVIDIA NeMo Customizer DPO training format.
        """
        return {
            "id": self.example_id,
            "prompt": self.prompt,
            "chosen_response": self.chosen,
            "rejected_response": self.rejected,
            "metadata": self.metadata
        }


class DataFactory:
    """
    Synthesizes training data from golden execution traces.

    Integrates with NVIDIA NeMo Data Designer for:
    - Trace-to-training-example conversion
    - Data augmentation and variation
    - Quality filtering and deduplication
    - DPO pair generation
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an expert software development agent. You execute tasks by running bash commands, reading files, and making edits. You think step-by-step and verify your work.

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

    def __init__(
        self,
        config: Optional[DataFactoryConfig] = None,
        data_designer_client: Optional["DataDesignerClient"] = None
    ):
        """Initialize the Data Factory.

        Args:
            config: Factory configuration (if None, uses defaults + env vars)
            data_designer_client: Optional Data Designer client for synthetic data generation
        """
        # If no config provided, create one and apply env var overrides
        use_env_overrides = config is None
        self.config = config or DataFactoryConfig()

        # Always load API keys from environment if not set
        if not self.config.nim_api_key:
            self.config.nim_api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.config.nemo_api_key:
            self.config.nemo_api_key = os.environ.get("NEMO_API_KEY")
        if not self.config.anthropic_api_key:
            self.config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Only apply env var overrides for non-API settings when using default config
        if use_env_overrides:
            aug_provider = os.environ.get("AUGMENTATION_PROVIDER", "").lower()
            if aug_provider == "nim":
                self.config.augmentation_provider = AugmentationProvider.NIM
            elif aug_provider == "anthropic":
                self.config.augmentation_provider = AugmentationProvider.ANTHROPIC

            anthropic_model = os.environ.get("ANTHROPIC_AUGMENTATION_MODEL")
            if anthropic_model:
                self.config.anthropic_model = anthropic_model

            nim_model = os.environ.get("NIM_MODEL")
            if nim_model:
                self.config.nim_model = nim_model

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # HTTP client for API calls
        self.client = httpx.AsyncClient(timeout=120.0)

        # Data Designer client for schema-based generation
        self.data_designer_client = data_designer_client

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def process_gold_trace(self, trace_path: Path) -> Optional[TrainingExample]:
        """
        Convert a gold trace to a training example.

        Args:
            trace_path: Path to the gold trace JSON file

        Returns:
            TrainingExample or None if trace is invalid
        """
        try:
            with open(trace_path, 'r', encoding='utf-8', errors='replace') as f:
                trace_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading trace {trace_path}: {e}")
            return None

        # Must be a dict with metadata, not a raw list of steps
        if not isinstance(trace_data, dict):
            return None

        # Validate trace
        if not self._validate_trace(trace_data):
            return None

        # Extract components
        metadata = trace_data.get("metadata", {})
        trace_steps = trace_data.get("trace", [])

        # Build user prompt from initial task
        user_prompt = metadata.get("user_initial_prompt", "")
        if not user_prompt:
            return None

        # Build assistant response from trace
        assistant_response = self._trace_to_response(trace_steps)

        # Generate example ID
        example_id = hashlib.sha256(
            f"{user_prompt}{assistant_response}".encode()
        ).hexdigest()[:16]

        return TrainingExample(
            example_id=example_id,
            system_prompt=self.SYSTEM_PROMPT_TEMPLATE,
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            metadata={
                "source_trace": str(trace_path),
                "task_id": metadata.get("task_id"),
                "success_rate": trace_data.get("summary", {}).get("success_rate", 0),
                "total_steps": len(trace_steps)
            }
        )

    def _validate_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Validate a trace meets quality requirements."""
        # Check verification status
        if self.config.require_successful_verification:
            vp = trace_data.get("metadata", {}).get("verification_passed")
            if vp is False:  # Only reject explicitly failed verification
                return False

        # Check step count
        trace_steps = trace_data.get("trace", [])
        if len(trace_steps) < self.config.min_trace_steps:
            return False
        if len(trace_steps) > self.config.max_trace_steps:
            return False

        # Check for required fields
        metadata = trace_data.get("metadata", {})
        if not metadata.get("user_initial_prompt"):
            return False

        return True

    def _trace_to_response(self, trace_steps: List[Dict[str, Any]]) -> str:
        """
        Convert trace steps to an assistant response.

        Formats the trace as a reasoning + action sequence.
        """
        response_parts = []

        for i, step in enumerate(trace_steps, 1):
            tool_name = step.get("tool_name", "unknown")
            command = step.get("command", "")
            output = step.get("output", "")[:500]  # Truncate long outputs
            success = step.get("success")

            # Format based on tool type
            if tool_name.lower() == "bash":
                response_parts.append(f"**Step {i}: Execute command**")
                response_parts.append(f"```bash\n{command}\n```")
                if output:
                    response_parts.append(f"Output:\n```\n{output}\n```")
            elif tool_name.lower() in ("read", "write", "edit"):
                response_parts.append(f"**Step {i}: {tool_name.title()} file**")
                response_parts.append(f"```\n{command}\n```")

            response_parts.append("")  # Empty line between steps

        return "\n".join(response_parts)

    async def augment_example(
        self,
        example: TrainingExample,
        num_variations: int = 3
    ) -> List[TrainingExample]:
        """
        Generate variations of a training example using configured provider.

        Uses LLM to create semantically similar but syntactically different
        versions of the task and solution.

        Provider is selected via config.augmentation_provider:
        - ANTHROPIC: Higher quality, better instruction following
        - NIM: Cost-effective, good for code
        """
        # Route to appropriate provider
        if self.config.augmentation_provider == AugmentationProvider.ANTHROPIC:
            return await self._augment_with_anthropic(example, num_variations)
        else:
            return await self._augment_with_nim(example, num_variations)

    async def _augment_with_anthropic(
        self,
        example: TrainingExample,
        num_variations: int = 3
    ) -> List[TrainingExample]:
        """Generate variations using Anthropic Claude."""
        if not self.config.anthropic_api_key:
            print("No Anthropic API key, falling back to NIM")
            return await self._augment_with_nim(example, num_variations)

        variations = [example]  # Include original

        augmentation_prompt = f"""Generate {num_variations} variations of this software development task and solution.

Each variation should:
1. Rephrase the task naturally (different wording, same intent)
2. Use equivalent but different commands/approaches where possible
3. Maintain complete correctness and functionality
4. Be realistic tasks a developer might actually request

Original Task:
{example.user_prompt}

Original Solution:
{example.assistant_response}

Output ONLY a JSON array with {num_variations} objects, each having "task" and "solution" keys:
[{{"task": "...", "solution": "..."}}, ...]"""

        try:
            response = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.config.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.anthropic_model,
                    "max_tokens": 4096,
                    "messages": [
                        {"role": "user", "content": augmentation_prompt}
                    ]
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result["content"][0]["text"]

                # Parse variations from response
                try:
                    json_start = content.find("[")
                    json_end = content.rfind("]") + 1
                    if json_start >= 0 and json_end > json_start:
                        variation_data = json.loads(content[json_start:json_end])

                        for i, var in enumerate(variation_data):
                            var_example = TrainingExample(
                                example_id=f"{example.example_id}_var{i+1}",
                                system_prompt=example.system_prompt,
                                user_prompt=var.get("task", example.user_prompt),
                                assistant_response=var.get("solution", example.assistant_response),
                                metadata={
                                    **example.metadata,
                                    "augmented": True,
                                    "augmentation_provider": "anthropic",
                                    "variation_index": i + 1
                                }
                            )
                            variations.append(var_example)
                except json.JSONDecodeError:
                    print("Failed to parse Anthropic response as JSON")
            else:
                print(f"Anthropic API error: {response.status_code} - {response.text[:200]}")

        except Exception as e:
            print(f"Anthropic augmentation failed: {e}")

        return variations

    async def _augment_with_nim(
        self,
        example: TrainingExample,
        num_variations: int = 3
    ) -> List[TrainingExample]:
        """Generate variations using NVIDIA NIM."""
        if not self.config.nim_api_key:
            return [example]  # Return original if no API key

        variations = [example]  # Include original

        augmentation_prompt = f"""Given this software development task and solution, generate {num_variations} variations.
Each variation should:
1. Rephrase the task slightly differently
2. Keep the same general approach but vary command syntax where possible
3. Maintain correctness

Original Task:
{example.user_prompt}

Original Solution:
{example.assistant_response}

Generate {num_variations} variations in JSON format:
[{{"task": "...", "solution": "..."}}, ...]"""

        try:
            response = await self.client.post(
                f"{self.config.nim_endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.nim_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.nim_model,
                    "messages": [
                        {"role": "user", "content": augmentation_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Parse variations from response
                try:
                    json_start = content.find("[")
                    json_end = content.rfind("]") + 1
                    if json_start >= 0 and json_end > json_start:
                        variation_data = json.loads(content[json_start:json_end])

                        for i, var in enumerate(variation_data):
                            var_example = TrainingExample(
                                example_id=f"{example.example_id}_var{i+1}",
                                system_prompt=example.system_prompt,
                                user_prompt=var.get("task", example.user_prompt),
                                assistant_response=var.get("solution", example.assistant_response),
                                metadata={
                                    **example.metadata,
                                    "augmented": True,
                                    "augmentation_provider": "nim",
                                    "variation_index": i + 1
                                }
                            )
                            variations.append(var_example)
                except json.JSONDecodeError:
                    pass  # Keep original only

        except Exception as e:
            print(f"NIM augmentation failed: {e}")

        return variations

    async def generate_dpo_pairs(
        self,
        gold_trace_path: Path,
        failed_trace_path: Path
    ) -> Optional[DPOExample]:
        """
        Generate a DPO training pair from gold and failed traces.

        Args:
            gold_trace_path: Path to successful trace
            failed_trace_path: Path to failed trace

        Returns:
            DPOExample for preference learning
        """
        # Load both traces
        try:
            with open(gold_trace_path, 'r') as f:
                gold_trace = json.load(f)
            with open(failed_trace_path, 'r') as f:
                failed_trace = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading traces: {e}")
            return None

        # Extract prompts - they should be similar
        gold_prompt = gold_trace.get("metadata", {}).get("user_initial_prompt", "")
        failed_prompt = failed_trace.get("metadata", {}).get("user_initial_prompt", "")

        # Use gold prompt as the canonical prompt
        prompt = gold_prompt or failed_prompt
        if not prompt:
            return None

        # Convert traces to responses
        chosen = self._trace_to_response(gold_trace.get("trace", []))
        rejected = self._trace_to_response(failed_trace.get("trace", []))

        # Generate example ID
        example_id = hashlib.sha256(
            f"{prompt}{chosen}{rejected}".encode()
        ).hexdigest()[:16]

        return DPOExample(
            example_id=example_id,
            prompt=f"{self.SYSTEM_PROMPT_TEMPLATE}\n\nUser: {prompt}",
            chosen=chosen,
            rejected=rejected,
            metadata={
                "gold_trace": str(gold_trace_path),
                "failed_trace": str(failed_trace_path)
            }
        )

    async def process_trace_directory(
        self,
        gold_dir: Path,
        failed_dir: Optional[Path] = None
    ) -> Tuple[List[TrainingExample], List[DPOExample]]:
        """
        Process all traces in a directory.

        Args:
            gold_dir: Directory containing gold traces
            failed_dir: Optional directory containing failed traces (for DPO)

        Returns:
            Tuple of (training_examples, dpo_examples)
        """
        training_examples = []
        dpo_examples = []

        # Process gold traces
        gold_traces = list(Path(gold_dir).glob("*.json"))
        print(f"Processing {len(gold_traces)} gold traces...")

        for trace_path in gold_traces:
            example = self.process_gold_trace(trace_path)
            if example:
                # Augment if configured
                if self.config.strategy == SynthesisStrategy.AUGMENTED:
                    augmented = await self.augment_example(
                        example,
                        self.config.augmentation_factor
                    )
                    training_examples.extend(augmented)
                else:
                    training_examples.append(example)

        # Generate DPO pairs if failed traces available
        if failed_dir and Path(failed_dir).exists():
            failed_traces = list(Path(failed_dir).glob("*.json"))
            print(f"Processing {len(failed_traces)} failed traces for DPO...")

            # Match gold and failed traces by task similarity
            for failed_path in failed_traces:
                # Find a matching gold trace (simplified - in production use semantic matching)
                if gold_traces:
                    gold_path = gold_traces[0]  # Simplified matching
                    dpo_example = await self.generate_dpo_pairs(gold_path, failed_path)
                    if dpo_example:
                        dpo_examples.append(dpo_example)

        return training_examples, dpo_examples

    def save_training_batch(
        self,
        examples: List[TrainingExample],
        batch_name: str = "batch"
    ) -> Path:
        """
        Save training examples to a JSONL file.

        Args:
            examples: List of training examples
            batch_name: Name prefix for the batch file

        Returns:
            Path to the saved batch file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{batch_name}_{timestamp}.jsonl"
        output_path = Path(self.config.output_dir) / filename

        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        print(f"Saved {len(examples)} examples to {output_path}")
        return output_path

    def save_dpo_batch(
        self,
        examples: List[DPOExample],
        batch_name: str = "dpo_batch"
    ) -> Path:
        """
        Save DPO examples to a JSONL file.

        Args:
            examples: List of DPO examples
            batch_name: Name prefix for the batch file

        Returns:
            Path to the saved batch file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{batch_name}_{timestamp}.jsonl"
        output_path = Path(self.config.output_dir) / filename

        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        print(f"Saved {len(examples)} DPO examples to {output_path}")
        return output_path

    async def generate_synthetic_data(
        self,
        schema: "DataSchema",
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic training data using Data Designer schema.

        Args:
            schema: DataSchema defining the data structure
            output_path: Optional path to save generated data

        Returns:
            List of generated records
        """
        if not self.data_designer_client:
            print("Data Designer client not configured")
            return []

        try:
            records = await self.data_designer_client.generate(
                schema,
                output_path=output_path
            )
            print(f"Generated {len(records)} synthetic records")
            return records
        except Exception as e:
            print(f"Synthetic data generation failed: {e}")
            return []

    async def generate_coding_tasks(
        self,
        num_tasks: int = 100,
        languages: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None
    ) -> List[TrainingExample]:
        """
        Generate synthetic coding task training data.

        Uses predefined schema for diverse coding tasks.

        Args:
            num_tasks: Number of tasks to generate
            languages: Programming languages to include
            difficulties: Difficulty levels

        Returns:
            List of TrainingExample
        """
        # Import here to avoid circular dependency
        from .schema_builder import coding_task_schema

        schema = coding_task_schema(
            languages=languages,
            difficulties=difficulties,
            num_rows=num_tasks
        )

        records = await self.generate_synthetic_data(schema)

        examples = []
        for record in records:
            task_id = record.get("task_id", "")
            task_desc = record.get("task_description", "")
            solution = record.get("reference_solution", "")

            if task_desc and solution:
                example = TrainingExample(
                    example_id=f"synthetic_{task_id}",
                    system_prompt=self.SYSTEM_PROMPT_TEMPLATE,
                    user_prompt=task_desc,
                    assistant_response=f"```{record.get('language', 'python')}\n{solution}\n```",
                    metadata={
                        "synthetic": True,
                        "language": record.get("language"),
                        "difficulty": record.get("difficulty"),
                        "task_type": record.get("task_type")
                    }
                )
                examples.append(example)

        return examples

    async def generate_conversation_data(
        self,
        num_conversations: int = 100,
        personas: Optional[List[str]] = None,
        topics: Optional[List[str]] = None
    ) -> List[TrainingExample]:
        """
        Generate synthetic conversation training data.

        Args:
            num_conversations: Number of conversations to generate
            personas: Assistant personas to use
            topics: Topics for conversations

        Returns:
            List of TrainingExample
        """
        # Import here to avoid circular dependency
        from .schema_builder import conversation_schema

        schema = conversation_schema(
            personas=personas,
            topics=topics,
            num_rows=num_conversations
        )

        records = await self.generate_synthetic_data(schema)

        examples = []
        for i, record in enumerate(records):
            user_msg = record.get("user_message", "")
            assistant_msg = record.get("assistant_response", "")

            if user_msg and assistant_msg:
                example = TrainingExample(
                    example_id=f"conv_{record.get('conversation_id', i)}",
                    system_prompt=f"You are a {record.get('persona', 'helpful assistant')}.",
                    user_prompt=user_msg,
                    assistant_response=assistant_msg,
                    metadata={
                        "synthetic": True,
                        "persona": record.get("persona"),
                        "topic": record.get("topic")
                    }
                )
                examples.append(example)

        return examples


async def main():
    """Example usage of the Data Factory."""
    config = DataFactoryConfig(
        strategy=SynthesisStrategy.AUGMENTED,
        augmentation_factor=2
    )

    factory = DataFactory(config)

    try:
        # Process traces
        gold_dir = Path("data/gold_traces")
        failed_dir = Path("data/failed_traces")

        if gold_dir.exists():
            examples, dpo_examples = await factory.process_trace_directory(
                gold_dir, failed_dir
            )

            if examples:
                factory.save_training_batch(examples, "sft_batch")

            if dpo_examples:
                factory.save_dpo_batch(dpo_examples, "dpo_batch")
        else:
            print("No gold traces directory found")
    finally:
        await factory.close()


if __name__ == "__main__":
    asyncio.run(main())
