"""
Coding Agent SFT Pipeline

Generates supervised fine-tuning data for coding AI agents.
Produces task prompts, structured tool-use solutions, and quality scores
through DataDesigner's column DAG engine.

Pipeline:
  samplers (task_category, complexity, language, codebase_size)
  -> LLM text (task_prompt)
  -> LLM structured (solution as AgentSolution)
  -> expression (solution_text flattened)
  -> LLM judge (quality_score with 3 dimensions)
  -> filter (quality >= 3)
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bashgym.factory.data_designer import PipelineConfig

try:
    import data_designer.config as dd
    DATA_DESIGNER_AVAILABLE = True
except ImportError:
    DATA_DESIGNER_AVAILABLE = False


# =========================================================================
# Pydantic Models for Structured Output
# =========================================================================

class ToolCall(BaseModel):
    """Structured tool call for training data."""
    tool: str = Field(description="Tool name: read, edit, bash, write, glob, grep")
    arguments: str = Field(description="Tool arguments")
    reasoning: str = Field(description="Why this tool call is needed")


class AgentStep(BaseModel):
    """Single step in an agent solution."""
    thought: str = Field(description="Agent reasoning before action")
    tool_call: ToolCall = Field(description="The tool invocation")
    observation: str = Field(description="Expected tool output")


class AgentSolution(BaseModel):
    """Complete agent solution with tool-use trajectory."""
    plan: str = Field(description="High-level approach")
    steps: list[AgentStep] = Field(description="Ordered list of tool calls")
    summary: str = Field(description="What was accomplished")


# =========================================================================
# Pipeline Builder
# =========================================================================

def build_sft_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build the coding agent SFT training data pipeline.

    Creates a DataDesigner config with:
    - 4 sampler columns for statistical diversity
    - 1 LLM text column for task prompt generation
    - 1 LLM structured column for solution generation
    - 1 expression column for flattened solution text
    - 1 LLM judge column for quality scoring
    - 1 processor for filtering low-quality examples

    Args:
        config: PipelineConfig with model/temperature settings

    Returns:
        Configured DataDesignerConfigBuilder
    """
    if not DATA_DESIGNER_AVAILABLE:
        raise ImportError("data-designer>=0.5.0 is required")

    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="text-model",
                model=config.text_model,
                inference_parameters=dd.InferenceParameters(
                    temperature=config.temperature_text,
                    top_p=0.99,
                    max_tokens=2048,
                ),
            ),
            dd.ModelConfig(
                alias="code-model",
                model=config.code_model,
                inference_parameters=dd.InferenceParameters(
                    temperature=config.temperature_code,
                    top_p=0.95,
                    max_tokens=4096,
                ),
            ),
            dd.ModelConfig(
                alias="judge-model",
                model=config.judge_model,
                inference_parameters=dd.InferenceParameters(
                    temperature=config.temperature_judge,
                    max_tokens=1024,
                ),
            ),
        ],
        model_providers=[
            dd.ModelProvider(
                name=config.provider,
                endpoint=config.provider_endpoint,
                provider_type="openai",
                api_key=f"${{{_env_key_for_provider(config.provider)}}}",
            ),
        ],
    )

    # --- Sampling for statistical diversity ---

    builder.add_column(dd.SamplerColumnConfig(
        name="task_category",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "bug_fix", "feature", "refactor", "test", "docs",
                "config", "debug", "optimize", "security_fix",
            ],
            weights=[0.2, 0.25, 0.15, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="complexity",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["simple", "moderate", "complex"],
            weights=[0.3, 0.5, 0.2],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="language",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["python", "typescript", "javascript", "rust", "go", "bash"],
            weights=[0.35, 0.2, 0.15, 0.1, 0.1, 0.1],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="codebase_size",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["single_file", "small_project", "medium_project"],
            weights=[0.3, 0.5, 0.2],
        ),
    ))

    # --- Task prompt generation ---

    builder.add_column(dd.LLMTextColumnConfig(
        name="task_prompt",
        model_alias="text-model",
        prompt=(
            "You are generating training data for a coding AI agent.\n\n"
            "Seed task: {{ seed_task }}\n"
            "Category: {{ task_category }}\n"
            "Complexity: {{ complexity }}\n"
            "Language: {{ language }}\n"
            "Codebase size: {{ codebase_size }}\n\n"
            "Generate a realistic, specific coding task prompt that a developer "
            "would give to an AI coding assistant. The task should be a "
            "{{ complexity }} {{ task_category }} task in {{ language }}.\n\n"
            "Requirements:\n"
            "- Be specific about files, functions, or components involved\n"
            "- Include enough context for the agent to understand the codebase\n"
            "- Match the style and scope of the seed task\n"
            "- Output ONLY the task prompt, nothing else"
        ),
    ))

    # --- Solution generation as structured tool-use trajectory ---

    builder.add_column(dd.LLMStructuredColumnConfig(
        name="solution",
        model_alias="code-model",
        prompt=(
            "You are a coding AI agent. Solve this task step by step.\n\n"
            "Task: {{ task_prompt }}\n\n"
            "Show your complete solution as a sequence of tool calls. "
            "Available tools: read (read files), edit (modify files), "
            "bash (run commands), write (create files), glob (find files), "
            "grep (search content).\n\n"
            "Think carefully about each step. Provide your reasoning, "
            "the exact tool call, and the expected observation."
        ),
        output_format=AgentSolution,
    ))

    # --- Flatten solution to training-ready text ---

    builder.add_column(dd.ExpressionColumnConfig(
        name="solution_text",
        expr=(
            "Plan: {{ solution.plan }}\n\n"
            "{% for step in solution.steps %}"
            "Step {{ loop.index }}:\n"
            "Thought: {{ step.thought }}\n"
            "[{{ step.tool_call.tool }}] {{ step.tool_call.arguments }}\n"
            "Output: {{ step.observation }}\n\n"
            "{% endfor %}"
            "Summary: {{ solution.summary }}"
        ),
    ))

    # --- Quality validation ---

    builder.add_column(dd.LLMJudgeColumnConfig(
        name="quality_score",
        model_alias="judge-model",
        prompt=(
            "Evaluate this coding agent solution:\n\n"
            "Task: {{ task_prompt }}\n\n"
            "Solution:\n{{ solution_text }}"
        ),
        scores=[
            dd.Score(
                name="correctness",
                description="Does the solution correctly address the task?",
                options={
                    "5": "Perfect -- addresses every aspect correctly",
                    "4": "Minor issues that don't affect functionality",
                    "3": "Works but misses some requirements",
                    "2": "Significant logic errors",
                    "1": "Does not address the task",
                },
            ),
            dd.Score(
                name="tool_usage",
                description="Does the agent use appropriate tools in a logical sequence?",
                options={
                    "5": "Optimal tool usage -- reads before editing, tests after changes",
                    "4": "Good tool usage with minor inefficiencies",
                    "3": "Acceptable but could be improved",
                    "2": "Illogical tool sequence",
                    "1": "Wrong tools or missing critical steps",
                },
            ),
            dd.Score(
                name="completeness",
                description="Is the solution thorough and complete?",
                options={
                    "5": "Handles all requirements including edge cases",
                    "4": "Handles main requirements, minor gaps",
                    "3": "Partially complete",
                    "2": "Missing major parts",
                    "1": "Incomplete",
                },
            ),
        ],
    ))

    # --- Filter low-quality examples ---

    builder.add_processor(
        processor_type="filter",
        condition="quality_score.correctness >= 3 and quality_score.tool_usage >= 3",
    )

    return builder


def _env_key_for_provider(provider: str) -> str:
    """Map provider name to environment variable key."""
    mapping = {
        "nvidia": "NVIDIA_API_KEY",
        "nvidia-nim": "NVIDIA_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "local": "LOCAL_API_KEY",
    }
    return mapping.get(provider, "NVIDIA_API_KEY")
