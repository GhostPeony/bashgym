"""
External Dataset Pipeline

Ingests HuggingFace datasets, CSV, Parquet, or JSON files as seed data
and generates augmented training examples for coding AI agents.

Flexible enough to handle:
- Code instruction datasets (e.g. bigcode/starcoderdata)
- General instruction datasets (e.g. Open-Orca, Alpaca)
- Conversation datasets
- Q&A datasets

Pipeline:
  seed dataset (auto-mapped columns)
  -> samplers (augmentation_style)
  -> LLM text (augmented_task, augmented_response)
  -> LLM judge (quality_score)
  -> filter (quality >= 3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bashgym.factory.designer_pipelines import build_base_config

if TYPE_CHECKING:
    from bashgym.factory.data_designer import PipelineConfig

try:
    import data_designer.config as dd
except ImportError:
    pass


def build_external_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build pipeline for external dataset augmentation.

    Takes seed data from HuggingFace or local files and generates
    augmented, rewritten training examples in BashGym's agent format.

    Seed dataset columns are auto-detected. Common mappings:
    - instruction/prompt/question -> seed_task
    - input/context -> seed_context
    - output/response/answer -> seed_response

    Args:
        config: PipelineConfig with model/temperature settings

    Returns:
        Configured DataDesignerConfigBuilder
    """
    builder = build_base_config(config)

    # --- Augmentation style diversity ---

    builder.add_column(
        dd.SamplerColumnConfig(
            name="augmentation_style",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "rephrase",  # Reword the task
                    "extend",  # Add complexity
                    "simplify",  # Make it easier
                    "agent_format",  # Convert to agent tool-use style
                ],
                weights=[0.3, 0.25, 0.15, 0.3],
            ),
        )
    )

    builder.add_column(
        dd.SamplerColumnConfig(
            name="target_format",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["step_by_step", "tool_use", "explanation_first", "code_first"],
                weights=[0.3, 0.3, 0.2, 0.2],
            ),
        )
    )

    # --- Rewrite task prompt for agent training ---

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="task_prompt",
            model_alias="text-model",
            prompt=(
                "You are rewriting a coding task for training an AI coding agent.\n\n"
                "Original task: {{ seed_task }}\n"
                "{% if seed_context %}Context: {{ seed_context }}\n{% endif %}"
                "Augmentation style: {{ augmentation_style }}\n\n"
                "Rewrite this as a clear, specific task that a developer would give "
                "to an AI coding assistant. If the style is 'agent_format', include "
                "details about files and project structure. If 'extend', add additional "
                "requirements. If 'simplify', focus on the core ask.\n\n"
                "Output ONLY the rewritten task prompt."
            ),
        )
    )

    # --- Generate agent-style response ---

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="solution",
            model_alias="code-model",
            prompt=(
                "You are a coding AI agent. Solve this task showing your work.\n\n"
                "Task: {{ task_prompt }}\n"
                "{% if seed_response %}Reference approach: {{ seed_response }}\n{% endif %}"
                "Response format: {{ target_format }}\n\n"
                "If format is 'tool_use', show your solution as tool calls:\n"
                "[tool_name] arguments\n"
                "If 'step_by_step', number each step.\n"
                "If 'explanation_first', explain before showing code.\n"
                "If 'code_first', show the code then explain."
            ),
        )
    )

    # --- Quality check ---

    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="quality_score",
            model_alias="judge-model",
            prompt=(
                "Evaluate this task-solution pair for training a coding AI agent:\n\n"
                "Task: {{ task_prompt }}\n\n"
                "Solution:\n{{ solution }}"
            ),
            scores=[
                dd.Score(
                    name="relevance",
                    description="Does the solution address the task?",
                    options={
                        "5": "Perfectly addresses the task",
                        "4": "Mostly relevant, minor drift",
                        "3": "Partially relevant",
                        "2": "Significant mismatch",
                        "1": "Completely off-topic",
                    },
                ),
                dd.Score(
                    name="training_value",
                    description="Would this be a good training example?",
                    options={
                        "5": "Excellent training example",
                        "4": "Good training example",
                        "3": "Acceptable",
                        "2": "Low value, too generic or too niche",
                        "1": "Not suitable for training",
                    },
                ),
            ],
        )
    )

    # --- Filter ---

    builder.add_processor(
        processor_type="filter",
        condition="quality_score.relevance >= 3 and quality_score.training_value >= 3",
    )

    return builder
