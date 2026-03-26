"""
Coding Agent DPO Pipeline

Generates Direct Preference Optimization training data by producing
two solutions for each task at different temperatures, judging both,
and using expression columns to assign chosen/rejected labels.

Pipeline:
  samplers (task_category)
  -> LLM text (task_prompt)
  -> LLM text x2 (solution_a @ temp=0.9, solution_b @ temp=0.5)
  -> LLM judge x2 (judge_a, judge_b)
  -> expressions (chosen, rejected based on judge scores)
  -> filter (scores must differ for meaningful preference signal)
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


def build_dpo_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build DPO preference pair pipeline.

    Generates two solutions for each task, judges both,
    and uses expression columns to assign chosen/rejected.

    The intentional temperature difference (0.9 vs 0.5) creates
    natural quality variation - higher temperature solutions are
    more creative but less reliable, producing a realistic
    distribution of preference pairs.

    Args:
        config: PipelineConfig with model/temperature settings

    Returns:
        Configured DataDesignerConfigBuilder
    """
    builder = build_base_config(config)

    # DPO needs two solution models at different temperatures for preference pairs
    builder.model_configs.extend(
        [
            dd.ModelConfig(
                alias="solution-model-a",
                model=config.code_model,
                inference_parameters=dd.InferenceParameters(
                    temperature=0.9,  # Higher temp for more variation
                    max_tokens=4096,
                ),
            ),
            dd.ModelConfig(
                alias="solution-model-b",
                model=config.code_model,
                inference_parameters=dd.InferenceParameters(
                    temperature=0.5,  # Lower temp for different style
                    max_tokens=4096,
                ),
            ),
        ]
    )

    # --- Sampling for diversity ---

    builder.add_column(
        dd.SamplerColumnConfig(
            name="task_category",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["bug_fix", "feature", "refactor", "test", "debug"],
            ),
        )
    )

    builder.add_column(
        dd.SamplerColumnConfig(
            name="complexity",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["simple", "moderate", "complex"],
                weights=[0.2, 0.5, 0.3],
            ),
        )
    )

    builder.add_column(
        dd.SamplerColumnConfig(
            name="language",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["python", "typescript", "javascript", "rust", "go"],
                weights=[0.35, 0.25, 0.15, 0.15, 0.1],
            ),
        )
    )

    # --- Task prompt ---

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="task_prompt",
            model_alias="text-model",
            prompt=(
                "Seed: {{ seed_task }}\n"
                "Category: {{ task_category }}\n"
                "Complexity: {{ complexity }}\n"
                "Language: {{ language }}\n\n"
                "Generate a specific, realistic {{ complexity }} {{ task_category }} "
                "coding task in {{ language }}. Be precise about the files and "
                "functions involved. Output ONLY the task prompt."
            ),
        )
    )

    # --- Two independent solutions with different temperatures ---

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="solution_a",
            model_alias="solution-model-a",
            prompt=(
                "You are a coding AI agent. Solve this task step by step, "
                "showing your tool calls (read, edit, bash, write, glob, grep).\n\n"
                "Task: {{ task_prompt }}\n\n"
                "Think through each step carefully."
            ),
        )
    )

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="solution_b",
            model_alias="solution-model-b",
            prompt=(
                "You are a coding AI agent. Solve this task step by step, "
                "showing your tool calls (read, edit, bash, write, glob, grep).\n\n"
                "Task: {{ task_prompt }}\n\n"
                "Think through each step carefully."
            ),
        )
    )

    # --- Judge both solutions ---

    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="judge_a",
            model_alias="judge-model",
            prompt=(
                "Evaluate this coding agent solution:\n\n"
                "Task: {{ task_prompt }}\n\n"
                "Solution:\n{{ solution_a }}"
            ),
            scores=[
                dd.Score(
                    name="quality",
                    description="Overall solution quality",
                    options={
                        "5": "Excellent - correct, complete, and well-structured",
                        "4": "Good - mostly correct with minor issues",
                        "3": "Acceptable - works but has notable gaps",
                        "2": "Below average - significant issues",
                        "1": "Poor - incorrect or incomplete",
                    },
                ),
            ],
        )
    )

    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="judge_b",
            model_alias="judge-model",
            prompt=(
                "Evaluate this coding agent solution:\n\n"
                "Task: {{ task_prompt }}\n\n"
                "Solution:\n{{ solution_b }}"
            ),
            scores=[
                dd.Score(
                    name="quality",
                    description="Overall solution quality",
                    options={
                        "5": "Excellent - correct, complete, and well-structured",
                        "4": "Good - mostly correct with minor issues",
                        "3": "Acceptable - works but has notable gaps",
                        "2": "Below average - significant issues",
                        "1": "Poor - incorrect or incomplete",
                    },
                ),
            ],
        )
    )

    # --- Expression columns to pick chosen/rejected ---

    builder.add_column(
        dd.ExpressionColumnConfig(
            name="chosen",
            expr=(
                "{% if judge_a.quality >= judge_b.quality %}"
                "{{ solution_a }}"
                "{% else %}"
                "{{ solution_b }}"
                "{% endif %}"
            ),
        )
    )

    builder.add_column(
        dd.ExpressionColumnConfig(
            name="rejected",
            expr=(
                "{% if judge_a.quality >= judge_b.quality %}"
                "{{ solution_b }}"
                "{% else %}"
                "{{ solution_a }}"
                "{% endif %}"
            ),
        )
    )

    # --- Filter: only keep pairs where scores differ ---

    builder.add_processor(
        processor_type="filter",
        condition="judge_a.quality != judge_b.quality",
    )

    return builder
