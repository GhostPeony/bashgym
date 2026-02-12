"""
Unstructured Data Pipeline

Processes raw code files, documentation, and repositories into seed data
that feeds the DataDesigner pipeline. Converts unstructured content into
coding agent training examples.

Supports:
- Source code files (.py, .ts, .js, .rs, .go, etc.)
- Documentation (.md, .txt, .rst)
- Mixed directories (code repos)

Pipeline:
  seed (extracted file content + metadata)
  -> samplers (task_type, interaction_depth)
  -> LLM text (task_from_code, context_summary)
  -> LLM text (agent_solution)
  -> LLM judge (quality_score)
  -> filter (quality >= 3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bashgym.factory.data_designer import PipelineConfig

try:
    import data_designer.config as dd
    DATA_DESIGNER_AVAILABLE = True
except ImportError:
    DATA_DESIGNER_AVAILABLE = False


def build_unstructured_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build pipeline for unstructured data to training examples.

    Takes raw code and documentation as seed, then generates realistic
    coding agent training data grounded in real codebase patterns.

    Seed columns expected (from DataDesignerPipeline._extract_seeds_from_unstructured):
    - seed_task: Brief description of the file
    - seed_context: File contents (truncated)
    - seed_language: Detected language
    - seed_file_type: File extension

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

    # --- Task type diversity ---

    builder.add_column(dd.SamplerColumnConfig(
        name="task_type",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "explain_code",       # Explain what this code does
                "fix_bug",            # Find and fix a bug in this code
                "add_feature",        # Add a feature to this code
                "write_tests",        # Write tests for this code
                "refactor",           # Refactor this code
                "optimize",           # Optimize performance
                "add_docs",           # Add documentation
            ],
            weights=[0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="interaction_depth",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["shallow", "moderate", "deep"],
            weights=[0.2, 0.5, 0.3],
        ),
    ))

    # --- Generate a realistic task from the code ---

    builder.add_column(dd.LLMTextColumnConfig(
        name="task_prompt",
        model_alias="text-model",
        prompt=(
            "You are creating a coding task based on real code.\n\n"
            "File type: {{ seed_language }} ({{ seed_file_type }})\n"
            "Task type: {{ task_type }}\n"
            "Interaction depth: {{ interaction_depth }}\n\n"
            "Code:\n```{{ seed_language }}\n{{ seed_context[:3000] }}\n```\n\n"
            "Generate a realistic task prompt that a developer would give to an "
            "AI coding assistant about this code. The task should:\n"
            "- Be a {{ task_type }} task\n"
            "- Have {{ interaction_depth }} depth (shallow=1-2 steps, "
            "moderate=3-5 steps, deep=5+ steps)\n"
            "- Reference specific functions, classes, or patterns from the code\n"
            "- Be actionable and specific\n\n"
            "Output ONLY the task prompt."
        ),
    ))

    # --- Summarize the code context ---

    builder.add_column(dd.LLMTextColumnConfig(
        name="context_summary",
        model_alias="text-model",
        prompt=(
            "Briefly summarize this {{ seed_language }} code in 2-3 sentences. "
            "Focus on its purpose, key components, and structure.\n\n"
            "```{{ seed_language }}\n{{ seed_context[:3000] }}\n```"
        ),
    ))

    # --- Generate agent solution ---

    builder.add_column(dd.LLMTextColumnConfig(
        name="solution",
        model_alias="code-model",
        prompt=(
            "You are a coding AI agent working with {{ seed_language }} code.\n\n"
            "Context: {{ context_summary }}\n\n"
            "Task: {{ task_prompt }}\n\n"
            "Solve this step by step using tool calls. Available tools:\n"
            "- [read] file_path -- Read file contents\n"
            "- [edit] file_path: old -> new -- Edit a file\n"
            "- [bash] command -- Run a shell command\n"
            "- [write] file_path -- Create/overwrite a file\n"
            "- [grep] pattern path -- Search for content\n\n"
            "Show your reasoning at each step."
        ),
    ))

    # --- Quality scoring ---

    builder.add_column(dd.LLMJudgeColumnConfig(
        name="quality_score",
        model_alias="judge-model",
        prompt=(
            "Evaluate this code-grounded training example:\n\n"
            "Context: {{ context_summary }}\n\n"
            "Task: {{ task_prompt }}\n\n"
            "Solution:\n{{ solution }}"
        ),
        scores=[
            dd.Score(
                name="grounding",
                description="Is the task grounded in the actual code?",
                options={
                    "5": "Directly references specific code elements",
                    "4": "Well-grounded with minor generic parts",
                    "3": "Partially grounded",
                    "2": "Mostly generic, loosely connected",
                    "1": "Not grounded in the code at all",
                },
            ),
            dd.Score(
                name="solution_quality",
                description="Is the solution correct and complete?",
                options={
                    "5": "Correct, complete, well-reasoned",
                    "4": "Minor issues, overall good",
                    "3": "Acceptable but incomplete",
                    "2": "Significant issues",
                    "1": "Incorrect or irrelevant",
                },
            ),
        ],
    ))

    # --- Filter ---

    builder.add_processor(
        processor_type="filter",
        condition="quality_score.grounding >= 3 and quality_score.solution_quality >= 3",
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
