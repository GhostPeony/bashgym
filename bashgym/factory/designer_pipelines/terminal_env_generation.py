"""Terminal environment generation pipeline.

Generates TMax-style executable environment drafts: task instructions, source
files, verifier artifacts, build hints, rollout limits, and quality flags. This
is the Data Designer entry point for producing rows that can later be converted
into ``bashgym.environments.EnvironmentSpec`` bundles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from bashgym.factory.designer_pipelines import build_base_config

if TYPE_CHECKING:
    from bashgym.factory.data_designer import PipelineConfig

try:
    import data_designer.config as dd
except ImportError:
    pass


class EnvironmentFileDraft(BaseModel):
    """One file to include in the generated environment bundle."""

    path: str = Field(description="Relative POSIX path inside the task workspace")
    content: str = Field(description="Complete file content")
    purpose: str = Field(description="Why this file exists in the task")


class TerminalEnvironmentDraft(BaseModel):
    """Structured executable terminal environment draft."""

    summary: str = Field(description="One-sentence environment summary")
    files: list[EnvironmentFileDraft] = Field(description="Workspace files required for the task")
    verifier_path: str = Field(description="Relative path to the verifier script or test file")
    verifier_command: str = Field(description="Command that evaluates task success")
    setup_commands: list[str] = Field(description="Commands needed before rollout/evaluation")
    expected_solution_shape: str = Field(description="What a correct agent likely changes or produces")


def build_terminal_env_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build a terminal-environment generation pipeline."""
    builder = build_base_config(config)

    builder.add_column(
        dd.SamplerColumnConfig(
            name="domain",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "system_administration",
                    "security",
                    "data_processing",
                    "file_operations",
                    "software_engineering",
                    "debugging",
                    "data_querying",
                    "scientific_computing",
                    "data_science",
                ],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="skill_type",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "bash",
                    "python",
                    "debugging",
                    "test_repair",
                    "text_processing",
                    "service_orchestration",
                    "forensics",
                    "data_validation",
                ],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="persona",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "developer repairing a failing CI task",
                    "data engineer cleaning a brittle report",
                    "security analyst validating suspicious files",
                    "researcher reproducing a command-line workflow",
                    "operator debugging a local service",
                ],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="fixture_kind",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["none", "text_files", "csv", "sqlite", "service", "binary_blob"],
                weights=[0.15, 0.25, 0.25, 0.15, 0.1, 0.1],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="language",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["bash", "python", "javascript", "typescript", "go", "rust"],
                weights=[0.2, 0.45, 0.12, 0.12, 0.06, 0.05],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="task_complexity",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["simple", "moderate", "complex"],
                weights=[0.25, 0.55, 0.2],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="command_complexity",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["bash_only", "bash_plus_code", "multi_service"],
                weights=[0.35, 0.5, 0.15],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="verifier_kind",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "exact_success",
                    "metric_threshold",
                    "adversarial_corpus",
                    "fuzz_equivalence",
                    "multi_protocol",
                ],
                weights=[0.35, 0.25, 0.15, 0.15, 0.1],
            ),
        )
    )

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="task_prompt",
            model_alias="text-model",
            prompt=(
                "Generate a realistic terminal-agent task.\n\n"
                "Domain: {{ domain }}\n"
                "Skill focus: {{ skill_type }}\n"
                "Persona: {{ persona }}\n"
                "Fixture kind: {{ fixture_kind }}\n"
                "Language: {{ language }}\n"
                "Task complexity: {{ task_complexity }}\n"
                "Command complexity: {{ command_complexity }}\n"
                "Verifier kind: {{ verifier_kind }}\n\n"
                "Write only the user-facing task instruction. It must be solvable "
                "inside an isolated terminal workspace and should not mention the "
                "hidden verifier."
            ),
        )
    )

    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="environment_draft",
            model_alias="code-model",
            prompt=(
                "Create an executable terminal-agent environment for this task.\n\n"
                "Task: {{ task_prompt }}\n"
                "Domain: {{ domain }}\n"
                "Skill: {{ skill_type }}\n"
                "Fixture: {{ fixture_kind }}\n"
                "Language: {{ language }}\n"
                "Verifier kind: {{ verifier_kind }}\n\n"
                "Return source files and a verifier. Keep dependencies lightweight. "
                "The verifier must be runnable from the workspace and should score "
                "real task success, not just string formatting."
            ),
            output_format=TerminalEnvironmentDraft,
        )
    )

    builder.add_column(
        dd.ExpressionColumnConfig(
            name="environment_summary",
            expr=(
                "Task: {{ task_prompt }}\n"
                "Summary: {{ environment_draft.summary }}\n"
                "Verifier: {{ environment_draft.verifier_command }}\n"
                "Expected solution: {{ environment_draft.expected_solution_shape }}"
            ),
        )
    )

    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="quality_score",
            model_alias="judge-model",
            prompt=(
                "Evaluate this generated executable terminal environment.\n\n"
                "{{ environment_summary }}\n\n"
                "Files:\n{{ environment_draft.files | tojson }}\n"
                "Setup commands: {{ environment_draft.setup_commands | tojson }}"
            ),
            scores=[
                dd.Score(
                    name="executability",
                    description="Can this environment plausibly build/run in an isolated terminal?",
                    options={
                        "5": "Clear, runnable, lightweight, and self-contained",
                        "4": "Runnable with minor assumptions",
                        "3": "Possibly runnable but has gaps",
                        "2": "Likely broken",
                        "1": "Not executable",
                    },
                ),
                dd.Score(
                    name="verifier_quality",
                    description="Does the verifier measure real task success?",
                    options={
                        "5": "Strong verifier with meaningful outcome signal",
                        "4": "Good verifier with small gaps",
                        "3": "Acceptable but shallow",
                        "2": "Weak or gameable",
                        "1": "No meaningful verifier",
                    },
                ),
                dd.Score(
                    name="learnability",
                    description="Is the task neither trivial nor impossible for terminal RL?",
                    options={
                        "5": "Excellent learnable challenge",
                        "4": "Good challenge",
                        "3": "Usable but may need calibration",
                        "2": "Too easy or too hard",
                        "1": "Not useful for training",
                    },
                ),
            ],
        )
    )

    builder.add_column(
        dd.ExpressionColumnConfig(
            name="passes_quality",
            dtype="bool",
            expr=(
                "{{ quality_score.executability.score >= 3 "
                "and quality_score.verifier_quality.score >= 3 "
                "and quality_score.learnability.score >= 3 }}"
            ),
        )
    )

    return builder
