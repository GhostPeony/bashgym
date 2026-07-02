"""Session Distillation Hint Pipeline.

Optional Data Designer model-reader path for BashGym Session Distillation.
The default production path is the deterministic heuristic reader in
``bashgym.factory.session_distillation``; this pipeline is for teams that want a
model to propose or audit targeted hints from trace failure rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from bashgym.factory.designer_pipelines import build_base_config

if TYPE_CHECKING:
    from bashgym.factory.data_designer import PipelineConfig

try:
    import data_designer.config as dd
except ImportError:
    pass


class SessionDistillationReaderOutput(BaseModel):
    """Structured model-reader output for one failed trace span."""

    hint_text: str = Field(
        ...,
        description="A short corrective hint that should be inserted before the target action.",
    )
    mistake_type: Literal[
        "missing_command",
        "missing_path",
        "missing_dependency",
        "permission_error",
        "syntax_error",
        "failed_action",
    ] = Field(..., description="Local mistake class found in the failed action.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Reader confidence that the hint is useful."
    )
    reason: str = Field(..., description="Brief evidence for why this hint targets the failure.")


def build_session_distillation_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build a model-reader pipeline for Session Distillation hints.

    Expected seed columns:
    ``trace_id``, ``session_id``, ``step_index``, ``original_context``,
    ``target_text``, ``target_type``, ``verifier_outcome``, and
    ``failure_reason``.
    """

    builder = build_base_config(config)

    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="session_distillation_hint",
            model_alias="code-model",
            output_format=SessionDistillationReaderOutput,
            system_prompt=(
                "You are a careful coding-agent trace reader. Your job is to find the "
                "smallest useful hint for a failed action. Do not rewrite the action. "
                "Do not invent missing context. Produce one concise hint that would help "
                "the same model rescore the same target tokens under better context."
            ),
            prompt=(
                "Original context:\n{{ original_context }}\n\n"
                "Target action tokens:\n{{ target_text }}\n\n"
                "Target type: {{ target_type }}\n"
                "Verifier outcome: {{ verifier_outcome }}\n"
                "Failure evidence:\n{{ failure_reason }}\n\n"
                "Return the local hint, mistake type, confidence, and evidence."
            ),
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="hint_text",
            expr="{{ session_distillation_hint.hint_text }}",
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="reader_confidence",
            dtype="float",
            expr="{{ session_distillation_hint.confidence }}",
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="recommended_for_session_distillation",
            dtype="bool",
            expr="{{ session_distillation_hint.confidence >= 0.6 }}",
        )
    )

    return builder
