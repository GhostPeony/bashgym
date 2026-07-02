"""
Coding Agent Trace-Distillation Pipeline

Distills native agent rollouts (Claude Code, Codex, Hermes, Pi, ATIF) into
standalone SFT instruction/response pairs, modeled on NVIDIA's official
"Agent Rollout Trace Distillation" recipe.

The seed dataset is attached by ``DataDesignerPipeline.from_agent_rollouts``
(an ``AgentRolloutSeedSource``), which normalizes every rollout format into the
same row schema (``messages``, ``final_assistant_message``, ``project_path``,
``tool_call_count``, ...). This pipeline turns each rollout into a training
example via:

  AgentRolloutSeedSource (raw rollout rows)
  -> LLM structured (trace_digest as AgentRolloutTraceDigest)
  -> LLM structured (sft_record as AgentRolloutFinetuningRecord)
  -> LLM judge (sft_quality_judge_result, 5 dimensions 0-4)
  -> expressions (sft_instruction / sft_response / sft_skill_tags /
     trace_training_value / recommended_for_sft)

Model aliases reuse the shared base config (``text-model`` for digesting,
``code-model`` for the SFT record, ``judge-model`` for scoring). For a true
distillation run, point all three at a strong teacher (e.g.
``nvidia/nemotron-3-super-120b-a12b`` or a private compute-served model) via PipelineConfig.
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


# =========================================================================
# Pydantic Models for Structured Output (from the NVIDIA recipe)
# =========================================================================


class AgentRolloutTraceDigest(BaseModel):
    """Structured summary of one agent rollout."""

    user_goal: str = Field(
        ..., description="Standalone summary of the concrete user or delegated agent task."
    )
    repository_context: str = Field(
        ..., description="The repo, codebase, or environment context that shaped the task."
    )
    task_type: str = Field(..., description="Short label for the kind of work in the trace.")
    notable_actions: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Most important assistant actions, tools, or repo operations from the trace.",
    )
    useful_outcome: str = Field(
        ..., description="The most useful result, conclusion, or next step learned from the trace."
    )
    training_value: Literal["high", "medium", "low"] = Field(
        ..., description="Whether this trace is a good source for assistant fine-tuning."
    )
    quality_notes: str = Field(
        ...,
        description="Short note about anything that makes the trace useful, narrow, noisy, or partial.",
    )


class AgentRolloutFinetuningRecord(BaseModel):
    """A standalone SFT instruction/response distilled from a rollout."""

    instruction: str = Field(
        ..., description="A standalone user request suitable for SFT of a coding assistant."
    )
    response: str = Field(
        ...,
        description="A grounded assistant response that helps with the instruction without inventing unsupported details.",
    )
    skill_tags: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Short tags describing the skills demonstrated in the example.",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Approximate difficulty of the resulting training example."
    )


def _sft_judge_scores() -> list:
    """Five-dimension SFT quality rubric (0-4), from the NVIDIA recipe."""
    return [
        dd.Score(
            name="groundedness",
            description="Is the candidate example grounded in the trace digest rather than generic filler?",
            options={
                4: "Strongly grounded in the trace digest with concrete task fidelity.",
                3: "Mostly grounded but slightly generic or overgeneralized.",
                2: "Partially grounded but missing important trace-specific substance.",
                1: "Weakly grounded and mostly generic.",
                0: "Not grounded in the trace digest.",
            },
        ),
        dd.Score(
            name="standalone_task",
            description="Would a new reader understand the instruction without seeing the trace?",
            options={
                4: "Fully standalone and immediately understandable.",
                3: "Mostly standalone with minor missing context.",
                2: "Understandable but noticeably dependent on hidden trace context.",
                1: "Hard to understand without the trace.",
                0: "Not standalone.",
            },
        ),
        dd.Score(
            name="response_quality",
            description="How helpful, technically specific, and instruction-following is the response?",
            options={
                4: "Highly useful, technically specific, and directly responsive.",
                3: "Useful overall with minor omissions or verbosity.",
                2: "Partially helpful but shallow, vague, or uneven.",
                1: "Low-quality response with major gaps.",
                0: "Unhelpful or incorrect response.",
            },
        ),
        dd.Score(
            name="faithfulness",
            description="Does the candidate avoid inventing details beyond what the digest justifies?",
            options={
                4: "Faithful to the digest; no meaningful unsupported details invented.",
                3: "Mostly faithful with minor speculative details.",
                2: "Noticeable invented details or overconfident extrapolation.",
                1: "Many unsupported implementation details are fabricated.",
                0: "Severely unfaithful to the digest.",
            },
        ),
        dd.Score(
            name="training_utility",
            description="Would this example be worth keeping in an SFT dataset for a coding assistant?",
            options={
                4: "Very strong SFT example worth keeping.",
                3: "Reasonably useful SFT example.",
                2: "Marginal example; probably not worth the tokens.",
                1: "Poor SFT example.",
                0: "Should not be kept.",
            },
        ),
    ]


# =========================================================================
# Pipeline Builder
# =========================================================================


def build_distill_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build the agent-rollout trace-distillation pipeline.

    Note: the seed dataset (an ``AgentRolloutSeedSource``) is attached by
    ``DataDesignerPipeline.from_agent_rollouts``; this builder only defines the
    column DAG that consumes the normalized rollout fields.

    Args:
        config: PipelineConfig with model/temperature settings.

    Returns:
        Configured DataDesignerConfigBuilder.
    """
    builder = build_base_config(config)

    # --- Step 1: digest the raw rollout into a structured summary ---
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="trace_digest",
            model_alias="text-model",
            output_format=AgentRolloutTraceDigest,
            system_prompt=(
                "You analyze AI coding-agent traces and distill them into faithful, "
                "reusable summaries. Only describe what the trace supports; never invent details."
            ),
            prompt=(
                "Summarize this agent rollout for use as a fine-tuning source.\n\n"
                "Project: {{ project_path }}\n"
                "Tool calls: {{ tool_call_count }}\n"
                "Final assistant message: {{ final_assistant_message }}\n\n"
                "Conversation:\n{{ messages }}"
            ),
        )
    )

    # --- Step 2: produce a standalone SFT instruction/response from the digest ---
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="sft_record",
            model_alias="code-model",
            output_format=AgentRolloutFinetuningRecord,
            system_prompt=(
                "You convert distilled trace summaries into high-quality, standalone "
                "supervised fine-tuning examples for a coding assistant. The instruction "
                "must be understandable on its own; the response must stay grounded in the digest."
            ),
            prompt=(
                "Trace digest:\n"
                "- User goal: {{ trace_digest.user_goal }}\n"
                "- Repository context: {{ trace_digest.repository_context }}\n"
                "- Task type: {{ trace_digest.task_type }}\n"
                "- Notable actions: {{ trace_digest.notable_actions }}\n"
                "- Useful outcome: {{ trace_digest.useful_outcome }}\n"
                "- Quality notes: {{ trace_digest.quality_notes }}\n\n"
                "Write a standalone instruction a developer might ask, and a grounded "
                "assistant response that accomplishes it."
            ),
        )
    )

    # --- Step 3: judge the candidate example on five dimensions (0-4) ---
    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="sft_quality_judge_result",
            model_alias="judge-model",
            system_prompt=(
                "You are a strict reviewer scoring candidate SFT examples distilled from "
                "agent traces. Penalize generic, ungrounded, or unfaithful examples."
            ),
            prompt=(
                "Trace digest:\n"
                "- User goal: {{ trace_digest.user_goal }}\n"
                "- Useful outcome: {{ trace_digest.useful_outcome }}\n\n"
                "Candidate SFT example:\n"
                "- Instruction: {{ sft_record.instruction }}\n"
                "- Response: {{ sft_record.response }}"
            ),
            scores=_sft_judge_scores(),
        )
    )

    # --- Step 4: flatten to training-ready columns ---
    builder.add_column(
        dd.ExpressionColumnConfig(name="sft_instruction", expr="{{ sft_record.instruction }}")
    )
    builder.add_column(
        dd.ExpressionColumnConfig(name="sft_response", expr="{{ sft_record.response }}")
    )
    builder.add_column(
        dd.ExpressionColumnConfig(name="sft_skill_tags", expr="{{ sft_record.skill_tags }}")
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="trace_training_value", expr="{{ trace_digest.training_value }}"
        )
    )

    # recommended_for_sft: every judge dimension >= 4 and the digest rates the trace 'high'.
    # Judge sub-scores are nested in the judge dict column: <judge>.<score>.score.
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="recommended_for_sft",
            dtype="bool",
            expr=(
                "{{ sft_quality_judge_result.groundedness.score >= 4 "
                "and sft_quality_judge_result.standalone_task.score >= 4 "
                "and sft_quality_judge_result.response_quality.score >= 4 "
                "and sft_quality_judge_result.faithfulness.score >= 4 "
                "and sft_quality_judge_result.training_utility.score >= 4 "
                "and trace_training_value == 'high' }}"
            ),
        )
    )

    return builder
