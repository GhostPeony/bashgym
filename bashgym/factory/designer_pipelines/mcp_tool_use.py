"""
MCP Tool-Use Pipeline

Generates *real* tool-use trajectories: an LLM column is granted the BashGym
sandbox MCP tools (bash/read_file/write_file/edit_file/grep/list_files) and
actually executes them in a Docker (or guarded-local) workspace while solving a
generated coding task. Unlike ``tool_use_sft`` (which produces a simulated
structured conversation), here the tool calls and observations are genuine.

Pipeline:
  samplers (task_category, language)
  -> LLM text (task_prompt)            [no tools]
  -> LLM text (agent_transcript)       [tool_alias=sandbox -> real execution]
  -> LLM judge (quality_score)
  -> expression (passes_quality)

Requires data-designer>=0.6.x with MCP support (ToolConfig/LocalStdioMCPProvider).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bashgym.factory.designer_pipelines import (
    HAS_MCP,
    build_base_config,
    build_sandbox_tool_config,
)

if TYPE_CHECKING:
    from bashgym.factory.data_designer import PipelineConfig

try:
    import data_designer.config as dd
except ImportError:
    pass


def build_mcp_tool_use_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build the real-tool-use pipeline (MCP-backed sandbox execution)."""
    if not HAS_MCP:
        raise RuntimeError(
            "mcp_tool_use requires data-designer>=0.6.x with MCP support "
            "(ToolConfig / LocalStdioMCPProvider)."
        )

    # This pipeline always uses tools; flag it so the DataDesigner instance
    # attaches the sandbox MCP provider (build_mcp_providers reads enable_tools).
    config.enable_tools = True
    builder = build_base_config(config, tool_configs=[build_sandbox_tool_config(config)])

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
            name="language",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["python", "bash", "javascript"],
                weights=[0.6, 0.2, 0.2],
            ),
        )
    )

    # --- Task prompt (no tools) ---
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="task_prompt",
            model_alias="text-model",
            prompt=(
                "Generate a small, self-contained {{ language }} coding task "
                "({{ task_category }}) that can be completed from scratch in an empty "
                "workspace using shell and file tools. State the task in 1-3 sentences; "
                "output only the task."
            ),
        )
    )

    # --- Agent transcript WITH real tools ---
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="agent_transcript",
            model_alias="code-model",
            tool_alias=config.mcp_tool_alias,
            prompt=(
                "You are a coding agent with these tools in a fresh workspace: "
                "bash, read_file, write_file, edit_file, grep, list_files.\n\n"
                "Task: {{ task_prompt }}\n\n"
                "Actually solve it by calling the tools (create files, run commands, "
                "verify with bash). Then give a brief summary of what you did and how "
                "you verified it. Use real tool calls, not pretend ones."
            ),
        )
    )

    # --- Quality scoring ---
    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="quality_score",
            model_alias="judge-model",
            prompt=(
                "Evaluate this agent's real tool-use solution.\n\n"
                "Task: {{ task_prompt }}\n\nTranscript:\n{{ agent_transcript }}"
            ),
            scores=[
                dd.Score(
                    name="task_success",
                    description="Did the agent actually accomplish the task?",
                    options={
                        "5": "Fully solved and verified with tools",
                        "4": "Solved with minor gaps",
                        "3": "Partially solved",
                        "2": "Attempted but mostly unsuccessful",
                        "1": "Did not solve the task",
                    },
                ),
                dd.Score(
                    name="tool_use",
                    description="Did the agent use the tools effectively and realistically?",
                    options={
                        "5": "Excellent, realistic tool use with verification",
                        "4": "Good tool use",
                        "3": "Acceptable tool use",
                        "2": "Weak or illogical tool use",
                        "1": "Little or no real tool use",
                    },
                ),
            ],
        )
    )

    # --- Quality flag (filtered at export) ---
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="passes_quality",
            dtype="bool",
            expr="{{ quality_score.task_success.score >= 3 and quality_score.tool_use.score >= 3 }}",
        )
    )

    return builder
