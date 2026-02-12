"""
Tool-Use SFT Pipeline

Generates structured tool-call training data with Pydantic schema enforcement.
Produces realistic multi-turn agent conversations with tool calls, observations,
and agent responses.

Pipeline:
  samplers (tool_focus, num_tools, scenario_type)
  -> LLM structured (conversation as ToolUseConversation)
  -> expression (formatted messages for NeMo training)
  -> LLM judge (interaction_quality)
  -> filter (quality >= 3)
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from bashgym.factory.data_designer import PipelineConfig

try:
    import data_designer.config as dd
    DATA_DESIGNER_AVAILABLE = True
except ImportError:
    DATA_DESIGNER_AVAILABLE = False


# =========================================================================
# Pydantic Models for Structured Tool-Use Output
# =========================================================================

class ToolCallRequest(BaseModel):
    """A single tool call request from the agent."""
    name: Literal["read", "edit", "bash", "write", "glob", "grep"]
    arguments: dict = Field(description="Tool-specific arguments")


class ToolCallResponse(BaseModel):
    """Expected response from a tool call."""
    success: bool
    output: str = Field(description="Tool output or error message")


class ToolUseTurn(BaseModel):
    """A single turn in a tool-use conversation."""
    thought: str = Field(description="Agent's reasoning about what to do next")
    tool_call: ToolCallRequest = Field(description="The tool invocation")
    tool_response: ToolCallResponse = Field(description="Simulated tool output")


class ToolUseConversation(BaseModel):
    """A complete tool-use conversation."""
    user_request: str = Field(description="The user's coding task")
    turns: list[ToolUseTurn] = Field(description="Sequence of tool-use turns")
    final_response: str = Field(description="Agent's summary to the user")


# =========================================================================
# Pipeline Builder
# =========================================================================

def build_tool_use_pipeline(config: PipelineConfig) -> dd.DataDesignerConfigBuilder:
    """Build pipeline for structured tool-use training data.

    Generates realistic multi-turn agent conversations with:
    - Diverse tool focus (read-heavy, edit-heavy, bash-heavy, mixed)
    - Varying conversation lengths (1-5 tool calls)
    - Structured Pydantic output for consistent training format
    - Quality scoring on interaction coherence

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
                alias="main-model",
                model=config.code_model,
                inference_parameters=dd.InferenceParameters(
                    temperature=0.7,
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

    # --- Sampling for diversity ---

    builder.add_column(dd.SamplerColumnConfig(
        name="tool_focus",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["read_heavy", "edit_heavy", "bash_heavy", "mixed"],
            weights=[0.2, 0.3, 0.2, 0.3],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="num_tools",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["1", "2", "3", "4", "5"],
            weights=[0.1, 0.2, 0.3, 0.25, 0.15],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="scenario_type",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "file_exploration", "bug_investigation", "code_modification",
                "project_setup", "test_execution", "dependency_management",
            ],
            weights=[0.15, 0.2, 0.25, 0.15, 0.15, 0.1],
        ),
    ))

    builder.add_column(dd.SamplerColumnConfig(
        name="language",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["python", "typescript", "javascript", "rust", "go"],
            weights=[0.35, 0.25, 0.15, 0.15, 0.1],
        ),
    ))

    # --- Generate structured tool-use conversation ---

    builder.add_column(dd.LLMStructuredColumnConfig(
        name="conversation",
        model_alias="main-model",
        prompt=(
            "Generate a realistic coding AI agent conversation that uses "
            "{{ num_tools }} tool calls. The conversation should focus on "
            "{{ tool_focus }} operations for a {{ scenario_type }} scenario "
            "in {{ language }}.\n\n"
            "Seed context: {{ seed_task }}\n\n"
            "Available tools:\n"
            "- read: Read file contents. Args: {file_path: str}\n"
            "- edit: Edit a file. Args: {file_path: str, old_string: str, new_string: str}\n"
            "- bash: Run a command. Args: {command: str}\n"
            "- write: Create/overwrite a file. Args: {file_path: str, content: str}\n"
            "- glob: Find files by pattern. Args: {pattern: str}\n"
            "- grep: Search file content. Args: {pattern: str, path: str}\n\n"
            "The user request should be a natural coding task. The agent should "
            "think about what to do, make appropriate tool calls, observe realistic "
            "results, and provide a helpful final response."
        ),
        output_format=ToolUseConversation,
    ))

    # --- Format to NeMo messages ---

    builder.add_column(dd.ExpressionColumnConfig(
        name="formatted_response",
        expr=(
            "{% for turn in conversation.turns %}"
            "**Thought:** {{ turn.thought }}\n\n"
            "[{{ turn.tool_call.name }}] "
            "{{ turn.tool_call.arguments | tojson }}\n\n"
            "**Output:**\n```\n{{ turn.tool_response.output }}\n```\n\n"
            "{% endfor %}"
            "{{ conversation.final_response }}"
        ),
    ))

    # --- Quality scoring ---

    builder.add_column(dd.LLMJudgeColumnConfig(
        name="interaction_quality",
        model_alias="judge-model",
        prompt=(
            "Evaluate this tool-use agent interaction:\n\n"
            "User request: {{ conversation.user_request }}\n\n"
            "Agent response:\n{{ formatted_response }}"
        ),
        scores=[
            dd.Score(
                name="coherence",
                description="Are the tool calls logically sequenced and do observations make sense?",
                options={
                    "5": "Perfect logical flow, realistic observations",
                    "4": "Minor gaps but overall coherent",
                    "3": "Acceptable but some illogical steps",
                    "2": "Confusing tool sequence",
                    "1": "Incoherent",
                },
            ),
            dd.Score(
                name="tool_appropriateness",
                description="Are the right tools used for the task?",
                options={
                    "5": "Optimal tool selection throughout",
                    "4": "Good choices, minor inefficiencies",
                    "3": "Acceptable, some wrong tools",
                    "2": "Frequently wrong tool choice",
                    "1": "Completely wrong tools",
                },
            ),
        ],
    ))

    # --- Filter low quality ---

    builder.add_processor(
        processor_type="filter",
        condition="interaction_quality.coherence >= 3 and interaction_quality.tool_appropriateness >= 3",
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
