"""
Dataset format contracts for bashgym training methods.

Defines the exact schema each training strategy expects so converters
and validators have a single source of truth.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class DatasetFormat(Enum):
    """Supported training dataset formats."""

    SFT = "sft"               # Supervised fine-tuning (OpenAI messages format)
    DPO = "dpo"               # Direct Preference Optimization
    GRPO = "grpo"             # Group Relative Policy Optimization
    DISTILLATION = "distillation"  # Teacher → student knowledge transfer


@dataclass
class FieldSpec:
    """Specification for a single dataset field."""

    name: str
    type: type | tuple[type, ...]
    required: bool = True
    description: str = ""
    validator: Optional[Callable] = None  # Optional fn(value) -> bool


@dataclass
class FormatContract:
    """Schema contract for a dataset format."""

    format: DatasetFormat
    fields: list[FieldSpec]
    description: str = ""

    def required_fields(self) -> list[str]:
        return [f.name for f in self.fields if f.required]

    def optional_fields(self) -> list[str]:
        return [f.name for f in self.fields if not f.required]

    def field_by_name(self, name: str) -> FieldSpec | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None


# =========================================================================
# Validators
# =========================================================================


def _is_message_list(v) -> bool:
    """Check value is a list of messages with role+content.

    Accepts both:
      - Plain string content: {"role": "user", "content": "hi"}
      - Multimodal list content: {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    """
    if not isinstance(v, list) or len(v) == 0:
        return False
    valid_roles = {"system", "user", "assistant", "tool"}
    for msg in v:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") not in valid_roles:
            return False
        # content can be None for assistant with tool_calls
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            continue
        content = msg.get("content")
        # Accept string content
        if isinstance(content, str):
            continue
        # Accept multimodal list-of-parts content
        if isinstance(content, list):
            if not all(isinstance(p, dict) and "type" in p for p in content):
                return False
            continue
        return False
    return True


def _is_nonempty_string(v) -> bool:
    return isinstance(v, str) and len(v.strip()) > 0


def _is_string_or_messages(v) -> bool:
    """GRPO prompt can be a string OR a list of messages."""
    if isinstance(v, str):
        return len(v.strip()) > 0
    if isinstance(v, list):
        return _is_message_list(v)
    return False


# =========================================================================
# Format Contracts
# =========================================================================


SFT_CONTRACT = FormatContract(
    format=DatasetFormat.SFT,
    description="Supervised fine-tuning with OpenAI-format messages and tool calls",
    fields=[
        FieldSpec(
            name="messages",
            type=list,
            required=True,
            description="List of {role, content, tool_calls?} dicts",
            validator=_is_message_list,
        ),
        FieldSpec(
            name="tools",
            type=list,
            required=False,
            description="Tool schemas available to the model",
        ),
        FieldSpec(
            name="metadata",
            type=dict,
            required=False,
            description="Provenance and other metadata",
        ),
    ],
)


GRPO_CONTRACT = FormatContract(
    format=DatasetFormat.GRPO,
    description="GRPO training with prompts and optional verification info",
    fields=[
        FieldSpec(
            name="prompt",
            type=(str, list),
            required=True,
            description="Either a plain text prompt OR a list of chat messages",
            validator=_is_string_or_messages,
        ),
        FieldSpec(
            name="tests",
            type=str,
            required=False,
            description="Test code for verification reward (optional)",
        ),
        FieldSpec(
            name="reward_target",
            type=(str, dict),
            required=False,
            description="Optional gold answer or expected behavior",
        ),
        FieldSpec(
            name="metadata",
            type=dict,
            required=False,
            description="Provenance and domain info",
        ),
    ],
)


DPO_CONTRACT = FormatContract(
    format=DatasetFormat.DPO,
    description="Direct Preference Optimization with chosen/rejected pairs",
    fields=[
        FieldSpec(
            name="prompt",
            type=(str, list),
            required=True,
            description="The prompt that produced both responses",
            validator=_is_string_or_messages,
        ),
        FieldSpec(
            name="chosen",
            type=(str, list),
            required=True,
            description="Preferred response (text or message list)",
            validator=_is_string_or_messages,
        ),
        FieldSpec(
            name="rejected",
            type=(str, list),
            required=True,
            description="Less preferred response",
            validator=_is_string_or_messages,
        ),
        FieldSpec(
            name="metadata",
            type=dict,
            required=False,
            description="Why this pair was selected",
        ),
    ],
)


DISTILLATION_CONTRACT = FormatContract(
    format=DatasetFormat.DISTILLATION,
    description="Teacher response distillation training",
    fields=[
        FieldSpec(
            name="prompt",
            type=(str, list),
            required=True,
            description="Input prompt",
            validator=_is_string_or_messages,
        ),
        FieldSpec(
            name="teacher_response",
            type=str,
            required=True,
            description="Teacher model output to distill into student",
            validator=_is_nonempty_string,
        ),
        FieldSpec(
            name="messages",
            type=list,
            required=False,
            description="Optional structured messages",
        ),
        FieldSpec(
            name="metadata",
            type=dict,
            required=False,
        ),
    ],
)


CONTRACTS: dict[DatasetFormat, FormatContract] = {
    DatasetFormat.SFT: SFT_CONTRACT,
    DatasetFormat.GRPO: GRPO_CONTRACT,
    DatasetFormat.DPO: DPO_CONTRACT,
    DatasetFormat.DISTILLATION: DISTILLATION_CONTRACT,
}


def get_contract(format: DatasetFormat | str) -> FormatContract:
    """Look up a contract by format enum or string name."""
    if isinstance(format, str):
        format = DatasetFormat(format.lower())
    return CONTRACTS[format]
