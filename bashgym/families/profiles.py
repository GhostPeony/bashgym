"""Declarative per-family training recipes.

A ``ModelFamilyProfile`` is the single source of every model-family-specific fact
the trainer, exporter, and evaluator need to fine-tune, export, and serve a base
model correctly: chat/tool-call format, LoRA target/exclude modules, attention
implementation, named correctness patches, GGUF template source, and the default
training backend.

Supporting a new open model = add one ``ModelFamilyProfile`` to ``REGISTRY`` —
no edits to trainer/export/eval code.

NOTE: distinct from ``bashgym.models.ModelProfile``, which is metadata about an
*already-trained* model artifact. This is the recipe for training a base family.
"""

from __future__ import annotations

from dataclasses import dataclass

# Standard attention + MLP projection modules — shared by Gemma/Qwen/Llama/Mistral.
_DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# Towers to skip on multimodal checkpoints so LoRA only touches the text stack.
_MULTIMODAL_EXCLUDES = ("vision_tower", "multi_modal_projector", "audio_tower")


@dataclass(frozen=True)
class ModelFamilyProfile:
    """Everything family-specific needed to train/export/serve a base model."""

    family: str
    # Lowercased substrings; if any is found in the model id, this profile matches.
    match: tuple[str, ...]
    tool_call_format: str = "openai_json"  # openai_json|gemma4_delimited|qwen_xml|hermes
    lora_target_modules: tuple[str, ...] = _DEFAULT_LORA_TARGETS
    lora_exclude_modules: tuple[str, ...] = ()
    attn_implementation: str = (
        "sdpa"  # sdpa works on GB10/sm_121; flash_attention_2 where supported
    )
    dtype: str = "bfloat16"
    patches: tuple[str, ...] = ()  # names resolved by bashgym.families.patches
    thinking: bool = False  # needs a *-thinking chat template
    chat_template_override: str | None = None
    default_backend: str = "auto"  # auto|unsloth|plain|trl_vllm
    gguf_template_source: str = "base"  # "base" = pull chat template from base tokenizer
    stop_tokens: tuple[str, ...] = ()

    def matches(self, model_id: str) -> bool:
        mid = model_id.lower()
        return any(m in mid for m in self.match)


GEMMA4 = ModelFamilyProfile(
    family="gemma4",
    match=("gemma-4", "gemma4"),
    tool_call_format="gemma4_delimited",
    lora_exclude_modules=_MULTIMODAL_EXCLUDES,
    patches=("gemma4_clippable_linear",),
    thinking=True,
)

QWEN3 = ModelFamilyProfile(
    family="qwen3",
    match=("qwen3", "qwen-3"),
    tool_call_format="qwen_xml",
    thinking=True,
)

QWEN25 = ModelFamilyProfile(
    family="qwen2.5",
    match=("qwen2.5", "qwen2_5", "qwen-2.5"),
    tool_call_format="qwen_xml",
)

LLAMA3 = ModelFamilyProfile(
    family="llama3",
    match=("llama-3", "llama3"),
    tool_call_format="openai_json",
)

# Explicit fallback: empty match tuple so it never matches in the loop; returned last.
GENERIC = ModelFamilyProfile(
    family="generic",
    match=(),
    tool_call_format="openai_json",
)

# Order matters: most specific first, GENERIC is the explicit fallback.
REGISTRY: tuple[ModelFamilyProfile, ...] = (GEMMA4, QWEN3, QWEN25, LLAMA3, GENERIC)


def resolve_family_profile(base_model: str) -> ModelFamilyProfile:
    """Resolve a base-model id (e.g. 'google/gemma-4-31B-it') to its family profile.

    Returns ``GENERIC`` when no specific family matches.
    """
    for profile in REGISTRY:
        if profile.match and profile.matches(base_model):
            return profile
    return GENERIC
