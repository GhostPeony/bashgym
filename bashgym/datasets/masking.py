"""Loss masking policy for SFT: train on the agent's actions, not the observations.

Standard practice for agentic SFT (SWE-World et al.): compute loss only on assistant
tokens — never on user prompts, tool outputs, or system context — so the model learns
to ACT, not to predict the environment's observations. This module is the single
source of truth for that policy: which turns are trainable, and a guard the export
uses to confirm observations are masked. The SFT trainer applies it at the token level
(Unsloth ``train_on_responses_only`` / TRL completion-only collator).
"""

from __future__ import annotations

TRAINABLE_ROLES = ("assistant",)
OBSERVATION_ROLES = ("system", "user", "tool")


def trainable_message_mask(messages) -> list[bool]:
    """Per-message mask: True where loss should be computed (assistant turns only)."""
    return [isinstance(m, dict) and m.get("role") in TRAINABLE_ROLES for m in messages]


def observations_masked(messages) -> bool:
    """True iff the example has trainable assistant content AND no observation turn
    (system/user/tool) is trainable — the invariant an SFT export must satisfy.
    """
    mask = trainable_message_mask(messages)
    if not any(mask):
        return False  # nothing for the model to learn to produce
    for m, trainable in zip(messages, mask):
        role = m.get("role") if isinstance(m, dict) else None
        if trainable and role in OBSERVATION_ROLES:
            return False
    return True
