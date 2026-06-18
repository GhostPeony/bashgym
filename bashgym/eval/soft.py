"""SERA-style soft/graded scoring for agent tool-call trajectories.

Binary exact-match is too sparse for agent eval and RL: a trajectory that gets
4 of 5 steps right scores the same 0 as one that gets none, so the gradient
between "almost solved it" and "did nothing" is lost. Following SERA (Soft-
Verified Efficient Repository Agents, arXiv:2601.20789), this grades a whole
tool-call SEQUENCE with continuous partial credit — usable both as a held-out
eval metric and, later, as a dense GRPO reward.

Per-call credit gates argument credit on the tool name: calling the wrong tool
earns nothing (its args are incomparable), while the right tool with imperfect
args earns a base credit plus its argument F1. Trajectory credit aligns gold and
predicted steps positionally and normalizes by the longer length, so both
omitted steps and hallucinated extra steps cost credit symmetrically.
"""

from __future__ import annotations

from .metrics import score_tool_call


def soft_call_score(predicted: dict, gold: dict, *, name_weight: float = 0.4) -> float:
    """Graded [0, 1] score for one predicted call vs gold.

    Wrong tool -> 0.0 (arguments of a different tool are not comparable). Right
    tool -> ``name_weight`` (credit for selecting correctly) plus the remaining
    ``1 - name_weight`` scaled by per-argument F1. Exact match -> 1.0.
    """
    s = score_tool_call(predicted, gold)
    if not s["name_match"]:
        return 0.0
    return name_weight + (1.0 - name_weight) * s["arg_f1"]


def soft_trajectory_score(
    predicted_calls: list[dict], gold_calls: list[dict], *, name_weight: float = 0.4
) -> float:
    """Graded [0, 1] score for a predicted tool-call SEQUENCE vs the gold sequence.

    Positional alignment: gold step ``i`` is graded against predicted step ``i``.
    The sum of per-step credit is divided by ``max(len(gold), len(predicted))`` so
    that missing steps (predicted too short) and extra steps (predicted too long)
    are both penalized. Two empty sequences score 1.0; one empty scores 0.0.
    """
    if not gold_calls and not predicted_calls:
        return 1.0
    n = max(len(gold_calls), len(predicted_calls))
    if n == 0:
        return 1.0
    total = 0.0
    for pred, gold in zip(predicted_calls, gold_calls):
        total += soft_call_score(pred, gold, name_weight=name_weight)
    return total / n
