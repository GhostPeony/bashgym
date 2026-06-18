"""Mix self-collected traces with public datasets at a configurable ratio.

Best practice (SWE-Master / Nemotron recipes): keep the self-collected slice at a
target fraction (~10-30%) of the SFT mix — small enough that the public bulk teaches
verified issue-fixing, large enough that the model keeps our workflow style. If the
self pool is too small for the ratio, upsample it (with repetition).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class MixReport:
    n_self_in: int
    n_public_in: int
    n_self_out: int
    n_public_out: int
    self_fraction: float


def mix_datasets(
    self_examples,
    public_examples,
    *,
    self_fraction: float = 0.2,
    upsample: bool = True,
    seed: int = 0,
):
    """Return ``(mixed, report)`` where self-examples are ~``self_fraction`` of total.

    Uses all public examples and sizes the self slice to hit the target fraction,
    upsampling (with repetition) when the self pool is too small and ``upsample``.
    """
    if not 0.0 < self_fraction < 1.0:
        raise ValueError("self_fraction must be in (0, 1)")

    self_examples = list(self_examples)
    public_examples = list(public_examples)
    rng = random.Random(seed)
    n_pub = len(public_examples)

    if n_pub == 0:
        return self_examples, MixReport(len(self_examples), 0, len(self_examples), 0, 1.0)

    # target_self / (target_self + n_pub) = self_fraction
    target_self = round(self_fraction / (1 - self_fraction) * n_pub)

    if len(self_examples) >= target_self:
        chosen_self = rng.sample(self_examples, target_self)
    elif upsample and self_examples:
        chosen_self = [rng.choice(self_examples) for _ in range(target_self)]
    else:
        chosen_self = self_examples

    mixed = public_examples + chosen_self
    rng.shuffle(mixed)
    total = len(mixed)
    return mixed, MixReport(
        n_self_in=len(self_examples),
        n_public_in=n_pub,
        n_self_out=len(chosen_self),
        n_public_out=n_pub,
        self_fraction=(len(chosen_self) / total) if total else 0.0,
    )
