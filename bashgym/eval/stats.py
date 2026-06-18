"""Session-clustered paired bootstrap for held-out eval.

A candidate fine-tune is "better" only if its per-example advantage over the base
survives resampling. Examples from the same session are NOT independent, so the
bootstrap resamples whole sessions (clusters), per Anthropic's "Adding Error Bars
to Evals" (arXiv:2411.00640) — clustering at the unit of correlation keeps the CI
honest instead of falsely tight.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    mean: float  # observed mean paired delta (candidate - base)
    ci_low: float
    ci_high: float
    n: int  # number of paired examples
    n_clusters: int

    @property
    def significant(self) -> bool:
        """The 95% CI excludes zero (candidate reliably differs from base)."""
        return self.ci_low > 0 or self.ci_high < 0

    @property
    def better(self) -> bool:
        """The candidate is reliably better: the whole CI is above zero."""
        return self.ci_low > 0


def paired_bootstrap(
    deltas: list[float],
    clusters: list,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Cluster bootstrap of paired per-example deltas (candidate - base).

    ``deltas[i]`` and ``clusters[i]`` are aligned; ``clusters`` is typically the
    session id. Each resample draws ``n_clusters`` clusters with replacement and
    pools their deltas; the CI is the ``alpha/2 .. 1-alpha/2`` percentiles of the
    resampled means.
    """
    if len(deltas) != len(clusters):
        raise ValueError("deltas and clusters must be the same length")
    if not deltas:
        raise ValueError("no data to bootstrap")

    by_cluster: dict = {}
    for d, c in zip(deltas, clusters):
        by_cluster.setdefault(c, []).append(d)
    cluster_keys = list(by_cluster.keys())

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        pooled: list[float] = []
        for _ in range(len(cluster_keys)):
            pooled.extend(by_cluster[rng.choice(cluster_keys)])
        means.append(sum(pooled) / len(pooled))
    means.sort()

    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    observed = sum(deltas) / len(deltas)
    return BootstrapResult(
        mean=observed, ci_low=lo, ci_high=hi, n=len(deltas), n_clusters=len(cluster_keys)
    )
