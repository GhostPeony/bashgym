"""Benchmark decontamination for training data.

Before mixing in public datasets (or exporting), drop any example that overlaps a
benchmark we evaluate on. The 2026 standard: zero shared 13-grams and <0.7 3-gram
Jaccard versus the benchmark corpus (an optional embedding-cosine gate can be added
on top). Without this, training on data that leaks SWE-bench/HumanEval inflates the
very scores we use to decide "is it better?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens for n-gram comparison."""
    return re.findall(r"\w+", text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple]:
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class DecontaminationReport:
    kept: int = 0
    dropped: int = 0
    drop_reasons: dict = field(default_factory=dict)


class Decontaminator:
    """Flags/removes examples overlapping a benchmark corpus via n-gram gates."""

    def __init__(
        self,
        benchmark_texts,
        *,
        big_n: int = 13,
        small_n: int = 3,
        jaccard_threshold: float = 0.7,
    ):
        self.big_n = big_n
        self.small_n = small_n
        self.jaccard_threshold = jaccard_threshold
        self._bench_big: set[tuple] = set()
        self._bench_small_sets: list[set[tuple]] = []
        for t in benchmark_texts:
            toks = _tokens(t)
            self._bench_big |= ngrams(toks, big_n)
            small = ngrams(toks, small_n)
            if small:
                self._bench_small_sets.append(small)

    def contamination_reason(self, text: str) -> str | None:
        """Return a reason string if ``text`` is contaminated, else None."""
        toks = _tokens(text)
        # Any shared long n-gram is an exact leak.
        if ngrams(toks, self.big_n) & self._bench_big:
            return "13gram_overlap"
        # High 3-gram Jaccard against any single benchmark item.
        ex_small = ngrams(toks, self.small_n)
        if ex_small:
            for b in self._bench_small_sets:
                jaccard = len(ex_small & b) / len(ex_small | b)
                if jaccard >= self.jaccard_threshold:
                    return "3gram_jaccard"
        return None

    def filter(self, examples, text_of) -> tuple[list, DecontaminationReport]:
        """Split ``examples`` into kept (clean) + a drop report. ``text_of(ex)->str``."""
        kept: list = []
        report = DecontaminationReport()
        for ex in examples:
            reason = self.contamination_reason(text_of(ex))
            if reason:
                report.dropped += 1
                report.drop_reasons[reason] = report.drop_reasons.get(reason, 0) + 1
            else:
                kept.append(ex)
        report.kept = len(kept)
        return kept, report
