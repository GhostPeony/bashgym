"""Contamination filtering for executable environment specs."""

from __future__ import annotations

from bashgym.datasets.decontaminate import DecontaminationReport, Decontaminator
from bashgym.environments.contracts import EnvironmentSpec


def environment_text(env: EnvironmentSpec) -> str:
    """Text surface used for benchmark-overlap checks."""
    parts = [
        env.id,
        env.instruction,
        env.domain,
        " ".join(env.skills),
        " ".join(f"{axis.name}:{axis.value}" for axis in env.axes),
    ]
    for path, content in sorted(env.files.items()):
        parts.append(path)
        parts.append(content[:2000])
    return "\n".join(p for p in parts if p)


def filter_contaminated_environments(
    envs: list[EnvironmentSpec],
    benchmark_texts: list[str],
    *,
    big_n: int = 13,
    small_n: int = 3,
    jaccard_threshold: float = 0.7,
) -> tuple[list[EnvironmentSpec], DecontaminationReport]:
    """Drop environments overlapping benchmark text via n-gram gates."""
    decontaminator = Decontaminator(
        benchmark_texts,
        big_n=big_n,
        small_n=small_n,
        jaccard_threshold=jaccard_threshold,
    )
    return decontaminator.filter(envs, text_of=environment_text)
