"""Immutable evaluation and comparison evidence persistence tests."""

from bashgym.campaigns.evaluation import (
    DevelopmentGateContract,
    compare_development_evaluations,
)
from tests.campaigns.test_evaluation import _artifact, _rows
from tests.campaigns.test_worker import START, active_repository


def test_evaluation_and_gate_evidence_are_content_addressed_and_replay_safe(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    champion = _artifact(digest="a" * 64, rows=_rows(count=18, videos=3, rank=1))
    candidate = _artifact(digest="b" * 64, rows=_rows(count=18, videos=3, rank=2))
    comparison = compare_development_evaluations(
        champion, candidate, DevelopmentGateContract(bootstrap_samples=100)
    )

    champion_id = repository.store_retrieval_evaluation(
        "workspace-a", "campaign-1", champion, now=START
    )
    assert (
        repository.store_retrieval_evaluation(
            "workspace-a", "campaign-1", champion, now=START
        )
        == champion_id
    )
    assert repository.get_retrieval_evaluation("workspace-a", champion_id) == champion

    decision_id = repository.store_development_comparison(
        "workspace-a", "campaign-1", comparison, now=START
    )
    assert (
        repository.store_development_comparison(
            "workspace-a", "campaign-1", comparison, now=START
        )
        == decision_id
    )
    assert repository.get_development_comparison("workspace-a", decision_id) == comparison
