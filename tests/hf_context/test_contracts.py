from datetime import datetime

import pytest
from pydantic import ValidationError

from bashgym._compat import UTC
from bashgym.integrations.huggingface.context_contracts import (
    Comparability,
    CompletionOutcome,
    EvalSettings,
    EvidenceAssessment,
    EvidenceKind,
    EvidenceRecord,
    Freshness,
    HFContextBundle,
    Lifecycle,
    Provenance,
    Visibility,
    classify_comparability,
    rank_evidence,
)

NOW = datetime(2026, 7, 10, 12, tzinfo=UTC)


def evidence(
    evidence_id: str,
    *,
    assessment: EvidenceAssessment | None = None,
    popularity: int = 0,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=EvidenceKind.MODEL,
        resource_id=f"org/{evidence_id}",
        revision="a" * 40,
        canonical_url=f"https://huggingface.co/org/{evidence_id}",
        summary="A model used by the fixture.",
        facts={"downloads": popularity},
        provenance=Provenance(source="huggingface_hub", retrieved_at=NOW),
        assessment=assessment or EvidenceAssessment(),
    )


def test_contracts_forbid_unknown_fields_and_freeze_evidence():
    item = evidence("model-a")

    with pytest.raises(ValidationError):
        EvidenceRecord(**{**item.model_dump(), "raw_token": "secret"})
    with pytest.raises(ValidationError):
        item.summary = "mutated"  # type: ignore[misc]


def test_bundle_hash_is_deterministic_and_covers_evidence():
    payload = dict(
        bundle_id="hfctx_fixture",
        version=1,
        workspace_id="workspace-a",
        lifecycle=Lifecycle.READY,
        freshness=Freshness.FRESH,
        completion_outcome=CompletionOutcome.COMPLETE,
        intent="Find a comparable coding model",
        evidence=(evidence("model-a"),),
        selected_evidence_ids=("model-a",),
        created_at=NOW,
    )
    first = HFContextBundle(**payload)
    second = HFContextBundle(**payload)

    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64

    changed = HFContextBundle(**{**payload, "intent": "Find a dataset"})
    assert changed.content_hash != first.content_hash


def test_bundle_rejects_unknown_selected_evidence():
    with pytest.raises(ValidationError, match="selected evidence"):
        HFContextBundle(
            bundle_id="hfctx_fixture",
            version=1,
            workspace_id="workspace-a",
            lifecycle=Lifecycle.READY,
            intent="fixture",
            evidence=(evidence("model-a"),),
            selected_evidence_ids=("missing",),
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, Comparability.COMPARABLE),
        ({"backend": "accelerate"}, Comparability.PARTIAL),
        ({"few_shot": 0}, Comparability.ORIENTATION_ONLY),
        ({"prompt_template": None}, Comparability.ORIENTATION_ONLY),
        ({"harness_version": None, "backend": None}, Comparability.ORIENTATION_ONLY),
    ],
)
def test_eval_comparability_is_conservative(changes, expected):
    reference = EvalSettings(
        benchmark_id="mmlu",
        task_revision="v1",
        metric="accuracy",
        prompt_template="chatml",
        few_shot=5,
        harness="lighteval",
        harness_version="0.8.0",
        backend="vllm",
        sampling={"temperature": 0},
    )
    candidate = reference.model_copy(update=changes)
    assert classify_comparability(candidate, reference) is expected


def test_ranking_uses_assessment_then_popularity_then_stable_id():
    high_relevance = evidence(
        "z-model",
        assessment=EvidenceAssessment(task_relevance=3, compatibility=2, confidence=0.8),
        popularity=10,
    )
    popular = evidence(
        "b-model",
        assessment=EvidenceAssessment(task_relevance=2, compatibility=3, confidence=1.0),
        popularity=1000,
    )
    stable_a = evidence(
        "a-model",
        assessment=EvidenceAssessment(task_relevance=2, compatibility=3, confidence=1.0),
        popularity=1000,
    )
    excluded = evidence(
        "excluded",
        assessment=EvidenceAssessment(
            task_relevance=3,
            compatibility=3,
            confidence=1.0,
            constraint_violations=("license",),
        ),
        popularity=9999,
    )

    ranked = rank_evidence([popular, high_relevance, excluded, stable_a])
    assert [item.evidence_id for item in ranked] == ["z-model", "a-model", "b-model"]


def test_visibility_defaults_public_and_never_contains_secret_fields():
    item = evidence("public")
    assert item.visibility is Visibility.PUBLIC
    assert "token" not in item.model_dump_json().lower()
