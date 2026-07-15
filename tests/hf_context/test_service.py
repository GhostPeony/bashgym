import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from bashgym.integrations.huggingface.context_contracts import (
    CompletionOutcome,
    EvidenceKind,
    EvidenceRecord,
    Freshness,
    HFContextBundle,
    Lifecycle,
    Provenance,
)
from bashgym.integrations.huggingface.context_persistence import HFContextRepository
from bashgym.integrations.huggingface.context_service import HFContextService

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hf_context"


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeSources:
    def __init__(self, *, fail_datasets: bool = False, model_revision: str | None = None):
        self.fail_datasets = fail_datasets
        self.model_revision = model_revision

    def discover_models(self, query: str, *, limit: int):
        assert query
        model = fixture("rich_model.json")
        if self.model_revision:
            model["sha"] = self.model_revision
        return [model]

    def discover_datasets(self, query: str, *, limit: int):
        if self.fail_datasets:
            raise TimeoutError("dataset source timed out with hf_secret_should_not_leak")
        return [fixture("multi_config_dataset.json")]


def service(tmp_path, *, fail_datasets: bool = False):
    repository = HFContextRepository(tmp_path / "hf-context.sqlite3")
    repository.initialize()
    return HFContextService(repository, sources=FakeSources(fail_datasets=fail_datasets))


def test_discovery_finalizes_ranked_bundle_with_models_datasets_and_evals(tmp_path):
    current = service(tmp_path).discover(
        workspace_id="workspace-a",
        intent="code generation",
        task="text-generation",
    )

    assert current.lifecycle is Lifecycle.READY
    assert current.completion_outcome is CompletionOutcome.COMPLETE
    assert {item.kind.value for item in current.evidence} == {"model", "dataset", "evaluation"}
    assert len(current.evidence) <= 12
    assert all(item.canonical_url.startswith("https://") for item in current.evidence)


def test_partial_source_failure_keeps_usable_results_and_safe_status(tmp_path):
    current = service(tmp_path, fail_datasets=True).discover(
        workspace_id="workspace-a",
        intent="code generation",
    )

    assert current.completion_outcome is CompletionOutcome.PARTIAL
    assert any(status.source == "datasets" and status.status == "failed" for status in current.source_status)
    assert "hf_secret_should_not_leak" not in current.model_dump_json()
    assert any(item.kind.value == "model" for item in current.evidence)


def test_pin_creates_immutable_next_version_and_projection_uses_selection(tmp_path):
    current_service = service(tmp_path)
    first = current_service.discover(workspace_id="workspace-a", intent="code generation")
    selected = first.evidence[0].evidence_id

    second = current_service.pin(
        "workspace-a",
        first.bundle_id,
        first.version,
        selected_evidence_ids=[selected],
        expected_version=1,
    )
    projection = current_service.markdown("workspace-a", first.bundle_id, second.version)

    assert second.version == 2
    assert second.selected_evidence_ids == (selected,)
    assert first.selected_evidence_ids == ()
    assert second.content_hash != first.content_hash
    assert projection.bundle_id == first.bundle_id
    assert projection.version == 2


def test_active_summary_contains_no_external_excerpt(tmp_path):
    current_service = service(tmp_path)
    first = current_service.discover(workspace_id="workspace-a", intent="code generation")
    current_service.activate("workspace-a", first.bundle_id, first.version)

    summary = current_service.active_summary("workspace-a")
    assert summary == {
        "bundle_id": first.bundle_id,
        "version": 1,
        "intent": "code generation",
        "freshness": "fresh",
        "lifecycle": "ready",
        "evidence_counts": {"dataset": 1, "evaluation": 1, "model": 1},
        "warning_count": 0,
    }
    assert "excerpt" not in json.dumps(summary)


def test_eval_preview_is_non_executing_and_idempotent(tmp_path):
    current_service = service(tmp_path)
    first = current_service.discover(workspace_id="workspace-a", intent="code generation")

    preview = current_service.prepare_eval("workspace-a", first.bundle_id, first.version)
    repeated = current_service.prepare_eval("workspace-a", first.bundle_id, first.version)

    assert preview == repeated
    assert preview["execute"] is False
    assert preview["bundle_id"] == first.bundle_id
    assert preview["tasks"] == ["openai_humaneval"]


def test_cancel_wins_against_worker_and_preserves_model_checkpoint(tmp_path):
    entered_datasets = threading.Event()
    release_datasets = threading.Event()

    class BlockingSources(FakeSources):
        def discover_datasets(self, query: str, *, limit: int):
            entered_datasets.set()
            assert release_datasets.wait(timeout=5)
            return super().discover_datasets(query, limit=limit)

    repository = HFContextRepository(tmp_path / "hf-context.sqlite3")
    repository.initialize()
    current_service = HFContextService(repository, sources=BlockingSources())
    collecting = current_service.begin_discovery(
        workspace_id="workspace-a", intent="code generation"
    )
    result: list = []
    worker = threading.Thread(
        target=lambda: result.append(
            current_service.run_discovery(
                "workspace-a", collecting.bundle_id, collecting.version
            )
        )
    )
    worker.start()
    assert entered_datasets.wait(timeout=5)

    cancelled = current_service.cancel(
        "workspace-a", collecting.bundle_id, collecting.version
    )
    release_datasets.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert cancelled.completion_outcome is CompletionOutcome.CANCELLED
    assert {item.kind.value for item in cancelled.evidence} == {"model", "evaluation"}
    assert result[0].content_hash == cancelled.content_hash


def test_refresh_creates_new_version_and_records_revision_drift(tmp_path):
    repository = HFContextRepository(tmp_path / "hf-context.sqlite3")
    repository.initialize()
    sources = FakeSources(model_revision="a" * 40)
    current_service = HFContextService(repository, sources=sources)
    first = current_service.discover(workspace_id="workspace-a", intent="code generation")
    model_id = next(item.evidence_id for item in first.evidence if item.kind.value == "model")
    pinned = current_service.pin(
        "workspace-a",
        first.bundle_id,
        first.version,
        selected_evidence_ids=[model_id],
        expected_version=1,
    )
    sources.model_revision = "b" * 40

    collecting, previous = current_service.begin_refresh(
        "workspace-a", pinned.bundle_id, pinned.version, expected_version=2
    )
    refreshed = current_service.run_discovery(
        "workspace-a", collecting.bundle_id, collecting.version, previous=previous
    )

    assert refreshed.version == 3
    assert refreshed.selected_evidence_ids == (model_id,)
    assert any("Revision drift" in warning for warning in refreshed.warnings)
    assert current_service.get("workspace-a", first.bundle_id, 1).content_hash == first.content_hash


def test_read_marks_schema_two_bundle_stale_without_changing_content_hash(tmp_path):
    repository = HFContextRepository(tmp_path / "hf-context.sqlite3")
    repository.initialize()
    old = datetime(2026, 7, 1, tzinfo=UTC)
    stored = HFContextBundle(
        bundle_id="hfctx_stale",
        version=1,
        workspace_id="workspace-a",
        lifecycle=Lifecycle.READY,
        freshness=Freshness.FRESH,
        completion_outcome=CompletionOutcome.COMPLETE,
        intent="Compare old model metadata",
        evidence=(
            EvidenceRecord(
                evidence_id="hf_model_old",
                kind=EvidenceKind.MODEL,
                resource_id="org/old-model",
                canonical_url="https://huggingface.co/org/old-model",
                provenance=Provenance(
                    source="huggingface_hub",
                    source_url="https://huggingface.co/org/old-model",
                    retrieved_at=old,
                ),
            ),
        ),
        created_at=old,
        ready_at=old,
    )
    repository.create_lineage(stored)

    observed = HFContextService(repository, sources=FakeSources()).get(
        "workspace-a", "hfctx_stale", 1
    )
    assert observed.freshness is Freshness.STALE
    assert observed.content_hash == stored.content_hash
