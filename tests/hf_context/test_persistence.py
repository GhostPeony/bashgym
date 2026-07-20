from datetime import datetime

import pytest

from bashgym._compat import UTC
from bashgym.integrations.huggingface.context_contracts import (
    CompletionOutcome,
    EvidenceKind,
    EvidenceRecord,
    Freshness,
    HFContextBundle,
    Lifecycle,
    Provenance,
    Visibility,
)
from bashgym.integrations.huggingface.context_persistence import (
    BundleNotFoundError,
    BundleRevisionConflictError,
    HFContextRepository,
    ImmutableBundleError,
)

NOW = datetime(2026, 7, 10, 12, tzinfo=UTC)


def bundle(
    *,
    workspace_id: str = "workspace-a",
    bundle_id: str = "hfctx_fixture",
    version: int = 1,
    lifecycle: Lifecycle = Lifecycle.COLLECTING,
    intent: str = "Find coding models",
) -> HFContextBundle:
    return HFContextBundle(
        bundle_id=bundle_id,
        version=version,
        workspace_id=workspace_id,
        lifecycle=lifecycle,
        freshness=Freshness.FRESH,
        completion_outcome=(CompletionOutcome.COMPLETE if lifecycle is Lifecycle.READY else None),
        intent=intent,
        created_at=NOW,
        ready_at=NOW if lifecycle is Lifecycle.READY else None,
    )


@pytest.fixture
def repository(tmp_path):
    repo = HFContextRepository(tmp_path / "hf-context.sqlite3")
    repo.initialize()
    return repo


def test_repository_enables_wal_foreign_keys_and_migrations(repository):
    assert repository.journal_mode() == "wal"
    assert repository.foreign_keys_enabled() is True
    assert repository.schema_version() == 1


def test_collecting_bundle_finalizes_once_and_ready_versions_are_immutable(repository):
    repository.create_lineage(bundle())
    ready = bundle(lifecycle=Lifecycle.READY)
    assert repository.finalize_version(ready).lifecycle is Lifecycle.READY

    with pytest.raises(ImmutableBundleError):
        repository.finalize_version(ready.model_copy(update={"intent": "changed"}))


def test_optimistic_head_conflict_is_atomic_for_new_versions(repository):
    repository.create_lineage(bundle(lifecycle=Lifecycle.READY))
    second = bundle(version=2, lifecycle=Lifecycle.READY, intent="Pinned selection")
    assert repository.create_version(second, expected_head=1).version == 2

    with pytest.raises(BundleRevisionConflictError) as caught:
        repository.create_version(
            bundle(version=3, lifecycle=Lifecycle.READY, intent="Concurrent refresh"),
            expected_head=1,
        )
    assert caught.value.current == 2


def test_cross_workspace_lookup_is_indistinguishable_from_missing(repository):
    repository.create_lineage(bundle(lifecycle=Lifecycle.READY))

    with pytest.raises(BundleNotFoundError):
        repository.get_version("workspace-b", "hfctx_fixture", 1)
    with pytest.raises(BundleNotFoundError):
        repository.get_version("workspace-a", "hfctx_missing", 1)


def test_active_pointer_targets_exact_ready_version_and_deactivates(repository):
    repository.create_lineage(bundle(lifecycle=Lifecycle.READY))
    repository.create_version(
        bundle(version=2, lifecycle=Lifecycle.READY, intent="Version two"), expected_head=1
    )

    active = repository.activate("workspace-a", "hfctx_fixture", 1)
    assert active.version == 1
    assert repository.get_active("workspace-a").version == 1
    repository.deactivate("workspace-a")
    assert repository.get_active("workspace-a") is None


def test_tombstone_clears_active_pointer_and_history_is_workspace_scoped(repository):
    repository.create_lineage(bundle(lifecycle=Lifecycle.READY))
    repository.activate("workspace-a", "hfctx_fixture", 1)
    repository.delete_lineage("workspace-a", "hfctx_fixture")

    assert repository.get_active("workspace-a") is None
    assert repository.list_versions("workspace-a") == []
    with pytest.raises(BundleNotFoundError):
        repository.get_version("workspace-a", "hfctx_fixture", 1)


def test_eval_preview_is_idempotent_per_exact_bundle_version(repository):
    repository.create_lineage(bundle(lifecycle=Lifecycle.READY))
    preview = {"model_id": "org/model", "tasks": ["humaneval"]}
    first = repository.put_eval_preview("workspace-a", "hfctx_fixture", 1, "a" * 64, preview)
    second = repository.put_eval_preview("workspace-a", "hfctx_fixture", 1, "a" * 64, preview)
    assert first == second == preview


def test_collecting_checkpoint_can_be_cancelled_once_and_never_overwritten(repository):
    repository.create_lineage(bundle())
    checkpoint = bundle(intent="Usable partial evidence")
    assert repository.update_collecting(checkpoint).intent == "Usable partial evidence"

    cancelled = repository.cancel_version("workspace-a", "hfctx_fixture", 1)
    assert cancelled.lifecycle is Lifecycle.READY
    assert cancelled.completion_outcome is CompletionOutcome.CANCELLED
    assert repository.cancel_version("workspace-a", "hfctx_fixture", 1) == cancelled

    with pytest.raises(ImmutableBundleError):
        repository.update_collecting(checkpoint)
    with pytest.raises(ImmutableBundleError):
        repository.finalize_version(bundle(lifecycle=Lifecycle.READY))


def test_collecting_refresh_uses_optimistic_head(repository):
    repository.create_lineage(bundle(lifecycle=Lifecycle.READY))
    collecting = bundle(version=2, lifecycle=Lifecycle.COLLECTING, intent="Refresh")
    repository.create_collecting_version(collecting, expected_head=1)

    with pytest.raises(BundleRevisionConflictError):
        repository.create_collecting_version(
            bundle(version=2, lifecycle=Lifecycle.COLLECTING, intent="Concurrent"),
            expected_head=1,
        )
    assert (
        repository.get_version("workspace-a", "hfctx_fixture", 2).lifecycle is Lifecycle.COLLECTING
    )


def test_retention_keeps_active_and_collecting_versions(repository):
    first = bundle(bundle_id="hfctx_00")
    repository.create_lineage(first)
    repository.finalize_version(bundle(bundle_id="hfctx_00", lifecycle=Lifecycle.READY))
    repository.activate("workspace-a", "hfctx_00", 1)

    for index in range(1, 22):
        bundle_id = f"hfctx_{index:02d}"
        repository.create_lineage(bundle(bundle_id=bundle_id))
        repository.finalize_version(bundle(bundle_id=bundle_id, lifecycle=Lifecycle.READY))

    repository.create_lineage(bundle(bundle_id="hfctx_collecting"))

    assert len(repository.list_versions("workspace-a", limit=100)) == 20
    assert repository.get_active("workspace-a").bundle_id == "hfctx_00"
    assert (
        repository.get_version("workspace-a", "hfctx_collecting", 1).lifecycle
        is Lifecycle.COLLECTING
    )
    with pytest.raises(BundleNotFoundError):
        repository.get_version("workspace-a", "hfctx_01", 1)
    with pytest.raises(BundleNotFoundError):
        repository.get_version("workspace-a", "hfctx_02", 1)
    assert repository.get_version("workspace-a", "hfctx_03", 1).bundle_id == "hfctx_03"


def test_workspace_private_excerpt_is_rejected_before_sqlite_persistence():
    with pytest.raises(ValueError, match="revocable encrypted payloads"):
        EvidenceRecord(
            evidence_id="hf_model_private",
            kind=EvidenceKind.MODEL,
            resource_id="org/private-model",
            canonical_url="https://huggingface.co/org/private-model",
            excerpt="private marker must never enter sqlite",
            visibility=Visibility.WORKSPACE_PRIVATE,
            provenance=Provenance(
                source="huggingface_hub",
                source_url="https://huggingface.co/org/private-model",
            ),
        )
