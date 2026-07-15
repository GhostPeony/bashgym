"""Application service for discovery, pinning, projection, and safe Eval previews."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any
from uuid import uuid4

from .context_contracts import (
    CompletionOutcome,
    EvidenceKind,
    Freshness,
    HFContextBundle,
    Lifecycle,
    SourceStatus,
    canonical_hash,
    rank_evidence,
    utc_now,
)
from .context_persistence import HFContextRepository, ImmutableBundleError
from .context_projection import ProjectionResult, project_bundle_markdown
from .context_sources import (
    HFContextSourceClient,
    normalize_dataset,
    normalize_model,
    normalize_model_card_evals,
)


class HFContextService:
    def __init__(self, repository: HFContextRepository, *, sources: Any | None = None):
        self.repository = repository
        self.sources = sources or HFContextSourceClient()

    def begin_discovery(
        self,
        *,
        workspace_id: str,
        intent: str,
        task: str | None = None,
        target: dict[str, Any] | None = None,
        origin: dict[str, str] | None = None,
    ) -> HFContextBundle:
        bundle_id = f"hfctx_{uuid4().hex}"
        correlation_id = f"hfctx_corr_{uuid4().hex}"
        collecting = HFContextBundle(
            bundle_id=bundle_id,
            version=1,
            workspace_id=workspace_id,
            lifecycle=Lifecycle.COLLECTING,
            freshness=Freshness.FRESH,
            intent=intent,
            task=task,
            target=target or {},
            correlation_id=correlation_id,
            origin=origin or {},
        )
        self.repository.create_lineage(collecting)
        return collecting

    def discover(
        self,
        *,
        workspace_id: str,
        intent: str,
        task: str | None = None,
        target: dict[str, Any] | None = None,
        origin: dict[str, str] | None = None,
    ) -> HFContextBundle:
        """Compatibility wrapper for callers that still need a synchronous result."""

        collecting = self.begin_discovery(
            workspace_id=workspace_id,
            intent=intent,
            task=task,
            target=target,
            origin=origin,
        )
        return self.run_discovery(workspace_id, collecting.bundle_id, collecting.version)

    @staticmethod
    def _search_query(bundle: HFContextBundle) -> str:
        if bundle.task:
            return bundle.task.strip()
        words = [word for word in bundle.intent.split() if len(word) > 2]
        return " ".join(words[:6]) or bundle.intent

    def _checkpoint(self, bundle: HFContextBundle) -> HFContextBundle:
        try:
            return self.repository.update_collecting(bundle)
        except ImmutableBundleError:
            return self.get(bundle.workspace_id, bundle.bundle_id, bundle.version)

    def run_discovery(
        self,
        workspace_id: str,
        bundle_id: str,
        version: int,
        *,
        previous: HFContextBundle | None = None,
    ) -> HFContextBundle:
        collecting = self.get(workspace_id, bundle_id, version)
        if collecting.lifecycle is Lifecycle.READY:
            return collecting
        if previous is None and version > 1:
            previous = self.get(workspace_id, bundle_id, version - 1)

        query = self._search_query(collecting)
        evidence = []
        statuses: list[SourceStatus] = []
        failed = False

        try:
            raw_models = self.sources.discover_models(query, limit=3)
            models = [normalize_model(item, intent=query) for item in raw_models]
            evaluations = [
                evaluation
                for item in raw_models
                for evaluation in normalize_model_card_evals(item)
            ]
            evidence.extend(rank_evidence(models)[:5])
            evidence.extend(rank_evidence(evaluations)[:3])
            statuses.append(
                SourceStatus(source="models", status="complete", result_count=len(models))
            )
            statuses.append(
                SourceStatus(
                    source="evaluations", status="complete", result_count=len(evaluations)
                )
            )
        except Exception:  # noqa: BLE001 - error text may contain credentials
            failed = True
            statuses.extend(
                [
                    SourceStatus(
                        source="models",
                        status="failed",
                        error_code="hf_source_unavailable",
                        safe_message="Model discovery is temporarily unavailable.",
                    ),
                    SourceStatus(
                        source="evaluations",
                        status="failed",
                        error_code="hf_source_unavailable",
                        safe_message="Evaluation evidence is temporarily unavailable.",
                    ),
                ]
            )

        checkpoint = HFContextBundle.model_validate(
            {
                **collecting.model_dump(mode="python", exclude={"content_hash"}),
                "evidence": tuple(evidence[:12]),
                "selected_evidence_ids": tuple(
                    evidence_id
                    for evidence_id in collecting.selected_evidence_ids
                    if any(item.evidence_id == evidence_id for item in evidence)
                ),
                "source_status": tuple(statuses),
            }
        )
        checkpoint = self._checkpoint(checkpoint)
        if checkpoint.lifecycle is Lifecycle.READY:
            return checkpoint

        try:
            raw_datasets = self.sources.discover_datasets(query, limit=3)
            datasets = [normalize_dataset(item, intent=query) for item in raw_datasets]
            evidence.extend(rank_evidence(datasets)[:4])
            statuses.append(
                SourceStatus(source="datasets", status="complete", result_count=len(datasets))
            )
        except Exception:  # noqa: BLE001 - return a visibility-safe partial status
            failed = True
            statuses.append(
                SourceStatus(
                    source="datasets",
                    status="failed",
                    error_code="hf_source_timeout",
                    safe_message="Dataset discovery did not complete.",
                )
            )

        warnings = list(collecting.warnings)
        if previous is not None:
            old_revisions = {
                (item.kind, item.resource_id): item.revision for item in previous.evidence
            }
            for item in evidence:
                old_revision = old_revisions.get((item.kind, item.resource_id))
                if old_revision and item.revision and old_revision != item.revision:
                    warnings.append(
                        f"Revision drift: {item.resource_id} changed from {old_revision} to {item.revision}."
                    )

        selected = tuple(
            evidence_id
            for evidence_id in collecting.selected_evidence_ids
            if any(item.evidence_id == evidence_id for item in evidence)
        )
        final_checkpoint = HFContextBundle.model_validate(
            {
                **collecting.model_dump(mode="python", exclude={"content_hash"}),
                "evidence": tuple(evidence[:12]),
                "selected_evidence_ids": selected,
                "warnings": tuple(warnings),
                "source_status": tuple(statuses),
            }
        )
        final_checkpoint = self._checkpoint(final_checkpoint)
        if final_checkpoint.lifecycle is Lifecycle.READY:
            return final_checkpoint

        ready_at = utc_now()
        ready = HFContextBundle(
            bundle_id=bundle_id,
            version=version,
            workspace_id=workspace_id,
            lifecycle=Lifecycle.READY,
            freshness=Freshness.FRESH,
            completion_outcome=(CompletionOutcome.PARTIAL if failed else CompletionOutcome.COMPLETE),
            intent=collecting.intent,
            task=collecting.task,
            target=collecting.target,
            evidence=tuple(evidence[:12]),
            selected_evidence_ids=selected,
            warnings=tuple(warnings),
            source_status=tuple(statuses),
            correlation_id=collecting.correlation_id,
            origin=collecting.origin,
            created_at=collecting.created_at,
            ready_at=ready_at,
        )
        try:
            return self.repository.finalize_version(ready)
        except ImmutableBundleError:
            return self.get(workspace_id, bundle_id, version)

    def begin_refresh(
        self,
        workspace_id: str,
        bundle_id: str,
        version: int,
        *,
        expected_version: int,
    ) -> tuple[HFContextBundle, HFContextBundle]:
        previous = self.get(workspace_id, bundle_id, version)
        if previous.lifecycle is not Lifecycle.READY:
            raise ImmutableBundleError("only ready bundle versions can be refreshed")
        collecting = HFContextBundle(
            bundle_id=bundle_id,
            version=expected_version + 1,
            workspace_id=workspace_id,
            lifecycle=Lifecycle.COLLECTING,
            freshness=Freshness.FRESH,
            intent=previous.intent,
            task=previous.task,
            target=previous.target,
            evidence=previous.evidence,
            selected_evidence_ids=previous.selected_evidence_ids,
            correlation_id=f"hfctx_corr_{uuid4().hex}",
            origin=previous.origin,
        )
        return (
            self.repository.create_collecting_version(
                collecting, expected_head=expected_version
            ),
            previous,
        )

    def cancel(
        self, workspace_id: str, bundle_id: str, version: int
    ) -> HFContextBundle:
        return self.repository.cancel_version(workspace_id, bundle_id, version)

    @staticmethod
    def _with_current_freshness(bundle: HFContextBundle) -> HFContextBundle:
        if bundle.lifecycle is Lifecycle.COLLECTING or bundle.schema_version == "1":
            return bundle
        retrieved_at = min(
            (item.provenance.retrieved_at for item in bundle.evidence),
            default=bundle.ready_at or bundle.created_at,
        )
        has_short_ttl = any(
            item.kind in {EvidenceKind.MODEL, EvidenceKind.DATASET}
            for item in bundle.evidence
        )
        ttl = timedelta(hours=6 if has_short_ttl else 24)
        freshness = Freshness.STALE if utc_now() - retrieved_at > ttl else Freshness.FRESH
        if freshness is bundle.freshness:
            return bundle
        return HFContextBundle.model_validate(
            {
                **bundle.model_dump(mode="python", exclude={"content_hash", "freshness"}),
                "freshness": freshness,
                "content_hash": bundle.content_hash,
            }
        )

    def get(self, workspace_id: str, bundle_id: str, version: int) -> HFContextBundle:
        return self._with_current_freshness(
            self.repository.get_version(workspace_id, bundle_id, version)
        )

    def history(self, workspace_id: str, *, limit: int = 20) -> list[HFContextBundle]:
        return [
            self._with_current_freshness(bundle)
            for bundle in self.repository.list_versions(workspace_id, limit=limit)
        ]

    def pin(
        self,
        workspace_id: str,
        bundle_id: str,
        version: int,
        *,
        selected_evidence_ids: list[str],
        expected_version: int,
    ) -> HFContextBundle:
        current = self.get(workspace_id, bundle_id, version)
        next_bundle = HFContextBundle.model_validate(
            {
                **current.model_dump(mode="python", exclude={"content_hash"}),
                "version": expected_version + 1,
                "selected_evidence_ids": tuple(selected_evidence_ids),
                "created_at": utc_now(),
            }
        )
        return self.repository.create_version(next_bundle, expected_head=expected_version)

    def activate(self, workspace_id: str, bundle_id: str, version: int) -> HFContextBundle:
        return self.repository.activate(workspace_id, bundle_id, version)

    def deactivate(self, workspace_id: str) -> None:
        self.repository.deactivate(workspace_id)

    def delete(self, workspace_id: str, bundle_id: str) -> None:
        self.repository.delete_lineage(workspace_id, bundle_id)

    def active(self, workspace_id: str) -> HFContextBundle | None:
        bundle = self.repository.get_active(workspace_id)
        return self._with_current_freshness(bundle) if bundle is not None else None

    def active_summary(self, workspace_id: str) -> dict[str, Any] | None:
        bundle = self.active(workspace_id)
        if bundle is None:
            return None
        counts = Counter(item.kind.value for item in bundle.evidence)
        return {
            "bundle_id": bundle.bundle_id,
            "version": bundle.version,
            "intent": bundle.intent,
            "freshness": bundle.freshness.value,
            "lifecycle": bundle.lifecycle.value,
            "evidence_counts": dict(sorted(counts.items())),
            "warning_count": len(bundle.warnings),
        }

    def markdown(self, workspace_id: str, bundle_id: str, version: int) -> ProjectionResult:
        return project_bundle_markdown(self.get(workspace_id, bundle_id, version))

    def prepare_eval(self, workspace_id: str, bundle_id: str, version: int) -> dict[str, Any]:
        bundle = self.get(workspace_id, bundle_id, version)
        selected = set(bundle.selected_evidence_ids)
        items = [item for item in bundle.evidence if not selected or item.evidence_id in selected]
        tasks = sorted(
            {
                str(item.facts.get("dataset_type"))
                for item in items
                if item.kind is EvidenceKind.EVALUATION and item.facts.get("dataset_type")
            }
        )
        models = [item.resource_id for item in items if item.kind is EvidenceKind.MODEL]
        preview = {
            "bundle_id": bundle.bundle_id,
            "version": bundle.version,
            "execute": False,
            "model_id": bundle.target.get("model_id") or (models[0] if models else None),
            "tasks": tasks,
            "unknowns": [
                "Prompt template and harness version must be confirmed before execution."
            ],
        }
        action_hash = canonical_hash(preview)
        return self.repository.put_eval_preview(
            workspace_id, bundle_id, version, action_hash, preview
        )


__all__ = ["HFContextService"]
