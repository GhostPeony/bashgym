"""Content-addressed keys for completed stage results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from bashgym.campaigns.contracts import SealedActionResult, StageKind, canonical_hash

RESULT_KEY_SCHEMA = "campaign_result_key.v1"
REUSED_FROM_ATTEMPT_KEY = "reused_from_attempt_id"
REUSED_FROM_ACTION_KEY = "reused_from_action_id"
# Version one reuses data builds only. Every evaluation stage seals evidence that names its
# producing identity, and the readers validate that name against the consuming attempt:
# contract evaluation embeds the producing proposal id, and development evaluation embeds the
# producing campaign, study, action, attempt, and candidate digest. Reused evidence would carry
# the source's identity and fail those readers, so evaluations never reuse.
REUSABLE_STAGES = frozenset({StageKind.DATA_BUILD})
REMOTE_IDENTITY_KEYS = frozenset(
    {
        "remote_resident_model",
        "remote_resident_dataset",
        "source_training",
        "sealed_stage_artifact_inputs",
        "diagnostic_proposal_id",
        "capacity_policy",
        "budget_unit",
        "budget_reservation",
        "profile_id",
        "profile_revision",
    }
)


NO_REUSE_LINKS: Mapping[str, str] = MappingProxyType({})


def attempts_with_reuse_sources(
    attempt_ids: Sequence[str], reuse_links: Mapping[str, str]
) -> frozenset[str]:
    """Widen one experiment's attempts with the attempts that executed its reused stages.

    A study whose stage was reused registers no artifact identity of its own under
    its attempt, so both projections that join on attempt identity must search the
    producing attempt as well.
    """

    own = tuple(attempt_ids)
    return frozenset(own) | frozenset(
        reuse_links[attempt_id] for attempt_id in own if attempt_id in reuse_links
    )


def reuse_enabled(*, stage: StageKind, executor_kind: str, runtime: Mapping[str, Any]) -> bool:
    """Data builds reuse on registered remote compute; fake runs opt in; nothing else does."""

    if stage not in REUSABLE_STAGES:
        return False
    if executor_kind == "fake":
        return runtime.get("memoize") is True
    return executor_kind == "ssh_remote"


def stage_result_key(
    *,
    stage: StageKind,
    executor_kind: str,
    manifest_digest: str,
    stage_input: Mapping[str, Any],
    recipe_digest: str,
    executor_config: Mapping[str, Any],
    upstream_outputs: tuple[tuple[str, str, str], ...],
) -> str:
    """Hash content only: identities of the producing study or attempt are excluded."""

    executor_content = {
        key: value for key, value in executor_config.items() if key not in REMOTE_IDENTITY_KEYS
    }
    return canonical_hash(
        {
            "schema_version": RESULT_KEY_SCHEMA,
            "stage": stage.value,
            "executor_kind": executor_kind,
            "manifest_digest": manifest_digest,
            "stage_input": dict(stage_input),
            "recipe_digest": recipe_digest,
            "executor_content": executor_content,
            "upstream_outputs": sorted(upstream_outputs),
        }
    )


def reused_from_attempt_id(manifest: SealedActionResult) -> str | None:
    value = manifest.remote_process_identity.get(REUSED_FROM_ATTEMPT_KEY)
    return value if isinstance(value, str) and value else None


__all__ = [
    "NO_REUSE_LINKS",
    "REMOTE_IDENTITY_KEYS",
    "RESULT_KEY_SCHEMA",
    "REUSABLE_STAGES",
    "REUSED_FROM_ACTION_KEY",
    "REUSED_FROM_ATTEMPT_KEY",
    "attempts_with_reuse_sources",
    "reuse_enabled",
    "reused_from_attempt_id",
    "stage_result_key",
]
