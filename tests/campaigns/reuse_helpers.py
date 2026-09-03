"""Shared fixtures for campaign result-reuse tests.

Reuse tests need to derive the content key a scheduled action would carry and to
rewrite stored reuse evidence the way a damaged or tampered database row reads.
Both belong beside the tests rather than inside any one test module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.result_reuse import manifest_content_digest, stage_result_key
from bashgym.campaigns.runtime import CampaignRuntimeRepository

WORKSPACE_ID = "workspace-a"


def derived_data_build_result_key(
    repository: CampaignRuntimeRepository,
    campaign_id: str,
    *,
    stage_input: Mapping[str, Any],
    executor_config: Mapping[str, Any],
    recipe_digest: str,
) -> str:
    """Derive one remote data build's content key exactly as `next_action_spec` does."""

    campaign = repository.get_campaign(WORKSPACE_ID, campaign_id)
    revision = repository.get_manifest_revision(
        WORKSPACE_ID, campaign_id, campaign.manifest_revision
    )
    return stage_result_key(
        stage=StageKind.DATA_BUILD,
        executor_kind="ssh_remote",
        manifest_content_digest=manifest_content_digest(revision.manifest),
        stage_input=stage_input,
        recipe_digest=recipe_digest,
        executor_config=executor_config,
        upstream_outputs=(),
    )


def rewrite_result_manifest(
    repository: CampaignRuntimeRepository,
    attempt_id: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """Rewrite one stored result manifest the way a tampered database row would read."""

    with repository._connection(immediate=True) as connection:
        row = connection.execute(
            "SELECT result_json FROM campaign_attempts WHERE workspace_id = ? AND attempt_id = ?",
            (WORKSPACE_ID, attempt_id),
        ).fetchone()
        payload = json.loads(row["result_json"])
        mutate(payload)
        connection.execute(
            "UPDATE campaign_attempts SET result_json = ?"
            " WHERE workspace_id = ? AND attempt_id = ?",
            (json.dumps(payload), WORKSPACE_ID, attempt_id),
        )


def set_reuse_link(
    repository: CampaignRuntimeRepository, attempt_id: str, source_attempt_id: str
) -> None:
    """Point one stored manifest's reuse link at another attempt."""

    def mutate(payload: dict[str, Any]) -> None:
        payload["remote_process_identity"]["reused_from_attempt_id"] = source_attempt_id

    rewrite_result_manifest(repository, attempt_id, mutate)


def set_action_status(repository: CampaignRuntimeRepository, action_id: str, status: str) -> None:
    """Move one action row's status without touching its attempts."""

    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_actions SET status = ? WHERE workspace_id = ? AND action_id = ?",
            (status, WORKSPACE_ID, action_id),
        )
