"""Transport-neutral campaign-service authority tests."""

import pytest

from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.contracts import AutonomyProfile, canonical_hash
from bashgym.campaigns.persistence import CampaignRepository
from bashgym.campaigns.service import CampaignService
from tests.campaigns.test_persistence import campaign, manifest


def test_hermes_can_create_only_an_exact_server_approved_template(tmp_path):
    repository = CampaignRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="hermes-agent",
        autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
        workspace_ids=("workspace-a",),
    )
    principal = auth.authenticate_access(auth.exchange_refresh(refresh.raw_token).raw_token)
    approved = manifest()
    service = CampaignService(
        repository,
        approved_template_hashes={
            "memexai-approved-v1": canonical_hash(approved.model_dump(mode="json"))
        },
    )

    owned_campaign = campaign().model_copy(update={"owner_actor_id": "hermes-agent"})
    created = service.create(
        owned_campaign,
        approved,
        principal=principal,
        correlation_id="correlation-create",
        idempotency_key="create-approved",
        approved_template_id="memexai-approved-v1",
    )
    assert created.campaign.owner_actor_id == "hermes-agent"

    arbitrary = approved.model_copy(update={"compute_profile_id": "unapproved-cloud"})
    with pytest.raises(PermissionError, match="campaign.create"):
        service.create(
            campaign(campaign_id="campaign-2").model_copy(
                update={"owner_actor_id": "hermes-agent"}
            ),
            arbitrary,
            principal=principal,
            correlation_id="correlation-unapproved",
            idempotency_key="create-unapproved",
            approved_template_id="memexai-approved-v1",
        )


def test_request_body_cannot_change_campaign_owner_identity(tmp_path):
    repository = CampaignRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    principal = auth.authenticate_access(auth.exchange_refresh(refresh.raw_token).raw_token)

    with pytest.raises(PermissionError, match="owner_must_match"):
        CampaignService(repository).create(
            campaign().model_copy(update={"owner_actor_id": "spoofed-actor"}),
            manifest(),
            principal=principal,
            correlation_id="correlation-spoof",
            idempotency_key="create-spoof",
        )
