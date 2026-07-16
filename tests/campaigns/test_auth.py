"""Campaign credential binding, exchange, and immediate revocation tests."""

from datetime import timedelta

import pytest

from bashgym.campaigns.auth import CampaignAuthenticationError, CampaignAuthService
from bashgym.campaigns.contracts import AutonomyProfile, Capability, CredentialKind
from bashgym.campaigns.persistence import CampaignRepository


@pytest.fixture
def auth(tmp_path):
    repository = CampaignRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    return repository, CampaignAuthService(repository)


def test_refresh_exchange_resolves_stored_hermes_identity_and_capabilities(auth):
    _repository, service = auth
    refresh = service.issue_refresh_credential(
        actor_id="hermes-agent",
        autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
        workspace_ids=("workspace-b", "workspace-a", "workspace-a"),
    )
    access = service.exchange_refresh(refresh.raw_token)
    principal = service.authenticate_access(access.raw_token)

    assert principal.actor_id == "hermes-agent"
    assert principal.autonomy_profile == AutonomyProfile.HERMES_BOUNDED
    assert principal.authorization_revision == 1
    assert principal.workspace_ids == ("workspace-a", "workspace-b")
    assert Capability.COMPUTE_TRAIN_WITHIN_BUDGET in principal.capabilities
    assert Capability.PROMOTION_DECIDE not in principal.capabilities


def test_workspace_and_capability_checks_fail_closed(auth):
    _repository, service = auth
    refresh = service.issue_refresh_credential(
        actor_id="hermes-agent",
        autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
        workspace_ids=("workspace-a",),
    )
    principal = service.authenticate_access(service.exchange_refresh(refresh.raw_token).raw_token)

    with pytest.raises(PermissionError, match="campaign_workspace_forbidden"):
        principal.require("workspace-b", Capability.CAMPAIGN_READ)
    with pytest.raises(PermissionError, match="promotion.decide"):
        principal.require("workspace-a", Capability.PROMOTION_DECIDE)


def test_parent_revocation_invalidates_all_descendants_on_next_request(auth):
    _repository, service = auth
    refresh = service.issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    first = service.exchange_refresh(refresh.raw_token)
    second = service.exchange_refresh(refresh.raw_token)
    assert service.authenticate_access(first.raw_token).actor_id == "codex-agent"

    descendants = service.revoke_credential(refresh.credential_id, reason="operator revoked")

    assert descendants == 2
    for raw_token in (first.raw_token, second.raw_token):
        with pytest.raises(CampaignAuthenticationError):
            service.authenticate_access(raw_token)
    with pytest.raises(CampaignAuthenticationError):
        service.exchange_refresh(refresh.raw_token)
    revoked = _repository.get_actor_credential(refresh.credential_id)
    assert revoked is not None and revoked.authorization_revision == 2


def test_profile_or_scope_revision_is_durable_and_resolved_on_existing_access(auth):
    repository, service = auth
    refresh = service.issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    access = service.exchange_refresh(refresh.raw_token)
    before = service.authenticate_access(access.raw_token)

    revision = service.revise_credential_authorization(
        refresh.credential_id,
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-b", "workspace-a"),
    )
    after = service.authenticate_access(access.raw_token)

    assert before.authorization_revision == 1
    assert revision == 2
    assert after.authorization_revision == 2
    assert after.workspace_ids == ("workspace-a", "workspace-b")
    stored = repository.get_actor_credential(refresh.credential_id)
    assert stored is not None and stored.authorization_revision == 2


def test_missing_malformed_and_expired_access_tokens_are_indistinguishable(auth):
    _repository, service = auth
    refresh = service.issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    expired = service.exchange_refresh(refresh.raw_token, access_ttl=timedelta(seconds=-1))

    for raw_token in (None, "not-a-token", expired.raw_token):
        with pytest.raises(CampaignAuthenticationError) as error:
            service.authenticate_access(raw_token)
        assert str(error.value) == "campaign_auth_required"


def test_raw_refresh_and_access_tokens_never_reach_sqlite(auth):
    repository, service = auth
    refresh = service.issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    access = service.exchange_refresh(refresh.raw_token)
    database_bytes = repository.db_path.read_bytes()

    assert refresh.raw_token.encode() not in database_bytes
    assert access.raw_token.encode() not in database_bytes
    assert refresh.raw_token.rsplit(".", 1)[1].encode() not in database_bytes
    assert access.raw_token.rsplit(".", 1)[1].encode() not in database_bytes


def test_desktop_bootstrap_is_idempotent_hashed_and_exchanges_for_local_user(auth):
    repository, service = auth
    bootstrap = "bgcb.launch-123." + "desktop-secret-material-" * 2

    assert service.install_desktop_bootstrap(bootstrap) == "launch-123"
    assert service.install_desktop_bootstrap(bootstrap) == "launch-123"
    access = service.exchange_desktop_bootstrap(bootstrap)
    principal = service.authenticate_access(access.raw_token)

    assert access.kind == CredentialKind.ACCESS
    assert principal.actor_id == "desktop-user"
    assert principal.autonomy_profile == AutonomyProfile.DESKTOP_USER
    assert principal.workspace_ids == ("desktop-local",)
    principal.require("any-local-workspace", Capability.CAMPAIGN_CREATE)
    database_bytes = repository.db_path.read_bytes()
    assert bootstrap.encode() not in database_bytes
    assert bootstrap.rsplit(".", 1)[1].encode() not in database_bytes


def test_refresh_and_bootstrap_prefixes_cannot_be_substituted(auth):
    _repository, service = auth
    refresh = service.issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    bootstrap = "bgcb.launch-456." + "another-desktop-secret-" * 2
    service.install_desktop_bootstrap(bootstrap)

    with pytest.raises(CampaignAuthenticationError):
        service.exchange_desktop_bootstrap(refresh.raw_token.replace("bgcr.", "bgcb.", 1))
    with pytest.raises(CampaignAuthenticationError):
        service.exchange_refresh(bootstrap.replace("bgcb.", "bgcr.", 1))
