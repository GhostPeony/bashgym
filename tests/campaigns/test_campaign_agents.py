"""Campaign-scoped agent grant, credential, replay, and projection tests."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.campaign_agent_contracts import (
    CampaignAgentActionContext,
    CampaignAgentAttachRequest,
    CampaignAgentCapability,
    CampaignAgentFamily,
    CampaignAgentGrantRequest,
    CampaignAgentHeartbeatRequest,
    CampaignAgentPublicViewQuery,
    CampaignAgentRevokeRequest,
    CampaignAgentScope,
)
from bashgym.campaigns.campaign_agents import (
    PROHIBITED_CAMPAIGN_AGENT_CAPABILITIES,
    CampaignAgentAuthorizationError,
    CampaignAgentBrokerUnavailableError,
    CampaignAgentConflictError,
    CampaignAgentCredentialError,
    CampaignAgentError,
    CampaignAgentIntegrityError,
    CampaignAgentRepository,
    CampaignAgentService,
    mapped_campaign_capabilities,
)
from bashgym.campaigns.contracts import (
    ActorPrincipal,
    AutonomyProfile,
    Campaign,
    CampaignKind,
    CampaignManifest,
    Capability,
    CredentialKind,
    ManifestRevision,
    TargetModelContract,
    canonical_hash,
)
from bashgym.campaigns.persistence import CampaignRepository

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
TEST_SEAL_KEY = b"campaign-agent-test-authority-key!"


def _sealer() -> ArtifactSealer:
    return ArtifactSealer(TEST_SEAL_KEY, key_version="campaign-agent-test-v1")


def _campaign(workspace_id: str = "workspace-a", campaign_id: str = "campaign-1") -> Campaign:
    return Campaign(
        workspace_id=workspace_id,
        campaign_id=campaign_id,
        title="Operator-selected model campaign",
        kind=CampaignKind.GENERAL,
        objective="Improve an explicitly bound trainable model.",
        target_model=TargetModelContract(
            target_contract_key="operator-model-binding-v1",
            base_model_ref="registry://trainable/model@immutable-revision",
            task="operator-selected-task",
        ),
        owner_actor_id="desktop-user",
    )


def _seed_campaign(repository: CampaignRepository, workspace="workspace-a", campaign="campaign-1"):
    value = _campaign(workspace, campaign)
    manifest = CampaignManifest(
        approved_data_scopes=("approved-data",),
        compute_profile_id="registered-private-compute",
        budget_limits={"gpu_hours": 1.0},
        evaluation_plan={"suite": "registered-eval"},
        promotion_gates={"primary_metric": 0.0},
    )
    repository.create_campaign(
        value,
        ManifestRevision(
            workspace_id=workspace,
            campaign_id=campaign,
            revision=1,
            manifest=manifest,
            actor_id="desktop-user",
            correlation_id=f"create-{campaign}",
        ),
        actor_id="desktop-user",
        credential_kind=CredentialKind.ACCESS,
        correlation_id=f"create-{campaign}",
        idempotency_key=f"create-{campaign}",
    )


def _principal(profile=AutonomyProfile.DESKTOP_USER, workspace="workspace-a") -> ActorPrincipal:
    return ActorPrincipal(
        actor_id="desktop-user" if profile == AutonomyProfile.DESKTOP_USER else "codex-agent",
        autonomy_profile=profile,
        credential_id="credential-1",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=(workspace,),
        capabilities=frozenset(Capability),
        expires_at=NOW + timedelta(hours=1),
    )


def _grant_request(**updates) -> CampaignAgentGrantRequest:
    values = {
        "scope": CampaignAgentScope(workspace_id="workspace-a", campaign_id="campaign-1"),
        "agent_family": CampaignAgentFamily.CODEX,
        "agent_origin": "codex-session-origin",
        "agent_principal_id": "codex-campaign-agent",
        "session_id": "session-1",
        "requested_capabilities": (
            CampaignAgentCapability.CAMPAIGN_OBSERVE,
            CampaignAgentCapability.TRAINING_LAUNCH,
        ),
        "granted_capabilities": (CampaignAgentCapability.CAMPAIGN_OBSERVE,),
        "idempotency_key": "grant-1",
    }
    values.update(updates)
    return CampaignAgentGrantRequest(**values)


@pytest.fixture
def setup(tmp_path):
    campaign_repository = CampaignRepository(tmp_path / "campaigns.sqlite3")
    campaign_repository.initialize()
    _seed_campaign(campaign_repository)
    _seed_campaign(campaign_repository, campaign="campaign-2")
    repository = CampaignAgentRepository(campaign_repository.db_path, sealer=_sealer())
    repository.initialize()
    delivered = []

    def verified(scope, family, origin, session_id, principal_id):
        return (
            scope.workspace_id == "workspace-a"
            and scope.campaign_id.startswith("campaign-")
            and family in {CampaignAgentFamily.CODEX, CampaignAgentFamily.HERMES}
            and origin.endswith("-origin")
            and session_id.startswith("session-")
            and principal_id.endswith("-agent")
        )

    service = CampaignAgentService(
        campaign_repository,
        repository,
        origin_verifier=verified,
        credential_broker=delivered.append,
    )
    return campaign_repository, repository, service, delivered


def _attach_request(grant, **updates) -> CampaignAgentAttachRequest:
    values = {
        "scope": grant.scope,
        "agent_family": grant.agent_family,
        "agent_origin": grant.agent_origin,
        "agent_principal_id": grant.agent_principal_id,
        "session_id": grant.session_id,
        "requested_capabilities": grant.requested_capabilities,
        "granted_capabilities": grant.granted_capabilities,
        "confirmation_receipt": grant,
        "base_attachment_version": None,
        "idempotency_key": "attach-1",
    }
    values.update(updates)
    return CampaignAgentAttachRequest(**values)


def _action_context(grant, **updates) -> CampaignAgentActionContext:
    values = {
        "scope": grant.scope,
        "agent_family": grant.agent_family,
        "agent_origin": grant.agent_origin,
        "agent_principal_id": grant.agent_principal_id,
        "session_id": grant.session_id,
    }
    values.update(updates)
    return CampaignAgentActionContext(**values)


def test_only_a_human_with_verified_origin_can_issue_exact_grant(setup):
    _campaigns, _repository, service, _delivered = setup
    request = _grant_request()

    grant = service.issue_grant(_principal(), request, now=NOW)
    replay = service.issue_grant(_principal(), request, now=NOW + timedelta(seconds=1))

    assert replay == grant
    assert grant.human_principal_id == "desktop-user"
    assert grant.agent_origin == request.agent_origin
    assert grant.granted_capabilities == request.granted_capabilities
    with pytest.raises(CampaignAgentAuthorizationError, match="human"):
        service.issue_grant(_principal(AutonomyProfile.CODEX_TRUSTED), request, now=NOW)
    with pytest.raises(CampaignAgentAuthorizationError, match="origin"):
        service.issue_grant(
            _principal(),
            _grant_request(agent_origin="unverified", idempotency_key="grant-2"),
            now=NOW,
        )


def test_campaign_agent_schema_is_restart_safe_and_checksum_guarded(setup):
    _campaigns, repository, _service, _delivered = setup
    repository.initialize()
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE campaign_agent_schema_migrations SET checksum = 'tampered' WHERE version = 1"
        )

    with pytest.raises(CampaignAgentError, match="checksum"):
        repository.initialize()


def test_campaign_agent_schema_initialization_is_concurrency_safe(tmp_path):
    path = tmp_path / "concurrent-campaigns.sqlite3"

    def initialize(_index):
        CampaignAgentRepository(path, sealer=_sealer()).initialize()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(initialize, range(24)))
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM campaign_agent_schema_migrations").fetchone()[
                0
            ]
            == 3
        )


def test_grant_replay_conflicts_when_same_key_changes_scope_or_authority(setup):
    _campaigns, _repository, service, _delivered = setup
    service.issue_grant(_principal(), _grant_request(), now=NOW)

    with pytest.raises(CampaignAgentConflictError, match="idempotency"):
        service.issue_grant(
            _principal(),
            _grant_request(granted_capabilities=(CampaignAgentCapability.TRAINING_LAUNCH,)),
            now=NOW,
        )
    with pytest.raises(CampaignAgentConflictError, match="idempotency"):
        service.issue_grant(
            _principal().model_copy(update={"actor_id": "other-human"}),
            _grant_request(),
            now=NOW,
        )


def test_attach_is_brokered_and_public_projection_contains_no_bearer_or_canary(setup):
    _campaigns, repository, service, delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)

    view, replayed = service.attach(_principal(), _attach_request(grant), now=NOW)
    replay, was_replayed = service.attach(_principal(), _attach_request(grant), now=NOW)

    assert replayed is False and was_replayed is True
    assert replay == view
    assert len(delivered) == 1
    raw = delivered[0].raw_token
    serialized = json.dumps(view, sort_keys=True)
    database = repository.db_path.read_bytes()
    assert raw not in serialized
    assert raw not in repr(delivered[0])
    assert raw.encode() not in database
    assert raw.rsplit(".", 1)[1].encode() not in database
    assert set(view) == {"schema_version", "observed_at", "scope", "attachment", "audit_events"}
    assert view["attachment"]["provenance"]["agent_origin_status"] == "verified"
    assert "API-SECRET-CANARY" not in serialized
    with pytest.raises(CampaignAgentAuthorizationError, match="human principal"):
        service.attach(
            _principal().model_copy(update={"actor_id": "other-human"}),
            _attach_request(grant),
            now=NOW,
        )


def test_attach_fails_closed_without_trusted_broker_before_mutation(setup):
    campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    unavailable = CampaignAgentService(
        campaigns,
        repository,
        origin_verifier=service.origin_verifier,
        credential_broker=None,
    )

    with pytest.raises(CampaignAgentBrokerUnavailableError):
        unavailable.attach(_principal(), _attach_request(grant), now=NOW)
    assert (
        repository.public_view(
            CampaignAgentPublicViewQuery(workspace_id="workspace-a", campaign_id="campaign-1"),
            now=NOW,
        )["attachment"]
        is None
    )


def test_broker_delivery_failure_rolls_back_attachment_without_exposing_secret(setup):
    campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    rejected = []

    def reject(credential):
        rejected.append(credential)
        raise RuntimeError("BROKER-SECRET-CANARY")

    rejecting = CampaignAgentService(
        campaigns,
        repository,
        origin_verifier=service.origin_verifier,
        credential_broker=reject,
    )
    with pytest.raises(CampaignAgentBrokerUnavailableError) as error:
        rejecting.attach(
            _principal(),
            _attach_request(grant, idempotency_key="attach-rejected"),
            now=NOW,
        )
    assert "BROKER-SECRET-CANARY" not in str(error.value)
    assert (
        repository.public_view(
            CampaignAgentPublicViewQuery(workspace_id="workspace-a", campaign_id="campaign-1"),
            now=NOW,
        )["attachment"]
        is None
    )
    with sqlite3.connect(repository.db_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM campaign_agent_pending_attachments"
            ).fetchone()[0]
            == 0
        )
    recovered = []
    retrying = CampaignAgentService(
        campaigns,
        repository,
        origin_verifier=service.origin_verifier,
        credential_broker=recovered.append,
    )
    view, replayed = retrying.attach(
        _principal(),
        _attach_request(grant, idempotency_key="attach-corrected"),
        now=NOW,
    )
    assert replayed is False and view["attachment"]["status"] == "attached"
    heartbeat = CampaignAgentHeartbeatRequest(
        scope=grant.scope,
        agent_family=grant.agent_family,
        agent_origin=grant.agent_origin,
        agent_principal_id=grant.agent_principal_id,
        session_id=grant.session_id,
    )
    with pytest.raises(CampaignAgentCredentialError):
        retrying.heartbeat(rejected[0].raw_token, heartbeat, now=NOW)
    assert retrying.heartbeat(recovered[0].raw_token, heartbeat, now=NOW)["attachment"]


def test_broker_delivery_runs_after_durable_prepare_without_sqlite_write_lock(setup):
    campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    observed = {}

    def broker(credential):
        with sqlite3.connect(repository.db_path, timeout=0.1) as connection:
            connection.execute("BEGIN IMMEDIATE")
            observed["pending"] = connection.execute(
                "SELECT COUNT(*) FROM campaign_agent_pending_attachments"
            ).fetchone()[0]
            observed["stable"] = connection.execute(
                "SELECT COUNT(*) FROM campaign_agent_attachments"
            ).fetchone()[0]
        observed["credential"] = credential

    brokered = CampaignAgentService(
        campaigns,
        repository,
        origin_verifier=service.origin_verifier,
        credential_broker=broker,
    )
    view, replayed = brokered.attach(_principal(), _attach_request(grant), now=NOW)

    assert replayed is False
    assert observed["pending"] == 1
    assert observed["stable"] == 0
    assert view["attachment"]["status"] == "attached"
    with sqlite3.connect(repository.db_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM campaign_agent_pending_attachments"
            ).fetchone()[0]
            == 0
        )


def test_pending_activation_revalidates_durable_binding_after_broker_delivery(setup):
    campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    brokered = []

    def tampering_broker(credential):
        brokered.append(credential)
        with sqlite3.connect(repository.db_path) as connection:
            connection.execute("UPDATE campaign_agent_pending_attachments SET request_json = '{}'")

    tampered = CampaignAgentService(
        campaigns,
        repository,
        origin_verifier=service.origin_verifier,
        credential_broker=tampering_broker,
    )
    with pytest.raises(CampaignAgentIntegrityError):
        tampered.attach(_principal(), _attach_request(grant), now=NOW)
    view = repository.public_view(
        CampaignAgentPublicViewQuery(workspace_id="workspace-a", campaign_id="campaign-1"),
        now=NOW,
    )
    assert view["attachment"] is None
    heartbeat = CampaignAgentHeartbeatRequest(
        scope=grant.scope,
        agent_family=grant.agent_family,
        agent_origin=grant.agent_origin,
        agent_principal_id=grant.agent_principal_id,
        session_id=grant.session_id,
    )
    with pytest.raises(CampaignAgentCredentialError):
        tampered.heartbeat(brokered[0].raw_token, heartbeat, now=NOW)


def test_scope_session_origin_and_grant_bindings_cannot_be_substituted(setup):
    _campaigns, _repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)

    with pytest.raises(CampaignAgentAuthorizationError, match="human principal"):
        service.attach(
            _principal().model_copy(update={"actor_id": "other-human"}),
            _attach_request(grant, idempotency_key="bad-human"),
            now=NOW,
        )

    for field, value in (
        ("agent_family", CampaignAgentFamily.HERMES),
        ("agent_origin", "hermes-session-origin"),
        ("session_id", "session-2"),
        ("agent_principal_id", "other-agent"),
        (
            "granted_capabilities",
            (CampaignAgentCapability.TRAINING_LAUNCH,),
        ),
        ("scope", CampaignAgentScope(workspace_id="workspace-a", campaign_id="campaign-2")),
    ):
        with pytest.raises(CampaignAgentAuthorizationError, match="binding"):
            service.attach(
                _principal(),
                _attach_request(grant, **{field: value}, idempotency_key=f"bad-{field}"),
                now=NOW,
            )


def test_heartbeat_requires_exact_credential_binding_and_current_grant_revision(setup):
    _campaigns, _repository, service, delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    service.attach(_principal(), _attach_request(grant), now=NOW)
    heartbeat = CampaignAgentHeartbeatRequest(
        scope=grant.scope,
        agent_family=grant.agent_family,
        agent_origin=grant.agent_origin,
        agent_principal_id=grant.agent_principal_id,
        session_id=grant.session_id,
        resume_cursor="public-cursor-7",
        resume_sequence=7,
        expected_resume_cursor=None,
    )

    live = service.heartbeat(delivered[0].raw_token, heartbeat, now=NOW + timedelta(seconds=5))
    assert live["attachment"]["provenance"]["liveness"] == "live"
    assert live["attachment"]["provenance"]["resume_cursor"] == "public-cursor-7"
    repeated = service.heartbeat(
        delivered[0].raw_token,
        heartbeat.model_copy(
            update={
                "resume_cursor": "public-cursor-8",
                "resume_sequence": 8,
                "expected_resume_cursor": "public-cursor-7",
            }
        ),
        now=NOW + timedelta(seconds=6),
    )
    assert len(repeated["audit_events"]) == len(live["audit_events"])
    omitted = service.heartbeat(
        delivered[0].raw_token,
        CampaignAgentHeartbeatRequest(
            scope=grant.scope,
            agent_family=grant.agent_family,
            agent_origin=grant.agent_origin,
            agent_principal_id=grant.agent_principal_id,
            session_id=grant.session_id,
        ),
        now=NOW + timedelta(seconds=7),
    )
    assert omitted["attachment"]["provenance"]["resume_cursor"] == "public-cursor-8"
    with pytest.raises(CampaignAgentConflictError, match="cursor"):
        service.heartbeat(
            delivered[0].raw_token,
            heartbeat.model_copy(
                update={
                    "resume_cursor": "public-cursor-6",
                    "resume_sequence": 6,
                    "expected_resume_cursor": "public-cursor-8",
                }
            ),
            now=NOW + timedelta(seconds=8),
        )
    competing = [
        heartbeat.model_copy(
            update={
                "resume_cursor": f"public-cursor-9-{suffix}",
                "resume_sequence": 9,
                "expected_resume_cursor": "public-cursor-8",
            }
        )
        for suffix in ("a", "b")
    ]

    def advance(request):
        try:
            service.heartbeat(delivered[0].raw_token, request, now=NOW + timedelta(seconds=9))
            return "advanced"
        except CampaignAgentConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(advance, competing)) == ["advanced", "conflict"]
    with pytest.raises(CampaignAgentCredentialError):
        service.heartbeat(
            delivered[0].raw_token,
            heartbeat.model_copy(update={"session_id": "session-2"}),
            now=NOW + timedelta(seconds=6),
        )

    service.issue_grant(
        _principal(),
        _grant_request(idempotency_key="grant-escalated"),
        now=NOW + timedelta(seconds=10),
    )
    invalidated = service.public_view(
        _principal(),
        CampaignAgentPublicViewQuery(workspace_id="workspace-a", campaign_id="campaign-1"),
        now=NOW + timedelta(seconds=11),
    )
    assert invalidated["attachment"]["status"] == "revoked"
    assert invalidated["attachment"]["provenance"]["credential_status"] == "revoked"
    with pytest.raises(CampaignAgentCredentialError):
        service.heartbeat(delivered[0].raw_token, heartbeat, now=NOW + timedelta(seconds=12))


def test_heartbeat_cannot_regress_durable_last_seen_time(setup):
    _campaigns, repository, service, delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    service.attach(_principal(), _attach_request(grant), now=NOW)
    heartbeat = CampaignAgentHeartbeatRequest(
        scope=grant.scope,
        agent_family=grant.agent_family,
        agent_origin=grant.agent_origin,
        agent_principal_id=grant.agent_principal_id,
        session_id=grant.session_id,
    )

    service.heartbeat(delivered[0].raw_token, heartbeat, now=NOW + timedelta(seconds=10))
    service.heartbeat(delivered[0].raw_token, heartbeat, now=NOW + timedelta(seconds=5))

    with sqlite3.connect(repository.db_path) as connection:
        assert (
            connection.execute("SELECT last_seen_at FROM campaign_agent_attachments").fetchone()[0]
            == "2026-07-16T12:00:10Z"
        )


def test_protected_action_authority_is_exact_and_capability_bounded(setup):
    _campaigns, _repository, service, delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    service.attach(_principal(), _attach_request(grant), now=NOW)

    authorization = service.authorize_action(
        delivered[0].raw_token,
        _action_context(grant),
        required_capability=CampaignAgentCapability.CAMPAIGN_OBSERVE,
        now=NOW + timedelta(seconds=1),
    )

    principal = authorization.principal
    authorization.require_scope(grant.scope.workspace_id, grant.scope.campaign_id)
    assert principal.actor_id == grant.agent_principal_id
    assert principal.autonomy_profile is AutonomyProfile.CODEX_TRUSTED
    assert principal.workspace_ids == (grant.scope.workspace_id,)
    assert principal.capabilities == frozenset({Capability.CAMPAIGN_READ})
    assert delivered[0].raw_token not in repr(authorization)
    with pytest.raises(CampaignAgentAuthorizationError, match="scope"):
        authorization.require_scope(grant.scope.workspace_id, "campaign-2")
    with pytest.raises(CampaignAgentAuthorizationError, match="capability"):
        service.authorize_action(
            delivered[0].raw_token,
            _action_context(grant),
            required_capability=CampaignAgentCapability.TRAINING_LAUNCH,
            now=NOW + timedelta(seconds=1),
        )


def test_protected_action_principal_is_narrowed_to_the_adapter_capability(setup):
    _campaigns, _repository, service, delivered = setup
    grant = service.issue_grant(
        _principal(),
        _grant_request(
            granted_capabilities=(
                CampaignAgentCapability.CAMPAIGN_OBSERVE,
                CampaignAgentCapability.TRAINING_LAUNCH,
            ),
            idempotency_key="grant-multi-capability",
        ),
        now=NOW,
    )
    service.attach(
        _principal(),
        _attach_request(grant, idempotency_key="attach-multi-capability"),
        now=NOW,
    )

    authorization = service.authorize_action(
        delivered[0].raw_token,
        _action_context(grant),
        required_capability=CampaignAgentCapability.CAMPAIGN_OBSERVE,
        now=NOW + timedelta(seconds=1),
    )

    assert authorization.principal.capabilities == frozenset({Capability.CAMPAIGN_READ})


def test_protected_action_rechecks_provenance_revocation_and_grant_revision(setup):
    _campaigns, _repository, service, delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    attached, _ = service.attach(_principal(), _attach_request(grant), now=NOW)
    context = _action_context(grant)

    verified_origin = service.origin_verifier
    service.origin_verifier = lambda *_args: False
    with pytest.raises(CampaignAgentAuthorizationError, match="origin"):
        service.authorize_action(
            delivered[0].raw_token,
            context,
            required_capability=CampaignAgentCapability.CAMPAIGN_OBSERVE,
            now=NOW + timedelta(seconds=1),
        )
    service.origin_verifier = verified_origin

    for field, value in (
        ("agent_family", CampaignAgentFamily.HERMES),
        ("agent_origin", "other-origin"),
        ("agent_principal_id", "other-agent"),
        ("session_id", "session-2"),
        ("scope", CampaignAgentScope(workspace_id="workspace-a", campaign_id="campaign-2")),
    ):
        with pytest.raises((CampaignAgentCredentialError, CampaignAgentAuthorizationError)):
            service.authorize_action(
                delivered[0].raw_token,
                context.model_copy(update={field: value}),
                required_capability=CampaignAgentCapability.CAMPAIGN_OBSERVE,
                now=NOW + timedelta(seconds=1),
            )

    service.revoke(
        _principal(),
        CampaignAgentRevokeRequest(
            scope=grant.scope,
            attachment_id=attached["attachment"]["attachment_id"],
            attachment_version=1,
            idempotency_key="revoke-action-authority",
        ),
        now=NOW + timedelta(seconds=2),
    )
    with pytest.raises(CampaignAgentCredentialError):
        service.authorize_action(
            delivered[0].raw_token,
            context,
            required_capability=CampaignAgentCapability.CAMPAIGN_OBSERVE,
            now=NOW + timedelta(seconds=3),
        )


def test_hermes_action_authority_uses_the_bounded_profile(setup):
    _campaigns, _repository, service, delivered = setup
    grant = service.issue_grant(
        _principal(),
        _grant_request(
            agent_family=CampaignAgentFamily.HERMES,
            agent_origin="hermes-session-origin",
            agent_principal_id="hermes-campaign-agent",
            idempotency_key="grant-hermes-action",
        ),
        now=NOW,
    )
    service.attach(
        _principal(),
        _attach_request(grant, idempotency_key="attach-hermes-action"),
        now=NOW,
    )

    authorization = service.authorize_action(
        delivered[0].raw_token,
        _action_context(grant),
        required_capability=CampaignAgentCapability.CAMPAIGN_OBSERVE,
        now=NOW + timedelta(seconds=1),
    )

    assert authorization.principal.autonomy_profile is AutonomyProfile.HERMES_BOUNDED


def test_revoke_then_reattach_renews_credential_and_survives_restart(setup):
    campaigns, repository, service, delivered = setup
    first_grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    first, _ = service.attach(_principal(), _attach_request(first_grant), now=NOW)
    first_token = delivered[0].raw_token
    revoked, replayed = service.revoke(
        _principal(),
        CampaignAgentRevokeRequest(
            scope=first_grant.scope,
            attachment_id=first["attachment"]["attachment_id"],
            attachment_version=1,
            idempotency_key="revoke-1",
        ),
        now=NOW + timedelta(seconds=10),
    )
    assert replayed is False
    assert revoked["attachment"]["status"] == "revoked"
    assert revoked["attachment"]["provenance"]["credential_revocation_revision"] == 1

    second_grant = service.issue_grant(
        _principal(), _grant_request(idempotency_key="grant-2"), now=NOW + timedelta(seconds=11)
    )
    attached, _ = service.attach(
        _principal(),
        _attach_request(
            second_grant,
            base_attachment_version=2,
            idempotency_key="attach-2",
        ),
        now=NOW + timedelta(seconds=12),
    )
    assert attached["attachment"]["attachment_version"] == 3
    assert [item["kind"] for item in attached["attachment"]["receipts"]] == [
        "attach",
        "revoke",
        "attach",
    ]
    assert delivered[1].credential_id != delivered[0].credential_id
    assert delivered[1].issued_at >= datetime.fromisoformat(
        revoked["attachment"]["provenance"]["revoked_at"].replace("Z", "+00:00")
    )

    heartbeat = CampaignAgentHeartbeatRequest(
        scope=second_grant.scope,
        agent_family=second_grant.agent_family,
        agent_origin=second_grant.agent_origin,
        agent_principal_id=second_grant.agent_principal_id,
        session_id=second_grant.session_id,
    )
    restarted = CampaignAgentService(
        campaigns,
        CampaignAgentRepository(repository.db_path, sealer=_sealer()),
        origin_verifier=service.origin_verifier,
        credential_broker=delivered.append,
    )
    with pytest.raises(CampaignAgentCredentialError):
        restarted.heartbeat(first_token, heartbeat, now=NOW + timedelta(seconds=13))
    assert (
        restarted.heartbeat(delivered[1].raw_token, heartbeat, now=NOW + timedelta(seconds=13))[
            "attachment"
        ]["provenance"]["liveness"]
        == "live"
    )


def test_same_second_revoke_and_reattach_use_strict_canonical_transition_order(setup):
    _campaigns, _repository, service, _delivered = setup
    first_grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    first, _ = service.attach(_principal(), _attach_request(first_grant), now=NOW)
    revoked, _ = service.revoke(
        _principal(),
        CampaignAgentRevokeRequest(
            scope=first_grant.scope,
            attachment_id=first["attachment"]["attachment_id"],
            attachment_version=1,
            idempotency_key="revoke-same-second",
        ),
        now=NOW,
    )
    second_grant = service.issue_grant(
        _principal(), _grant_request(idempotency_key="grant-same-second"), now=NOW
    )
    attached, _ = service.attach(
        _principal(),
        _attach_request(
            second_grant,
            base_attachment_version=2,
            idempotency_key="attach-same-second",
        ),
        now=NOW,
    )

    times = [
        datetime.fromisoformat(receipt["occurred_at"].replace("Z", "+00:00"))
        for receipt in attached["attachment"]["receipts"]
    ]
    assert times == sorted(set(times))
    assert times[1] == datetime.fromisoformat(
        revoked["attachment"]["provenance"]["revoked_at"].replace("Z", "+00:00")
    )
    assert datetime.fromisoformat(attached["observed_at"].replace("Z", "+00:00")) >= times[-1]


def test_public_reads_are_campaign_scoped_and_audit_resume_is_bounded(setup):
    _campaigns, _repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    attached, _ = service.attach(_principal(), _attach_request(grant), now=NOW)
    latest_sequence = attached["audit_events"][0]["sequence"]

    resumed = service.public_view(
        _principal(),
        CampaignAgentPublicViewQuery(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            after_sequence=latest_sequence,
            limit=1,
        ),
        now=NOW,
    )
    sibling = service.public_view(
        _principal(),
        CampaignAgentPublicViewQuery(workspace_id="workspace-a", campaign_id="campaign-2"),
        now=NOW,
    )
    assert resumed["audit_events"] == []
    assert sibling["attachment"] is None
    with pytest.raises(PermissionError, match="workspace"):
        service.public_view(
            _principal(),
            CampaignAgentPublicViewQuery(workspace_id="workspace-b", campaign_id="campaign-1"),
            now=NOW,
        )


def test_audit_and_receipt_resume_pages_are_ascending_explicit_and_complete(setup):
    _campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    attached, _ = service.attach(_principal(), _attach_request(grant), now=NOW)
    service.revoke(
        _principal(),
        CampaignAgentRevokeRequest(
            scope=grant.scope,
            attachment_id=attached["attachment"]["attachment_id"],
            attachment_version=1,
            idempotency_key="revoke-page",
        ),
        now=NOW + timedelta(seconds=1),
    )

    first = repository.audit_page(grant.scope, after_sequence=0, limit=1)
    second = repository.audit_page(grant.scope, after_sequence=first["next_cursor"], limit=50)
    receipts = repository.receipt_page(grant.scope, after_version=0, limit=1)

    assert first["has_more"] is True
    assert first["next_cursor"] == first["items"][-1]["sequence"]
    assert [item["sequence"] for item in first["items"] + second["items"]] == sorted(
        item["sequence"] for item in first["items"] + second["items"]
    )
    assert second["has_more"] is False
    assert receipts["has_more"] is True
    assert receipts["next_cursor"] == 1


def test_forged_unkeyed_grant_digest_is_rejected_on_replay(setup):
    _campaigns, repository, service, _delivered = setup
    service.issue_grant(_principal(), _grant_request(), now=NOW)
    with sqlite3.connect(repository.db_path) as connection:
        row = connection.execute(
            "SELECT receipt_id, payload_json FROM campaign_agent_grants"
        ).fetchone()
        payload = json.loads(row[1])
        payload["session_id"] = "forged-session"
        connection.execute(
            "UPDATE campaign_agent_grants SET payload_json = ?, receipt_digest = ? WHERE receipt_id = ?",
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                f"sha256:{canonical_hash(payload)}",
                row[0],
            ),
        )

    with pytest.raises(CampaignAgentIntegrityError, match="grant receipt"):
        service.issue_grant(_principal(), _grant_request(), now=NOW)


def test_forged_unkeyed_transition_digest_is_rejected_before_projection(setup):
    _campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    service.attach(_principal(), _attach_request(grant), now=NOW)
    with sqlite3.connect(repository.db_path) as connection:
        row = connection.execute("SELECT * FROM campaign_agent_receipts").fetchone()
        forged_payload = {
            "receipt_id": row[0],
            "kind": row[4],
            "actor_id": "forged-human",
            "occurred_at": row[6],
            "idempotency_key": row[7],
            "attachment_version": row[8],
        }
        connection.execute(
            "UPDATE campaign_agent_receipts SET actor_id = ?, receipt_digest = ? WHERE receipt_id = ?",
            ("forged-human", f"sha256:{canonical_hash(forged_payload)}", row[0]),
        )

    with pytest.raises(CampaignAgentIntegrityError, match="transition receipt"):
        repository.public_view(
            CampaignAgentPublicViewQuery(workspace_id="workspace-a", campaign_id="campaign-1"),
            now=NOW,
        )


def test_projection_fails_closed_when_attachment_authority_is_tampered(setup):
    _campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    service.attach(_principal(), _attach_request(grant), now=NOW)
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute("UPDATE campaign_agent_attachments SET agent_origin = 'forged-origin'")

    with pytest.raises(CampaignAgentIntegrityError, match="attachment projection"):
        repository.public_view(
            CampaignAgentPublicViewQuery(workspace_id="workspace-a", campaign_id="campaign-1"),
            now=NOW,
        )


def test_projection_fails_closed_when_audit_event_is_tampered(setup):
    _campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    service.attach(_principal(), _attach_request(grant), now=NOW)
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE campaign_agent_events SET message_code = 'forged-secret-bearing-message'"
        )

    with pytest.raises(CampaignAgentIntegrityError, match="audit projection"):
        repository.public_view(
            CampaignAgentPublicViewQuery(workspace_id="workspace-a", campaign_id="campaign-1"),
            now=NOW,
        )


def test_attachment_lifetime_uses_bounded_rolling_window_without_exhaustion(setup):
    _campaigns, repository, service, _delivered = setup
    grant = service.issue_grant(_principal(), _grant_request(), now=NOW)
    view, _ = service.attach(_principal(), _attach_request(grant), now=NOW)
    for version in range(1, 25, 2):
        view, _ = service.revoke(
            _principal(),
            CampaignAgentRevokeRequest(
                scope=grant.scope,
                attachment_id=view["attachment"]["attachment_id"],
                attachment_version=version,
                idempotency_key=f"revoke-{version + 1}",
            ),
            now=NOW,
        )
        if version + 1 >= 25:
            break
        grant = service.issue_grant(
            _principal(),
            _grant_request(idempotency_key=f"grant-{version + 2}"),
            now=NOW,
        )
        view, _ = service.attach(
            _principal(),
            _attach_request(
                grant,
                base_attachment_version=version + 1,
                idempotency_key=f"attach-{version + 2}",
            ),
            now=NOW,
        )

    assert view["attachment"]["attachment_version"] == 25
    assert len(view["attachment"]["receipts"]) <= 20
    assert view["attachment"]["receipts"][0]["kind"] == "attach"
    assert view["attachment"]["receipt_window"]["has_earlier"] is True
    first = repository.receipt_page(grant.scope, after_version=0, limit=10)
    second = repository.receipt_page(grant.scope, after_version=first["next_cursor"], limit=20)
    assert len(first["items"] + second["items"]) == 25


def test_fixed_agent_capability_map_excludes_every_prohibited_authority():
    mapped = mapped_campaign_capabilities(tuple(CampaignAgentCapability))

    assert mapped.isdisjoint(PROHIBITED_CAMPAIGN_AGENT_CAPABILITIES)
    assert Capability.COMPUTE_AMEND_BUDGET not in mapped
    assert Capability.PROMOTION_DECIDE not in mapped
    assert Capability.ARTIFACT_PUBLISH_HF not in mapped
    assert Capability.COMPUTE_FORCE_STOP not in mapped
