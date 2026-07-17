"""Desktop-attested campaign-agent session and encrypted delivery tests."""

from __future__ import annotations

import base64
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import ValidationError

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.campaign_agent_contracts import (
    CampaignAgentFamily,
    CampaignAgentHostSessionRegistrationRequest,
    CampaignAgentScope,
)
from bashgym.campaigns.campaign_agent_sessions import (
    CampaignAgentSessionAuthorizationError,
    CampaignAgentSessionConflictError,
    CampaignAgentSessionIntegrityError,
    CampaignAgentSessionOriginVerifier,
    CampaignAgentSessionRepository,
    CampaignAgentSessionService,
    EncryptedCampaignAgentCredentialBroker,
)
from bashgym.campaigns.campaign_agents import BrokeredCampaignAgentCredential
from bashgym.campaigns.contracts import (
    ActorPrincipal,
    AutonomyProfile,
    Capability,
    CredentialKind,
)

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _keypair() -> tuple[X25519PrivateKey, str]:
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private, _b64(public)


def _principal(
    *,
    actor_id: str = "desktop-user",
    profile: AutonomyProfile = AutonomyProfile.DESKTOP_USER,
) -> ActorPrincipal:
    return ActorPrincipal(
        actor_id=actor_id,
        autonomy_profile=profile,
        credential_id=f"access-{actor_id}",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=("workspace-a",),
        capabilities=frozenset(Capability),
        expires_at=NOW + timedelta(hours=1),
    )


def _request(public_key: str, **updates) -> CampaignAgentHostSessionRegistrationRequest:
    values = {
        "scope": CampaignAgentScope(workspace_id="workspace-a", campaign_id="campaign-1"),
        "agent_family": CampaignAgentFamily.CODEX,
        "agent_origin": "electron-main",
        "agent_principal_id": "codex-agent-1",
        "session_id": "pty-session-1",
        "ephemeral_public_key": public_key,
        "ttl_seconds": 300,
        "idempotency_key": "register-1",
    }
    values.update(updates)
    return CampaignAgentHostSessionRegistrationRequest(**values)


def _credential(request, credential_id="credential-1") -> BrokeredCampaignAgentCredential:
    return BrokeredCampaignAgentCredential(
        attachment_id=f"attachment-{credential_id}",
        credential_id=credential_id,
        raw_token=f"bgag.{credential_id}.secret",
        workspace_id=request.scope.workspace_id,
        campaign_id=request.scope.campaign_id,
        agent_family=request.agent_family,
        agent_origin=request.agent_origin,
        session_id=request.session_id,
        agent_principal_id=request.agent_principal_id,
        granted_capabilities=(),
        authorization_revision=1,
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )


@pytest.fixture
def sessions(tmp_path):
    repository = CampaignAgentSessionRepository(
        tmp_path / "campaigns.sqlite3",
        sealer=ArtifactSealer(b"session-authority-test-key-material", key_version="test-v1"),
    )
    repository.initialize()
    service = CampaignAgentSessionService(repository)
    return repository, service


def test_registration_requires_desktop_authority_and_is_exact_replay_safe(sessions):
    repository, service = sessions
    _private, public_key = _keypair()
    request = _request(public_key)

    receipt = service.register(_principal(), request, now=NOW)
    replay = service.register(_principal(), request, now=NOW + timedelta(seconds=2))

    assert replay == receipt
    assert receipt.public_key_digest.startswith("sha256:")
    assert not hasattr(receipt, "ephemeral_public_key")
    with pytest.raises(CampaignAgentSessionAuthorizationError):
        service.register(
            _principal(profile=AutonomyProfile.CODEX_TRUSTED),
            _request(public_key, idempotency_key="register-agent"),
            now=NOW,
        )

    _other_private, other_key = _keypair()
    with pytest.raises(CampaignAgentSessionConflictError):
        service.register(
            _principal(),
            _request(other_key, idempotency_key="register-1"),
            now=NOW,
        )
    with pytest.raises(CampaignAgentSessionConflictError):
        service.register(
            _principal(),
            _request(
                public_key,
                session_id="pty-session-substitution",
                idempotency_key="register-3",
            ),
            now=NOW,
        )
    assert repository.get_live_exact(request, now=NOW).registration_id == receipt.registration_id


def test_same_desktop_can_rotate_live_exact_session_and_revokes_pending_delivery(sessions):
    repository, service = sessions
    _old_private, old_key = _keypair()
    old_request = _request(old_key)
    old_receipt = service.register(_principal(), old_request, now=NOW)
    EncryptedCampaignAgentCredentialBroker(repository, clock=lambda: NOW)(
        _credential(old_request)
    )
    _new_private, new_key = _keypair()
    renewal = _request(new_key, idempotency_key="renew-session-1")

    new_receipt = service.register(
        _principal(), renewal, now=NOW + timedelta(minutes=2)
    )

    assert new_receipt.registration_id != old_receipt.registration_id
    assert repository.get_registration(old_receipt.registration_id)["status"] == "revoked"
    with sqlite3.connect(repository.db_path) as connection:
        connection.row_factory = sqlite3.Row
        revoked_envelope = connection.execute(
            "SELECT * FROM campaign_agent_delivery_envelopes WHERE registration_id=?",
            (old_receipt.registration_id,),
        ).fetchone()
    assert revoked_envelope is not None
    assert revoked_envelope["state"] == "revoked"
    repository._verify_delivery(revoked_envelope)
    with pytest.raises(CampaignAgentSessionAuthorizationError):
        service.claim(_principal(), old_receipt.registration_id, now=NOW + timedelta(minutes=2))
    assert repository.get_live_exact(renewal, now=NOW + timedelta(minutes=2)).registration_id == (
        new_receipt.registration_id
    )

    _third_private, third_key = _keypair()
    with pytest.raises(CampaignAgentSessionAuthorizationError):
        service.register(
            _principal(actor_id="other-desktop"),
            _request(third_key, idempotency_key="foreign-renewal"),
            now=NOW + timedelta(minutes=3),
        )
    with pytest.raises(CampaignAgentSessionConflictError):
        service.register(
            _principal(),
            _request(new_key, idempotency_key="reused-renewal-key"),
            now=NOW + timedelta(minutes=3),
        )


def test_expired_exact_tuple_can_reregister_and_resolves_only_new_live_row(sessions):
    repository, service = sessions
    _old_private, old_key = _keypair()
    old_request = _request(old_key, ttl_seconds=30)
    old_receipt = service.register(_principal(), old_request, now=NOW)
    _new_private, new_key = _keypair()
    new_request = _request(new_key, idempotency_key="register-2", ttl_seconds=60)

    new_receipt = service.register(
        _principal(), new_request, now=NOW + timedelta(seconds=31)
    )
    resolved = repository.find_live_exact(
        new_request.scope,
        new_request.agent_family,
        new_request.agent_origin,
        new_request.session_id,
        new_request.agent_principal_id,
        now=NOW + timedelta(seconds=31),
    )

    assert resolved is not None
    assert resolved["registration_id"] == new_receipt.registration_id
    assert resolved["registration_id"] != old_receipt.registration_id


def test_registration_rejects_unusable_low_order_x25519_key():
    all_zero_public_key = _b64(b"\x00" * 32)

    with pytest.raises(ValidationError, match="usable X25519 public key required"):
        _request(all_zero_public_key)


def test_origin_verification_is_exact_expiring_revocable_and_tamper_evident(sessions):
    repository, service = sessions
    _private, public_key = _keypair()
    request = _request(public_key, ttl_seconds=60)
    receipt = service.register(_principal(), request, now=NOW)
    verifier = CampaignAgentSessionOriginVerifier(repository, clock=lambda: NOW)

    assert verifier(
        request.scope,
        request.agent_family,
        request.agent_origin,
        request.session_id,
        request.agent_principal_id,
    )
    assert not verifier(
        CampaignAgentScope(workspace_id="workspace-a", campaign_id="campaign-other"),
        request.agent_family,
        request.agent_origin,
        request.session_id,
        request.agent_principal_id,
    )
    assert not CampaignAgentSessionOriginVerifier(
        repository, clock=lambda: NOW + timedelta(seconds=61)
    )(
        request.scope,
        request.agent_family,
        request.agent_origin,
        request.session_id,
        request.agent_principal_id,
    )

    service.revoke(_principal(), receipt.registration_id, now=NOW + timedelta(seconds=10))
    assert not verifier(
        request.scope,
        request.agent_family,
        request.agent_origin,
        request.session_id,
        request.agent_principal_id,
    )

    _private2, key2 = _keypair()
    receipt2 = service.register(
        _principal(), _request(key2, session_id="pty-session-2", idempotency_key="register-2"), now=NOW
    )
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE campaign_agent_host_sessions SET agent_origin = 'tampered' WHERE registration_id = ?",
            (receipt2.registration_id,),
        )
        connection.commit()
    with pytest.raises(CampaignAgentSessionIntegrityError):
        repository.get_registration(receipt2.registration_id)


def test_broker_persists_only_ciphertext_and_same_desktop_claims_once(sessions):
    repository, service = sessions
    private, public_key = _keypair()
    request = _request(public_key)
    receipt = service.register(_principal(), request, now=NOW)
    broker = EncryptedCampaignAgentCredentialBroker(repository, clock=lambda: NOW)
    raw_token = "bgag.credential-1.SECRET-CANARY"
    credential = BrokeredCampaignAgentCredential(
        attachment_id="attachment-1",
        credential_id="credential-1",
        raw_token=raw_token,
        workspace_id=request.scope.workspace_id,
        campaign_id=request.scope.campaign_id,
        agent_family=request.agent_family,
        agent_origin=request.agent_origin,
        session_id=request.session_id,
        agent_principal_id=request.agent_principal_id,
        granted_capabilities=(),
        authorization_revision=1,
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )

    broker(credential)
    assert raw_token not in repository.db_path.read_bytes().decode(errors="ignore")
    with pytest.raises(CampaignAgentSessionAuthorizationError):
        service.claim(_principal(actor_id="other-desktop"), receipt.registration_id, now=NOW)

    envelope = service.claim(_principal(), receipt.registration_id, now=NOW)
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

    shared = private.exchange(X25519PublicKey.from_public_bytes(_decode(envelope.ephemeral_public_key)))
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_decode(envelope.hkdf_salt),
        info=envelope.hkdf_info.encode(),
    ).derive(shared)
    plaintext = ChaCha20Poly1305(key).decrypt(
        _decode(envelope.nonce),
        _decode(envelope.ciphertext),
        envelope.aad_json.encode(),
    )
    payload = json.loads(plaintext)
    assert payload["raw_token"] == raw_token
    assert payload["registration_id"] == receipt.registration_id
    assert raw_token not in envelope.model_dump_json()
    with pytest.raises(CampaignAgentSessionConflictError):
        service.claim(_principal(), receipt.registration_id, now=NOW)


def test_ciphertext_tampering_and_concurrent_double_claim_fail_closed(sessions):
    repository, service = sessions
    _private, public_key = _keypair()
    request = _request(public_key)
    receipt = service.register(_principal(), request, now=NOW)
    broker = EncryptedCampaignAgentCredentialBroker(repository, clock=lambda: NOW)

    def deliver(credential_id: str):
        broker(
            BrokeredCampaignAgentCredential(
                attachment_id=f"attachment-{credential_id}",
                credential_id=credential_id,
                raw_token=f"bgag.{credential_id}.secret",
                workspace_id=request.scope.workspace_id,
                campaign_id=request.scope.campaign_id,
                agent_family=request.agent_family,
                agent_origin=request.agent_origin,
                session_id=request.session_id,
                agent_principal_id=request.agent_principal_id,
                granted_capabilities=(),
                authorization_revision=1,
                issued_at=NOW,
                expires_at=NOW + timedelta(hours=1),
            )
        )

    deliver("credential-1")
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE campaign_agent_delivery_envelopes SET ciphertext = ciphertext || 'A'"
        )
        connection.commit()
    with pytest.raises(CampaignAgentSessionIntegrityError):
        service.claim(_principal(), receipt.registration_id, now=NOW)

    _private2, key2 = _keypair()
    receipt2 = service.register(
        _principal(), _request(key2, session_id="pty-session-2", idempotency_key="register-2"), now=NOW
    )
    request2 = _request(key2, session_id="pty-session-2", idempotency_key="register-2")
    broker2 = EncryptedCampaignAgentCredentialBroker(repository, clock=lambda: NOW)
    credential = BrokeredCampaignAgentCredential(
        attachment_id="attachment-2",
        credential_id="credential-2",
        raw_token="bgag.credential-2.secret",
        workspace_id=request2.scope.workspace_id,
        campaign_id=request2.scope.campaign_id,
        agent_family=request2.agent_family,
        agent_origin=request2.agent_origin,
        session_id=request2.session_id,
        agent_principal_id=request2.agent_principal_id,
        granted_capabilities=(),
        authorization_revision=1,
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )
    broker2(credential)

    def claim():
        try:
            return service.claim(_principal(), receipt2.registration_id, now=NOW).envelope_id
        except CampaignAgentSessionConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _value: claim(), range(2)))
    assert results.count("conflict") == 1


def test_broker_rejects_missing_expired_or_revoked_registration(sessions):
    repository, service = sessions
    _private, public_key = _keypair()
    request = _request(public_key, ttl_seconds=60)
    receipt = service.register(_principal(), request, now=NOW)
    credential = BrokeredCampaignAgentCredential(
        attachment_id="attachment-1",
        credential_id="credential-1",
        raw_token="bgag.credential-1.secret",
        workspace_id=request.scope.workspace_id,
        campaign_id=request.scope.campaign_id,
        agent_family=request.agent_family,
        agent_origin=request.agent_origin,
        session_id=request.session_id,
        agent_principal_id=request.agent_principal_id,
        granted_capabilities=(),
        authorization_revision=1,
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )

    with pytest.raises(CampaignAgentSessionAuthorizationError):
        EncryptedCampaignAgentCredentialBroker(
            repository, clock=lambda: NOW + timedelta(seconds=61)
        )(credential)
    service.revoke(_principal(), receipt.registration_id, now=NOW)
    with pytest.raises(CampaignAgentSessionAuthorizationError):
        EncryptedCampaignAgentCredentialBroker(repository, clock=lambda: NOW)(credential)
