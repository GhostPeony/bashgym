"""Opaque, revocable actor credentials for campaign-only authorization."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import timedelta
from uuid import uuid4

from bashgym.campaigns.contracts import (
    CODEX_CAPABILITIES,
    DESKTOP_LOCAL_SCOPE,
    HERMES_CAPABILITIES,
    ActorPrincipal,
    AutonomyProfile,
    CredentialKind,
    IssuedCredential,
    utc_now,
)
from bashgym.campaigns.persistence import (
    CampaignPersistenceError,
    CampaignRepository,
    StoredAccessToken,
    StoredCredential,
)


class CampaignAuthenticationError(PermissionError):
    """Stable fail-closed error that does not disclose credential state."""

    code = "campaign_auth_required"

    def __init__(self) -> None:
        super().__init__(self.code)


_TOKEN_PREFIX = {
    CredentialKind.REFRESH: "bgcr",
    CredentialKind.ACCESS: "bgca",
    CredentialKind.DESKTOP_BOOTSTRAP: "bgcb",
}
_PBKDF2_ITERATIONS = 310_000
_DESKTOP_ACTOR_ID = "desktop-user"


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _derive(secret: str, salt: bytes) -> str:
    return _encode(
        hashlib.pbkdf2_hmac(
            "sha256",
            secret.encode("utf-8"),
            salt,
            _PBKDF2_ITERATIONS,
        )
    )


def _new_token(kind: CredentialKind, token_id: str) -> tuple[str, str, str]:
    secret = secrets.token_urlsafe(32)
    salt = secrets.token_bytes(16)
    raw = f"{_TOKEN_PREFIX[kind]}.{token_id}.{secret}"
    return raw, _encode(salt), _derive(secret, salt)


def _decode_salt(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _parse(raw_token: str, expected_kind: CredentialKind) -> tuple[str, str]:
    try:
        prefix, token_id, secret = raw_token.split(".", 2)
    except ValueError as exc:
        raise CampaignAuthenticationError() from exc
    if prefix != _TOKEN_PREFIX[expected_kind] or not token_id or len(secret) < 32:
        raise CampaignAuthenticationError()
    return token_id, secret


def _verify(secret: str, salt: str, expected_hash: str) -> bool:
    actual = _derive(secret, _decode_salt(salt))
    return hmac.compare_digest(actual, expected_hash)


def capabilities_for(profile: AutonomyProfile):
    """Return the code-owned, non-upgradeable capability matrix."""

    if profile == AutonomyProfile.HERMES_BOUNDED:
        return HERMES_CAPABILITIES
    if profile in {AutonomyProfile.CODEX_TRUSTED, AutonomyProfile.DESKTOP_USER}:
        return CODEX_CAPABILITIES
    raise CampaignAuthenticationError()


class CampaignAuthService:
    """Issue, exchange, resolve, and revoke campaign credentials."""

    def __init__(self, repository: CampaignRepository):
        self.repository = repository

    def issue_refresh_credential(
        self,
        *,
        actor_id: str,
        autonomy_profile: AutonomyProfile,
        workspace_ids: tuple[str, ...],
        ttl: timedelta = timedelta(days=30),
    ) -> IssuedCredential:
        """Provision a profile-bound refresh credential and return it exactly once."""

        normalized_workspaces = tuple(sorted(set(workspace_ids)))
        if not normalized_workspaces:
            raise ValueError("a campaign credential requires at least one workspace")
        # Force validation of the server-owned profile matrix before persisting it.
        capabilities_for(autonomy_profile)
        credential_id = f"cred-{uuid4().hex}"
        raw, salt, token_hash = _new_token(CredentialKind.REFRESH, credential_id)
        now = utc_now()
        expires_at = now + ttl
        self.repository.insert_actor_credential(
            StoredCredential(
                credential_id=credential_id,
                actor_id=actor_id,
                autonomy_profile=autonomy_profile.value,
                credential_kind=CredentialKind.REFRESH.value,
                workspace_ids=normalized_workspaces,
                token_salt=salt,
                token_hash=token_hash,
                issued_at=now,
                expires_at=expires_at,
                token_not_before=now,
                revoked_at=None,
            )
        )
        return IssuedCredential(
            credential_id=credential_id,
            raw_token=raw,
            kind=CredentialKind.REFRESH,
            expires_at=expires_at,
        )

    def install_desktop_bootstrap(
        self,
        raw_bootstrap_token: str,
        *,
        ttl: timedelta = timedelta(hours=24),
    ) -> str:
        """Install one Electron-launch bootstrap without persisting its raw secret."""

        credential_id, secret = _parse(
            raw_bootstrap_token, CredentialKind.DESKTOP_BOOTSTRAP
        )
        existing = self.repository.get_actor_credential(credential_id)
        now = utc_now()
        if existing is not None:
            if (
                existing.actor_id != _DESKTOP_ACTOR_ID
                or existing.autonomy_profile != AutonomyProfile.DESKTOP_USER.value
                or existing.credential_kind != CredentialKind.DESKTOP_BOOTSTRAP.value
                or existing.workspace_ids != (DESKTOP_LOCAL_SCOPE,)
                or existing.revoked_at is not None
                or existing.expires_at <= now
                or not _verify(secret, existing.token_salt, existing.token_hash)
            ):
                raise CampaignAuthenticationError()
            return credential_id
        salt = secrets.token_bytes(16)
        self.repository.insert_actor_credential(
            StoredCredential(
                credential_id=credential_id,
                actor_id=_DESKTOP_ACTOR_ID,
                autonomy_profile=AutonomyProfile.DESKTOP_USER.value,
                credential_kind=CredentialKind.DESKTOP_BOOTSTRAP.value,
                workspace_ids=(DESKTOP_LOCAL_SCOPE,),
                token_salt=_encode(salt),
                token_hash=_derive(secret, salt),
                issued_at=now,
                expires_at=now + ttl,
                token_not_before=now,
                revoked_at=None,
            )
        )
        return credential_id

    def _exchange_parent(
        self,
        raw_parent_token: str,
        *,
        parent_kind: CredentialKind,
        access_ttl: timedelta,
    ) -> IssuedCredential:
        credential_id, secret = _parse(raw_parent_token, parent_kind)
        parent = self.repository.get_actor_credential(credential_id)
        now = utc_now()
        if (
            parent is None
            or parent.credential_kind != parent_kind.value
            or parent.revoked_at is not None
            or parent.expires_at <= now
            or parent.token_not_before > now
            or not _verify(secret, parent.token_salt, parent.token_hash)
        ):
            raise CampaignAuthenticationError()
        access_token_id = f"access-{uuid4().hex}"
        raw, salt, token_hash = _new_token(CredentialKind.ACCESS, access_token_id)
        expires_at = min(now + access_ttl, parent.expires_at)
        try:
            self.repository.insert_access_token(
                StoredAccessToken(
                    access_token_id=access_token_id,
                    credential_id=credential_id,
                    token_salt=salt,
                    token_hash=token_hash,
                    issued_at=now,
                    expires_at=expires_at,
                    revoked_at=None,
                )
            )
        except CampaignPersistenceError as exc:
            raise CampaignAuthenticationError() from exc
        return IssuedCredential(
            credential_id=access_token_id,
            raw_token=raw,
            kind=CredentialKind.ACCESS,
            expires_at=expires_at,
        )

    def exchange_refresh(
        self,
        raw_refresh_token: str,
        *,
        access_ttl: timedelta = timedelta(hours=1),
    ) -> IssuedCredential:
        """Exchange a valid parent without accepting actor/profile/workspace overrides."""

        return self._exchange_parent(
            raw_refresh_token,
            parent_kind=CredentialKind.REFRESH,
            access_ttl=access_ttl,
        )

    def exchange_desktop_bootstrap(
        self,
        raw_bootstrap_token: str,
        *,
        access_ttl: timedelta = timedelta(minutes=15),
    ) -> IssuedCredential:
        """Exchange the current Electron launch secret for an in-memory access token."""

        return self._exchange_parent(
            raw_bootstrap_token,
            parent_kind=CredentialKind.DESKTOP_BOOTSTRAP,
            access_ttl=access_ttl,
        )

    def authenticate_access(self, raw_access_token: str | None) -> ActorPrincipal:
        """Resolve authority from storage on every request, including parent revocation."""

        if not raw_access_token:
            raise CampaignAuthenticationError()
        access_token_id, secret = _parse(raw_access_token, CredentialKind.ACCESS)
        resolved = self.repository.get_access_with_parent(access_token_id)
        now = utc_now()
        if resolved is None:
            raise CampaignAuthenticationError()
        access, parent = resolved
        if (
            access.revoked_at is not None
            or parent.revoked_at is not None
            or access.expires_at <= now
            or parent.expires_at <= now
            or access.issued_at < parent.token_not_before
            or not _verify(secret, access.token_salt, access.token_hash)
        ):
            raise CampaignAuthenticationError()
        profile = AutonomyProfile(parent.autonomy_profile)
        return ActorPrincipal(
            actor_id=parent.actor_id,
            autonomy_profile=profile,
            credential_id=parent.credential_id,
            credential_kind=CredentialKind.ACCESS,
            workspace_ids=parent.workspace_ids,
            capabilities=capabilities_for(profile),
            expires_at=access.expires_at,
        )

    def revoke_credential(self, credential_id: str, *, reason: str) -> int:
        """Immediately revoke the refresh credential and every live child."""

        return self.repository.revoke_actor_credential(
            credential_id,
            audit_event_id=f"auth-{uuid4().hex}",
            reason=reason,
        )


__all__ = [
    "CampaignAuthService",
    "CampaignAuthenticationError",
    "capabilities_for",
]
