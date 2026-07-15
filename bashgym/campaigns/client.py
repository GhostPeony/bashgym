"""Authenticated JSON client for campaign CLI and local agent projections.

The client accepts a credential *reference* only.  The referenced refresh token
is resolved immediately before exchange, while short-lived access credentials
remain process-local and are refreshed once after an HTTP 401 response.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from bashgym import secrets as secret_store
from bashgym.mcp.policy import validate_secret_ref_name

_DOMAIN_REJECTION_CODES = frozenset(
    {
        "campaign_artifact_mismatch",
        "campaign_budget_exceeded",
        "campaign_compute_unavailable",
        "campaign_gate_failed",
        "campaign_protected_lease_denied",
    }
)
_CONFLICT_CODES = frozenset(
    {
        "campaign_action_already_claimed",
        "campaign_idempotency_conflict",
        "campaign_invalid_transition",
        "campaign_version_conflict",
    }
)


def campaign_exit_code(status_code: int | None, error_code: str) -> int:
    """Map stable campaign/HTTP errors to a shell-friendly exit code."""

    if status_code == 401 or error_code == "campaign_auth_required":
        return 3
    if (
        status_code == 403
        or error_code.startswith("campaign_capability_required")
        or error_code
        in {
            "campaign_scope_denied",
            "campaign_workspace_forbidden",
        }
    ):
        return 4
    if status_code == 404 or error_code in {
        "campaign_not_found",
        "campaign_attempt_not_found",
    }:
        return 5
    if status_code == 409 or error_code in _CONFLICT_CODES:
        return 6
    if error_code in _DOMAIN_REJECTION_CODES:
        return 7
    if status_code is None or status_code >= 500:
        return 8
    return 2


class CampaignClientError(RuntimeError):
    """Secret-free campaign client failure with stable CLI projection fields."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.exit_code = campaign_exit_code(status_code, code)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "status_code": self.status_code,
        }


SecretResolver = Callable[[str], str | None]
_SAFE_HEADER_NAMES = {
    "idempotency-key": "Idempotency-Key",
    "x-correlation-id": "X-Correlation-ID",
}
_SAFE_HEADER_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")


class CampaignApiClient:
    """Synchronous campaign API client with refresh-token exchange and 401 retry."""

    def __init__(
        self,
        *,
        api_base: str,
        credential_ref: str,
        secret_resolver: SecretResolver = secret_store.get_secret,
        timeout: float = 15.0,
    ) -> None:
        normalized_base = api_base.strip().rstrip("/")
        normalized_ref = credential_ref.strip()
        if not normalized_base:
            raise ValueError("campaign API base is required")
        if not normalized_ref:
            raise ValueError("campaign credential reference is required")
        if normalized_ref.startswith(("bgca.", "bgcb.", "bgcr.")):
            raise ValueError("pass a credential reference, never a raw campaign token")
        validate_secret_ref_name(normalized_ref)
        self.api_base = normalized_base
        self.credential_ref = normalized_ref
        self._secret_resolver = secret_resolver
        self._timeout = timeout
        self._access_token: str | None = None

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Send one authenticated API request, refreshing once after a 401."""

        access_token = self._access_token or self._exchange_refresh()
        try:
            return self._send(
                method,
                path,
                query=query,
                payload=payload,
                headers=headers,
                token=access_token,
            )
        except CampaignClientError as exc:
            if exc.status_code != 401:
                raise
        self._access_token = None
        access_token = self._exchange_refresh()
        return self._send(
            method,
            path,
            query=query,
            payload=payload,
            headers=headers,
            token=access_token,
        )

    def _exchange_refresh(self) -> str:
        refresh_token = self._secret_resolver(self.credential_ref)
        if not refresh_token:
            raise CampaignClientError(
                "campaign_auth_required",
                "Campaign credential reference is not configured.",
                status_code=401,
            )
        response = self._send(
            "POST",
            "/campaign-auth/exchange",
            token=refresh_token,
        )
        if not isinstance(response, dict):
            raise CampaignClientError(
                "campaign_auth_response_invalid",
                "Campaign credential exchange returned an invalid response.",
                status_code=502,
            )
        raw_token = response.get("raw_token")
        if not isinstance(raw_token, str) or not raw_token.startswith("bgca."):
            raise CampaignClientError(
                "campaign_auth_response_invalid",
                "Campaign credential exchange returned an invalid response.",
                status_code=502,
            )
        self._access_token = raw_token
        return raw_token

    def _send(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        token: str,
    ) -> Any:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("campaign API paths must be relative to the configured API base")
        query_items = {key: value for key, value in (query or {}).items() if value is not None}
        encoded_query = urllib.parse.urlencode(query_items, doseq=True)
        url = f"{self.api_base}{path}"
        if encoded_query:
            url = f"{url}?{encoded_query}"
        data = json.dumps(dict(payload)).encode("utf-8") if payload is not None else None
        request_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
        }
        for name, value in (headers or {}).items():
            canonical_name = _SAFE_HEADER_NAMES.get(name.strip().lower())
            if canonical_name is None:
                raise ValueError(f"campaign request header is not allowed: {name}")
            normalized_value = value.strip()
            if not _SAFE_HEADER_VALUE.fullmatch(normalized_value):
                raise ValueError(f"campaign request header has an invalid value: {canonical_name}")
            request_headers[canonical_name] = normalized_value
        request = urllib.request.Request(
            url,
            data=data,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise CampaignClientError(
                "campaign_api_unavailable",
                "The BashGym campaign API is unavailable.",
                retryable=True,
            ) from exc
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise CampaignClientError(
                "campaign_response_invalid",
                "The BashGym campaign API returned invalid JSON.",
                status_code=502,
            ) from exc

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> CampaignClientError:
        try:
            body = json.loads(exc.read().decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, dict):
            code = str(detail.get("code") or f"campaign_http_{exc.code}")
            message = str(detail.get("message") or "Campaign request failed.")
            retryable = bool(detail.get("retryable", exc.code >= 500))
        elif isinstance(detail, str):
            code = f"campaign_http_{exc.code}"
            message = detail
            retryable = exc.code >= 500
        else:
            code = (
                str(body.get("code") or f"campaign_http_{exc.code}")
                if isinstance(body, dict)
                else f"campaign_http_{exc.code}"
            )
            message = (
                str(body.get("message") or "Campaign request failed.")
                if isinstance(body, dict)
                else "Campaign request failed."
            )
            retryable = exc.code >= 500
        if exc.code == 401:
            code = "campaign_auth_required"
            message = "Campaign authentication is required."
            retryable = False
        return CampaignClientError(
            code,
            message,
            status_code=exc.code,
            retryable=retryable,
        )


__all__ = [
    "CampaignApiClient",
    "CampaignClientError",
    "campaign_exit_code",
]
