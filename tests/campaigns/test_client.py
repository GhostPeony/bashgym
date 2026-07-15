from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse

import pytest

from bashgym.campaigns.client import (
    CampaignApiClient,
    CampaignClientError,
    campaign_exit_code,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def http_error(request, status, code, message):
    return urllib.error.HTTPError(
        request.full_url,
        status,
        message,
        hdrs=None,
        fp=io.BytesIO(json.dumps({"detail": {"code": code, "message": message}}).encode("utf-8")),
    )


def test_client_resolves_reference_exchanges_and_sends_bounded_query(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/campaign-auth/exchange"):
            assert request.get_header("Authorization") == "Bearer bgcr.parent.refresh-secret"
            return Response({"raw_token": "bgca.access.short-lived-secret"})
        return Response({"events": [], "next_cursor": 12})

    monkeypatch.setattr("bashgym.campaigns.client.urllib.request.urlopen", urlopen)
    client = CampaignApiClient(
        api_base="http://localhost:8003/api/",
        credential_ref="BASHGYM_CAMPAIGN_CODEX_REFRESH",
        secret_resolver=lambda ref: (
            "bgcr.parent.refresh-secret" if ref == "BASHGYM_CAMPAIGN_CODEX_REFRESH" else None
        ),
    )
    result = client.request_json(
        "GET",
        "/campaigns/campaign-1/events",
        query={"workspace_id": "workspace-a", "after_cursor": 10, "limit": 200},
    )

    assert result == {"events": [], "next_cursor": 12}
    assert len(requests) == 2
    parsed = urllib.parse.urlparse(requests[-1].full_url)
    assert parsed.path == "/api/campaigns/campaign-1/events"
    assert urllib.parse.parse_qs(parsed.query) == {
        "workspace_id": ["workspace-a"],
        "after_cursor": ["10"],
        "limit": ["200"],
    }
    assert requests[-1].get_header("Authorization") == "Bearer bgca.access.short-lived-secret"
    assert "secret" not in requests[-1].full_url


def test_client_reexchanges_once_after_resource_401(monkeypatch):
    accesses = iter(["bgca.first.access-secret", "bgca.second.access-secret"])
    resource_calls = 0
    exchanges = 0

    def urlopen(request, timeout):
        nonlocal resource_calls, exchanges
        if request.full_url.endswith("/campaign-auth/exchange"):
            exchanges += 1
            return Response({"raw_token": next(accesses)})
        resource_calls += 1
        if resource_calls == 1:
            raise http_error(request, 401, "anything", "do not disclose credential state")
        assert request.get_header("Authorization") == "Bearer bgca.second.access-secret"
        return Response({"campaign_id": "campaign-1"})

    monkeypatch.setattr("bashgym.campaigns.client.urllib.request.urlopen", urlopen)
    client = CampaignApiClient(
        api_base="http://localhost:8003/api",
        credential_ref="CAMPAIGN_REFRESH",
        secret_resolver=lambda _ref: "bgcr.parent.refresh-secret",
    )

    assert client.request_json("GET", "/campaigns/campaign-1") == {"campaign_id": "campaign-1"}
    assert exchanges == 2
    assert resource_calls == 2


def test_client_allows_only_safe_mutation_headers(monkeypatch):
    captured = []

    def urlopen(request, timeout):
        if request.full_url.endswith("/campaign-auth/exchange"):
            return Response({"raw_token": "bgca.access.short-lived-secret"})
        captured.append(request)
        return Response({"status": "active"})

    monkeypatch.setattr("bashgym.campaigns.client.urllib.request.urlopen", urlopen)
    client = CampaignApiClient(
        api_base="http://localhost:8003/api",
        credential_ref="CAMPAIGN_REFRESH",
        secret_resolver=lambda _ref: "bgcr.parent.refresh-secret",
    )

    client.request_json(
        "POST",
        "/campaigns/campaign-1/start",
        payload={"workspace_id": "workspace-a", "expected_version": 2},
        headers={"Idempotency-Key": "start-123", "X-Correlation-ID": "workflow-4"},
    )
    assert captured[0].get_header("Idempotency-key") == "start-123"
    assert captured[0].get_header("X-correlation-id") == "workflow-4"

    with pytest.raises(ValueError, match="not allowed"):
        client.request_json(
            "POST",
            "/campaigns/campaign-1/start",
            headers={"Authorization": "Bearer attacker"},
        )
    with pytest.raises(ValueError, match="invalid value"):
        client.request_json(
            "POST",
            "/campaigns/campaign-1/start",
            headers={"Idempotency-Key": "bgcr.raw token"},
        )


def test_missing_reference_and_raw_token_reference_fail_closed(monkeypatch):
    monkeypatch.setattr(
        "bashgym.campaigns.client.urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not be used"),
    )
    missing = CampaignApiClient(
        api_base="http://localhost:8003/api",
        credential_ref="MISSING_REFRESH",
        secret_resolver=lambda _ref: None,
    )
    with pytest.raises(CampaignClientError) as caught:
        missing.request_json("GET", "/campaigns")
    assert caught.value.code == "campaign_auth_required"
    assert caught.value.exit_code == 3

    with pytest.raises(ValueError, match="reference"):
        CampaignApiClient(
            api_base="http://localhost:8003/api",
            credential_ref="bgcr.raw.refresh-secret",
        )
    with pytest.raises(ValueError, match="secret reference names"):
        CampaignApiClient(
            api_base="http://localhost:8003/api",
            credential_ref="literal secret with spaces",
        )


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (401, "campaign_auth_required", 3),
        (403, "campaign_capability_required:campaign.start", 4),
        (404, "campaign_not_found", 5),
        (409, "campaign_version_conflict", 6),
        (422, "campaign_budget_exceeded", 7),
        (503, "campaign_http_503", 8),
        (400, "campaign_invalid_input", 2),
    ],
)
def test_error_exit_code_contract(status, code, expected):
    assert campaign_exit_code(status, code) == expected
