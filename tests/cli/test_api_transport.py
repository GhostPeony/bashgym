"""Operator-facing transport failures must remain actionable and secret-free."""

import json
import urllib.error

import pytest

from bashgym.cli import main


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    "failure,code",
    [
        (TimeoutError("private transport detail"), "api_request_timeout"),
        (urllib.error.URLError("private transport detail"), "api_connection_unavailable"),
        (ConnectionResetError("private transport detail"), "api_connection_unavailable"),
    ],
)
def test_api_transport_failure_is_structured_without_automatic_retry(
    monkeypatch, capsys, method, failure, code
):
    calls = []

    def unavailable(request, **kwargs):
        calls.append(request)
        raise failure

    monkeypatch.setattr("bashgym.cli._open_api_url", unavailable)
    assert main(["api", method, "/api/health", "--json"]) == 8
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["ok"] is False
    assert result["error"]["code"] == code
    assert result["error"]["outcome"] == "unknown"
    assert "doctor" in result["error"]["next_action"]
    assert "private transport detail" not in captured.out + captured.err
    assert "Traceback" not in captured.err
    assert len(calls) == 1


def test_api_response_read_timeout_uses_same_error_contract(monkeypatch, capsys):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            raise TimeoutError("private response detail")

    monkeypatch.setattr("bashgym.cli._open_api_url", lambda *a, **k: Response())
    assert main(["api", "GET", "/api/health", "--json"]) == 8
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "api_request_timeout"
