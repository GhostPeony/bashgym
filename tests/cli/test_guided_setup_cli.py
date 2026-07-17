import json

from bashgym.cli import main


class RecordingCampaignClient:
    def __init__(self):
        self.calls = []

    def request_json(self, method, path, *, query=None, payload=None, headers=None):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "query": query,
                "payload": payload,
                "headers": headers,
            }
        )
        return {"ok": True, "path": path}


def connection():
    return [
        "--workspace-id",
        "workspace-a",
        "--credential-ref",
        "CAMPAIGN_REFRESH",
        "--json",
    ]


def test_agent_manifest_exposes_skill_install_and_guided_setup_commands(capsys):
    assert main(["manifest", "--json"]) == 0

    commands = json.loads(capsys.readouterr().out)["commands"]
    for command in (
        "operator skills install",
        "operator skills check",
        "campaign setup-context",
        "campaign setup-step",
        "campaign setup-doctor",
        "campaign setup-validate",
        "campaign setup-create",
    ):
        assert command in commands


def test_guided_setup_cli_routes_context_and_ordered_session_step(monkeypatch, capsys):
    client = RecordingCampaignClient()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)

    assert main(["campaign", "setup-context", *connection()]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1] == {
        "method": "GET",
        "path": "/campaigns/setup/context",
        "query": {"workspace_id": "workspace-a", "session_id": None},
        "payload": None,
        "headers": None,
    }

    assert (
        main(
            [
                "campaign",
                "setup-step",
                *connection(),
                "--session-id",
                "setupsess_0123456789abcdef0123456789abcdef",
                "--expected-version",
                "0",
                "--step",
                "template",
                "--selection-id",
                "autoresearch-template-v1",
                "--idempotency-key",
                "setup-template-1",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1] == {
        "method": "POST",
        "path": "/campaigns/setup/session",
        "query": None,
        "payload": {
            "workspace_id": "workspace-a",
            "session_id": "setupsess_0123456789abcdef0123456789abcdef",
            "expected_version": 0,
            "step": "template",
            "selection_id": "autoresearch-template-v1",
        },
        "headers": {"Idempotency-Key": "setup-template-1"},
    }


def test_guided_setup_cli_routes_doctor_validate_and_atomic_create(monkeypatch, capsys, tmp_path):
    client = RecordingCampaignClient()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    draft = {
        "template_id": "autoresearch-template-v1",
        "installation_id": "ins_0123456789abcdef0123456789abcdef",
        "bindings": {
            "model": "model-a",
            "data": "data-a",
            "compute": "compute-a",
            "evaluation": "evaluation-a",
        },
    }
    draft_path = tmp_path / "guided-setup-draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    assert main(["campaign", "setup-doctor", *connection(), "--draft", str(draft_path)]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/setup/doctor"
    assert client.calls[-1]["payload"] == {"workspace_id": "workspace-a", **draft}
    assert client.calls[-1]["headers"] is None

    assert (
        main(
            [
                "campaign",
                "setup-validate",
                *connection(),
                "--draft",
                str(draft_path),
                "--idempotency-key",
                "validate-setup-1",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/setup/validate"
    assert client.calls[-1]["payload"] == {"workspace_id": "workspace-a", **draft}
    assert client.calls[-1]["headers"] == {"Idempotency-Key": "validate-setup-1"}

    assert (
        main(
            [
                "campaign",
                "setup-create",
                *connection(),
                "--campaign",
                "campaign-a",
                "--title",
                "Bounded AutoResearch campaign",
                "--validation-receipt",
                "setuprcpt_0123456789abcdef0123456789abcdef",
                "--idempotency-key",
                "create-setup-1",
                "--correlation-id",
                "guided-setup-1",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1] == {
        "method": "POST",
        "path": "/campaigns/setup/create",
        "query": None,
        "payload": {
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-a",
            "title": "Bounded AutoResearch campaign",
            "validation_receipt_id": "setuprcpt_0123456789abcdef0123456789abcdef",
        },
        "headers": {
            "Idempotency-Key": "create-setup-1",
            "X-Correlation-ID": "guided-setup-1",
        },
    }
