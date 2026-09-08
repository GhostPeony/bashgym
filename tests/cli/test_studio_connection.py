"""A saved studio endpoint is reused without environment overrides."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bashgym import cli, studio


@pytest.fixture
def root(tmp_path, monkeypatch):
    from bashgym.api import database

    secrets = {}
    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))
    monkeypatch.delenv("BASHGYM_API_BASE", raising=False)
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr("bashgym.secrets.get_secret", secrets.get)
    monkeypatch.setattr("bashgym.secrets.set_secret", secrets.__setitem__)
    monkeypatch.setattr("bashgym.operator_skills.install_skills", lambda **_: {"verified": True})
    return tmp_path, secrets


def test_explicit_init_port_replays_without_changing_authority(root):
    directory, secrets = root
    first = studio.initialize(directory, api_port=8124, start_service=False)
    original_profile = studio.read_profile(directory)
    original_secrets = dict(secrets)
    second = studio.initialize(directory, start_service=False)
    third = studio.initialize(directory, api_port=8124, start_service=False)
    assert first["browser_url"] == "http://127.0.0.1:8124"
    assert second["browser_url"] == third["browser_url"] == first["browser_url"]
    assert second["replayed"] and third["replayed"]
    assert studio.read_profile(directory) == original_profile
    assert secrets == original_secrets
    with pytest.raises(ValueError, match="studio_profile_conflict"):
        studio.initialize(directory, api_port=8125, start_service=False)
    assert studio.read_profile(directory) == original_profile
    assert secrets == original_secrets


@pytest.mark.parametrize("port", [0, 65536, -1, True])
def test_invalid_port_is_rejected_before_writing_state(root, port):
    directory, secrets = root
    with pytest.raises(ValueError, match="studio_api_port_invalid"):
        studio.initialize(directory, api_port=port, start_service=False)
    assert not (directory / "studio.v1.json").exists()
    assert not secrets


def test_init_cli_wires_port_and_saved_profile_into_both_clients(root, capsys):
    directory, _ = root
    args = cli.build_parser().parse_args(["init", "--api-port", "8124", "--no-service", "--json"])
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["browser_url"] == "http://127.0.0.1:8124"
    connection = SimpleNamespace(
        api_base=None, credential_ref=studio.read_profile(directory)["credential_ref"]
    )
    assert cli._workspace_api_base(connection) == "http://127.0.0.1:8124/api"
    with patch("bashgym.campaigns.client.CampaignApiClient") as client:
        cli._campaign_client(connection)
    assert client.call_args.kwargs["api_base"] == "http://127.0.0.1:8124/api"


def test_api_base_precedence_is_explicit_then_environment_then_profile(root, monkeypatch):
    directory, _ = root
    studio.initialize(directory, api_port=8124, start_service=False)
    for resolver in (cli._workspace_api_base, cli._campaign_api_base):
        assert resolver(SimpleNamespace(api_base=None)) == "http://127.0.0.1:8124/api"
        monkeypatch.setenv("BASHGYM_API_BASE", "http://localhost:8126")
        assert resolver(SimpleNamespace(api_base=None)) == "http://localhost:8126/api"
        assert (
            resolver(SimpleNamespace(api_base="http://localhost:8127"))
            == "http://localhost:8127/api"
        )
        monkeypatch.delenv("BASHGYM_API_BASE")


def test_invalid_saved_url_is_not_used_or_echoed(root):
    directory, _ = root
    studio.initialize(directory, start_service=False)
    profile = studio.read_profile(directory)
    profile["api_base"] = "https://operator:private-credential@example.test/api"
    studio._write_profile(directory, profile)
    with pytest.raises(ValueError, match="studio_profile_invalid") as error:
        cli._campaign_api_base(SimpleNamespace(api_base=None))
    assert "private-credential" not in str(error.value)
    assert (
        cli._campaign_api_base(SimpleNamespace(api_base="http://localhost:8127"))
        == "http://localhost:8127/api"
    )


def test_health_and_generated_service_use_saved_loopback_port(root, monkeypatch):
    from bashgym.campaigns import worker_service

    directory, _ = root
    studio.initialize(directory, api_port=8124, start_service=False)
    with patch("bashgym.campaigns.worker_service.probe_api_health", return_value={}) as probe:
        studio._health(directory)
    assert probe.call_args.kwargs["port"] == 8124
    assert probe.call_args.kwargs["host"] == "127.0.0.1"

    original_builder = worker_service.build_api_service_definition

    def builder(**kwargs):
        return original_builder(
            **kwargs, home=directory / "home", target=worker_service.WorkerPlatform.LINUX
        )

    with (
        patch("bashgym.studio._health", return_value={"healthy": False}),
        patch("bashgym.studio._wait_for_service", return_value={"healthy": True}),
        patch("bashgym.studio._setup_context", return_value={}),
        patch("bashgym.campaigns.worker_service.build_api_service_definition", side_effect=builder),
        patch("bashgym.campaigns.worker_service.ApiServiceManager") as manager,
    ):
        studio.initialize(directory)
    definition = manager.return_value.install.call_args.args[0]
    argv = definition.launch_argv
    assert argv[argv.index("--port") + 1] == "8124"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


def test_preparation_compiler_accepts_saved_connection_without_flag(root, capsys):
    directory, _ = root
    studio.initialize(directory, api_port=8124, start_service=False)
    profile = studio.read_profile(directory)
    stops = directory / "stops.json"
    stops.write_text(
        json.dumps(
            {
                "max_attempts": 1,
                "budget_unit": "gpu_hours",
                "max_total_cost": 1,
                "minimum_improvement": 0,
            }
        )
    )
    args = SimpleNamespace(
        template_id="template",
        definition_digest="a" * 64,
        expected_version=0,
        onboarding_id="onboarding",
        campaign_id="campaign",
        campaign_title="Fixture",
        stop_rules=stops,
        controller_lease_key_ref="CONTROLLER_REF",
        write_inputs=False,
        credential_ref=profile["credential_ref"],
        api_base=None,
        workspace_id=profile["workspace_id"],
        session_id="session",
        json=True,
    )
    with patch("bashgym.campaigns.studio_preparation.build_registered_preparation") as compiler:
        compiler.return_value.summary.return_value = {"prepared": True}
        assert cli.cmd_research_prepare(args) == 0
    compiler.assert_called_once()
    assert json.loads(capsys.readouterr().out)["prepared"] is True
