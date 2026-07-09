import pytest
from fastapi.testclient import TestClient

from bashgym.api import agent_routes
from bashgym.api.routes import app


class FakeHermesResponse:
    def __init__(self, status_code: int, data):
        self.status_code = status_code
        self._data = data
        self.text = str(data)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._data


class FakeHermesClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, headers=None):
        self.calls.append({"method": "GET", "url": url, "headers": headers or {}})
        if url.endswith("/health/detailed"):
            return FakeHermesResponse(
                200,
                {
                    "status": "ok",
                    "platform": "hermes-agent",
                    "version": "0.16.0",
                    "gateway_state": "running",
                    "active_agents": 0,
                },
            )
        if url.endswith("/health"):
            return FakeHermesResponse(200, {"status": "ok"})
        if url.endswith("/capabilities"):
            return FakeHermesResponse(200, {"features": ["runs", "skills"]})
        if url.endswith("/models"):
            return FakeHermesResponse(200, {"data": [{"id": "hermes-agent"}]})
        if url.endswith("/skills"):
            return FakeHermesResponse(200, {"skills": [{"name": "training"}]})
        if url.endswith("/toolsets"):
            return FakeHermesResponse(200, {"toolsets": [{"name": "bashgym"}]})
        return FakeHermesResponse(404, {"error": "not found"})

    async def post(self, url, headers=None, json=None):
        self.calls.append(
            {"method": "POST", "url": url, "headers": headers or {}, "json": json or {}}
        )
        if url.endswith("/responses"):
            return FakeHermesResponse(
                200,
                {
                    "id": "resp_123",
                    "status": "completed",
                    "output_text": "I can see the BashGym workspace context.",
                },
            )
        return FakeHermesResponse(404, {"error": "not found"})


class FakeRuntimeFailureThenChatClient(FakeHermesClient):
    calls = []

    async def post(self, url, headers=None, json=None):
        self.calls.append(
            {"method": "POST", "url": url, "headers": headers or {}, "json": json or {}}
        )
        if url.endswith("/responses"):
            return FakeHermesResponse(
                200,
                {
                    "id": "resp_failed",
                    "status": "completed",
                    "output_text": (
                        "API call failed after 3 retries: ValidationException: "
                        "provided model identifier is invalid"
                    ),
                },
            )
        if url.endswith("/chat/completions"):
            return FakeHermesResponse(
                200,
                {
                    "id": "chat_123",
                    "choices": [{"message": {"content": "Chat fallback worked."}}],
                },
            )
        return FakeHermesResponse(404, {"error": "not found"})


class FakeTunnelProcess:
    pid = 43210
    terminated = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def _client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_agent_endpoint_env(monkeypatch):
    for key in (
        "HERMES_API_BASE",
        "HERMES_API_KEY",
        "HERMES_API_SERVER_KEY",
        "HERMES_ENDPOINT_LABEL",
        "HERMES_MODEL",
        "HERMES_SESSION_KEY",
        "HERMES_HOME",
        "AGENT_ENDPOINT_HERMES_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_agent_endpoint_default_profile_is_public(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")

    response = _client().get("/api/agent/endpoints")

    assert response.status_code == 200
    data = response.json()
    assert data["endpoints"][0]["id"] == "hermes"
    assert data["endpoints"][0]["base_url"] == "http://127.0.0.1:8642/v1"
    assert data["endpoints"][0]["api_key_configured"] is False


def test_agent_endpoint_save_stores_secret_without_echo(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")

    response = _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Local Hermes",
            "base_url": "http://127.0.0.1:8642",
            "model": "hermes-agent",
            "api_key": "secret-token-123",
        },
    )

    assert response.status_code == 200
    assert "secret-token-123" not in response.text
    data = response.json()
    assert data["base_url"] == "http://127.0.0.1:8642/v1"
    assert data["api_key_configured"] is True

    listed = _client().get("/api/agent/endpoints")
    assert "secret-token-123" not in listed.text
    assert listed.json()["endpoints"][0]["api_key_configured"] is True


def test_agent_endpoint_save_stores_model_options(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")

    response = _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Local Hermes",
            "base_url": "http://127.0.0.1:8642",
            "model": "hermes-qwen3.6-27b-dense",
            "model_options": [
                "hermes-qwen3.6-27b-dense",
                "hermes-qwen3.6-chat",
                "hermes-qwen3.6-27b-dense",
            ],
            "api_key": "secret-token-123",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "hermes-qwen3.6-27b-dense"
    assert data["model_options"] == [
        "hermes-qwen3.6-27b-dense",
        "hermes-qwen3.6-chat",
    ]


def test_agent_endpoint_discovery_uses_official_hermes_surfaces(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeHermesClient)
    FakeHermesClient.calls = []

    _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Local Hermes",
            "base_url": "http://127.0.0.1:8642/v1",
            "model": "hermes-agent",
            "api_key": "secret-token-123",
        },
    )
    response = _client().post("/api/agent/endpoints/hermes/discover")

    assert response.status_code == 200
    assert "secret-token-123" not in response.text
    data = response.json()
    assert data["ok"] is True
    assert data["summary"] == {"models": 1, "skills": 1, "toolsets": 1}
    assert {
        call["url"].replace("http://127.0.0.1:8642/", "") for call in FakeHermesClient.calls
    } == {
        "health",
        "health/detailed",
        "v1/capabilities",
        "v1/models",
        "v1/skills",
        "v1/toolsets",
    }
    health_call = next(call for call in FakeHermesClient.calls if call["url"].endswith("/health"))
    assert health_call["url"] == "http://127.0.0.1:8642/health"
    assert all(
        call["headers"]["authorization"] == "Bearer secret-token-123"
        for call in FakeHermesClient.calls
    )


def test_endpoint_name_extraction_keeps_large_skill_lists():
    data = {"data": [{"name": f"skill-{index}"} for index in range(32)]}

    names = agent_routes._extract_named_endpoint_items(data, ("skills", "data"))

    assert len(names) == 32
    assert names[0] == "skill-0"
    assert names[-1] == "skill-31"


def test_agent_endpoint_chat_sends_workspace_context(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeHermesClient)
    FakeHermesClient.calls = []

    _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Local Hermes",
            "base_url": "http://127.0.0.1:8642/v1",
            "model": "hermes-agent",
            "api_key": "secret-token-123",
        },
    )
    response = _client().post(
        "/api/agent/endpoints/hermes/chat",
        json={
            "message": "What should I run next?",
            "context": "Training run: grpo step 12 loss 0.4",
            "conversation": "bashgym-canvas-agent-node",
        },
    )

    assert response.status_code == 200
    assert "secret-token-123" not in response.text
    data = response.json()
    assert data["response"] == "I can see the BashGym workspace context."
    post = next(call for call in FakeHermesClient.calls if call["method"] == "POST")
    assert post["url"].endswith("/responses")
    assert post["headers"]["authorization"] == "Bearer secret-token-123"
    assert post["headers"]["x-hermes-session-key"] == "bashgym-canvas-agent-node"
    assert "Training run: grpo" in post["json"]["input"]
    assert "What should I run next?" in post["json"]["input"]


def test_agent_endpoint_chat_falls_back_when_responses_returns_runtime_failure(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeRuntimeFailureThenChatClient)
    FakeRuntimeFailureThenChatClient.calls = []

    _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Local Hermes",
            "base_url": "http://127.0.0.1:8642/v1",
            "model": "hermes-qwen3.6-27b-dense",
            "api_key": "secret-token-123",
        },
    )
    response = _client().post(
        "/api/agent/endpoints/hermes/chat",
        json={
            "message": "What should I run next?",
            "context": "Training run: grpo step 12 loss 0.4",
            "conversation": "bashgym-canvas-agent-node",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Chat fallback worked."
    assert data["raw_status"] == "chat.completions"
    urls = [call["url"] for call in FakeRuntimeFailureThenChatClient.calls]
    assert any(url.endswith("/responses") for url in urls)
    assert any(url.endswith("/chat/completions") for url in urls)


def test_hermes_tunnel_connect_saves_forwarded_profile(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.shutil, "which", lambda _: "ssh")
    monkeypatch.setattr(agent_routes, "_find_available_local_port", lambda: 18642)

    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append({"command": command, "kwargs": kwargs})
        return FakeTunnelProcess()

    async def fake_probe(base_url, api_key=None):
        assert base_url == "http://127.0.0.1:18642/v1"
        assert api_key == "remote-token"
        return True, None

    monkeypatch.setattr(agent_routes.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(agent_routes, "_probe_hermes_health", fake_probe)
    agent_routes._HERMES_TUNNELS.clear()

    response = _client().post(
        "/api/agent/hermes/tunnel/connect",
        json={
            "endpoint_id": "hermes",
            "label": "Remote Hermes",
            "ssh_target": "remote-gpu",
            "remote_host": "127.0.0.1",
            "remote_port": 8642,
            "model": "hermes-qwen3.6-chat",
            "model_options": ["hermes-qwen3.6-chat", "hermes-agent"],
            "session_key": "bashgym-canvas",
            "api_key": "remote-token",
        },
    )

    assert response.status_code == 200
    assert "remote-token" not in response.text
    data = response.json()
    assert data["active"] is True
    assert data["healthy"] is True
    assert data["ssh_target"] == "remote-gpu"
    assert data["local_base_url"] == "http://127.0.0.1:18642/v1"
    assert data["profile"]["base_url"] == "http://127.0.0.1:18642/v1"
    assert data["profile"]["api_key_configured"] is True
    command = popen_calls[0]["command"]
    assert command[:3] == ["ssh", "-N", "-L"]
    assert "127.0.0.1:18642:127.0.0.1:8642" in command
    assert command[-1] == "remote-gpu"


def test_hermes_tunnel_rejects_unsafe_ssh_target(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")

    response = _client().post(
        "/api/agent/hermes/tunnel/connect",
        json={
            "endpoint_id": "hermes",
            "ssh_target": "-oProxyCommand=bad",
            "api_key": "remote-token",
        },
    )

    assert response.status_code == 400


def test_hermes_tunnel_status_infers_existing_forward_after_reload(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    agent_routes._HERMES_TUNNELS.clear()

    async def fake_probe(base_url, api_key=None):
        assert base_url == "http://127.0.0.1:18642/v1"
        assert api_key == "remote-token"
        return True, None

    monkeypatch.setattr(agent_routes, "_probe_hermes_health", fake_probe)
    _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Remote Hermes",
            "base_url": "http://127.0.0.1:18642/v1",
            "model": "hermes-qwen3.6-chat",
            "api_key": "remote-token",
        },
    )

    response = _client().get("/api/agent/hermes/tunnel/status")

    assert response.status_code == 200
    data = response.json()
    assert data["active"] is True
    assert data["healthy"] is True
    assert data["local_base_url"] == "http://127.0.0.1:18642/v1"
    assert data["local_port"] == 18642


def test_toolkit_inventory_lists_local_skills_and_tools(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "youtube-retrieval"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: youtube-retrieval",
                "description: Work with YouTube transcript retrieval evals.",
                "---",
                "",
                "# YouTube Retrieval",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "evals.md").write_text("retrieval notes", encoding="utf-8")
    monkeypatch.setenv("BASHGYM_SKILL_DIRS", str(skills_root))
    agent_routes._TOOLKIT_CACHE.clear()

    response = _client().get(
        "/api/agent/toolkit",
        params={"include_remote": "false", "refresh": "true"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is False
    assert any(skill["name"] == "youtube-retrieval" for skill in data["skills"])
    youtube = next(skill for skill in data["skills"] if skill["name"] == "youtube-retrieval")
    assert youtube["resource_counts"]["references"] == 1
    assert any(tool["name"] == "import_traces" for tool in data["tools"])
    assert data["endpoint_capabilities"][0]["warnings"] == [
        "Remote probing disabled for this request"
    ]

    cached = _client().get("/api/agent/toolkit", params={"include_remote": "false"})
    assert cached.status_code == 200
    assert cached.json()["cached"] is True


def test_toolkit_scan_cap_is_internal_not_user_warning(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    for index in range(3):
        skill_dir = skills_root / f"skill-{index}"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: skill-{index}\ndescription: Test skill {index}.\n---\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        agent_routes,
        "_toolkit_skill_root_candidates",
        lambda: [("test", skills_root)],
    )

    root_infos, _skills, warnings = agent_routes._scan_skill_roots(max_skills=2)

    assert root_infos[0].skill_count == 2
    assert not any("Stopped local skill scan" in warning for warning in warnings)


def test_toolkit_inventory_probes_authenticated_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeHermesClient)
    agent_routes._TOOLKIT_CACHE.clear()
    FakeHermesClient.calls = []

    _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Local Hermes",
            "base_url": "http://127.0.0.1:8642/v1",
            "model": "hermes-agent",
            "api_key": "secret-token-123",
        },
    )

    response = _client().get("/api/agent/toolkit", params={"refresh": "true"})

    assert response.status_code == 200
    assert "secret-token-123" not in response.text
    data = response.json()
    endpoint = next(
        item for item in data["endpoint_capabilities"] if item["endpoint_id"] == "hermes"
    )
    assert endpoint["ok"] is True
    assert endpoint["skills"] == 1
    assert endpoint["toolsets"] == 1
    assert endpoint["skill_names"] == ["training"]
    assert endpoint["toolset_names"] == ["bashgym"]


def test_hermes_setup_status_reports_missing_local_gateway(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(agent_routes.shutil, "which", lambda _: None)

    async def fake_probe(_base_url, _api_key=None):
        return False, "connect failed"

    monkeypatch.setattr(agent_routes, "_probe_hermes_health", fake_probe)

    response = _client().get("/api/agent/hermes/setup-status")

    assert response.status_code == 200
    data = response.json()
    assert data["installed"] is False
    assert data["env_exists"] is False
    assert "Install Hermes CLI" in data["setup_needed"]
    assert "Start Hermes gateway" in data["setup_needed"]


def test_hermes_quick_setup_writes_env_saves_profile_and_redacts_key(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(agent_routes.shutil, "which", lambda _: "hermes")
    monkeypatch.setattr(
        agent_routes, "_hermes_gateway_command", lambda _: ["hermes", "gateway", "run"]
    )
    monkeypatch.setattr(
        agent_routes, "_start_hermes_gateway", lambda _: str(tmp_path / "gateway.log")
    )
    config_sets = []
    monkeypatch.setattr(
        agent_routes,
        "_hermes_config_set",
        lambda command, key, value: config_sets.append((command, key, value)) or True,
    )

    probe_calls = []

    async def fake_probe(base_url, api_key=None):
        probe_calls.append((base_url, api_key))
        return len(probe_calls) > 1, None

    monkeypatch.setattr(agent_routes, "_probe_hermes_health", fake_probe)

    response = _client().post(
        "/api/agent/hermes/quick-setup",
        json={
            "profile_id": "hermes",
            "label": "Local Hermes",
            "base_url": "http://127.0.0.1:8642",
            "model": "hermes-agent",
            "session_key": "bashgym-test",
            "api_key": "setup-secret-token",
            "write_env": True,
            "start_gateway": True,
        },
    )

    assert response.status_code == 200
    assert "setup-secret-token" not in response.text
    data = response.json()
    assert data["status"]["gateway_healthy"] is True
    assert data["status"]["profile"]["api_key_configured"] is True
    assert "Enabled Hermes API server via hermes config" in data["actions"]
    assert "Started Hermes gateway process" in data["actions"]
    assert ("hermes", "API_SERVER_ENABLED", "true") in config_sets
    assert all(item[1] != "API_SERVER_KEY" for item in config_sets)

    env_text = (tmp_path / "hermes" / ".env").read_text(encoding="utf-8")
    assert "API_SERVER_ENABLED=true" in env_text
    assert "API_SERVER_KEY=setup-secret-token" in env_text

    listed = _client().get("/api/agent/endpoints")
    assert "setup-secret-token" not in listed.text
    assert listed.json()["endpoints"][0]["api_key_configured"] is True


def test_hermes_setup_status_uses_cli_discovered_windows_paths(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.shutil, "which", lambda _: "hermes")
    monkeypatch.setattr(
        agent_routes, "_hermes_gateway_command", lambda _: ["hermes", "gateway", "run"]
    )
    hermes_dir = tmp_path / "AppData" / "Local" / "hermes"
    env_path = hermes_dir / ".env"
    config_path = hermes_dir / "config.yaml"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "API_SERVER_ENABLED=true\nAPI_SERVER_KEY=setup-secret-token\n", encoding="utf-8"
    )
    config_path.write_text("model:\n  default: hermes-agent\n", encoding="utf-8")

    def fake_cli_path(command, *args):
        if args == ("env-path",):
            return env_path
        if args == ("path",):
            return config_path
        return None

    async def fake_probe(base_url, api_key=None):
        assert api_key == "setup-secret-token"
        return False, "connect failed"

    monkeypatch.setattr(agent_routes, "_hermes_cli_path", fake_cli_path)
    monkeypatch.setattr(agent_routes, "_probe_hermes_health", fake_probe)

    response = _client().get("/api/agent/hermes/setup-status")

    assert response.status_code == 200
    data = response.json()
    assert data["installed"] is True
    assert data["env_path"] == str(env_path)
    assert data["hermes_home"] == str(hermes_dir)
    assert data["config_path"] == str(config_path)
    assert data["configured_model"] == "hermes-agent"
    assert data["env_api_enabled"] is True
    assert data["env_key_present"] is True
