import asyncio
import json

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


class FakeSkillToolCallingClient(FakeHermesClient):
    calls = []

    async def post(self, url, headers=None, json=None):
        self.calls.append(
            {"method": "POST", "url": url, "headers": headers or {}, "json": json or {}}
        )
        if url.endswith("/responses") and "previous_response_id" not in (json or {}):
            return FakeHermesResponse(
                200,
                {
                    "id": "resp_tool_1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "skill_lab_prepare",
                            "arguments": '{"skill":"factory"}',
                        }
                    ],
                },
            )
        if url.endswith("/responses"):
            return FakeHermesResponse(
                200,
                {
                    "id": "resp_tool_2",
                    "status": "completed",
                    "output_text": "I opened factory in Skill Lab.",
                },
            )
        return FakeHermesResponse(404, {"error": "not found"})


class FakeHermesStreamResponse:
    def __init__(self, status_code: int, lines: list[str], data=None):
        self.status_code = status_code
        self._lines = lines
        self._data = data or {}
        self.headers = {"content-type": "text/event-stream"}
        self.text = json.dumps(self._data)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self.text.encode("utf-8")

    def json(self):
        return self._data


class FakeStreamingHermesClient(FakeHermesClient):
    calls = []

    def stream(self, method, url, headers=None, json=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers or {}, "json": json or {}}
        )
        return FakeHermesStreamResponse(
            200,
            [
                "event: response.created",
                'data: {"type":"response.created","response":{"id":"resp_stream"}}',
                "",
                "event: response.output_text.delta",
                'data: {"type":"response.output_text.delta","delta":"Hello "}',
                "",
                "event: response.output_text.delta",
                'data: {"type":"response.output_text.delta","delta":"**GhostWork**"}',
                "",
                "event: response.completed",
                ('data: {"type":"response.completed","response":{"id":"resp_stream","output":[]}}'),
                "",
            ],
        )


class FakeAuthRejectingHermesClient(FakeHermesClient):
    calls = []

    async def get(self, url, headers=None):
        self.calls.append({"method": "GET", "url": url, "headers": headers or {}})
        if url.endswith("/health"):
            return FakeHermesResponse(200, {"status": "ok"})
        return FakeHermesResponse(
            401,
            {
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    def stream(self, method, url, headers=None, json=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers or {}, "json": json or {}}
        )
        return FakeHermesStreamResponse(
            401,
            [],
            {
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )


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


class FakeTrainingResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"run_id": "run_agent", "status": "pending", "strategy": "sft"}


class FakeTrainingClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        return FakeTrainingResponse()


def _client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolate_agent_endpoint_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("BASHGYM_DISABLE_KEYRING", "1")
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
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
    monkeypatch.setattr(agent_routes, "_get_agent_api_key", lambda _endpoint_id: None)

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


def test_agent_endpoint_discovery_does_not_treat_public_health_as_connected(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeAuthRejectingHermesClient)
    FakeAuthRejectingHermesClient.calls = []

    _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Remote Hermes",
            "base_url": "http://127.0.0.1:18642/v1",
            "model": "hermes-agent",
            "api_key": "stale-token",
        },
    )
    response = _client().post("/api/agent/endpoints/hermes/discover")

    assert response.status_code == 200
    data = response.json()
    assert data["probes"]["health"]["ok"] is True
    assert data["probes"]["capabilities"]["status_code"] == 401
    assert data["ok"] is False
    assert data["warnings"][0].startswith("Hermes rejected the saved API server key")


def test_authenticated_hermes_health_probe_rejects_invalid_key(monkeypatch):
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeAuthRejectingHermesClient)
    FakeAuthRejectingHermesClient.calls = []

    healthy, error = asyncio.run(
        agent_routes._probe_hermes_health(
            "http://127.0.0.1:18642/v1",
            "stale-token",
        )
    )

    assert healthy is False
    assert error == "Saved API server key was rejected"
    assert FakeAuthRejectingHermesClient.calls[0]["url"].endswith("/v1/capabilities")


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
    assert "x-hermes-session-key" not in post["headers"]
    assert post["json"]["conversation"] == "bashgym-canvas-agent-node"
    assert post["json"]["store"] is True
    user_input = post["json"]["input"][0]
    assert user_input["role"] == "user"
    assert user_input["content"][0]["type"] == "input_text"
    input_text = user_input["content"][0]["text"]
    assert "<bashgym_authoritative_context>" in input_text
    assert "Training run: grpo" in input_text
    assert "What should I run next?" in input_text
    assert "live runtime, then the durable BashGym ledger" in post["json"]["instructions"]
    assert "Never blend projects" in post["json"]["instructions"]


def test_agent_endpoint_executes_skill_lab_function_calls_with_canvas_scope(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeSkillToolCallingClient)
    FakeSkillToolCallingClient.calls = []
    executions = []

    async def fake_execute(name, arguments, **kwargs):
        executions.append((name, arguments, kwargs))
        return {"status": "prepared", "skill_id": "skill-1"}

    monkeypatch.setattr(agent_routes, "execute_skill_lab_tool", fake_execute)
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
            "message": "Help me evaluate the factory skill",
            "workspace_id": "workspace-main",
            "origin": {"panel_id": "hermes-panel"},
        },
    )

    assert response.status_code == 200
    assert response.json()["response"] == "I opened factory in Skill Lab."
    assert executions[0][0] == "skill_lab_prepare"
    assert executions[0][1]["workspace_id"] == "workspace-main"
    assert executions[0][1]["origin"]["panel_id"] == "hermes-panel"
    second_call = FakeSkillToolCallingClient.calls[-1]["json"]
    assert second_call["previous_response_id"] == "resp_tool_1"
    assert second_call["input"][0]["type"] == "function_call_output"


async def test_hermes_cannot_self_confirm_skill_lab_side_effects(monkeypatch):
    confirmations = []

    async def fake_execute(name, arguments, **kwargs):
        confirmations.append(arguments["confirmed"])
        return {"status": "confirmation_required"}

    monkeypatch.setattr(agent_routes, "execute_skill_lab_tool", fake_execute)
    await agent_routes._execute_hermes_skill_lab_calls(
        {
            "workspace_id": "main",
            "origin": {"agent": "hermes"},
            "user_message": "Can you help me evaluate this skill?",
        },
        [
            {
                "id": "call-1",
                "name": "skill_lab_run",
                "arguments": {
                    "skill": "factory",
                    "endpoint_id": "hermes",
                    "confirmed": True,
                },
            }
        ],
    )

    assert confirmations == [False]


def test_agent_endpoint_chat_stream_relays_deltas_and_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeStreamingHermesClient)
    FakeStreamingHermesClient.calls = []

    _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Local Hermes",
            "base_url": "http://127.0.0.1:8642/v1",
            "model": "hermes-agent",
            "session_key": "workspace-memory",
            "api_key": "secret-token-123",
        },
    )
    response = _client().post(
        "/api/agent/endpoints/hermes/chat/stream",
        json={
            "message": "Render markdown",
            "context": "Canvas: GhostWork",
            "conversation": "bashgym-canvas-agent-node",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: delta\ndata: {"delta": "Hello "}' in response.text
    assert 'event: delta\ndata: {"delta": "**GhostWork**"}' in response.text
    assert 'event: done\ndata: {"endpoint_id": "hermes"' in response.text
    assert "secret-token-123" not in response.text

    call = FakeStreamingHermesClient.calls[0]
    assert call["url"].endswith("/responses")
    assert call["json"]["stream"] is True
    assert call["json"]["store"] is True
    assert call["json"]["conversation"] == "bashgym-canvas-agent-node"
    assert call["headers"]["x-hermes-session-key"] == "workspace-memory"


def test_agent_endpoint_chat_stream_explains_rejected_gateway_key(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeAuthRejectingHermesClient)
    FakeAuthRejectingHermesClient.calls = []

    _client().put(
        "/api/agent/endpoints/hermes",
        json={
            "label": "Remote Hermes",
            "base_url": "http://127.0.0.1:18642/v1",
            "model": "hermes-agent",
            "api_key": "stale-token",
        },
    )
    response = _client().post(
        "/api/agent/endpoints/hermes/chat/stream",
        json={"message": "Describe your BashGym skillset"},
    )

    assert response.status_code == 200
    assert "Hermes rejected the saved API server key" in response.text
    assert "invalid_api_key" not in response.text


def test_agent_training_tool_forwards_canvas_provenance(monkeypatch):
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeTrainingClient)
    FakeTrainingClient.calls = []

    tracking_context = {
        "workspace_id": "workspace-a",
        "project_id": "project-a",
        "project_display_name": "Project A",
        "experiment_id": "experiment-a",
        "experiment_name": "Experiment A",
        "objective": "Improve held-out quality.",
        "task_type": "terminal-agent",
        "model_id": "model-a",
        "model_version_id": "model-a-v1",
        "model_source_uri": "hf://example/model-a",
        "model_config_digest": "a" * 64,
        "dataset_id": "dataset-a",
        "dataset_version_id": "dataset-a-v1",
        "dataset_source_uri": "file://data/dataset-a.manifest.json",
        "dataset_content_digest": "b" * 64,
        "environment_id": "environment-a",
        "environment_runtime_digest": "c" * 64,
    }
    result = json.loads(
        asyncio.run(
            agent_routes._execute_tool(
                "start_training",
                {
                    "strategy": "sft",
                    "model": "Qwen/Qwen3-Coder",
                    "dataset_path": "data/train.jsonl",
                    "compute_target": "ssh:pony0",
                    "correlation_id": "training-intent-1",
                    "tracking_context": tracking_context,
                    "config": {
                        "checkpoint_limit": 1,
                        "artifact_retention": "adapter_only",
                        "auto_push_hf": True,
                        "hf_private": True,
                        "hf_upload_artifact": "adapter",
                    },
                    "origin": {"panel_id": "panel-1", "terminal_id": "term-1"},
                },
            )
        )
    )

    assert result["run_id"] == "run_agent"
    post = FakeTrainingClient.calls[0]
    assert post["url"] == "http://localhost:8003/api/training/start"
    assert post["json"] == {
        "strategy": "sft",
        "base_model": "Qwen/Qwen3-Coder",
        "dataset_path": "data/train.jsonl",
        "compute_target": "ssh:pony0",
        "use_remote_ssh": True,
        "device_id": "pony0",
        "correlation_id": "training-intent-1",
        "tracking": tracking_context,
        "checkpoint_limit": 1,
        "artifact_retention": "adapter_only",
        "auto_push_hf": True,
        "hf_private": True,
        "hf_upload_artifact": "adapter",
        "origin": {
            "kind": "agent",
            "agent": "hermes",
            "panel_id": "panel-1",
            "terminal_id": "term-1",
        },
    }


def test_agent_reads_project_isolated_experiment_context(monkeypatch, tmp_path):
    from bashgym.ledger.persistence import ExperimentLedgerRepository
    from tests.ledger.test_persistence import run_spec, seed_project

    database = tmp_path / "campaigns" / "campaigns.sqlite3"
    repository = ExperimentLedgerRepository(database)
    repository.initialize()
    seed_project(repository)
    repository.register_run(run_spec())
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)

    projects = json.loads(
        asyncio.run(
            agent_routes._execute_tool("list_experiment_projects", {"workspace_id": "workspace-a"})
        )
    )
    context = json.loads(
        asyncio.run(
            agent_routes._execute_tool(
                "get_experiment_context",
                {
                    "workspace_id": "workspace-a",
                    "project_id": "project-a",
                    "recent_limit": 10,
                },
            )
        )
    )

    assert projects["projects"][0]["project_id"] == "project-a"
    assert context["project_id"] == "project-a"
    assert context["recent_runs"][0]["run_id"] == "run-1"


def test_agent_training_tool_rejects_unknown_config_field():
    result = json.loads(
        asyncio.run(
            agent_routes._execute_tool(
                "start_training",
                {
                    "strategy": "sft",
                    "model": "Qwen/Test",
                    "config": {"made_up_setting": True},
                },
            )
        )
    )

    assert result == {"error": "start_training config contains unsupported fields: made_up_setting"}


def test_agent_data_designer_tool_forwards_runtime_and_canvas_inputs(monkeypatch):
    monkeypatch.setattr(agent_routes.httpx, "AsyncClient", FakeTrainingClient)
    FakeTrainingClient.calls = []

    asyncio.run(
        agent_routes._execute_tool(
            "start_data_designer",
            {
                "pipeline": "coding_agent_sft",
                "num_records": 24,
                "seed_source": "data/gold.jsonl",
                "seed_type": "file",
                "model": "Hermes/Test",
                "provider": "vllm",
                "provider_endpoint": "http://192.0.2.10:8889/v1",
                "origin": {"panel_id": "designer-1", "terminal_id": "term-1"},
            },
        )
    )

    post = FakeTrainingClient.calls[0]
    assert post["url"] == "http://localhost:8003/api/factory/designer/create"
    assert post["json"] == {
        "pipeline": "coding_agent_sft",
        "num_records": 24,
        "seed_source": "data/gold.jsonl",
        "seed_type": "file",
        "text_model": "Hermes/Test",
        "provider": "vllm",
        "provider_endpoint": "http://192.0.2.10:8889/v1",
        "origin": {
            "kind": "agent",
            "agent": "hermes",
            "panel_id": "designer-1",
            "terminal_id": "term-1",
        },
    }


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


def test_toolkit_discovers_active_hermes_skills_without_optional_catalog(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    active = hermes_home / "skills" / "user-skill"
    optional = hermes_home / "optional-skills" / "catalog-skill"
    active.mkdir(parents=True)
    optional.mkdir(parents=True)
    (active / "SKILL.md").write_text(
        "---\nname: user-skill\ndescription: An installed Hermes skill.\n---\n",
        encoding="utf-8",
    )
    (optional / "SKILL.md").write_text(
        "---\nname: catalog-skill\ndescription: An optional template.\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(agent_routes, "_hermes_command", lambda: None)

    roots = dict(agent_routes._toolkit_skill_root_candidates())
    assert roots["hermes"] == hermes_home / "skills"
    monkeypatch.setattr(
        agent_routes,
        "_toolkit_skill_root_candidates",
        lambda: [("hermes", roots["hermes"])],
    )
    _root_infos, skills, _warnings = agent_routes._scan_skill_roots()
    names = {skill.name for skill in skills}
    assert "user-skill" in names
    assert "catalog-skill" not in names
    user_skill = next(skill for skill in skills if skill.name == "user-skill")
    assert user_skill.source == "hermes"
    assert user_skill.available_sources == ["hermes"]


def test_local_hermes_setup_syncs_live_gateway_key(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(agent_routes, "_hermes_command", lambda: None)
    env_path = tmp_path / "hermes" / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text("API_SERVER_KEY=live-local-key\n", encoding="utf-8")
    agent_routes._set_agent_api_key("hermes", "saved-profile-key")

    async def healthy(_base_url, api_key=None):
        assert api_key == "live-local-key"
        return True, None

    monkeypatch.setattr(agent_routes, "_probe_hermes_health", healthy)
    asyncio.run(agent_routes._hermes_setup_status())
    headers = agent_routes._agent_headers(
        {"id": "hermes", "kind": "hermes", "base_url": "http://127.0.0.1:8642/v1"}
    )

    assert headers["authorization"] == "Bearer live-local-key"


def test_local_hermes_setup_does_not_replace_tunnel_key(monkeypatch, tmp_path):
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "state")
    monkeypatch.setattr("bashgym.secrets.get_secrets_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(agent_routes, "_hermes_command", lambda: None)
    env_path = tmp_path / "hermes" / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text("API_SERVER_KEY=local-key\n", encoding="utf-8")
    agent_routes._save_agent_endpoint_profile(
        "hermes",
        label="Remote Hermes",
        base_url="http://127.0.0.1:18642/v1",
        model="hermes-agent",
        session_key="bashgym-canvas",
        api_key="tunnel-key",
    )

    async def healthy(_base_url, api_key=None):
        assert api_key == "tunnel-key"
        return True, None

    monkeypatch.setattr(agent_routes, "_probe_hermes_health", healthy)
    asyncio.run(agent_routes._hermes_setup_status())

    assert agent_routes._get_agent_api_key("hermes") == "tunnel-key"


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
