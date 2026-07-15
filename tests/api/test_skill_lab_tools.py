import json

import httpx

from bashgym.agent.skill_lab_tools import SKILL_LAB_TOOL_NAMES, SkillLabToolClient

SKILL = {
    "skill_id": "skill-1",
    "name": "factory",
    "description": "Generate datasets",
    "source": "workspace",
    "revision": "abc123",
    "path": "C:/workspace/factory/SKILL.md",
}


async def test_prepare_saves_contract_and_materializes_workspace_node():
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/skill-lab/skills":
            return httpx.Response(200, json={"skills": [SKILL]})
        if request.method == "PUT" and request.url.path.endswith("/contracts/skill-1"):
            return httpx.Response(
                200,
                json={
                    "skill_id": "skill-1",
                    "workspace_id": "main",
                    "endpoint_id": "hermes",
                    "cases": [{"name": "target"}, {"name": "negative"}],
                    "thresholds": {},
                    "updated_at": "now",
                },
            )
        if request.method == "POST" and request.url.path == "/api/workspace/events":
            payload = json.loads(request.content)
            events.append(payload)
            return httpx.Response(200, json={"ok": True, "event": payload})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as http_client:
        bridge = SkillLabToolClient(
            api_base="http://test",
            workspace_id="main",
            origin={"terminal_id": "term-1", "agent": "codex"},
            client=http_client,
        )
        result = await bridge.execute(
            "skill_lab_prepare",
            {
                "skill": "factory",
                "endpoint_id": "hermes",
                "cases": [
                    {
                        "name": "target",
                        "prompt": "generate data",
                        "should_invoke": True,
                        "expected_patterns": ["dataset"],
                    },
                    {
                        "name": "negative",
                        "prompt": "show git status",
                        "should_invoke": False,
                        "expected_patterns": ["branch"],
                    },
                ],
            },
        )

    assert result["status"] == "prepared"
    assert events[0]["workspace_id"] == "main"
    assert events[0]["source"]["terminal_id"] == "term-1"
    assert events[0]["suggested_nodes"][0]["recipe"] == "skill_lab"
    assert events[0]["suggested_nodes"][0]["config"]["selectedSkillId"] == "skill-1"


async def test_run_preview_requires_confirmation_before_model_calls():
    requested_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append((request.method, request.url.path))
        if request.url.path == "/api/skill-lab/skills":
            return httpx.Response(200, json={"skills": [SKILL]})
        if request.url.path.endswith("/contracts/skill-1"):
            return httpx.Response(
                200,
                json={
                    "cases": [
                        {"name": "target", "prompt": "one", "should_invoke": True},
                        {"name": "negative", "prompt": "two", "should_invoke": False},
                    ],
                    "thresholds": {},
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as http_client:
        bridge = SkillLabToolClient(api_base="http://test", client=http_client)
        preview = await bridge.execute(
            "skill_lab_run",
            {"skill": "skill-1", "endpoint_id": "hermes", "confirmed": False},
        )

    assert preview["status"] == "confirmation_required"
    assert preview["model_calls"] == 6
    assert ("POST", "/api/skill-lab/runs") not in requested_paths
    assert {
        "skill_lab_context",
        "skill_lab_prepare",
        "skill_lab_save_skill",
        "skill_lab_run",
    }.issubset(SKILL_LAB_TOOL_NAMES)
