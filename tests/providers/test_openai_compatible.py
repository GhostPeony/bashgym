"""Tests for the generic OpenAI-compatible provider (cloud GPU portability)."""

import json

import httpx
import pytest

from bashgym.providers.openai_compatible import PRESETS, OpenAICompatibleProvider


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_handler(request):
    path = request.url.path
    if path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"completion_tokens": 5},
            },
        )
    if path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})
    return httpx.Response(404)


class TestForPlatform:
    def test_known_platforms_resolve(self):
        for name, url in PRESETS.items():
            p = OpenAICompatibleProvider.for_platform(name, api_key="k")
            assert p.provider_type == name
            assert p._base_url == url.rstrip("/")

    def test_vllm_is_local_no_key_required(self):
        p = OpenAICompatibleProvider.for_platform("vllm")
        assert p.is_local is True
        assert p.requires_api_key is False

    def test_cloud_platform_requires_key(self):
        p = OpenAICompatibleProvider.for_platform("together", api_key="k")
        assert p.is_local is False
        assert p.requires_api_key is True

    def test_unknown_platform_raises(self):
        with pytest.raises(ValueError, match="unknown platform"):
            OpenAICompatibleProvider.for_platform("not_a_platform")


class TestGenerate:
    async def test_generate_success(self):
        p = OpenAICompatibleProvider(
            "together",
            "https://api.together.xyz/v1",
            api_key="k",
            default_model="m",
            client=_client(_ok_handler),
        )
        resp = await p.generate([{"role": "user", "content": "hi"}])
        assert resp.success is True
        assert resp.content == "hello"
        assert resp.tokens_used == 5
        assert resp.provider_type == "together"

    async def test_error_status_is_unsuccessful(self):
        def handler(request):
            return httpx.Response(401, text="unauthorized")

        p = OpenAICompatibleProvider(
            "together", "https://x/v1", api_key="bad", client=_client(handler)
        )
        resp = await p.generate([{"role": "user", "content": "hi"}])
        assert resp.success is False
        assert "401" in resp.error

    async def test_system_prompt_prepended(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
            )

        p = OpenAICompatibleProvider("x", "https://x/v1", api_key="k", client=_client(handler))
        await p.generate([{"role": "user", "content": "hi"}], system_prompt="be terse")
        assert [m["role"] for m in captured["body"]["messages"]] == ["system", "user"]

    async def test_local_sends_no_auth_header(self):
        captured = {}

        def handler(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "x"}}], "usage": {}}
            )

        p = OpenAICompatibleProvider.for_platform("vllm", default_model="m")
        p._client = _client(handler)
        await p.generate([{"role": "user", "content": "hi"}])
        assert captured["auth"] is None


class TestHealthAndModels:
    async def test_health_check_ok(self):
        p = OpenAICompatibleProvider(
            "together", "https://x/v1", api_key="k", client=_client(_ok_handler)
        )
        h = await p.health_check()
        assert h.available is True
        assert "m1" in h.models_loaded

    async def test_list_models(self):
        p = OpenAICompatibleProvider(
            "together", "https://x/v1", api_key="k", client=_client(_ok_handler)
        )
        models = await p.list_models()
        assert {m.id for m in models} == {"m1", "m2"}
        assert all(m.provider_type == "together" for m in models)
