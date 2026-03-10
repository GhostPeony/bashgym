"""Tests for ModelRouter integration with ProviderRegistry."""

import pytest
from bashgym.gym.router import ModelRouter, RouterConfig, RoutingStrategy, ModelType, ModelConfig, ModelResponse
from bashgym.providers.registry import ProviderRegistry
from bashgym.providers.base import InferenceProvider, ProviderResponse, HealthStatus, ProviderModel


class FakeProvider(InferenceProvider):
    def __init__(self, ptype, content="response"):
        self._ptype = ptype
        self._content = content

    @property
    def provider_type(self): return self._ptype
    @property
    def requires_api_key(self): return False
    @property
    def is_local(self): return self._ptype == "ollama"

    async def generate(self, messages, model, **kwargs):
        return ProviderResponse(
            content=self._content, model_name=model,
            provider_type=self._ptype, latency_ms=10.0,
            tokens_used=5, success=True,
        )

    async def health_check(self):
        return HealthStatus(available=True)

    async def list_models(self):
        return [ProviderModel(id=f"{self._ptype}/test", name="test", provider_type=self._ptype)]


class TestRouterWithRegistry:
    def setup_method(self):
        self.registry = ProviderRegistry()
        self.teacher = FakeProvider("anthropic", content="teacher says")
        self.student = FakeProvider("ollama", content="student says")
        self.registry.register(self.teacher)
        self.registry.register(self.student)
        self.registry.map_model("claude-sonnet", "anthropic")
        self.registry.map_model("qwen2.5-coder:7b", "ollama")

    def _config(self, strategy):
        """Create a RouterConfig with instrumentation disabled for testing."""
        return RouterConfig(
            strategy=strategy,
            enable_guardrails=False,
            enable_profiling=False,
        )

    def test_router_accepts_registry(self):
        config = self._config(RoutingStrategy.TEACHER_ONLY)
        router = ModelRouter(config=config, registry=self.registry)
        assert router.registry is self.registry

    @pytest.mark.asyncio
    async def test_teacher_generates_via_registry(self):
        config = self._config(RoutingStrategy.TEACHER_ONLY)
        router = ModelRouter(config=config, registry=self.registry)
        # Remove any auto-loaded models so only our test model is present
        router.models.clear()
        router.register_model(ModelConfig(
            name="claude-sonnet", model_type=ModelType.TEACHER,
            endpoint="", api_key="test",
        ))

        response = await router.generate("Hello")
        assert response.success is True
        assert response.content == "teacher says"
        assert isinstance(response, ModelResponse)

    @pytest.mark.asyncio
    async def test_student_generates_via_registry(self):
        config = self._config(RoutingStrategy.STUDENT_ONLY)
        router = ModelRouter(config=config, registry=self.registry)
        # Remove any auto-loaded models so only our test model is present
        router.models.clear()
        router.register_model(ModelConfig(
            name="qwen2.5-coder:7b", model_type=ModelType.STUDENT,
            endpoint="",
        ))

        response = await router.generate("Simple fix")
        assert response.content == "student says"
        assert response.source == "ollama"


class TestRouterBackwardCompat:
    def test_router_without_registry(self):
        router = ModelRouter()
        assert router.registry is None

    def test_route_still_works_without_registry(self):
        config = RouterConfig(strategy=RoutingStrategy.TEACHER_ONLY)
        router = ModelRouter(config=config)
        # Register a model so routing doesn't fail
        router.register_model(ModelConfig(
            name="claude-sonnet", model_type=ModelType.TEACHER,
            endpoint="https://api.anthropic.com/v1/messages", api_key="test",
        ))
        decision = router.route("test prompt")
        assert decision.selected_model == "claude-sonnet"
