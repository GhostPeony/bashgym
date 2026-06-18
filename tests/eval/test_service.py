"""Hermetic tests for the eval orchestration service (no network).

The held-out runner is exercised through injected stub predictors via the
``predictor_factory`` seam; endpoint resolution and forgetting ingest use fakes.
"""

from __future__ import annotations

import pytest

from bashgym.eval import service
from bashgym.eval.heldout import first_gold_tool_call


def _example(session: str, tool: str = "read", path: str = "a.txt") -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"please {tool} {path}"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": tool, "arguments": f'{{"path": "{path}"}}'}}],
            },
        ],
        "metadata": {"session_id": session},
    }


def _factory(cfg: service.EndpointConfig):
    """candidate predicts the gold call perfectly; base always misses."""
    if cfg.model == "candidate":
        return lambda ex: first_gold_tool_call(ex)
    return lambda ex: {"function": {"name": "wrong_tool", "arguments": "{}"}}


# ── run_heldout ──────────────────────────────────────────────────────────────


def test_run_heldout_candidate_ships():
    examples = [_example(f"s{i}", path=f"f{i}.txt") for i in range(12)]
    base = service.EndpointConfig(base_url="http://x/v1", model="base")
    cand = service.EndpointConfig(base_url="http://x/v1", model="candidate")

    report = service.run_heldout(
        examples, base, cand, metric="exact_match", predictor_factory=_factory
    )

    assert report.base_pass_rate == 0.0
    assert report.candidate_pass_rate == 1.0
    assert report.trace_delta == pytest.approx(1.0)
    assert report.n_clusters == 12
    assert report.ship is True
    assert report.verdict.reasons == []


def test_run_heldout_no_gain_does_not_ship():
    examples = [_example(f"s{i}") for i in range(8)]
    base = service.EndpointConfig(base_url="http://x/v1", model="candidate")  # both perfect
    cand = service.EndpointConfig(base_url="http://x/v1", model="candidate")

    report = service.run_heldout(examples, base, cand, predictor_factory=_factory)

    assert report.trace_delta == pytest.approx(0.0)
    assert report.ship is False


def test_run_heldout_empty_raises():
    cfg = service.EndpointConfig(base_url="http://x/v1", model="candidate")
    with pytest.raises(ValueError, match="no held-out examples"):
        service.run_heldout([], cfg, cfg, predictor_factory=_factory)


def test_run_heldout_forgetting_blocks_ship():
    examples = [_example(f"s{i}", path=f"f{i}.txt") for i in range(12)]
    base = service.EndpointConfig(base_url="http://x/v1", model="base")
    cand = service.EndpointConfig(base_url="http://x/v1", model="candidate")

    report = service.run_heldout(
        examples,
        base,
        cand,
        predictor_factory=_factory,
        forgetting_drops={"mmlu": 0.2},  # big regression
    )
    assert report.ship is False
    assert any("mmlu" in r for r in report.verdict.reasons)


# ── resolve_endpoint ─────────────────────────────────────────────────────────


def test_resolve_endpoint_explicit():
    cfg = service.resolve_endpoint(base_url="http://h/v1", model="m", api_key="k")
    assert cfg.base_url == "http://h/v1" and cfg.model == "m" and cfg.api_key == "k"


def test_resolve_endpoint_requires_model():
    with pytest.raises(ValueError, match="model"):
        service.resolve_endpoint(base_url="http://h/v1")


def test_resolve_endpoint_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        service.resolve_endpoint(model="m")


class _FakeProvider:
    base_url = "http://together/v1"
    api_key = "secret"
    default_model = "meta/Llama-3"


class _FakeRegistry:
    def __init__(self, prov):
        self._prov = prov

    def get_provider(self, name):
        return self._prov


def test_resolve_endpoint_from_connected_provider():
    reg = _FakeRegistry(_FakeProvider())
    cfg = service.resolve_endpoint(provider_registry=reg, provider="together")
    assert cfg.base_url == "http://together/v1"
    assert cfg.api_key == "secret"
    assert cfg.model == "meta/Llama-3"


def test_resolve_endpoint_provider_override_model():
    reg = _FakeRegistry(_FakeProvider())
    cfg = service.resolve_endpoint(provider_registry=reg, provider="together", model="custom")
    assert cfg.model == "custom"


def test_resolve_endpoint_provider_not_connected():
    reg = _FakeRegistry(None)
    with pytest.raises(ValueError, match="not connected"):
        service.resolve_endpoint(provider_registry=reg, provider="together")


# ── load_jsonl_examples ──────────────────────────────────────────────────────


def test_load_jsonl_skips_blank_and_bad(tmp_path):
    p = tmp_path / "holdout.jsonl"
    p.write_text(
        '{"messages": []}\n'
        "\n"
        "not json\n"
        '{"messages": [], "metadata": {"session_id": "s"}}\n',
        encoding="utf-8",
    )
    examples = service.load_jsonl_examples(p)
    assert len(examples) == 2


def test_load_jsonl_limit(tmp_path):
    p = tmp_path / "holdout.jsonl"
    p.write_text("\n".join('{"messages": []}' for _ in range(10)), encoding="utf-8")
    assert len(service.load_jsonl_examples(p, limit=3)) == 3


def test_load_jsonl_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        service.load_jsonl_examples(tmp_path / "nope.jsonl")


# ── thresholds / benchmark commands / forgetting ─────────────────────────────


def test_thresholds_from_defaults_and_overrides():
    default = service.thresholds_from()
    assert default.min_trace_delta == 0.05
    custom = service.thresholds_from(min_trace_delta=0.1, max_forgetting_drop=0.02)
    assert custom.min_trace_delta == 0.1
    assert custom.max_forgetting_drop == 0.02
    assert custom.require_ci_excludes_zero is True


def test_benchmark_commands_selected_only():
    cmds = service.benchmark_commands("http://h/v1", "m", include=("forgetting", "bfcl"))
    assert set(cmds) == {"forgetting", "bfcl"}
    assert "http://h/v1" in " ".join(cmds["forgetting"])
    assert "bfcl" in cmds["bfcl"][0]


def test_ingest_forgetting_computes_drops():
    base = {"results": {"mmlu": {"acc,none": 0.70}, "gsm8k": {"acc,none": 0.50}}}
    cand = {"results": {"mmlu": {"acc,none": 0.65}, "gsm8k": {"acc,none": 0.55}}}
    report = service.ingest_forgetting(base, cand)
    assert report.drops["mmlu"] == pytest.approx(0.05)
    assert report.drops["gsm8k"] == pytest.approx(-0.05)
    assert report.regressed == {"mmlu": pytest.approx(0.05)}


class _RecordingRegistry:
    def __init__(self):
        self.calls = []

    def add_benchmark_result(self, model_id, name, score, passed, total, metrics):
        self.calls.append((model_id, name, score))
        return object()  # non-None = recorded


def test_record_forgetting_writes_candidate_scores():
    report = service.ingest_forgetting(
        {"results": {"mmlu": {"acc,none": 0.7}}},
        {"results": {"mmlu": {"acc,none": 0.6}}},
    )
    reg = _RecordingRegistry()
    recorded = service.record_forgetting(reg, "model-x", report)
    assert recorded == ["mmlu"]
    assert reg.calls == [("model-x", "mmlu", pytest.approx(0.6))]
