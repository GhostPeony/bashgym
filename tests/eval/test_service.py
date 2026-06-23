"""Hermetic tests for the eval orchestration service (no network).

The held-out runner is exercised through injected stub predictors via the
``predictor_factory`` seam; endpoint resolution and forgetting ingest use fakes.
"""

from __future__ import annotations

import shlex
import subprocess
import sys

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


def _py(code: str) -> str:
    if sys.platform != "win32":
        return shlex.join([sys.executable, "-c", code])
    return subprocess.list2cmdline([sys.executable, "-c", code])


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


def test_ingest_and_record_external_benchmarks():
    report = service.ingest_external_benchmarks(
        {"resolved": 6, "total": 10},
        benchmark_name="harbor_terminal_bench",
    )
    reg = _RecordingRegistry()

    recorded = service.record_external_benchmarks(reg, "model-x", report)

    assert report.scores == {"harbor_terminal_bench": pytest.approx(0.6)}
    assert recorded == ["harbor_terminal_bench"]
    assert reg.calls == [("model-x", "harbor_terminal_bench", pytest.approx(0.6))]


# ── model environment rollouts ───────────────────────────────────────────────


def test_run_model_environment_rollout_passk_uses_served_policy(tmp_path):
    env = {
        "id": "env_model_service",
        "instruction": "Create answer.txt containing ok.",
        "verifier": {
            "command": _py(
                "from pathlib import Path; "
                "raise SystemExit(0 if Path('answer.txt').read_text() == 'ok' else 1)"
            )
        },
    }
    command = _py("from pathlib import Path; Path('answer.txt').write_text('ok')")

    def complete_factory(cfg):
        assert cfg.model == "candidate"

        def complete(messages):
            if len(messages) <= 2:
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "run_command",
                                            "arguments": {"command": command},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            return '{"command":"submit"}'

        return complete

    report, rollouts, sampling_report = service.run_model_environment_rollout_passk(
        [env],
        service.EndpointConfig(base_url="http://h/v1", model="candidate"),
        workspace_root=tmp_path,
        attempts_per_environment=2,
        k_values=[1, 2],
        complete_factory=complete_factory,
    )

    assert report.pass_at_k["pass@1"] == pytest.approx(1.0)
    assert report.pass_at_k["pass@2"] == pytest.approx(1.0)
    assert len(rollouts) == 2
    assert rollouts[0].attempt.metadata["rollout_source"] == "served_model"
    assert sampling_report is None


def test_run_model_environment_rollout_passk_active_sampling_filters_zero_std(tmp_path):
    varied_env = {
        "id": "env_varied",
        "instruction": "Varied reward group: create answer.txt containing ok.",
        "verifier": {
            "command": _py(
                "from pathlib import Path; "
                "raise SystemExit(0 if Path('answer.txt').read_text() == 'ok' else 1)"
            )
        },
    }
    zero_env = {
        "id": "env_zero",
        "instruction": "Zero reward group: this verifier always fails.",
        "verifier": {"command": _py("raise SystemExit(1)")},
    }
    write_ok = _py("from pathlib import Path; Path('answer.txt').write_text('ok')")
    varied_first_calls = 0

    def complete_factory(_cfg):
        def complete(messages):
            nonlocal varied_first_calls
            first_prompt = messages[1]["content"]
            if len(messages) > 2:
                return '{"command":"submit"}'
            if "Varied reward group" in first_prompt:
                varied_first_calls += 1
                if varied_first_calls == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "run_command",
                                                "arguments": {"command": write_ok},
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
            return '{"command":"submit"}'

        return complete

    report, rollouts, sampling_report = service.run_model_environment_rollout_passk(
        [zero_env, varied_env],
        service.EndpointConfig(base_url="http://h/v1", model="candidate"),
        workspace_root=tmp_path,
        attempts_per_environment=2,
        k_values=[1, 2],
        filter_zero_std_groups=True,
        active_sampling=True,
        target_prompt_groups=1,
        complete_factory=complete_factory,
    )

    assert report.n_environments == 1
    assert set(report.to_dict()["per_environment"]) == {"env_varied"}
    assert {rollout.attempt.environment_id for rollout in rollouts} == {"env_varied"}
    assert sampling_report is not None
    assert sampling_report["selected_environment_ids"] == ["env_varied"]
    assert sampling_report["dropped_environment_ids"] == ["env_zero"]
    assert sampling_report["zero_std_groups_dropped"] == 1
    assert sampling_report["maintained_batch"] is True
    assert rollouts[0].attempt.metadata["active_sampling_selected"] is True


def test_run_model_environment_rollout_passk_caps_prompt_observations_not_raw_logs(tmp_path):
    long_stdout = "x" * 1200
    command = _py(f"print({long_stdout!r})")
    env = {
        "id": "env_long_stdout",
        "instruction": "Produce noisy output before submitting.",
        "verifier": {"command": _py("raise SystemExit(0)")},
    }
    seen_messages = []

    def complete_factory(_cfg):
        def complete(messages):
            seen_messages.append(messages)
            if len(messages) <= 2:
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "run_command",
                                            "arguments": {"command": command},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            return '{"command":"submit"}'

        return complete

    _report, rollouts, _sampling_report = service.run_model_environment_rollout_passk(
        [env],
        service.EndpointConfig(base_url="http://h/v1", model="candidate"),
        workspace_root=tmp_path,
        k_values=[1],
        max_observation_chars=80,
        complete_factory=complete_factory,
    )

    assert len(seen_messages) == 2
    next_prompt = seen_messages[1][-1]["content"]
    assert "...[truncated]" in next_prompt
    assert long_stdout not in next_prompt
    assert long_stdout in rollouts[0].observations[0].stdout
    assert rollouts[0].attempt.metadata["max_observation_chars"] == 80


def test_summarize_dppo_readiness_counts_behavior_and_train_logprobs():
    class _Attempt:
        metadata = {
            "behavior_logprob_tokens": 4,
            "train_logprob_tokens": 0,
        }

    class _Rollout:
        attempt = _Attempt()

    report = service.summarize_dppo_readiness([_Rollout()])

    assert report["attempts"] == 1
    assert report["attempts_with_behavior_logprobs"] == 1
    assert report["behavior_logprob_tokens"] == 4
    assert report["rollout_logprobs_ready"] is True
    assert report["optimizer_logprobs_ready"] is False
    assert report["needs_train_logprob_replay"] is True
