"""Tests for the deploy gate."""

from bashgym.eval.gate import GateThresholds, evaluate_gate


class TestEvaluateGate:
    def test_ships_on_reliable_improvement_no_forgetting(self):
        v = evaluate_gate(
            trace_delta=0.08,
            ci_low=0.02,
            ci_excludes_zero=True,
            forgetting_drops={"mmlu": 0.0, "gsm8k": 0.01},
        )
        assert v.ship and v.reasons == []

    def test_blocks_when_delta_too_small(self):
        v = evaluate_gate(trace_delta=0.01, ci_low=0.005, ci_excludes_zero=True)
        assert not v.ship and any("trace delta" in r for r in v.reasons)

    def test_blocks_when_improvement_not_significant(self):
        v = evaluate_gate(trace_delta=0.10, ci_low=-0.01, ci_excludes_zero=False)
        assert not v.ship and any("not statistically reliable" in r for r in v.reasons)

    def test_blocks_on_catastrophic_forgetting(self):
        v = evaluate_gate(
            trace_delta=0.10,
            ci_low=0.05,
            ci_excludes_zero=True,
            forgetting_drops={"ifeval": 0.09},
        )
        assert not v.ship and any("forgetting" in r for r in v.reasons)

    def test_custom_thresholds(self):
        # A stricter bar that the same result no longer clears.
        strict = GateThresholds(min_trace_delta=0.20)
        v = evaluate_gate(trace_delta=0.10, ci_low=0.05, ci_excludes_zero=True, thresholds=strict)
        assert not v.ship
