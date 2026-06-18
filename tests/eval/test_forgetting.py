"""Tests for the forgetting/regression eval (lm-eval/NeMo-Evaluator -> gate)."""

from bashgym.eval.forgetting import (
    DEFAULT_FORGETTING_TASKS,
    compute_forgetting,
    lm_eval_command,
    parse_lm_eval_results,
)
from bashgym.eval.gate import evaluate_gate


# A realistic lm-eval / NeMo Evaluator results blob (note the ",none" filter suffix,
# the _stderr companions, and multiple metrics per task).
def _lm_eval_json(mmlu, gsm8k, ifeval_acc):
    return {
        "results": {
            "mmlu": {"acc,none": mmlu, "acc_stderr,none": 0.004, "alias": "mmlu"},
            "gsm8k": {"exact_match,strict-match": gsm8k, "exact_match_stderr,strict-match": 0.01},
            "ifeval": {"acc,none": ifeval_acc},
        }
    }


class TestParse:
    def test_picks_headline_metric_and_ignores_stderr_alias(self):
        scores = parse_lm_eval_results(_lm_eval_json(0.65, 0.40, 0.55))
        assert scores == {"mmlu": 0.65, "gsm8k": 0.40, "ifeval": 0.55}

    def test_metric_priority_exact_match_over_acc(self):
        # exact_match should win over acc when both are present
        blob = {"results": {"t": {"acc,none": 0.9, "exact_match,none": 0.3}}}
        assert parse_lm_eval_results(blob) == {"t": 0.3}

    def test_metric_priority_acc_norm_over_acc(self):
        # acc_norm (normalized) is the headline for MC tasks like HellaSwag/ARC
        blob = {"results": {"hellaswag": {"acc,none": 0.80, "acc_norm,none": 0.83}}}
        assert parse_lm_eval_results(blob) == {"hellaswag": 0.83}

    def test_empty_or_malformed(self):
        assert parse_lm_eval_results({}) == {}
        assert parse_lm_eval_results({"results": "nope"}) == {}


class TestComputeForgetting:
    def test_drops_base_minus_candidate(self):
        base = _lm_eval_json(0.65, 0.40, 0.55)
        cand = _lm_eval_json(0.62, 0.45, 0.55)  # mmlu down, gsm8k up, ifeval flat
        rep = compute_forgetting(base, cand)
        assert rep.drops == {"mmlu": 0.03, "gsm8k": -0.05, "ifeval": 0.0}
        assert rep.worst == ("mmlu", 0.03)
        assert rep.regressed == {"mmlu": 0.03}  # only mmlu actually regressed

    def test_accepts_already_parsed_dicts(self):
        rep = compute_forgetting({"mmlu": 0.7, "gsm8k": 0.5}, {"mmlu": 0.6, "gsm8k": 0.5})
        assert rep.drops == {"mmlu": 0.1, "gsm8k": 0.0}

    def test_only_shared_tasks(self):
        rep = compute_forgetting({"mmlu": 0.7, "extra": 0.9}, {"mmlu": 0.6, "other": 0.1})
        assert rep.drops == {"mmlu": 0.1}  # 'extra'/'other' not shared

    def test_to_dict_serializable(self):
        rep = compute_forgetting({"mmlu": 0.7}, {"mmlu": 0.6})
        d = rep.to_dict()
        assert d["drops"] == {"mmlu": 0.1} and d["worst"] == ["mmlu", 0.1]


class TestGateIntegration:
    def test_forgetting_drop_blocks_ship(self):
        # Candidate beats traces but regressed mmlu by 0.08 > the 0.05 gate threshold
        rep = compute_forgetting({"mmlu": 0.70}, {"mmlu": 0.62})
        verdict = evaluate_gate(
            trace_delta=0.20, ci_low=0.10, ci_excludes_zero=True, forgetting_drops=rep.drops
        )
        assert verdict.ship is False
        assert any("forgetting" in r and "mmlu" in r for r in verdict.reasons)

    def test_small_drop_passes_gate(self):
        rep = compute_forgetting({"mmlu": 0.70}, {"mmlu": 0.68})  # 0.02 < 0.05 threshold
        verdict = evaluate_gate(
            trace_delta=0.20, ci_low=0.10, ci_excludes_zero=True, forgetting_drops=rep.drops
        )
        assert verdict.ship is True


class TestLmEvalCommand:
    def test_builds_local_completions_argv(self):
        cmd = lm_eval_command("http://192.168.50.173:8100/v1/completions", model_name="cand-v3")
        assert cmd[0] == "lm_eval"
        assert "local-completions" in cmd
        joined = " ".join(cmd)
        assert "base_url=http://192.168.50.173:8100/v1/completions" in joined
        assert "model=cand-v3" in joined
        assert ",".join(DEFAULT_FORGETTING_TASKS) in joined

    def test_optional_fewshot_and_limit(self):
        cmd = lm_eval_command("http://x/v1", tasks=["mmlu"], num_fewshot=5, limit=100)
        assert "--num_fewshot" in cmd and "5" in cmd
        assert "--limit" in cmd and "100" in cmd
