"""Tests for external benchmark orchestration (bashgym/eval/benchmarks_ext.py)."""

import pytest

from bashgym.eval.benchmarks_ext import (
    BenchmarkSpec,
    bfcl_command,
    forgetting_suite_spec,
    harbor_terminal_bench_command,
    normalize_external_benchmark_results,
    parse_accuracy,
    parse_bfcl_results,
    parse_cua_rewardbench_results,
    parse_lm_eval,
    parse_resolved_rate,
    parse_rewardbench_results,
    parse_swebench_results,
    record_benchmarks,
    run_benchmarks,
    swebench_command,
    terminal_bench_command,
)


class TestParsers:
    def test_parse_lm_eval_averages_tasks(self):
        data = {"results": {"mmlu": {"acc,none": 0.6}, "gsm8k": {"exact_match,none": 0.4}}}
        r = parse_lm_eval("forgetting", data)
        assert abs(r.score - 0.5) < 1e-9
        assert r.metrics == {"mmlu": 0.6, "gsm8k": 0.4}

    def test_parse_resolved_rate(self):
        r = parse_resolved_rate("swebench", {"resolved": 7, "total": 10})
        assert r.score == 0.7 and r.passed == 7 and r.total == 10

    def test_parse_resolved_rate_alt_keys_and_list(self):
        r = parse_resolved_rate("tb", {"n_resolved": ["a", "b", "c"], "n_instances": 4})
        assert r.passed == 3 and r.total == 4 and r.score == 0.75

    def test_parse_accuracy_fraction_and_percent(self):
        assert parse_accuracy("bfcl", {"overall_accuracy": 0.83}).score == 0.83
        assert abs(parse_accuracy("bfcl", {"accuracy": 83.0}).score - 0.83) < 1e-9

    def test_normalize_external_named_result_map(self):
        report = normalize_external_benchmark_results(
            {
                "terminal_bench": {"resolved": ["task_a", "task_b"], "total": 4},
                "bfcl_v4": {"summary": {"overall_accuracy": 72.5}},
            }
        )

        assert report.scores == {
            "terminal_bench": 0.5,
            "bfcl_v4": 0.725,
        }
        assert report.results[0].passed == 2
        assert report.results[0].total == 4

    def test_normalize_external_harbor_trial_rewards(self):
        report = normalize_external_benchmark_results(
            {
                "trials": [
                    {"result": {"reward": 1.0}},
                    {"result": {"reward": 0.0}},
                    {"result": {"reward": 1.0}},
                ]
            },
            benchmark_name="harbor_terminal_bench",
        )

        assert report.scores["harbor_terminal_bench"] == pytest.approx(2 / 3)
        result = report.results[0]
        assert result.passed == 2
        assert result.total == 3
        assert result.metrics["trials"] == 3.0

    def test_normalize_external_list_with_record_names(self):
        report = normalize_external_benchmark_results(
            [
                {"benchmark_name": "swebench_verified_lite", "score": "41.0%"},
                {"benchmark_name": "bfcl_v4", "accuracy": 0.81},
            ]
        )

        assert report.scores == {"swebench_verified_lite": 0.41, "bfcl_v4": 0.81}

    def test_parse_bfcl_v4_weighted_categories(self):
        result = parse_bfcl_results(
            "bfcl_v4",
            {
                "categories": {
                    "Agentic": {"accuracy": 0.6},
                    "Multi-Turn": {"accuracy": 0.5},
                    "Live": {"accuracy": 0.8},
                    "Non-Live": {"accuracy": 0.7},
                    "Hallucination Measurement": {"accuracy": 0.9},
                }
            },
        )

        assert result.score == pytest.approx(0.63)
        assert result.metrics["category.agentic"] == 0.6
        assert result.metrics["category.multi_turn"] == 0.5
        assert result.metrics["bfcl_v4_weighted_score"] == pytest.approx(0.63)

    def test_normalize_external_bfcl_csv_row_preserves_breakdown(self):
        report = normalize_external_benchmark_results(
            {
                "data_overall": [
                    {
                        "Model": "candidate",
                        "Overall Acc": "72.5%",
                        "Non_Live Overall Acc": "70%",
                        "Live Overall Acc": "80%",
                        "Multi Turn Overall Acc": "65%",
                    }
                ]
            },
            benchmark_name="bfcl_v4",
        )

        result = report.results[0]
        assert result.score == 0.725
        assert result.metrics["overall_acc"] == 0.725
        assert result.metrics["category.non_live"] == 0.7
        assert result.metrics["category.live"] == 0.8
        assert result.metrics["category.multi_turn"] == 0.65

    def test_parse_swebench_results_json(self):
        result = parse_swebench_results(
            "swebench_verified",
            {
                "resolved_instances": ["django__django-1", "sympy__sympy-2"],
                "submitted_instances": ["django__django-1", "sympy__sympy-2", "astropy__astropy-3"],
                "completed_instances": 3,
                "total_instances": 5,
                "error_count": 1,
            },
        )

        assert result.score == pytest.approx(2 / 3)
        assert result.passed == 2
        assert result.total == 3
        assert result.metrics["resolved_instances"] == 2.0
        assert result.metrics["submitted_instances"] == 3.0
        assert result.metrics["completed_instances"] == 3.0
        assert result.metrics["error_count"] == 1.0

    def test_parse_swebench_instance_results_repo_breakdown(self):
        report = normalize_external_benchmark_results(
            {
                "instance_results": [
                    {"instance_id": "sympy__sympy-20590", "resolved": True},
                    {"instance_id": "sympy__sympy-21612", "resolved": False},
                    {"instance_id": "django__django-11099", "status": "resolved"},
                ]
            },
            benchmark_name="swe-bench_verified",
        )

        result = report.results[0]
        assert result.name == "swe_bench_verified"
        assert result.score == pytest.approx(2 / 3)
        assert result.passed == 2
        assert result.total == 3
        assert result.metrics["repo.sympy_sympy.resolution_rate"] == 0.5
        assert result.metrics["repo.django_django.resolution_rate"] == 1.0

    def test_parse_rewardbench_subset_rows(self):
        result = parse_rewardbench_results(
            "RewardBench 2",
            {
                "results": [
                    {"subset": "Chat", "num_correct": 1, "total": 2},
                    {"subset": "Safety", "num_correct": 2, "total": 2},
                    {"subset": "Reasoning", "accuracy": "50%"},
                ]
            },
        )

        assert result.name == "rewardbench_2"
        assert result.score == pytest.approx((0.5 + 1.0 + 0.5) / 3)
        assert result.passed == 3
        assert result.total == 4
        assert result.metrics["subset.chat"] == 0.5
        assert result.metrics["subset.safety"] == 1.0
        assert result.metrics["subset.reasoning"] == 0.5
        assert result.metrics["rewardbench_subset_count"] == 3.0

    def test_normalize_external_rewardbench_hf_like_rows(self):
        report = normalize_external_benchmark_results(
            [
                {"subset": "Factuality", "results": 1.0},
                {"subset": "Factuality", "results": 0.0},
                {"subset": "Chat Hard", "results": 1.0},
            ],
            benchmark_name="rewardbench",
        )

        result = report.results[0]
        assert report.scores["rewardbench"] == pytest.approx(0.75)
        assert result.passed == 2
        assert result.total == 3
        assert result.metrics["subset.factuality"] == 0.5
        assert result.metrics["subset.chat_hard"] == 1.0

    def test_parse_cua_rewardbench_metrics_json(self):
        result = parse_cua_rewardbench_results(
            "CUARewardBench",
            {
                "trajectory_reward_metrics": {
                    "overall": {
                        "Overall Accuracy": 0.72,
                        "F1": 0.7,
                        "TP": 36,
                        "TN": 20,
                        "FP": 8,
                        "FN": 6,
                    },
                    "by_task_type": {
                        "spreadsheet": {"Overall Accuracy": 0.8},
                    },
                },
                "action_reward_metrics": {
                    "overall": {"Overall Accuracy": "65%", "F1": 0.6},
                    "by_reward_type": {
                        "redundant": {"Overall Accuracy": 0.7},
                    },
                },
            },
        )

        assert result.name == "cuarewardbench"
        assert result.score == 0.72
        assert result.passed == 56
        assert result.total == 70
        assert result.metrics["trajectory_overall_accuracy"] == 0.72
        assert result.metrics["action_overall_accuracy"] == 0.65
        assert (
            result.metrics["trajectory.by_task_type.spreadsheet.overall_accuracy"]
            == 0.8
        )
        assert result.metrics["action.by_reward_type.redundant.overall_accuracy"] == 0.7

    def test_normalize_external_cua_rewardbench_rows(self):
        report = normalize_external_benchmark_results(
            {
                "results": [
                    {"task_type": "browser", "correct": True},
                    {"task_type": "browser", "correct": False},
                    {"task_type": "desktop", "accuracy": 1.0},
                ]
            },
            benchmark_name="cua_rewardbench",
        )

        result = report.results[0]
        assert result.score == pytest.approx(2 / 3)
        assert result.passed == 2
        assert result.total == 3


class TestCommandBuilders:
    def test_terminal_bench(self):
        cmd = terminal_bench_command("openai/cand", n_attempts=3)
        assert cmd[0] == "tb" and "run" in cmd
        assert "--model" in cmd and "openai/cand" in cmd
        assert "3" in cmd

    def test_harbor_terminal_bench(self):
        cmd = harbor_terminal_bench_command(
            "openai/cand",
            n_concurrent=12,
            environment="daytona",
        )
        assert cmd[:2] == ["harbor", "run"]
        assert "--dataset" in cmd and "terminal-bench@2.0" in cmd
        assert "--model" in cmd and "openai/cand" in cmd
        assert "--n-concurrent" in cmd and "12" in cmd
        assert "--env" in cmd and "daytona" in cmd

    def test_bfcl(self):
        cmd = bfcl_command("cand", test_category="simple")
        assert cmd[:2] == ["bfcl", "generate"]
        assert "simple" in cmd

    def test_swebench_with_extra_args(self):
        cmd = swebench_command("cand", subset="lite", extra_args=["--workers", "4"])
        assert "mini-swe-agent" in cmd and "lite" in cmd
        assert cmd[-2:] == ["--workers", "4"]


class TestForgettingSuiteSpec:
    def test_builds_lm_eval_spec(self):
        spec = forgetting_suite_spec("http://x/v1", model_name="cand")
        assert spec.name == "forgetting"
        assert spec.argv[0] == "lm_eval"
        assert spec.parser is parse_lm_eval


class TestRunBenchmarks:
    def test_runs_and_aggregates(self):
        specs = [
            BenchmarkSpec("forgetting", ["lm_eval"], parse_lm_eval),
            BenchmarkSpec("swebench", ["mini-swe-agent"], parse_resolved_rate),
        ]

        def run_command(argv):
            if argv[0] == "lm_eval":
                return {"results": {"mmlu": {"acc,none": 0.6}}}
            return {"resolved": 5, "total": 10}

        report = run_benchmarks(specs, run_command)
        assert report.scores == {"forgetting": 0.6, "swebench": 0.5}
        assert report.failures == []

    def test_failing_harness_recorded_not_fatal(self):
        specs = [
            BenchmarkSpec("good", ["lm_eval"], parse_lm_eval),
            BenchmarkSpec("bad", ["tb"], parse_resolved_rate),
        ]

        def run_command(argv):
            if argv[0] == "tb":
                raise RuntimeError("harness not installed")
            return {"results": {"mmlu": {"acc,none": 0.7}}}

        report = run_benchmarks(specs, run_command)
        assert report.scores == {"good": 0.7}  # only the successful one
        assert report.failures == ["bad"]


class TestRecordBenchmarks:
    def test_records_successful_only(self):
        class _FakeRegistry:
            def __init__(self):
                self.calls = []

            def add_benchmark_result(self, model_id, name, score, passed, total, metrics):
                self.calls.append((model_id, name, score, passed, total))

        specs = [
            BenchmarkSpec("good", ["lm_eval"], parse_lm_eval),
            BenchmarkSpec("bad", ["tb"], parse_resolved_rate),
        ]

        def run_command(argv):
            if argv[0] == "tb":
                raise RuntimeError("boom")
            return {"results": {"mmlu": {"acc,none": 0.7}}}

        report = run_benchmarks(specs, run_command)
        reg = _FakeRegistry()
        record_benchmarks(reg, "model-1", report)
        assert len(reg.calls) == 1  # the failed 'bad' is skipped
        assert reg.calls[0][1] == "good" and reg.calls[0][2] == 0.7
