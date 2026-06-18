"""Tests for external benchmark orchestration (bashgym/eval/benchmarks_ext.py)."""

from bashgym.eval.benchmarks_ext import (
    BenchmarkSpec,
    bfcl_command,
    forgetting_suite_spec,
    parse_accuracy,
    parse_lm_eval,
    parse_resolved_rate,
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


class TestCommandBuilders:
    def test_terminal_bench(self):
        cmd = terminal_bench_command("openai/cand", n_attempts=3)
        assert cmd[0] == "tb" and "run" in cmd
        assert "--model" in cmd and "openai/cand" in cmd
        assert "3" in cmd

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
