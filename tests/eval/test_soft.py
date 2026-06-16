"""Tests for SERA-style soft/graded tool-call scoring (bashgym/eval/soft.py)."""

from bashgym.eval.heldout import evaluate_candidate, first_gold_tool_call
from bashgym.eval.soft import soft_call_score, soft_trajectory_score


def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


class TestSoftCallScore:
    def test_exact_is_one(self):
        assert soft_call_score(call("Bash", command="ls"), call("Bash", command="ls")) == 1.0

    def test_wrong_tool_is_zero(self):
        assert soft_call_score(call("Read", file="a"), call("Bash", command="ls")) == 0.0

    def test_right_tool_wrong_args_is_name_weight(self):
        s = soft_call_score(
            call("Bash", command="WRONG"), call("Bash", command="ls"), name_weight=0.4
        )
        assert abs(s - 0.4) < 1e-9

    def test_partial_args_between_name_weight_and_one(self):
        # one of two args matches -> arg_f1 = 0.5 -> 0.4 + 0.6*0.5 = 0.7
        s = soft_call_score(
            call("Bash", command="ls", timeout="9"),
            call("Bash", command="ls", timeout="5"),
        )
        assert abs(s - 0.7) < 1e-9


class TestSoftTrajectory:
    def test_identical_is_one(self):
        seq = [call("Bash", command="ls"), call("Read", file="a")]
        assert soft_trajectory_score(seq, seq) == 1.0

    def test_four_of_five_beats_binary_zero(self):
        # The whole point: 4/5 steps perfect + 1 wrong tool scores ~0.8, not 0.
        gold = [call("Bash", command=f"c{i}") for i in range(5)]
        pred = [call("Bash", command=f"c{i}") for i in range(4)] + [call("WrongTool")]
        assert abs(soft_trajectory_score(pred, gold) - 0.8) < 1e-9

    def test_missing_step_penalized(self):
        gold = [call("Bash", command=c) for c in ("a", "b", "c")]
        pred = [call("Bash", command=c) for c in ("a", "b")]
        assert abs(soft_trajectory_score(pred, gold) - 2 / 3) < 1e-9

    def test_extra_step_penalized(self):
        gold = [call("Bash", command=c) for c in ("a", "b")]
        pred = [call("Bash", command=c) for c in ("a", "b", "c")]
        assert abs(soft_trajectory_score(pred, gold) - 2 / 3) < 1e-9

    def test_empty_cases(self):
        assert soft_trajectory_score([], []) == 1.0
        assert soft_trajectory_score([], [call("Bash", command="x")]) == 0.0
        assert soft_trajectory_score([call("Bash", command="x")], []) == 0.0


class TestRunnerSoftMetric:
    def test_runner_accepts_soft_metric(self):
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "tool_calls": [call("Bash", command="ls")]},
                ],
                "metadata": {"session_id": f"s{i}"},
            }
            for i in range(6)
        ]

        def right_name_wrong_args(_ex):
            return call("Bash", command="WRONG")

        report = evaluate_candidate(
            examples, right_name_wrong_args, first_gold_tool_call, metric="soft", seed=0
        )
        # base = soft(right tool, wrong arg) = 0.4 ; candidate = soft(exact) = 1.0
        assert abs(report.base_pass_rate - 0.4) < 1e-9
        assert report.candidate_pass_rate == 1.0
        assert report.metric == "soft"
