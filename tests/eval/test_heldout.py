"""Tests for the held-out eval runner (bashgym/eval/heldout.py)."""

import pytest

from bashgym.eval.heldout import (
    evaluate_candidate,
    first_gold_tool_call,
    run_heldout_eval,
    score_predictions,
)


def _ex(session, name="Bash", args=None):
    args = {"command": "ls"} if args is None else args
    return {
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}],
            },
        ],
        "metadata": {"session_id": session},
    }


def _perfect(ex):  # predicts exactly the gold call
    return first_gold_tool_call(ex)


def _wrong(ex):
    return {"type": "function", "function": {"name": "WrongTool", "arguments": {}}}


def _right_name_wrong_args(ex):
    return {"type": "function", "function": {"name": "Bash", "arguments": {"command": "WRONG"}}}


class TestFirstGoldToolCall:
    def test_extracts_first_assistant_call(self):
        gold = first_gold_tool_call(_ex("s1", name="Grep"))
        assert gold["function"]["name"] == "Grep"

    def test_missing_returns_empty(self):
        assert first_gold_tool_call({"messages": [{"role": "user", "content": "hi"}]}) == {}
        assert first_gold_tool_call({}) == {}


class TestScorePredictions:
    def test_alignment_and_metric_validation(self):
        with pytest.raises(ValueError):
            score_predictions([_ex("s1")], [{}], [{}], metric="bogus")
        with pytest.raises(ValueError):
            score_predictions([_ex("s1")], [{}], [{}, {}])  # misaligned lengths

    def test_scores_base_and_candidate(self):
        examples = [_ex("s1")]
        evals = score_predictions(examples, [_wrong(examples[0])], [_perfect(examples[0])])
        assert len(evals) == 1
        assert evals[0].base_score == 0.0
        assert evals[0].candidate_score == 1.0
        assert evals[0].delta == 1.0
        assert evals[0].session_id == "s1"


class TestRunHeldoutEval:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            run_heldout_eval([])


class TestEvaluateCandidate:
    def test_ships_when_candidate_beats_base(self):
        examples = [_ex(f"s{i % 3}") for i in range(9)]
        report = evaluate_candidate(examples, _wrong, _perfect, seed=0)
        assert report.n == 9
        assert report.n_clusters == 3  # bootstrap clustered by the 3 sessions
        assert report.base_pass_rate == 0.0
        assert report.candidate_pass_rate == 1.0
        assert report.trace_delta == 1.0
        assert report.bootstrap.better is True
        assert report.bootstrap.significant is True
        assert report.ship is True
        assert report.verdict.reasons == []

    def test_no_ship_when_identical(self):
        examples = [_ex(f"s{i % 3}") for i in range(9)]
        report = evaluate_candidate(examples, _perfect, _perfect, seed=0)
        assert report.trace_delta == 0.0
        assert report.ship is False
        assert report.verdict.reasons  # at least one blocking reason

    def test_forgetting_blocks_ship(self):
        examples = [_ex(f"s{i % 3}") for i in range(9)]
        report = evaluate_candidate(
            examples, _wrong, _perfect, forgetting_drops={"mmlu": 0.2}, seed=0
        )
        assert report.trace_delta == 1.0  # better on traces...
        assert report.ship is False  # ...but regressed on a general benchmark
        assert any("forgetting" in r for r in report.verdict.reasons)

    def test_metric_selection_changes_verdict(self):
        examples = [_ex(f"s{i % 3}") for i in range(9)]
        # exact_match: base gets args wrong -> fails -> candidate ships
        r_exact = evaluate_candidate(
            examples, _right_name_wrong_args, _perfect, metric="exact_match", seed=0
        )
        assert r_exact.base_pass_rate == 0.0 and r_exact.ship is True
        # name_match: base already names the right tool -> no delta -> no ship
        r_name = evaluate_candidate(
            examples, _right_name_wrong_args, _perfect, metric="name_match", seed=0
        )
        assert r_name.base_pass_rate == 1.0
        assert r_name.trace_delta == 0.0
        assert r_name.ship is False

    def test_to_dict_is_serializable(self):
        examples = [_ex(f"s{i % 3}") for i in range(9)]
        report = evaluate_candidate(examples, _wrong, _perfect, seed=0)
        d = report.to_dict()
        assert d["ship"] is True
        assert d["n"] == 9
        assert d["bootstrap"]["better"] is True
        assert d["metric"] == "exact_match"
