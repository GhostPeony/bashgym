"""Tests for episode-level pass@k (bashgym/eval/passk.py)."""

import pytest

from bashgym.eval.passk import EpisodeResult, compute_pass_at_k, evaluate_pass_at_k, pass_at_k


class TestPassAtK:
    def test_k1_equals_success_rate(self):
        assert pass_at_k(5, 2, 1) == 0.4  # 2 of 5

    def test_zero_correct_is_zero(self):
        assert pass_at_k(5, 0, 1) == 0.0
        assert pass_at_k(5, 0, 3) == 0.0

    def test_all_correct_is_one(self):
        assert pass_at_k(5, 5, 1) == 1.0

    def test_fewer_failures_than_k_is_one(self):
        assert pass_at_k(5, 1, 5) == 1.0  # n-c = 4 < 5, any draw of 5 hits the pass

    def test_higher_k_rewards_coverage(self):
        assert pass_at_k(5, 2, 1) == 0.4
        assert abs(pass_at_k(5, 2, 3) - 0.9) < 1e-9  # 1 - C(3,3)/C(5,3)

    def test_degenerate_inputs(self):
        assert pass_at_k(0, 0, 1) == 0.0
        assert pass_at_k(5, 2, 0) == 0.0


class TestComputePassAtK:
    def test_per_task_and_mean(self):
        results = [
            EpisodeResult("a", [True, True, False, False, False]),  # c=2, n=5
            EpisodeResult("b", [False] * 5),  # c=0
        ]
        rep = compute_pass_at_k(results, k=1)
        assert rep.per_task == {"a": 0.4, "b": 0.0}
        assert rep.mean_pass_at_k == 0.2
        assert rep.n_tasks == 2

    def test_episode_result_counts(self):
        r = EpisodeResult("x", [True, False, True])
        assert r.n == 3 and r.c == 2


class TestEvaluatePassAtK:
    def test_with_injected_run_episode(self):
        tasks = [{"id": "t1"}, {"id": "t2"}]

        def run_episode(task, attempt):  # t1 passes on even attempts, t2 never
            return task["id"] == "t1" and attempt % 2 == 0

        rep = evaluate_pass_at_k(tasks, run_episode, n_samples=4, k=1)
        assert rep.per_task["t1"] == 0.5  # 2 of 4
        assert rep.per_task["t2"] == 0.0

    def test_n_samples_less_than_k_raises(self):
        with pytest.raises(ValueError):
            evaluate_pass_at_k([{"id": "t"}], lambda t, i: True, n_samples=2, k=5)

    def test_to_dict_serializable(self):
        rep = compute_pass_at_k([EpisodeResult("a", [True, False])], k=1)
        d = rep.to_dict()
        assert d["k"] == 1 and d["per_task"]["a"] == 0.5 and d["n_tasks"] == 1
