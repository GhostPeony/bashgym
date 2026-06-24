"""Tests for environment mix metrics."""

from bashgym.environments.contracts import EnvironmentAxis, EnvironmentSpec, VerifierSpec
from bashgym.environments.metrics import balance_score, summarize_environment_mix


def test_balance_score_uses_possible_bucket_count():
    assert balance_score(["a", "b"], possible_values=["a", "b"]) == 1.0
    assert balance_score(["a", "a"], possible_values=["a", "b"]) == 0.5


def test_summarize_environment_mix_tracks_distribution_and_pass_rates():
    envs = [
        EnvironmentSpec(
            id="a",
            instruction="task a",
            domain="data_processing",
            skills=["bash"],
            axes=[EnvironmentAxis("task_complexity", "simple")],
            verifier=VerifierSpec(kind="pytest"),
            metadata={"pass@1": 0.25, "pass@4": 0.5},
        ),
        EnvironmentSpec(
            id="b",
            instruction="task b",
            domain="security",
            skills=["bash", "debugging"],
            axes=[EnvironmentAxis("task_complexity", "complex")],
            verifier=VerifierSpec(kind="adversarial_corpus"),
            metadata={"pass@1": 75, "pass@4": 100},
        ),
    ]

    report = summarize_environment_mix(
        envs,
        possible_domains=["data_processing", "security"],
        possible_skills=["bash", "debugging"],
    )

    assert report.total == 2
    assert report.domain_distribution == {"data_processing": 1, "security": 1}
    assert report.skill_distribution == {"bash": 2, "debugging": 1}
    assert report.axis_balance["domain"] == 1.0
    assert report.verifier_distribution == {"adversarial_corpus": 1, "pytest": 1}
    assert report.mean_pass_rates["pass@1"] == 0.5
    assert report.mean_pass_rates["pass@4"] == 0.75
