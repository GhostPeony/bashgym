"""Tests for environment recipe AutoResearch search space."""

from pathlib import Path

import pytest

from bashgym.environments.contracts import (
    EnvironmentAxis,
    EnvironmentSpec,
    FixtureSpec,
    VerifierSpec,
)
from bashgym.gym.autoresearch import AutoResearchConfig, AutoResearcher, AutoResearchStatus
from bashgym.gym.environment_recipe_search_space import (
    PROPOSAL_SCHEMA_VERSION,
    EnvironmentRecipeSearchSpace,
)


def _env(
    env_id: str,
    *,
    domain: str,
    skills: list[str],
    verifier_kind: str,
    fixture_kind: str,
    pass_at_1: float,
    task_complexity: str = "moderate",
) -> EnvironmentSpec:
    return EnvironmentSpec(
        id=env_id,
        instruction=f"Complete {env_id}",
        domain=domain,
        skills=skills,
        axes=[
            EnvironmentAxis(name="task_complexity", value=task_complexity),
            EnvironmentAxis(name="command_complexity", value="single"),
            EnvironmentAxis(name="language", value="python"),
            EnvironmentAxis(name="fixture_kind", value=fixture_kind),
        ],
        fixtures=[FixtureSpec(path=f"{env_id}.txt", kind=fixture_kind)],
        verifier=VerifierSpec(kind=verifier_kind, command="./verify.sh", path="verify.sh"),
        metadata={"pass@1": pass_at_1},
    )


def _envs() -> list[EnvironmentSpec]:
    return [
        _env(
            "env_file_py",
            domain="file_ops",
            skills=["edit"],
            verifier_kind="pytest",
            fixture_kind="file",
            pass_at_1=0.25,
        ),
        _env(
            "env_shell_logs",
            domain="bash",
            skills=["search"],
            verifier_kind="script",
            fixture_kind="files",
            pass_at_1=0.4,
        ),
        _env(
            "env_repo_fix",
            domain="repo",
            skills=["debug"],
            verifier_kind="pytest",
            fixture_kind="repo",
            pass_at_1=0.3,
            task_complexity="complex",
        ),
        _env(
            "env_service",
            domain="service",
            skills=["http"],
            verifier_kind="unit",
            fixture_kind="service",
            pass_at_1=0.5,
        ),
        _env(
            "env_bash_multi",
            domain="bash",
            skills=["search", "reason"],
            verifier_kind="script",
            fixture_kind="archive",
            pass_at_1=0.1,
            task_complexity="complex",
        ),
    ]


def test_proposal_is_deterministic_for_seed():
    space = EnvironmentRecipeSearchSpace(_envs())
    genome = EnvironmentRecipeSearchSpace.create_default_genome(
        sample_size=3,
        pass_at_1_target=0.35,
        seed=7,
    )

    proposal_a = space.proposal_for(genome)
    proposal_b = space.proposal_for(genome)

    assert proposal_a["schema_version"] == PROPOSAL_SCHEMA_VERSION
    assert proposal_a["selected_count"] == 3
    assert proposal_a["selected_environment_ids"] == proposal_b["selected_environment_ids"]
    assert "domain" in proposal_a["mix_report"]["axis_balance"]


def test_mutate_preserves_recipe_bounds():
    space = EnvironmentRecipeSearchSpace(_envs(), mutation_rate=1.0, mutation_scale=0.5)
    genome = EnvironmentRecipeSearchSpace.create_default_genome(
        sample_size=3,
        pass_at_1_target=0.35,
        seed=0,
    )

    mutated = space.mutate(genome)

    assert 1 <= mutated["sample_size"] <= len(_envs())
    assert 0.0 <= mutated["pass_at_1_target"] <= 1.0
    assert set(mutated["axis_weights"]) >= {"domain", "skill", "fixture_kind"}
    assert all(value > 0 for value in mutated["axis_weights"].values())


def test_evaluate_returns_finite_loss():
    space = EnvironmentRecipeSearchSpace(_envs())
    genome = EnvironmentRecipeSearchSpace.create_default_genome(sample_size=4)

    metric = space.evaluate(genome, 1, 4)

    assert isinstance(metric, float)
    assert 0.0 <= metric < 10.0


@pytest.mark.asyncio
async def test_autoresearch_loop_can_optimize_recipe():
    space = EnvironmentRecipeSearchSpace(_envs(), mutation_rate=0.8, mutation_scale=0.3)
    base_genome = EnvironmentRecipeSearchSpace.create_default_genome(sample_size=3, seed=11)
    researcher = AutoResearcher(
        AutoResearchConfig(max_experiments=3, mode="simulate"),
        base_genome,
        search_space=space,
    )

    best_config, experiments = await researcher.run_loop(Path("data/environments"))
    proposal = space.proposal_for(best_config, metric=researcher.best_metric)

    assert researcher.status == AutoResearchStatus.COMPLETED
    assert len(experiments) == 3
    assert proposal["selected_count"] == best_config["sample_size"]
    assert researcher.best_metric < float("inf")
