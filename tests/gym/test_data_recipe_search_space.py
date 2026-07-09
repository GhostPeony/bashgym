from bashgym.gym.autoresearch import AutoResearchConfig, AutoResearcher, AutoResearchStatus
from bashgym.gym.data_recipe_search_space import (
    DATA_RECIPE_SCHEMA_VERSION,
    DataRecipeSearchSpace,
)
from bashgym.sources import SourceUse, get_source, list_sources


def test_data_recipe_default_genome_normalizes_weights():
    sources = [get_source("ultrafeedback_binarized"), get_source("helpsteer2")]
    space = DataRecipeSearchSpace(sources, goal=SourceUse.DPO)

    genome = DataRecipeSearchSpace.create_default_genome(space.source_ids, space.domains)

    assert round(sum(genome["source_weights"].values()), 4) == 1.0
    assert round(sum(genome["domain_weights"].values()), 4) == 1.0
    assert genome["quality_threshold"] == 0.7


def test_data_recipe_mutate_preserves_bounds():
    sources = list_sources()
    space = DataRecipeSearchSpace(sources, goal=SourceUse.DPO, mutation_rate=1.0)
    genome = DataRecipeSearchSpace.create_default_genome(space.source_ids, space.domains)

    mutated = space.mutate(genome)

    assert 1 <= mutated["sample_size"] <= 1_000_000
    assert 0.0 <= mutated["quality_threshold"] <= 1.0
    assert 0.0 <= mutated["synthetic_multiplier"] <= 10.0
    assert 0.0 <= mutated["decontam_jaccard_threshold"] <= 1.0
    assert round(sum(mutated["source_weights"].values()), 4) == 1.0


def test_data_recipe_eval_only_source_penalized_for_training_goal():
    sources = [get_source("harbor_terminal_bench"), get_source("ultrafeedback_binarized")]
    space = DataRecipeSearchSpace(sources, goal=SourceUse.SFT)
    eval_only = DataRecipeSearchSpace.create_default_genome(space.source_ids, space.domains)
    eval_only["source_weights"] = {
        "harbor_terminal_bench": 1.0,
        "ultrafeedback_binarized": 0.0,
    }
    trainable = DataRecipeSearchSpace.create_default_genome(space.source_ids, space.domains)
    trainable["source_weights"] = {
        "harbor_terminal_bench": 0.0,
        "ultrafeedback_binarized": 1.0,
    }

    assert space.evaluate(eval_only, 1, 2) > space.evaluate(trainable, 1, 2)


def test_data_recipe_proposal_contains_source_and_data_designer_plan():
    sources = [get_source("ultrafeedback_binarized"), get_source("helpsteer2")]
    space = DataRecipeSearchSpace(sources, goal=SourceUse.DPO)
    genome = DataRecipeSearchSpace.create_default_genome(space.source_ids, space.domains)

    proposal = space.proposal_for(genome)

    assert proposal["schema_version"] == DATA_RECIPE_SCHEMA_VERSION
    assert proposal["goal"] == "dpo"
    assert proposal["data_designer"]["pipeline"] == "from_source"
    assert {source["id"] for source in proposal["sources"]} == {
        "ultrafeedback_binarized",
        "helpsteer2",
    }


def test_data_recipe_search_space_runs_with_autoresearch_loop(tmp_path):
    sources = [get_source("ultrafeedback_binarized"), get_source("helpsteer2")]
    space = DataRecipeSearchSpace(sources, goal=SourceUse.DPO, mutation_rate=0.8)
    base_genome = DataRecipeSearchSpace.create_default_genome(space.source_ids, space.domains)
    researcher = AutoResearcher(
        config=AutoResearchConfig(
            search_params=["data_recipe"],
            max_experiments=3,
            mode="simulate",
            mutation_rate=0.8,
            eval_metric="data_recipe_loss",
        ),
        base_trainer_config=base_genome,
        search_space=space,
    )

    import asyncio

    asyncio.run(researcher.run_loop(tmp_path))

    assert researcher.status == AutoResearchStatus.COMPLETED
    assert len(researcher.experiments) == 3
    assert researcher.best_metric < float("inf")
