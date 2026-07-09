import json

from bashgym.sources import (
    SourceUse,
    get_source,
    prepare_source_manifest,
    recommend_sources,
    validate_source_use,
)
from bashgym.sources.catalog import validate_catalog


def test_curated_catalog_validates():
    assert validate_catalog() == {}


def test_eval_only_source_blocks_training_use_by_default():
    card = get_source("harbor_terminal_bench")

    verdict = validate_source_use(card, SourceUse.SFT)

    assert verdict["ok"] is False
    assert "eval_only_source_for_training" in verdict["blocking_codes"]
    assert "source_not_training_eligible" in verdict["blocking_codes"]


def test_training_eligible_preference_source_allows_dpo():
    card = get_source("ultrafeedback_binarized")

    verdict = validate_source_use(card, SourceUse.DPO)

    assert verdict["ok"] is True
    assert verdict["blocking_codes"] == []


def test_recommend_sources_for_dpo_excludes_eval_only_by_default():
    recommendations = recommend_sources(goal=SourceUse.DPO)
    ids = {item["source"]["id"] for item in recommendations}

    assert "ultrafeedback_binarized" in ids
    assert "helpsteer2" in ids
    assert "rewardbench" not in ids


def test_prepare_source_manifest_writes_metadata(tmp_path):
    card = get_source("helpsteer2")

    manifest = prepare_source_manifest(card, goal=SourceUse.REWARD_MODEL, output_dir=tmp_path)

    path = tmp_path / "source_manifest.json"
    assert manifest["use_verdict"]["ok"] is True
    assert manifest["manifest_path"] == str(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "bashgym.source_manifest.v1"
    assert persisted["source"]["id"] == "helpsteer2"
