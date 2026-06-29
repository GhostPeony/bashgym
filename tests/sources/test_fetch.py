import json

from bashgym.sources import fetch_source_records, get_source, prepare_source_artifacts


def test_fetch_source_records_writes_capped_hf_jsonl_with_metadata(tmp_path):
    def loader(huggingface_id, *, split, subset, revision):
        assert huggingface_id == "HuggingFaceH4/ultrafeedback_binarized"
        assert split == "train_prefs"
        assert subset == "default"
        assert revision == "main"
        return [
            {
                "id": "row-1",
                "prompt": "Fix a failing test.",
                "chosen": "Run pytest and patch the bug.",
                "rejected": "Skip the test.",
                "metadata": {"quality_score": 0.9},
            },
            {
                "id": "row-2",
                "prompt": "Explain git bisect.",
                "chosen": "Use binary search over commits.",
                "rejected": "Guess randomly.",
            },
        ]

    report = fetch_source_records(
        get_source("ultrafeedback_binarized"),
        output_dir=tmp_path,
        split="train_prefs",
        subset="default",
        revision="main",
        limit=1,
        loader=loader,
    )

    assert report["ok"] is True
    assert report["record_count"] == 1
    assert report["truncated"] is True
    assert tmp_path.joinpath("source_fetch_report.json").exists()

    records = [
        json.loads(line)
        for line in tmp_path.joinpath("source_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["metadata"]["quality_score"] == 0.9
    assert records[0]["metadata"]["source_fetch"]["huggingface_id"] == (
        "HuggingFaceH4/ultrafeedback_binarized"
    )
    assert records[0]["metadata"]["source_fetch"]["split"] == "train_prefs"


def test_fetch_source_records_blocks_cards_without_huggingface_id(tmp_path):
    report = fetch_source_records(get_source("bfcl"), output_dir=tmp_path)

    assert report["ok"] is False
    assert "source_has_no_huggingface_id" in report["errors"]
    assert tmp_path.joinpath("source_fetch_report.json").exists()


def test_fetched_source_records_feed_existing_artifact_adapter(tmp_path):
    def loader(_huggingface_id, *, split, subset, revision):
        return [
            {
                "id": "row-1",
                "prompt": "Fix a failing test.",
                "chosen": "Run pytest and patch the bug.",
                "rejected": "Skip the test.",
                "metadata": {"decontamination_status": "checked"},
            }
        ]

    fetch_report = fetch_source_records(
        get_source("ultrafeedback_binarized"),
        output_dir=tmp_path / "fetch",
        limit=5,
        loader=loader,
    )

    report = prepare_source_artifacts(
        get_source("ultrafeedback_binarized"),
        goal="dpo",
        input_path=fetch_report["records_path"],
        output_dir=tmp_path / "artifacts",
    )

    assert report["ok"] is True
    assert report["artifacts"][0]["artifact_type"] == "dpo_pairs"
    assert tmp_path.joinpath("artifacts", "dpo_pairs.jsonl").exists()
