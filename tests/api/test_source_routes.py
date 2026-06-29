import json
from pathlib import Path

from fastapi.testclient import TestClient

from bashgym.api.routes import app

client = TestClient(app)


def test_source_routes_list_and_inspect_catalog():
    response = client.get("/api/sources")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["schema_version"] == "bashgym.source_catalog.v1"
    assert any(source["id"] == "bfcl" for source in payload["sources"])

    response = client.get("/api/sources/helpsteer2")
    assert response.status_code == 200
    card = response.json()
    assert card["source"]["id"] == "helpsteer2"
    assert card["source"]["training_eligible"] is True


def test_source_routes_recommend_training_sources_without_eval_only():
    response = client.post("/api/sources/recommend", json={"goal": "dpo"})

    assert response.status_code == 200
    payload = response.json()
    ids = {item["source"]["id"] for item in payload["recommendations"]}
    assert "ultrafeedback_binarized" in ids
    assert "rewardbench" not in ids


def test_source_routes_block_eval_only_training_prepare():
    response = client.post("/api/sources/harbor_terminal_bench/prepare", json={"goal": "sft"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["source"]["id"] == "harbor_terminal_bench"
    assert "eval_only_source_for_training" in detail["use_verdict"]["blocking_codes"]


def test_source_routes_prepare_training_source_manifest(tmp_path):
    response = client.post(
        "/api/sources/ultrafeedback_binarized/prepare",
        json={"goal": "dpo", "output_dir": str(tmp_path)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["source"]["id"] == "ultrafeedback_binarized"
    assert tmp_path.joinpath("source_manifest.json").exists()


def test_source_routes_prepare_local_input_artifacts(tmp_path):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "id": "uf-1",
                "prompt": "Fix a failing test.",
                "chosen": "Run pytest and patch the failing function.",
                "rejected": "Claim success without running tests.",
                "metadata": {
                    "quality_score": 0.9,
                    "label_source": "fixture",
                    "decontamination_status": "checked",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/sources/ultrafeedback_binarized/prepare",
        json={
            "goal": "dpo",
            "input_path": str(source_path),
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["schema_version"] == "bashgym.source_artifact_prepare.v1"
    assert payload["artifacts"][0]["validation"]["ok"] is True
    assert tmp_path.joinpath("out", "dpo_pairs.jsonl").exists()


def test_source_routes_fetch_huggingface_source_records(tmp_path, monkeypatch):
    def fake_fetch(card, *, output_dir, split, subset=None, revision=None, limit=None):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        records_path = output_path / "source_records.jsonl"
        records_path.write_text(
            json.dumps(
                {
                    "id": "uf-1",
                    "prompt": "Fix a failing test.",
                    "chosen": "Run pytest and patch the failing function.",
                    "rejected": "Claim success without running tests.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = output_path / "source_fetch_report.json"
        return {
            "schema_version": "bashgym.source_fetch.v1",
            "ok": True,
            "source_id": card.id,
            "source_name": card.name,
            "huggingface_id": card.huggingface_id,
            "split": split,
            "subset": subset,
            "revision": revision,
            "limit": limit,
            "output_dir": str(output_path),
            "records_path": str(records_path),
            "report_path": str(report_path),
            "record_count": 1,
            "truncated": False,
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr("bashgym.api.source_routes.fetch_source_records", fake_fetch)

    response = client.post(
        "/api/sources/ultrafeedback_binarized/fetch",
        json={"output_dir": str(tmp_path / "fetch"), "split": "train_prefs", "limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["schema_version"] == "bashgym.source_fetch.v1"
    assert payload["split"] == "train_prefs"
    assert Path(payload["records_path"]).exists()


def test_source_routes_prepare_can_fetch_then_convert(tmp_path, monkeypatch):
    def fake_fetch(card, *, output_dir, split, subset=None, revision=None, limit=None):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        records_path = output_path / "source_records.jsonl"
        records_path.write_text(
            json.dumps(
                {
                    "id": "uf-1",
                    "prompt": "Fix a failing test.",
                    "chosen": "Run pytest and patch the failing function.",
                    "rejected": "Claim success without running tests.",
                    "metadata": {"decontamination_status": "checked"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "schema_version": "bashgym.source_fetch.v1",
            "ok": True,
            "source_id": card.id,
            "source_name": card.name,
            "huggingface_id": card.huggingface_id,
            "split": split,
            "subset": subset,
            "revision": revision,
            "limit": limit,
            "output_dir": str(output_path),
            "records_path": str(records_path),
            "report_path": str(output_path / "source_fetch_report.json"),
            "record_count": 1,
            "truncated": False,
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr("bashgym.api.source_routes.fetch_source_records", fake_fetch)

    response = client.post(
        "/api/sources/ultrafeedback_binarized/prepare",
        json={
            "goal": "dpo",
            "fetch": True,
            "output_dir": str(tmp_path / "out"),
            "split": "train_prefs",
            "limit": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["fetch_report"]["schema_version"] == "bashgym.source_fetch.v1"
    assert payload["artifacts"][0]["artifact_type"] == "dpo_pairs"
    assert tmp_path.joinpath("out", "dpo_pairs.jsonl").exists()
