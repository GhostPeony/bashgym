import json

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
