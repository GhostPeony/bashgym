"""Run-history and dataset-inspection endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app


@pytest.fixture
def client():
    return TestClient(app)


class TestTrainingRuns:
    def test_direct_training_rejects_ambiguous_cloud_label(self, client):
        resp = client.post(
            "/api/training/start",
            json={"strategy": "sft", "compute_target": "cloud"},
        )

        assert resp.status_code == 400
        assert "ambiguous" in resp.json()["detail"]

    def test_list_runs_returns_runs_key(self, client):
        resp = client.get("/api/training/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_runs_not_captured_by_run_id_route(self, client):
        """'/training/runs' must hit the literal route, not /training/{run_id}."""
        resp = client.get("/api/training/runs")
        assert resp.status_code == 200
        # The {run_id} route returns TrainingResponse (run_id key) or 404 — neither applies
        assert "runs" in resp.json()

    def test_metrics_404_for_unknown_run(self, client):
        resp = client.get("/api/training/runs/definitely-not-a-run/metrics")
        assert resp.status_code == 404

    def test_metrics_rejects_traversal(self, client):
        resp = client.get("/api/training/runs/..%2Fescape/metrics")
        assert resp.status_code in (400, 404)


class TestDatasetInspect:
    def test_inspect_rejects_path_outside_allowed_roots(self, client):
        resp = client.get(
            "/api/training/dataset/inspect",
            params={"path": "C:/Windows/System32/drivers/etc/hosts"},
        )
        assert resp.status_code == 400

    def test_inspect_missing_dataset_404(self, client, tmp_path):
        # default path may not exist in a clean checkout
        resp = client.get("/api/training/dataset/inspect")
        assert resp.status_code in (200, 404)

    def test_inspect_valid_dataset(self, client, tmp_path, monkeypatch):
        dataset = tmp_path / "train.jsonl"
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ]
        dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        # Point the configured data dir at tmp_path so the path is allowed
        from bashgym import config as bashgym_config

        settings = bashgym_config.get_settings()
        monkeypatch.setattr(settings.data, "data_dir", str(tmp_path))

        resp = client.get("/api/training/dataset/inspect", params={"path": str(dataset)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["examples"][0]["warnings"] == []


class TestRunCardEvidence:
    def test_list_and_validate_run_cards_surfaces_promotion_blockers(
        self, client, tmp_path, monkeypatch
    ):
        from bashgym import config as bashgym_config

        settings = bashgym_config.get_settings()
        monkeypatch.setattr(settings.data, "data_dir", str(tmp_path))
        run_card = {
            "schema_version": "bashgym.run_card.v1",
            "run_id": "run-ui",
            "training_method": "dpo",
            "base_model": "Qwen/Qwen3-Coder",
            "compute_target_id": "local_cpu_or_gpu",
            "training_plan_path": "plans/dpo.json",
            "source_manifest_path": "data/source_manifest.json",
            "preference_pairs_path": None,
            "reward_examples_path": None,
            "reward_eval_path": None,
            "dataset_card_path": None,
            "backend": None,
            "git_commit": None,
            "branch": None,
            "metrics_path": None,
            "release_evidence_path": None,
            "smoke_bundle_path": None,
            "claim_tier": "narrow_routing",
            "thresholds": {},
            "outputs": [],
            "known_limitations": [],
            "decision": "pending",
            "created_at": "2026-06-29T00:00:00+00:00",
        }
        path = tmp_path / "run_card.json"
        path.write_text(json.dumps(run_card), encoding="utf-8")

        listed = client.get("/api/training/runcards")
        assert listed.status_code == 200
        cards = listed.json()["run_cards"]
        assert any(card["run_id"] == "run-ui" for card in cards)

        validated = client.get(
            "/api/training/runcards/validate",
            params={"path": str(path), "promotion": "true"},
        )
        assert validated.status_code == 200
        payload = validated.json()
        assert payload["ok"] is False
        assert payload["promotion_explanation"]["ok"] is False
        assert "run-ui is not promotable" in payload["promotion_explanation"]["headline"]
        assert payload["promotion_explanation"]["next_actions"]
        codes = {finding["code"] for finding in payload["findings"]}
        assert "missing_metrics_path" in codes
        assert "missing_release_evidence_path" in codes
        assert "missing_preference_pairs_path" in codes

    def test_validate_run_card_rejects_paths_outside_allowed_roots(self, client):
        resp = client.get(
            "/api/training/runcards/validate",
            params={"path": "C:/Windows/System32/drivers/etc/hosts"},
        )

        assert resp.status_code == 400
