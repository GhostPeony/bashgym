"""Tests for executable environment API routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from bashgym.api.routes import app

client = TestClient(app)


def _record() -> dict:
    return {
        "task_id": "env_api_demo",
        "instruction": "Create hello.py and make it print hello.",
        "domain": "python",
        "skills": ["file_editing", "testing"],
        "verifier": {"kind": "unit_test", "command": "python hello.py | grep hello"},
        "files": {"README.md": "Write the script in this directory.\n"},
    }


def test_environment_pipelines_exposes_terminal_pipeline():
    response = client.get("/api/environments/pipelines")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["pipelines"][0]["name"] == "terminal_env_generation"
    assert "tmax_15k" in payload["external_sources"]


def test_normalize_environments_returns_mix_report_and_validation_warnings():
    response = client.post(
        "/api/environments/normalize",
        json={
            "source": "tmax",
            "source_uri": "fixture://demo",
            "records": [_record(), {"task_id": "bad_env"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["total"] == 2
    assert payload["report"]["domain_distribution"]["python"] == 1
    assert payload["environments"][0]["id"] == "env_api_demo"
    assert payload["environments"][0]["source_uri"] == "fixture://demo"
    assert payload["errors"][0]["id"] == "bad_env"
    assert "missing instruction" in payload["errors"][0]["validation_errors"]


def test_import_jsonl_loads_local_file(tmp_path):
    path = tmp_path / "envs.jsonl"
    path.write_text(
        '{"task_id":"env_one","instruction":"List files","domain":"shell","skill":"inspection"}\n'
        '{"task_id":"env_two","instruction":"Run tests","domain":"python","skill":"testing"}\n',
        encoding="utf-8",
    )

    response = client.post(
        "/api/environments/import-jsonl",
        json={"path": str(path), "source": "tmax"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [env["id"] for env in payload["environments"]] == ["env_one", "env_two"]
    assert payload["report"]["skill_distribution"] == {"inspection": 1, "testing": 1}


def test_import_jsonl_rejects_missing_file(tmp_path):
    response = client.post(
        "/api/environments/import-jsonl",
        json={"path": str(tmp_path / "missing.jsonl")},
    )

    assert response.status_code == 400
    assert "not found" in response.json()["detail"]


def test_decontaminate_environments_drops_benchmark_overlap():
    clean = {
        "id": "clean",
        "instruction": "Summarize a local log file.",
        "verifier": {"command": "./verify.sh"},
    }
    leaked = {
        "id": "leaked",
        "instruction": "install the package and run the tests before reporting success",
        "verifier": {"command": "./verify.sh"},
    }

    response = client.post(
        "/api/environments/decontaminate",
        json={
            "environments": [clean, leaked],
            "benchmark_texts": ["install the package and run the tests before reporting success"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [env["id"] for env in payload["environments"]] == ["clean"]
    assert payload["report"]["kept"] == 1
    assert payload["report"]["dropped"] == 1


def test_materialize_environment_bundle_writes_files(tmp_path):
    response = client.post(
        "/api/environments/materialize",
        json={
            "output_dir": str(tmp_path),
            "environment": {
                "id": "env_api_bundle",
                "instruction": "Create a hello script.",
                "files": {"hello.py": "print('hello')\n"},
                "verifier": {"command": "python hello.py | grep hello", "path": "verify.sh"},
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["build"]["env_id"] == "env_api_bundle"
    assert (tmp_path / "env_api_bundle" / "env.json").exists()
    assert (tmp_path / "env_api_bundle" / "hello.py").read_text(encoding="utf-8") == "print('hello')\n"

    response = client.post(
        "/api/environments/materialize",
        json={
            "output_dir": str(tmp_path),
            "environment": {
                "id": "env_api_bundle",
                "instruction": "Create a hello script.",
                "verifier": {"command": "./verify.sh"},
            },
        },
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]
