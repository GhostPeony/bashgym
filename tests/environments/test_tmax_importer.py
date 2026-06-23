"""Tests for TMax/Harbor-style environment import."""

import json
from pathlib import Path

from bashgym.environments.tmax_importer import TMAX_HF_DATASETS, TMaxImporter

FIXTURE = Path("tests/fixtures/tmax_envs/tmax_demo_001.jsonl")


def _record():
    return {
        "task_id": "tmax_demo_001",
        "task_instruction": "Use the terminal to repair the CSV summarizer.",
        "domain": "data_processing",
        "skills": ["bash", "python", "debugging"],
        "persona": "data engineer cleaning a flaky report",
        "language": "python",
        "task_complexity": "moderate",
        "command_complexity": "bash+python",
        "verifier_kind": "metric_threshold",
        "files": {
            "summarize.py": "print('todo')\n",
            "tests/test_summary.py": "def test_summary():\n    assert True\n",
        },
        "verifier": {
            "kind": "metric_threshold",
            "command": "pytest tests",
            "reward_type": "graded",
            "success_threshold": 0.8,
        },
        "dockerfile": "Dockerfile",
        "base_image": "python:3.11-slim",
        "pass@1": 42,
    }


def test_importer_normalizes_record_and_preserves_raw():
    env = TMaxImporter().from_records([_record()], source_uri="hf://allenai/TMax-15K/train")[0]

    assert env.id == "tmax_demo_001"
    assert env.instruction.startswith("Use the terminal")
    assert env.domain == "data_processing"
    assert env.skills == ["bash", "python", "debugging"]
    assert env.axis_value("persona") == "data engineer cleaning a flaky report"
    assert env.verifier.is_graded
    assert env.verifier.success_threshold == 0.8
    assert env.build.base_image == "python:3.11-slim"
    assert env.metadata["raw_record"]["task_id"] == "tmax_demo_001"


def test_importer_reads_jsonl(tmp_path):
    path = tmp_path / "tmax.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    envs = TMaxImporter(preserve_raw=False).from_jsonl(path)

    assert len(envs) == 1
    assert envs[0].id == "tmax_demo_001"
    assert "raw_record" not in envs[0].metadata


def test_importer_reads_checked_in_fixture():
    envs = TMaxImporter().from_jsonl(FIXTURE)

    assert len(envs) == 1
    assert envs[0].id == "tmax_demo_001"
    assert envs[0].axis_value("task_complexity") == "moderate"


def test_tmax_dataset_aliases_include_release_targets():
    assert TMAX_HF_DATASETS["tmax_15k"] == "allenai/TMax-15K"
    assert "open-instruct" in TMAX_HF_DATASETS["tmax_15k_open_instruct"]
