import json
import sys

from bashgym.cli import main
from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.rollout import (
    CommandObservation,
    EnvironmentRolloutResult,
    RolloutAttempt,
)
from bashgym.eval.dppo_replay import build_dppo_replay_records, write_dppo_records_jsonl


def test_manifest_json_is_agent_readable(capsys):
    assert main(["manifest", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "training plan" in payload["commands"]
    assert "training analyze" in payload["commands"]
    assert any(doc["topic"] == "capabilities" for doc in payload["docs"])
    assert any(doc["topic"] == "world-models" for doc in payload["docs"])
    assert all(isinstance(doc["exists"], bool) for doc in payload["docs"])
    assert payload["next"][0]["command"].startswith("bashgym ")


def test_training_docs_topic_returns_content_in_json(capsys):
    assert main(["training", "docs", "--topic", "glossary", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["topic"] == "glossary"
    assert "Training Glossary" in payload["content"]


def test_training_docs_capabilities_topic_maps_full_spread(capsys):
    assert main(["training", "docs", "--topic", "capabilities", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["topic"] == "capabilities"
    assert "Training Capability Map" in payload["content"]
    assert "DPPO replay" in payload["content"]
    assert "ECHO" in payload["content"]


def test_training_plan_world_model_contains_echo_rwml_defaults(capsys):
    assert main(["training", "plan", "--strategy", "world-model", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    settings = payload["starting_settings"]
    assert payload["strategy"] == "world-model"
    assert settings["echo_aux_lambda"] == 0.05
    assert settings["rwml_distance_threshold"] == 0.2
    assert settings["rwml_history_window"] == 4
    assert "world_model_records" in payload["watch"]


def test_training_plan_grpo_adjusts_group_size_for_hardware(capsys):
    assert main(["training", "plan", "--strategy", "grpo", "--hardware", "dgx", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["starting_settings"]["training_profile"] == "terminal_rl_tmax_like"
    assert payload["starting_settings"]["grpo_group_size"] == 32
    assert "frac_reward_zero_std" in payload["watch"]


def test_replay_summarize_reports_world_model_coverage(tmp_path, capsys):
    environment = EnvironmentSpec.from_dict(
        {
            "id": "env_cli",
            "instruction": "List files",
            "verifier": {"command": "test -f answer.txt"},
        }
    )
    rollout = EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_cli",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={},
        ),
        workspace=tmp_path,
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="answer.txt\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=None,
    )
    replay_path = tmp_path / "replay.jsonl"
    records = build_dppo_replay_records(
        [environment],
        [rollout],
        batch_id="batch-cli",
        include_world_model=True,
    )
    write_dppo_records_jsonl(replay_path, records)

    assert main(["replay", "summarize", str(replay_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    summary = payload["summary"]
    assert summary["records"] == 1
    assert summary["world_model_records"] == 1
    assert summary["world_model"]["rwml_transitions"] == 1
    assert summary["world_model"]["echo_observation_chars"] == len("answer.txt\n")


def test_training_analyze_combines_metrics_replay_and_release_evidence(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "loss": 2.0, "reward_std": 0.1}),
                json.dumps(
                    {
                        "step": 2,
                        "loss": 1.5,
                        "reward_std": 0.0,
                        "frac_reward_zero_std": 0.75,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps({"ship": False, "reasons": ["environment holdout: pass@1 too low"]}),
        encoding="utf-8",
    )

    environment = EnvironmentSpec.from_dict(
        {
            "id": "env_analyze",
            "instruction": "List files",
            "verifier": {"command": "test -f answer.txt"},
        }
    )
    rollout = EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_analyze",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={},
        ),
        workspace=tmp_path,
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="answer.txt\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=None,
    )
    replay_path = tmp_path / "replay.jsonl"
    records = build_dppo_replay_records(
        [environment],
        [rollout],
        batch_id="batch-analyze",
        include_world_model=True,
    )
    write_dppo_records_jsonl(replay_path, records)

    assert (
        main(
            [
                "training",
                "analyze",
                "--metrics",
                str(metrics_path),
                "--replay",
                str(replay_path),
                "--release-evidence",
                str(release_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    assert payload["schema_version"] == "bashgym.run_analysis.v1"
    assert payload["verdict"]["level"] == "blocked"
    assert payload["training_metrics"]["loss"]["last"] == 1.5
    assert payload["replay_summary"]["world_model_records"] == 1
    assert "zero_reward_variance" in codes
    assert "many_zero_std_groups" in codes
    assert "world_model_quality_missing" in codes
    assert "release_gate_blocked" in codes


def test_training_analyze_reads_run_id_from_models_dir(tmp_path, capsys):
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 2.0, "heldout_pass_at_1": 0.2}) + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "analyze",
                "--run-id",
                "run-001",
                "--models-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-001"
    assert payload["training_metrics"]["points"] == 1
    assert payload["verdict"]["summary"]["has_heldout_signal"] is True


def test_training_analyze_accepts_release_world_model_quality(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 1, "loss": 2.0, "heldout_pass_at_1": 0.2}) + "\n",
        encoding="utf-8",
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps(
            {
                "ship": True,
                "reasons": [],
                "release_gate": {
                    "ship": True,
                    "world_model_quality": {
                        "present": True,
                        "diagnostic_only": True,
                        "signal": "improving",
                        "metrics": {"echo_loss": 0.7, "echo_loss_delta": -0.2},
                        "findings": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    environment = EnvironmentSpec.from_dict(
        {
            "id": "env_quality",
            "instruction": "List files",
            "verifier": {"command": "test -f answer.txt"},
        }
    )
    rollout = EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_quality",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={},
        ),
        workspace=tmp_path,
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="answer.txt\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=None,
    )
    replay_path = tmp_path / "replay.jsonl"
    records = build_dppo_replay_records(
        [environment],
        [rollout],
        batch_id="batch-quality",
        include_world_model=True,
    )
    write_dppo_records_jsonl(replay_path, records)

    assert (
        main(
            [
                "training",
                "analyze",
                "--metrics",
                str(metrics_path),
                "--replay",
                str(replay_path),
                "--release-evidence",
                str(release_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    assert "world_model_quality_missing" not in codes
    assert payload["release_evidence"]["world_model_quality"]["signal"] == "improving"
    assert payload["verdict"]["summary"]["has_world_model_quality"] is True


def test_serve_forwards_server_arguments(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_serve_main() -> None:
        captured["argv"] = sys.argv[:]

    monkeypatch.setattr("bashgym.main.main", fake_serve_main)

    assert (
        main(
            [
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "8002",
                "--log-level",
                "warning",
                "--",
                "--workers",
                "1",
            ]
        )
        == 0
    )

    assert captured["argv"] == [
        "bashgym serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8002",
        "--log-level",
        "warning",
        "--workers",
        "1",
    ]
