"""Product-facing no-GPU AutoResearch smoke coverage."""

from __future__ import annotations

import json

from bashgym.campaigns.control_smoke import run_autoresearch_control_smoke
from bashgym.cli import main


def test_control_smoke_proves_persistence_without_unlocking_quality(tmp_path):
    report = run_autoresearch_control_smoke(tmp_path)

    assert report["ok"] is True
    assert report["decision"] == "ineligible"
    assert report["next_action"] == "submit_baseline"
    assert report["checks"]["quality_baseline_locked"] is True
    assert (tmp_path / "campaigns.sqlite3").is_file()


def test_control_smoke_cli_runs_without_server_or_credentials(tmp_path, capsys):
    assert main(["campaign", "control-smoke", "--output-dir", str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["retained"] is True
    assert payload["output_directory"] == str(tmp_path.resolve())
