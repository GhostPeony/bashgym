import json
from pathlib import Path

from bashgym.gym.run_analysis import analyze_run_artifacts

ROOT = Path(__file__).parents[2]


def test_run_analysis_recommends_only_existing_documentation(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"step": 1, "loss": 1.0}\n', encoding="utf-8")
    smoke_bundle = tmp_path / "smoke-bundle.json"
    smoke_bundle.write_text(
        json.dumps(
            {
                "schema_version": "bashgym.backend_smoke_bundle.v1",
                "contract_ready": True,
                "optimizer_ready": False,
                "backend_launch_ready": False,
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_run_artifacts(
        metrics_path=metrics,
        smoke_bundle_path=smoke_bundle,
    )

    assert analysis["docs"]
    assert all((ROOT / item["path"]).is_file() for item in analysis["docs"])
