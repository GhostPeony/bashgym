import json

from bashgym.campaigns.preparation_inventory import preparation_inventory
from bashgym.ledger.contracts import ModelSpec, ModelVersionSpec, ProjectSpec
from bashgym.ledger.persistence import ExperimentLedgerRepository


def test_empty_inventory_is_read_only_and_does_not_invent_assets(tmp_path):
    before = set(tmp_path.rglob("*"))
    result = preparation_inventory(tmp_path, "workspace")
    assert all(not value for value in result["candidates"].values())
    assert result["reason_codes"] == ["registered_assets_unavailable"]
    assert result["execution_verified"] is False
    assert set(tmp_path.rglob("*")) == before


def test_inventory_filters_workspace_omits_transport_and_preserves_identity(tmp_path):
    ledger = ExperimentLedgerRepository(tmp_path / "campaigns" / "campaigns.sqlite3")
    ledger.initialize()
    for workspace in ("allowed", "other"):
        ledger.register_project(
            ProjectSpec(
                workspace_id=workspace,
                project_id="project",
                display_name="Study",
                owner_actor_id="owner",
            )
        )
        ledger.register_model(
            ModelSpec(
                workspace_id=workspace,
                project_id="project",
                model_id="learner",
                display_name="Learner",
                task_type="coding",
            )
        )
        ledger.register_model_version(
            ModelVersionSpec(
                workspace_id=workspace,
                project_id="project",
                model_id="learner",
                model_version_id=f"{workspace}-version",
                source_uri="/private/model/location",
                config_digest="a" * 64,
                metadata={"host": "private-host.invalid"},
            )
        )
    before = ledger.db_path.read_bytes()
    first = preparation_inventory(tmp_path, "allowed")
    second = preparation_inventory(tmp_path, "allowed")
    models = first["candidates"]["models"]
    assert len(models) == 1
    assert models[0]["record_id"] == "allowed-version"
    assert models == second["candidates"]["models"]
    assert models[0]["execution_verified"] is False
    assert "private-host" not in json.dumps(first)
    assert "/private/" not in json.dumps(first)
    assert "other-version" not in json.dumps(first)
    assert ledger.db_path.read_bytes() == before


def test_inventory_labels_never_publish_paths_or_compute_topology():
    from bashgym.campaigns.preparation_inventory import _public_label

    assert _public_label("/private/model/location", "models") == "Registered model"
    assert _public_label("private-host.invalid", "data") == "Registered dataset"
    assert _public_label("192.168.1.8", "evaluation") == "Registered evaluation"
    assert _public_label("ghp_testcanary", "models") == "Registered model"
    assert _public_label("a" * 161, "models") == "Registered model"
    assert _public_label("Private device label", "compute") == "Registered execution environment"
    assert _public_label("Personal coding learner", "models") == "Personal coding learner"
