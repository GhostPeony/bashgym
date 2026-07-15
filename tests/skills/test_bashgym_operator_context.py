import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / "assistant"
    / "workspace"
    / "skills"
    / "bashgym-operator"
    / "scripts"
    / "operator_context.py"
)
SPEC = importlib.util.spec_from_file_location("bashgym_operator_context", SCRIPT)
assert SPEC and SPEC.loader
operator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = operator
SPEC.loader.exec_module(operator)


def _paths(tmp_path: Path) -> operator.OperatorPaths:
    return operator.OperatorPaths(
        bashgym_root=tmp_path / "bashgym",
        training_root=tmp_path / "training",
        observer=tmp_path / "training_runs.py",
        gbrain=tmp_path / "gbrain",
        activity_root=tmp_path / "activity",
        api_base_url="http://127.0.0.1:1",
        ledger_db=tmp_path / "campaigns.sqlite3",
    )


def test_doctor_does_not_claim_unavailable_campaign_control(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(operator, "_hf_jobs_cli", lambda: None)
    monkeypatch.setattr(operator, "_hf_token_configured", lambda: False)
    monkeypatch.setattr(operator, "_hf_cli_check", lambda *_args: False)
    project = paths.memexai_root
    (project / "scripts").mkdir(parents=True)
    (project / ".venv/bin").mkdir(parents=True)
    (project / ".venv/bin/python").write_text("", encoding="utf-8")
    (project / "scripts/train_embedding_retriever.py").write_text("", encoding="utf-8")

    doctor = operator._doctor(paths)

    assert doctor["abilities"]["launch_memexai_training"] is True
    assert doctor["abilities"]["launch_general_training"] is False
    assert doctor["abilities"]["mutate_desktop_campaign"] is False
    assert doctor["sources"]["training_cli"]["available"] is False
    assert doctor["abilities"]["launch_hf_jobs"] is False
    assert doctor["activation_lanes"]["huggingface_jobs"]["ready"] is False
    assert doctor["sources"]["gbrain_activity"]["requires_explicit_source"] is True
    assert doctor["abilities"]["read_experiment_ledger"] is False


def test_doctor_reports_hf_jobs_only_when_cli_and_credentials_exist(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(operator, "_hf_jobs_cli", lambda: "/usr/bin/hf jobs")
    monkeypatch.setattr(operator, "_hf_token_configured", lambda: True)
    monkeypatch.setattr(operator, "_hf_cli_check", lambda *_args: True)

    doctor = operator._doctor(paths)

    assert doctor["sources"]["hf_jobs_cli"]["available"] is True
    assert doctor["sources"]["hf_credentials"]["configured"] is True
    assert doctor["sources"]["hf_credentials"]["verified"] is True
    assert doctor["sources"]["hf_credentials"]["jobs_access"] is True
    assert doctor["abilities"]["launch_hf_jobs"] is True


def test_memexai_context_reads_manifests_but_not_protected_rows(tmp_path):
    paths = _paths(tmp_path)
    project = paths.memexai_root
    (project / "inputs").mkdir(parents=True)
    (project / "runs/candidate-c").mkdir(parents=True)
    (project / "system-ablations/protected-dev-v1/base").mkdir(parents=True)
    (project / "scripts").mkdir(parents=True)
    (project / ".venv/bin").mkdir(parents=True)
    paths.observer.write_text(
        "def discover_training_runs():\n"
        " return {'schema_version': 1, 'observed_at': 'now', "
        "'summary': {'count': 1, 'active': 0}, "
        "'runs': [{'id': 'run-1', 'name': 'candidate-c', 'status': 'completed', "
        "'artifact_path': '/secret'}]}\n",
        encoding="utf-8",
    )
    (project / "inputs/positive-aware-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "dataset.v1",
                "statistics": {"rows": 12},
                "output": {"sha256": "a" * 64, "path": "/private/train.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    (project / "inputs/heldout-dev-test.jsonl").write_text("not valid json", encoding="utf-8")
    (project / "runs/candidate-c/training_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "base_model_path": "/models/qwen",
                "train_pairs": 12,
                "training": {"loss": "cached_mnrl", "result_metrics": {"loss": 0.2}},
            }
        ),
        encoding="utf-8",
    )
    (
        project / "system-ablations/protected-dev-v1/base/retrieval_system_ablation_manifest.json"
    ).write_text(
        json.dumps(
            {
                "counts": {"queries": 18},
                "protocol": {"dense": {"model_id": "base"}, "selected_splits": ["dev"]},
                "systems": {"dense": {"metrics": {"exact_chunk_mrr": 0.4}}},
            }
        ),
        encoding="utf-8",
    )

    context = operator._memexai_context(paths, limit=8)

    assert context["runtime"]["runs"][0] == {
        "id": "run-1",
        "name": "candidate-c",
        "status": "completed",
    }
    assert context["dataset"]["statistics"]["rows"] == 12
    assert context["recent_training_runs"][0]["base_model"] == "qwen"
    assert context["development_comparisons"][0]["systems"]["dense"]["exact_chunk_mrr"] == 0.4
    assert context["protected_test"]["artifact_exists"] is True
    assert "not valid json" not in json.dumps(context)


def test_general_ledger_context_is_project_isolated(tmp_path):
    from bashgym.ledger.persistence import ExperimentLedgerRepository
    from tests.ledger.test_persistence import run_spec, seed_project

    paths = _paths(tmp_path)
    repository = ExperimentLedgerRepository(paths.experiment_ledger_db)
    repository.initialize()
    seed_project(repository)
    repository.register_run(run_spec())

    context = operator._ledger_context(
        paths,
        workspace_id="workspace-a",
        project_id="project-a",
        limit=10,
    )

    assert context["project_id"] == "project-a"
    assert context["recent_runs"][0]["run_id"] == "run-1"
    assert context["lineage"]["model_versions"][0]["model_version_id"] == "model-version-1"


def test_unscoped_context_requires_selection_instead_of_defaulting_to_embedding(tmp_path):
    from bashgym.ledger.contracts import ProjectSpec
    from bashgym.ledger.persistence import ExperimentLedgerRepository

    paths = _paths(tmp_path)
    paths.bashgym_root.mkdir(parents=True)
    repository = ExperimentLedgerRepository(paths.experiment_ledger_db)
    repository.initialize()
    for project_id in ("embedding-project", "general-llm-project"):
        repository.register_project(
            ProjectSpec(
                workspace_id="workspace-a",
                project_id=project_id,
                display_name=project_id,
                owner_actor_id="operator",
            )
        )

    context = operator._project_context(
        paths,
        workspace_id="workspace-a",
        project_id=None,
        limit=10,
    )

    assert context["project_id"] is None
    assert context["task_profile"] is None
    assert context["selection"]["requires_project"] is True
    assert context["authority"]["conflicts"][0]["code"] == "project_selection_required"


def test_operator_bundle_integrity_detects_a_skill_rewrite(tmp_path):
    root = tmp_path / "bashgym-operator"
    root.mkdir()
    skill = root / "SKILL.md"
    skill.write_text("canonical", encoding="utf-8")
    digest = __import__("hashlib").sha256(skill.read_bytes()).hexdigest()
    (root / "bundle.lock.json").write_text(
        json.dumps(
            {
                "schema_version": "bashgym.operator-bundle-lock.v1",
                "files": {"SKILL.md": digest},
            }
        ),
        encoding="utf-8",
    )

    assert operator._bundle_integrity(root)["verified"] is True
    skill.write_text("self-improved without review", encoding="utf-8")
    integrity = operator._bundle_integrity(root)
    assert integrity["verified"] is False
    assert integrity["mismatches"] == ["SKILL.md"]
