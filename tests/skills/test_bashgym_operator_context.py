import importlib.util
import json
import re
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

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


class _OperatorRedirectTarget(BaseHTTPRequestHandler):
    requests = 0

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
        self.__class__.requests += 1
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _OperatorRedirectSource(BaseHTTPRequestHandler):
    location = ""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
        self.send_response(302)
        self.send_header("Location", self.__class__.location)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


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


def test_operator_paths_use_the_portable_cli_api_base_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("BASHGYM_API_BASE_URL", "http://127.0.0.1:7000")
    monkeypatch.setenv("BASHGYM_API_BASE", "http://127.0.0.1:9000/api")

    paths = operator.OperatorPaths.from_environment()

    assert paths.api_base_url == "http://127.0.0.1:9000/api"


def test_operator_paths_reject_secret_bearing_api_base_without_echoing(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("BASHGYM_API_BASE", "http://operator:top-secret@localhost:8003/api")

    with pytest.raises(ValueError) as exc_info:
        operator.OperatorPaths.from_environment()

    assert "top-secret" not in str(exc_info.value)


def test_operator_health_and_workspace_reads_refuse_api_redirects(tmp_path):
    target = ThreadingHTTPServer(("127.0.0.1", 0), _OperatorRedirectTarget)
    source = ThreadingHTTPServer(("127.0.0.1", 0), _OperatorRedirectSource)
    _OperatorRedirectTarget.requests = 0
    _OperatorRedirectSource.location = (
        f"http://127.0.0.1:{target.server_port}/not-api?token=redirect-secret"
    )
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (target, source)
    ]
    for thread in threads:
        thread.start()
    paths = _paths(tmp_path)
    paths = operator.replace(
        paths,
        api_base_url=f"http://127.0.0.1:{source.server_port}/api",
    )
    try:
        assert operator._api_health(paths.api_base_url) is False
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            operator._workspace_context(paths, "workspace-a", "json")
    finally:
        for server in (source, target):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)

    assert _OperatorRedirectTarget.requests == 0
    message = str(exc_info.value)
    assert "redirect-secret" not in message
    assert "/not-api" not in message


def test_doctor_does_not_claim_unavailable_campaign_control(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(operator, "_campaign_cli", lambda _paths: None)
    monkeypatch.setattr(operator, "_training_cli", lambda _paths: None)
    monkeypatch.setattr(operator, "_api_health", lambda _base_url: False)
    monkeypatch.setattr(operator, "_hf_jobs_cli", lambda: None)
    monkeypatch.setattr(operator, "_hf_token_configured", lambda: False)
    monkeypatch.setattr(operator, "_hf_cli_check", lambda *_args: False)
    doctor = operator._doctor(paths)

    assert set(doctor["abilities"]) == {
        "inspect_local_runs",
        "search_curated_activity",
        "mutate_desktop_campaign",
        "launch_general_training",
        "read_experiment_ledger",
        "launch_hf_jobs",
    }
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


def test_selected_project_context_does_not_infer_a_task_profile(tmp_path):
    from bashgym.ledger.contracts import ProjectSpec
    from bashgym.ledger.persistence import ExperimentLedgerRepository

    paths = _paths(tmp_path)
    paths.training_root.mkdir(parents=True)
    repository = ExperimentLedgerRepository(paths.experiment_ledger_db)
    repository.initialize()
    repository.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            display_name="Project A",
            owner_actor_id="operator",
        )
    )

    context = operator._project_context(
        paths,
        workspace_id="workspace-a",
        project_id="project-a",
        limit=8,
    )

    assert context["project_id"] == "project-a"
    assert context["ledger"]["project"]["project_id"] == "project-a"
    assert context["task_profile"] is None
    project_source = next(
        source
        for source in context["authority"]["sources"]
        if source["source_id"] == "project_local_evidence"
    )
    assert project_source["freshness"] == "not_loaded"


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


def test_single_project_context_still_requires_explicit_project_selection(tmp_path):
    from bashgym.ledger.contracts import ProjectSpec
    from bashgym.ledger.persistence import ExperimentLedgerRepository

    paths = _paths(tmp_path)
    repository = ExperimentLedgerRepository(paths.experiment_ledger_db)
    repository.initialize()
    repository.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="only-project",
            display_name="Only Project",
            owner_actor_id="operator",
        )
    )

    context = operator._project_context(
        paths,
        workspace_id="workspace-a",
        project_id=None,
        limit=10,
    )

    assert context["selection"]["requires_project"] is True
    assert context["project_id"] is None


def test_project_context_filters_host_runs_to_exact_canonical_tracking_scope(tmp_path):
    paths = _paths(tmp_path)
    paths.observer.write_text(
        """
def discover_training_runs():
    return {
        "schema_version": "observer.v1",
        "observed_at": "2026-07-17T00:00:00Z",
        "runs": [
            {"id": "exact", "status": "running", "tracking_ids": {"workspace_id": "workspace-a", "project_id": "project-a", "secret": "do-not-copy"}},
            {"id": "other-project", "status": "running", "tracking_ids": {"workspace_id": "workspace-a", "project_id": "project-b"}},
            {"id": "other-workspace", "status": "failed", "tracking_ids": {"workspace_id": "workspace-b", "project_id": "project-a"}},
            {"id": "unscoped", "status": "running"},
        ],
    }
""",
        encoding="utf-8",
    )

    context = operator._project_context(
        paths,
        workspace_id="workspace-a",
        project_id="project-a",
        limit=10,
    )

    assert [run["id"] for run in context["runtime"]["runs"]] == ["exact"]
    assert context["runtime"]["runs"][0]["tracking_ids"] == {
        "workspace_id": "workspace-a",
        "project_id": "project-a",
    }
    assert context["runtime"]["scope"] == {
        "workspace_id": "workspace-a",
        "project_id": "project-a",
        "excluded_other_scope": 2,
        "excluded_unscoped": 1,
    }
    assert context["runtime"]["summary"]["count"] == 1


def test_unselected_project_context_excludes_all_host_runs(tmp_path):
    paths = _paths(tmp_path)
    paths.observer.write_text(
        """
def discover_training_runs():
    return {"runs": [{"id": "project-run", "tracking_ids": {"workspace_id": "workspace-a", "project_id": "project-a"}}]}
""",
        encoding="utf-8",
    )

    context = operator._project_context(
        paths,
        workspace_id="workspace-a",
        project_id=None,
        limit=10,
    )

    assert context["runtime"]["runs"] == []
    assert context["runtime"]["scope"]["project_id"] is None
    assert context["runtime"]["scope"]["excluded_other_scope"] == 1


def test_operator_helper_rejects_blank_workspace_and_project_ids():
    for arguments in (
        ["context", "--workspace-id", " "],
        ["context", "--workspace-id", "workspace-a", "--project", " "],
        ["workspace", "--workspace-id", "\t"],
        ["ledger", "--workspace-id", "workspace-a", "--project-id", " "],
    ):
        try:
            operator._parser().parse_args(arguments)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"blank identifier was accepted: {arguments}")

    with pytest.raises(ValueError, match="workspace_id"):
        operator._project_context(
            _paths(Path(".")), workspace_id=" ", project_id="project-a", limit=1
        )


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


def test_public_operator_bundle_has_no_private_project_or_machine_residue():
    bundle_root = SCRIPT.parents[1]
    lock = json.loads((bundle_root / "bundle.lock.json").read_text(encoding="utf-8"))
    public_files = [
        bundle_root / relative
        for relative in lock["files"]
        if not relative.startswith("..")
    ]
    public_files.append(bundle_root / "bundle.lock.json")

    private_term = re.compile(
        r"(?<![a-z0-9])(?:memexai|ponyo|gx10|cade)(?![a-z0-9])",
        re.IGNORECASE,
    )
    private_path = re.compile(
        r"(?:[a-z]:[\\/]users[\\/][^\\/\s\"']+|/(?:users|home)/(?![\"'])[^/\s\"']+)",
        re.IGNORECASE,
    )
    for path in public_files:
        content = path.read_text(encoding="utf-8")
        relative = path.relative_to(bundle_root).as_posix()
        searchable = f"{relative}\n{content}".casefold()
        assert private_term.search(searchable) is None, relative
        assert private_path.search(searchable) is None, relative
