"""Exercise an installed studio outside a checkout, without models or resident services.

Run with the release environment's Python and -I. The temporary service binds an
unused loopback port; all credentials, skill files and setup state are disposable.
This verifies the installed HTTP path, not a live training recipe or OS autostart.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def check() -> dict:
    import httpx
    import psutil

    import bashgym

    package = Path(bashgym.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Run with an installed release environment and Python -I")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="bashgym-studio-proof-") as temporary:
        root = Path(temporary)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("BASHGYM_") and key != "PYTHONPATH"
        }
        environment.update(
            BASHGYM_DIR=str(root / "state"),
            BASHGYM_DISABLE_KEYRING="1",
            HERMES_HOME=str(root / "hermes"),
        )

        def cli(*args: str, expected_code: int = 0) -> dict:
            result = subprocess.run(
                [sys.executable, "-I", "-m", "bashgym.cli", *args, "--json"],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=45,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != expected_code:
                # CLI error JSON has no secret values; do not echo successful init codes.
                raise RuntimeError(f"Installed CLI failed: {result.stdout}\n{result.stderr}")
            return json.loads(result.stdout)

        initial = cli("init", "--agent-host", "hermes", "--no-service")
        repeated = cli("init", "--agent-host", "hermes", "--no-service")
        assert initial["training_started"] is False
        assert repeated["replayed"] is True
        assert repeated["workspace_id"] == initial["workspace_id"]
        missing = cli("research", "prepare", "--template-id", "selected", expected_code=2)
        assert missing["compute_started"] is False and "stop_rules" in missing["missing_inputs"]
        skills = cli("operator", "skills", "check", "--host", "hermes")
        assert skills["verified"], skills
        from bashgym.factory.data_factory import TrainingExample
        from bashgym.factory.example_generator import ExampleGenerator, ExampleGeneratorConfig
        from bashgym.factory.export_artifacts import resolve_training_export

        generator = ExampleGenerator(ExampleGeneratorConfig(output_dir=str(root / "examples")))
        rows = [
            TrainingExample(
                str(index),
                "system",
                f"fixture task {index}",
                f"fixture answer {index}",
                metadata={"repo_id": f"fixture-repository-{index}"},
            )
            for index in range(2)
        ]
        exported = generator.export_for_nemo(rows, root / "exports", train_split=0.5)
        assert exported["train_count"] == exported["val_count"] == 1
        assert (
            resolve_training_export(root / "exports", "train", exported["export_id"])
            == exported["train"]
        )
        from bashgym.campaigns.export import CampaignExportSnapshot, export_campaign_evidence

        report = export_campaign_evidence(
            CampaignExportSnapshot(campaign={"campaign_id": "installed-fixture"}),
            root / "campaign-export",
            formats=("json", "markdown", "csv"),
        )
        assert {item["name"] for item in report["files"]} == {
            "campaign_evidence.json",
            "campaign_report.md",
            "attempts.csv",
            "artifacts.csv",
            "comparisons.csv",
        }
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with (root / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "bashgym.campaigns.worker_service",
                    "run-api",
                    "--port",
                    str(port),
                    "--data-dir",
                    str(root / "state"),
                ],
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                with httpx.Client(base_url=base, timeout=5, trust_env=False) as http:
                    deadline = time.monotonic() + 45
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError((root / "server.log").read_text(encoding="utf-8"))
                        try:
                            health = http.get("/api/health")
                            if health.status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Installed API startup timed out")
                        time.sleep(0.1)
                    assert health.json()["authentication_required"] is True
                    assert health.json()["studio_protocol"] == "bashgym.studio.v1"
                    page = http.get("/")
                    assert page.status_code == 200
                    assets = re.findall(r'(?:src|href)="([^\"]+\.(?:js|css))"', page.text)
                    assert assets, "Installed page has no built browser assets"
                    for asset in assets:
                        assert http.get(asset).status_code == 200, asset
                    for font in ("Fraunces.ttf", "SourceSans3.ttf", "IBMPlexMono-Regular.ttf"):
                        response = http.get(f"/fonts/studio/{font}")
                        assert response.status_code == 200 and len(response.content) > 1000
                    query = {"workspace_id": initial["workspace_id"]}
                    assert http.get("/api/campaigns/setup/context", params=query).status_code == 401
                    headers = {"X-Requested-With": "XMLHttpRequest", "Origin": base}
                    response = http.post(
                        "/api/auth/local/pair",
                        json={"code": repeated["pairing_code"]},
                        headers=headers,
                    )
                    assert response.status_code == 200, response.text
                    assert "httponly" in response.headers["set-cookie"].lower()
                    identity = http.get("/api/auth/me")
                    assert identity.status_code == 200, identity.text
                    assert identity.headers["Cache-Control"] == "no-store"
                    designer = http.get("/api/factory/designer/pipelines")
                    assert designer.status_code == 200, designer.text
                    readiness = designer.json()["readiness"]
                    assert readiness["scope"] == "backend_process_imports"
                    assert readiness["provider_verified"] is False
                    assert readiness["generation_verified"] is False
                    assert (
                        http.post(
                            "/api/auth/local/pair",
                            json={"code": repeated["pairing_code"]},
                            headers=headers,
                        ).status_code
                        == 401
                    )
                    context = http.get("/api/campaigns/setup/context", params=query)
                    assert context.status_code == 200, context.text
                    context = context.json()
                    session_id = "setupsess_0123456789abcdef0123456789abcdef"
                    for version, (step, selection) in enumerate(
                        [
                            ("template", context["templates"][0]["template_id"]),
                            ("installation", context["installations"][0]["installation_id"]),
                        ]
                    ):
                        response = http.post(
                            "/api/campaigns/setup/session",
                            json={
                                **query,
                                "session_id": session_id,
                                "expected_version": version,
                                "step": step,
                                "selection_id": selection,
                            },
                            headers={**headers, "Idempotency-Key": f"proof-step-{version}"},
                        )
                        assert response.status_code == 200, response.text
                    saved = response.json()["session"]
                    resumed = http.get(
                        "/api/campaigns/setup/context", params={**query, "session_id": session_id}
                    )
                    assert resumed.status_code == 200 and resumed.json()["session"] == saved
                    agent_resume = cli("research", "prepare", "--api-base", base + "/api")
                    assert agent_resume["session"] == saved
                    assert saved["ready_for_validation"] is False
                    assert (
                        http.get(
                            "/api/campaigns/setup/context", params={"workspace_id": "unauthorized"}
                        ).status_code
                        == 403
                    )
                    process = psutil.Process(server.pid)
                    service_processes = [process, *process.children(recursive=True)]
                    rss = sum(process.memory_info().rss for process in service_processes)
            finally:
                # Windows virtualenv Python may be a launcher with a server child.
                # Stop the complete owned tree, never just the launcher process.
                try:
                    process = psutil.Process(server.pid)
                    owned = [*process.children(recursive=True), process]
                except psutil.NoSuchProcess:
                    owned = []
                for process in owned:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, remaining = psutil.wait_procs(owned, timeout=10)
                for process in remaining:
                    process.kill()
                psutil.wait_procs(remaining, timeout=5)
                server.wait(timeout=15)
        installed_bytes = sum(p.stat().st_size for p in Path(sys.prefix).rglob("*") if p.is_file())
        return {
            "schema_version": "bashgym.studio_install_check.v1",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "python_version": sys.version.split()[0],
            "dependency_scope": "installed_release_base",
            "installed_environment_bytes": installed_bytes,
            "api_process_tree_rss_bytes_after_setup": rss,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "checks": [
                "installed_cli_init_and_repeat",
                "installed_hermes_skill_integrity",
                "grouped_trace_export_without_training_extras",
                "basic_campaign_export_without_reporting_extras",
                "authenticated_optional_designer_readiness",
                "installed_http_api_and_bundled_assets",
                "single_use_httponly_pairing",
                "unauthorized_and_wrong_workspace_http",
                "persisted_setup_steps_and_resume",
                "browser_draft_resumed_by_separate_installed_agent_credential",
            ],
            "optional_packages_present": {
                name: importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "data_designer", "docx", "reportlab")
            },
            "ready_certified": False,
            "training_started": False,
            "limitations": ["OS autostart untested", "No selected assets or live training"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
