from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import venv
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SKILL_PREFIX = "assistant/workspace/skills/"
PUBLIC_SKILL_FILES = frozenset(
    {
        "bashgym-operator/SKILL.md",
        "bashgym-operator/bundle.lock.json",
        "bashgym-operator/references/operator-contract.md",
        "bashgym-operator/scripts/curate_activity.py",
        "bashgym-operator/scripts/gbrain_bridge.py",
        "bashgym-operator/scripts/operator_context.py",
        "bashgym/SKILL.md",
        "bashgym/references/architecture-overview.md",
        "bashgym/references/eval-capabilities.md",
        "factory/SKILL.md",
        "models/SKILL.md",
        "system/SKILL.md",
        "traces/SKILL.md",
        "training/SKILL.md",
        "training/references/bashgym-launch-recipes.md",
        "training/references/bashgym-methods-and-evals.md",
        "training/references/compute-target-activation.md",
    }
)
PUBLIC_SKILL_NAMES = {
    "bashgym",
    "bashgym-operator",
    "factory",
    "models",
    "system",
    "traces",
    "training",
}
LEGACY_WHEEL_RESIDUE = {
    "bashgym/campaigns/campaign_agents.py": Counter({"capability.handoff_memexai_prepare": 1}),
    "bashgym/campaigns/contracts.py": Counter(
        {
            "allow_memexai_handoff": 2,
            "capability.handoff_memexai_prepare": 1,
            "handoff.memexai_prepare": 1,
            "handoff_memexai_prepare": 1,
            "memexai_query_format_ablation_manifest.v1": 1,
        }
    ),
    "bashgym/campaigns/executors.py": Counter({"memexai_youtube": 2}),
    "bashgym/campaigns/installation.py": Counter({"allow_memexai_handoff": 1}),
    "bashgym/campaigns/proposals.py": Counter({"capability.handoff_memexai_prepare": 1}),
}
LEGACY_RESIDUE_TOKEN = re.compile(
    r"[a-z0-9_.]*memexai[a-z0-9_.]*",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def public_skill_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    proof_dir = tmp_path_factory.mktemp("public-skill-wheel")
    source_dir = proof_dir / "source"
    source_dir.mkdir()
    publishable = (
        subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            check=True,
            cwd=ROOT,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    for relative in filter(None, publishable):
        source = ROOT / relative
        if source.is_file():
            destination = source_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    wheel_dir = proof_dir / "wheel"
    wheel_dir.mkdir()
    build = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--python",
            sys.executable,
            "--out-dir",
            str(wheel_dir),
            str(source_dir),
        ],
        check=False,
        cwd=wheel_dir,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, (
        "failed to build isolated BashGym wheel\n"
        f"stdout:\n{build.stdout}\n"
        f"stderr:\n{build.stderr}"
    )
    build_output = build.stdout + build.stderr
    assert "project.license as a TOML table is deprecated" not in build_output
    assert "License classifiers are deprecated" not in build_output
    wheels = list(wheel_dir.glob("bashgym-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_contains_only_the_explicit_public_skill_bundle(public_skill_wheel: Path):
    with zipfile.ZipFile(public_skill_wheel) as archive:
        packaged = {
            name.removeprefix(SKILL_PREFIX)
            for name in archive.namelist()
            if name.startswith(SKILL_PREFIX) and not name.endswith("/")
        }

    assert packaged == PUBLIC_SKILL_FILES
    assert not any("__pycache__" in name for name in packaged)
    assert not any("ponyo" in name.casefold() for name in packaged)


def test_wheel_public_skill_contents_have_no_private_project_or_machine_residue(
    public_skill_wheel: Path,
):
    private_term = re.compile(
        r"(?<![a-z0-9])(?:memexai|ponyo|gx10|cade)(?![a-z0-9])",
        re.IGNORECASE,
    )
    private_path = re.compile(
        r"(?:[a-z]:[\\/]users[\\/][^\\/\s\"']+|/(?:users|home)/(?![\"'])[^/\s\"']+)",
        re.IGNORECASE,
    )
    with zipfile.ZipFile(public_skill_wheel) as archive:
        for name in archive.namelist():
            if not name.startswith(SKILL_PREFIX) or name.endswith("/"):
                continue
            content = archive.read(name).decode("utf-8")
            searchable = f"{name}\n{content}".casefold()
            assert private_term.search(searchable) is None, name
            assert private_path.search(searchable) is None, name


def test_wheel_does_not_capture_unrelated_checkout_content(public_skill_wheel: Path):
    with zipfile.ZipFile(public_skill_wheel) as archive:
        distribution_root = next(
            name.split("/", 1)[0]
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        )
        data_root = distribution_root.removesuffix(".dist-info") + ".data"
        unexpected = [
            name
            for name in archive.namelist()
            if not name.startswith(
                (
                    "bashgym/",
                    SKILL_PREFIX,
                    f"{data_root}/data/share/bashgym/docs/training/",
                    f"{distribution_root}/",
                )
            )
        ]

    assert unexpected == []


def test_every_text_wheel_member_has_no_unreviewed_private_residue(
    public_skill_wheel: Path,
):
    private_term = re.compile(
        r"(?<![a-z0-9])(?:memexai|ponyo|gx10|cade|ghostwork)(?![a-z0-9])",
        re.IGNORECASE,
    )
    private_path = re.compile(
        r"(?:[a-z]:[\\/]users[\\/][^\\/\s\"']+|/(?:users|home)/(?![\"'])[^/\s\"']+)",
        re.IGNORECASE,
    )
    observed_legacy: dict[str, Counter[str]] = {}
    with zipfile.ZipFile(public_skill_wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            try:
                content = archive.read(name).decode("utf-8")
            except UnicodeDecodeError:
                continue
            searchable = f"{name}\n{content}"
            legacy = Counter(
                match.group(0).casefold() for match in LEGACY_RESIDUE_TOKEN.finditer(searchable)
            )
            if legacy:
                observed_legacy[name] = legacy
            scrubbed = LEGACY_RESIDUE_TOKEN.sub("", searchable)
            assert private_term.search(scrubbed) is None, name
            assert private_path.search(scrubbed) is None, name

    assert observed_legacy == LEGACY_WHEEL_RESIDUE


def test_installed_wheel_skill_loader_discovers_the_public_operator_skills(
    public_skill_wheel: Path,
    tmp_path: Path,
):
    site = tmp_path / "site"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            str(public_skill_wheel),
            "--no-deps",
            "--target",
            str(site),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(site)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from bashgym.api import agent_routes; "
                "roots, skills, warnings = agent_routes._scan_skill_roots(); "
                "print(json.dumps({"
                "'module': agent_routes.__file__, "
                "'workspace_exists': dict(agent_routes._toolkit_skill_root_candidates())"
                "['workspace'].is_dir(), "
                "'names': sorted(skill.name for skill in skills if skill.source == 'workspace'), "
                "'warnings': warnings}))"
            ),
        ],
        check=False,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "installed wheel could not load its public operator skills\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    payload = json.loads(result.stdout)

    assert Path(payload["module"]).is_relative_to(site)
    assert payload["workspace_exists"] is True
    assert set(payload["names"]) == PUBLIC_SKILL_NAMES
    assert payload["warnings"] == []


class _PortablePreflightHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str]] = []

    def _respond(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.__class__.requests.append((self.command, self.path))
        payload = json.dumps({"ok": True, "path": self.path}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
        self._respond()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
        self._respond()

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
        self._respond()

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
        self._respond()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_installed_wheel_documented_preflights_run_outside_the_checkout(
    public_skill_wheel: Path,
    tmp_path: Path,
):
    environment_dir = tmp_path / "venv"
    venv.EnvBuilder().create(environment_dir)
    scripts_dir = environment_dir / ("Scripts" if os.name == "nt" else "bin")
    python = scripts_dir / ("python.exe" if os.name == "nt" else "python")
    bashgym = scripts_dir / ("bashgym.exe" if os.name == "nt" else "bashgym")
    installed = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            str(public_skill_wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, (installed.stdout, installed.stderr)

    _PortablePreflightHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PortablePreflightHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_base = f"http://127.0.0.1:{server.server_port}/api"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "USERPROFILE": str(tmp_path / "home"),
            "BASHGYM_LEDGER_DB": str(tmp_path / "missing-ledger.sqlite3"),
        }
    )
    factory_config = tmp_path / "factory-config.json"
    factory_config.write_text('{"provider":"local","model":"model-a"}', encoding="utf-8")
    commands = [
        ["operator", "doctor"],
        ["operator", "context", "--workspace-id", "workspace-a"],
        [
            "operator",
            "workspace",
            "--workspace-id",
            "workspace-a",
            "--api-base",
            api_base,
        ],
        *[
            ["api", "GET", path, "--api-base", api_base]
            for path in (
                "/api/health",
                "/api/system/info",
                "/api/system/recommendations",
                "/api/ssh/preflight",
            )
        ],
        ["api", "GET", "/api/factory/seeds", "--api-base", api_base],
        [
            "api",
            "PUT",
            "/api/factory/config",
            "--data-file",
            str(factory_config),
            "--api-base",
            api_base,
        ],
        ["api", "GET", "/api/models", "--api-base", api_base],
        ["api", "DELETE", "/api/models/model-a", "--api-base", api_base],
        ["api", "GET", "/api/stats", "--api-base", api_base],
        [
            "api",
            "GET",
            "/api/traces",
            "--query",
            "status=gold",
            "--api-base",
            api_base,
        ],
        ["api", "POST", "/api/traces/sync", "--api-base", api_base],
    ]
    imported = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import bashgym, pathlib; print(pathlib.Path(bashgym.__file__).resolve())",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert imported.returncode == 0, (imported.stdout, imported.stderr)
    assert Path(imported.stdout.strip()).is_relative_to(environment_dir)
    try:
        for command in commands:
            result = subprocess.run(
                [str(bashgym), *command],
                cwd=tmp_path,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, (command, result.stdout, result.stderr)
            if command == ["operator", "doctor"]:
                doctor = json.loads(result.stdout)
                assert doctor["sources"]["critical_skill_integrity"]["verified"] is True
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert _PortablePreflightHandler.requests == [
        ("GET", "/api/workspace/context?workspace_id=workspace-a&format=json"),
        ("GET", "/api/health"),
        ("GET", "/api/system/info"),
        ("GET", "/api/system/recommendations"),
        ("GET", "/api/ssh/preflight"),
        ("GET", "/api/factory/seeds"),
        ("PUT", "/api/factory/config"),
        ("GET", "/api/models"),
        ("DELETE", "/api/models/model-a"),
        ("GET", "/api/stats"),
        ("GET", "/api/traces?status=gold"),
        ("POST", "/api/traces/sync"),
    ]
