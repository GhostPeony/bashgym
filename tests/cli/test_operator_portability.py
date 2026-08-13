from __future__ import annotations

import importlib.resources
import io
import json
import re
import shlex
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import bashgym.cli as cli
from bashgym.cli import build_parser, main


class _ApiHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, bytes]] = []

    def _respond(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        self.__class__.requests.append((self.command, self.path, body))
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


class _RedirectHandler(BaseHTTPRequestHandler):
    location = ""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
        self.send_response(302)
        self.send_header("Location", self.__class__.location)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.fixture
def api_server():
    _ApiHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_operator_context_requires_an_explicit_workspace_id(capsys: pytest.CaptureFixture[str]):
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["operator", "context"])

    assert exc_info.value.code == 2
    assert "--workspace-id" in capsys.readouterr().err


def test_operator_doctor_runs_from_an_arbitrary_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))

    assert main(["operator", "doctor"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "bashgym.operator.doctor.v1"
    assert "critical_skill_integrity" in payload["sources"]


def test_operator_loader_does_not_depend_on_namespace_resource_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        importlib.resources,
        "files",
        lambda _package: (_ for _ in ()).throw(NotADirectoryError("editable namespace")),
    )

    assert main(["operator", "doctor"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "bashgym.operator.doctor.v1"


def test_portable_api_command_normalizes_api_path_and_reads_payload_file(
    api_server: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    payload_path = tmp_path / "request.json"
    payload_path.write_text('{"workspace_id":"workspace-a"}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "api",
                "POST",
                "/api/training/start",
                "--data-file",
                str(payload_path),
                "--api-base",
                api_server,
            ]
        )
        == 0
    )

    response = json.loads(capsys.readouterr().out)
    assert response["ok"] is True
    assert len(_ApiHandler.requests) == 1
    method, path, body = _ApiHandler.requests[0]
    assert (method, path) == ("POST", "/api/training/start")
    assert json.loads(body) == {"workspace_id": "workspace-a"}


def test_portable_api_command_encodes_query_arguments_without_shell_metacharacters(
    api_server: str,
    capsys: pytest.CaptureFixture[str],
):
    assert (
        main(
            [
                "api",
                "GET",
                "/api/training/runcards/validate",
                "--query",
                "path=data/models/run-1/run_card.json",
                "--query",
                "promotion=true",
                "--api-base",
                api_server,
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert _ApiHandler.requests[-1][:2] == (
        "GET",
        "/api/training/runcards/validate?path=data%2Fmodels%2Frun-1%2Frun_card.json&promotion=true",
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["/api/health?token=raw-secret"],
        ["/api/health", "--query", "credential=raw-secret"],
        ["/api/health?api_key=raw-secret"],
        ["/api/health?api-key=raw-secret"],
        ["/api/health?bearer=raw-secret"],
        ["/api/health", "--query", "api_key=raw-secret"],
        ["/api/health", "--query", "api-key=raw-secret"],
        ["/api/health", "--query", "bearer=raw-secret"],
    ],
    ids=(
        "inline-token",
        "option-credential",
        "inline-api-key-underscore",
        "inline-api-key-hyphen",
        "inline-bearer",
        "option-api-key-underscore",
        "option-api-key-hyphen",
        "option-bearer",
    ),
)
def test_portable_api_command_rejects_secret_shaped_query_keys(
    arguments: list[str],
    api_server: str,
):
    with pytest.raises(ValueError, match="credentials and secrets"):
        main(["api", "GET", *arguments, "--api-base", api_server])

    assert _ApiHandler.requests == []


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_portable_api_command_supports_methods_used_by_public_skills(
    method: str,
    api_server: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    arguments = ["api", method, "/api/factory/config", "--api-base", api_server]
    if method == "PUT":
        payload = tmp_path / "factory-config.json"
        payload.write_text('{"provider":"local","model":"model-a"}', encoding="utf-8")
        arguments.extend(["--data-file", str(payload)])

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert _ApiHandler.requests[-1][0] == method


@pytest.mark.parametrize(
    "base",
    [
        "http://user:top-secret@127.0.0.1:8003/api",
        "http://127.0.0.1:8003/api?token=top-secret",
        "http://127.0.0.1:8003/api#top-secret",
    ],
)
def test_api_base_argument_rejects_credential_bearing_or_ambiguous_urls_without_echoing(
    base: str,
):
    with pytest.raises(ValueError) as exc_info:
        cli._workspace_api_base(cli.argparse.Namespace(api_base=base))

    assert "top-secret" not in str(exc_info.value)


def test_api_base_environment_is_normalized_without_echoing_secrets(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("BASHGYM_API_BASE", "https://operator:top-secret@example.test/api")

    with pytest.raises(ValueError) as exc_info:
        cli._workspace_api_base(cli.argparse.Namespace(api_base=None))

    assert "top-secret" not in str(exc_info.value)


def test_api_base_allows_plain_http_only_for_loopback():
    assert cli._normalize_api_base("http://localhost:8003") == "http://localhost:8003/api"
    assert cli._normalize_api_base("https://control.example.test/api") == (
        "https://control.example.test/api"
    )
    with pytest.raises(ValueError, match="loopback"):
        cli._normalize_api_base("http://control.example.test/api")


def test_api_http_errors_do_not_echo_url_or_response_secrets(
    monkeypatch: pytest.MonkeyPatch,
):
    error = urllib.error.HTTPError(
        "http://example.test/api/health?token=top-secret",
        401,
        "unauthorized",
        {},
        io.BytesIO(b'{"detail":"top-secret"}'),
    )
    monkeypatch.setattr(
        cli.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )

    with pytest.raises(RuntimeError) as exc_info:
        cli._workspace_http_json(
            cli.argparse.Namespace(api_base="https://example.test/api"),
            "/api/health",
        )

    message = str(exc_info.value)
    assert "top-secret" not in message
    assert "example.test" not in message


def test_api_requests_refuse_redirects_outside_the_configured_api_boundary(
    api_server: str,
):
    redirect = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    _RedirectHandler.location = f"{api_server.removesuffix('/api')}/not-api?token=redirect-secret"
    thread = threading.Thread(target=redirect.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="HTTP 302") as exc_info:
            cli._workspace_http_json(
                cli.argparse.Namespace(api_base=f"http://127.0.0.1:{redirect.server_port}/api"),
                "/api/health",
            )
    finally:
        redirect.shutdown()
        thread.join(timeout=5)
        redirect.server_close()

    assert _ApiHandler.requests == []
    message = str(exc_info.value)
    assert "redirect-secret" not in message
    assert "/not-api" not in message


def test_api_request_file_reader_is_single_open_and_bounded():
    class BoundedReader:
        read_sizes: list[int] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            self.read_sizes.append(size)
            return b'{"ok":true}'

    class OneOpenPath:
        opens = 0

        def open(self, mode: str):
            assert mode == "rb"
            self.opens += 1
            return BoundedReader()

    path = OneOpenPath()
    assert cli._read_api_json_file(path) == {"ok": True}
    assert path.opens == 1
    assert BoundedReader.read_sizes == [1024 * 1024 + 1]


def test_api_request_file_reader_rejects_oversize_and_invalid_utf8(tmp_path: Path):
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        cli._read_api_json_file(oversized)

    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b'{"value":"\xff"}')
    with pytest.raises(ValueError, match="UTF-8"):
        cli._read_api_json_file(invalid)


@pytest.mark.parametrize(
    "arguments",
    [
        ["operator", "context", "--workspace-id", " "],
        ["operator", "context", "--workspace-id", "workspace-a", "--project", " "],
        ["operator", "workspace", "--workspace-id", "\t"],
        ["operator", "ledger", "--workspace-id", "workspace-a", "--project-id", " "],
    ],
)
def test_operator_cli_rejects_blank_workspace_and_project_ids(arguments: list[str]):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(arguments)
    assert exc_info.value.code == 2


def test_operator_curator_runs_from_the_installed_command_and_arbitrary_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "bashgym.activity.v1",
                "kind": "evaluation",
                "workspace_id": "workspace-a",
                "entity_id": "eval-1",
                "status": "completed",
                "occurred_at": "2026-07-17T00:00:00Z",
                "summary": "Evaluation completed.",
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "activity"
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "operator",
                "curate",
                "receipt",
                str(receipt),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["receipt_count"] == 1
    assert len(list(output_root.rglob("*.md"))) == 1


def test_public_operator_preflight_docs_have_no_checkout_only_commands():
    root = Path(__file__).parents[2] / "assistant" / "workspace" / "skills"
    documents = [
        root / relative
        for relative in (
            "bashgym/SKILL.md",
            "bashgym/references/architecture-overview.md",
            "bashgym/references/eval-capabilities.md",
            "bashgym-operator/SKILL.md",
            "bashgym-operator/references/operator-contract.md",
            "factory/SKILL.md",
            "models/SKILL.md",
            "system/SKILL.md",
            "traces/SKILL.md",
            "training/SKILL.md",
            "training/references/bashgym-launch-recipes.md",
            "training/references/bashgym-methods-and-evals.md",
            "training/references/compute-target-activation.md",
        )
    ]

    for document in documents:
        content = document.read_text(encoding="utf-8")
        assert "scripts/api.sh" not in content, document
        assert "scripts/operator_context.py" not in content, document
        assert "scripts/curate_activity.py" not in content, document
        assert "scripts/gbrain_bridge.py" not in content, document
        assert "BASHGYM_API_BASE_URL" not in content, document
        assert re.search(r"(?m)^\s*scripts/api\.sh\b", content) is None, document
        assert re.search(r"(?m)^\s*bashgym api \w+ /(?!api(?:/|\s))", content) is None, document
        assert re.search(r"(?m)^\s*bashgym api .*(?:'\{|\"\{)", content) is None, document
        for line in content.splitlines():
            if line.startswith("bashgym api "):
                build_parser().parse_args(shlex.split(line)[1:])


def test_every_documented_training_plan_example_runs(capsys: pytest.CaptureFixture[str]):
    skill = (
        Path(__file__).parents[2] / "assistant" / "workspace" / "skills" / "training" / "SKILL.md"
    ).read_text(encoding="utf-8")
    commands = [line for line in skill.splitlines() if line.startswith("bashgym training plan ")]
    assert commands

    for command in commands:
        assert main(shlex.split(command)[1:]) == 0, command
        assert json.loads(capsys.readouterr().out)["ok"] is True
