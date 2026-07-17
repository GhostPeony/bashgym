from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import bashgym.operator_skills as operator_skills
from bashgym.cli import build_parser, main

SOURCE_SKILLS = Path(__file__).parents[2] / "assistant" / "workspace" / "skills"
ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def installed_wheel_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("operator-skill-install-wheel")
    wheel_root = root / "wheel"
    build = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--python",
            sys.executable,
            "--out-dir",
            str(wheel_root),
            str(ROOT),
        ],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, (build.stdout, build.stderr)
    wheels = list(wheel_root.glob("bashgym-*.whl"))
    assert len(wheels) == 1
    site = root / "site"
    install = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(site),
            "--no-deps",
            str(wheels[0]),
        ],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, (install.stdout, install.stderr)
    return site


def _locked_bundle_paths() -> set[str]:
    operator_root = SOURCE_SKILLS / "bashgym-operator"
    lock = json.loads((operator_root / "bundle.lock.json").read_text(encoding="utf-8"))
    paths = {
        (operator_root / relative).resolve().relative_to(SOURCE_SKILLS.resolve()).as_posix()
        for relative in lock["files"]
    }
    paths.add("bashgym-operator/bundle.lock.json")
    return paths


def _set_host_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    host: str,
) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for variable in ("CODEX_HOME", "CLAUDE_HOME", "CLAUDE_CONFIG_DIR", "HERMES_HOME"):
        monkeypatch.delenv(variable, raising=False)
    variable = {
        "codex": "CODEX_HOME",
        "claude": "CLAUDE_HOME",
        "hermes": "HERMES_HOME",
    }[host]
    agent_home = tmp_path / f"{host}-home"
    monkeypatch.setenv(variable, str(agent_home))
    return agent_home / "skills"


def test_operator_skills_parser_exposes_install_and_check_commands():
    parser = build_parser()

    install = parser.parse_args(["operator", "skills", "install", "--host", "codex"])
    check = parser.parse_args(["operator", "skills", "check", "--host", "hermes"])

    assert (install.operator_command, install.skills_command, install.host) == (
        "skills",
        "install",
        "codex",
    )
    assert (check.operator_command, check.skills_command, check.host) == (
        "skills",
        "check",
        "hermes",
    )


@pytest.mark.parametrize("host", ["codex", "claude", "hermes"])
def test_operator_skills_install_deploys_the_full_locked_bundle_without_removing_other_skills(
    host: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    skills_root = _set_host_home(monkeypatch, tmp_path, host)
    unrelated = skills_root / "my-unrelated-skill" / "SKILL.md"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep me", encoding="utf-8")

    assert main(["operator", "skills", "install", "--host", host]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "bashgym.operator-skills-install.v1"
    assert payload["host"] == host
    assert payload["verified"] is True
    assert payload["file_count"] == len(_locked_bundle_paths())
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    installed = {
        path.relative_to(skills_root).as_posix()
        for path in skills_root.rglob("*")
        if path.is_file() and path.name != ".bashgym-skill-bundle-receipt.json"
    }
    assert installed == _locked_bundle_paths() | {"my-unrelated-skill/SKILL.md"}


def test_operator_skills_check_fails_closed_for_file_receipt_and_inventory_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    skills_root = _set_host_home(monkeypatch, tmp_path, "codex")
    assert main(["operator", "skills", "install"]) == 0
    capsys.readouterr()

    (skills_root / "training" / "SKILL.md").write_text("tampered", encoding="utf-8")
    (skills_root / "training" / "unexpected.md").write_text("extra", encoding="utf-8")
    receipt_path = skills_root / operator_skills.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["bundle_id"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert main(["operator", "skills", "check"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "bashgym.operator-skills-check.v1"
    assert payload["verified"] is False
    assert payload["mismatches"] == [
        ".bashgym-skill-bundle-receipt.json",
        "training/SKILL.md",
        "training/unexpected.md",
    ]


def test_operator_skills_auto_detection_fails_closed_when_multiple_hosts_are_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_host_home(monkeypatch, tmp_path, "codex")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    with pytest.raises(ValueError, match="detection is ambiguous"):
        operator_skills.install_skills()


def test_operator_bundle_lock_covers_every_packaged_public_skill_file():
    tracked = subprocess.run(
        ["git", "ls-files", "assistant/workspace/skills"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[2],
    ).stdout.splitlines()
    expected = {
        Path(relative).relative_to("assistant/workspace/skills").as_posix() for relative in tracked
    }

    assert _locked_bundle_paths() == expected


@pytest.mark.parametrize(
    ("host", "home_variable"),
    [
        ("codex", "CODEX_HOME"),
        ("claude", "CLAUDE_HOME"),
        ("hermes", "HERMES_HOME"),
    ],
)
def test_installed_wheel_deploys_and_checks_the_full_bundle_in_each_agent_home(
    host: str,
    home_variable: str,
    installed_wheel_site: Path,
    tmp_path: Path,
):
    agent_home = tmp_path / f"{host}-home"
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    for variable in ("CODEX_HOME", "CLAUDE_HOME", "CLAUDE_CONFIG_DIR", "HERMES_HOME"):
        environment.pop(variable, None)
    environment.update(
        {
            "PYTHONPATH": str(installed_wheel_site),
            "HOME": str(tmp_path / "home"),
            "USERPROFILE": str(tmp_path / "home"),
            home_variable: str(agent_home),
        }
    )
    arbitrary_cwd = tmp_path / "arbitrary-cwd"
    arbitrary_cwd.mkdir()
    command = (
        "from bashgym.cli import main; "
        f"raise SystemExit(main(['operator','skills','{{action}}','--host','{host}']))"
    )

    install = subprocess.run(
        [sys.executable, "-c", command.format(action="install")],
        check=False,
        cwd=arbitrary_cwd,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, (install.stdout, install.stderr)
    install_payload = json.loads(install.stdout)
    assert install_payload["verified"] is True

    skills_root = agent_home / "skills"
    installed = {
        path.relative_to(skills_root).as_posix()
        for path in skills_root.rglob("*")
        if path.is_file() and path.name != operator_skills.RECEIPT_NAME
    }
    assert installed == _locked_bundle_paths()

    check = subprocess.run(
        [sys.executable, "-c", command.format(action="check")],
        check=False,
        cwd=arbitrary_cwd,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, (check.stdout, check.stderr)
    assert json.loads(check.stdout)["verified"] is True


def test_operator_skill_install_rolls_back_every_replaced_entry_on_swap_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    skills_root = _set_host_home(monkeypatch, tmp_path, "codex")
    assert operator_skills.install_skills(host="codex")["verified"] is True
    (skills_root / "bashgym" / "local-marker.txt").write_text("old", encoding="utf-8")
    before = {
        path.relative_to(skills_root).as_posix(): path.read_bytes()
        for path in skills_root.rglob("*")
        if path.is_file()
    }
    real_replace = operator_skills.os.replace

    def fail_during_factory_swap(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        if ".bashgym-skills-stage-" in source_path.as_posix() and source_path.name == "factory":
            raise OSError("simulated swap failure")
        real_replace(source, destination)

    monkeypatch.setattr(operator_skills.os, "replace", fail_during_factory_swap)

    with pytest.raises(OSError, match="simulated swap failure"):
        operator_skills.install_skills(host="codex")

    after = {
        path.relative_to(skills_root).as_posix(): path.read_bytes()
        for path in skills_root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(skills_root.glob(".bashgym-skill-backup-*"))
    assert not list(skills_root.glob(".bashgym-skills-stage-*"))
