"""Tests for materializing executable environment bundles."""

import pytest

from bashgym.environments.builder import audit_environment_manifest, materialize_environment
from bashgym.environments.contracts import BuildSpec, EnvironmentSpec, FixtureSpec, VerifierSpec
from bashgym.environments.loader import load_environment


def test_materialize_environment_writes_bundle(tmp_path):
    spec = EnvironmentSpec(
        id="env_build",
        instruction="Create a hello script.",
        build=BuildSpec(base_image="python:3.11-slim"),
        verifier=VerifierSpec(command="./verify.sh", path="verify.sh"),
        files={
            "hello.py": "print('hello')\n",
            "verify.sh": "#!/usr/bin/env bash\npython hello.py | grep hello\n",
        },
    )

    result = materialize_environment(spec, tmp_path)

    assert result.path == tmp_path / "env_build"
    assert (result.path / "env.json").exists()
    assert (result.path / ".bashgym_manifest.json").exists()
    assert "env.json" in result.protected_files
    assert "verify.sh" in result.protected_files
    assert (result.path / "hello.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert load_environment(result.path).id == "env_build"


def test_materialize_environment_blocks_path_escape(tmp_path):
    spec = EnvironmentSpec(
        id="env_escape",
        instruction="Bad task",
        files={"../escape.txt": "nope"},
    )

    with pytest.raises(ValueError, match="escapes task root"):
        materialize_environment(spec, tmp_path)


def test_materialize_environment_blocks_id_escape(tmp_path):
    spec = EnvironmentSpec(
        id="../env_escape",
        instruction="Bad task",
    )

    with pytest.raises(ValueError, match="environment id escapes output directory"):
        materialize_environment(spec, tmp_path)


def test_materialize_environment_refuses_overwrite(tmp_path):
    spec = EnvironmentSpec(id="env_once", instruction="Task")
    materialize_environment(spec, tmp_path)

    with pytest.raises(FileExistsError):
        materialize_environment(spec, tmp_path)


def test_materialize_environment_manifest_detects_protected_file_tamper(tmp_path):
    spec = EnvironmentSpec(
        id="env_manifest",
        instruction="Task",
        verifier=VerifierSpec(command="./verify.sh", path="verify.sh"),
        fixtures=[FixtureSpec(path="private/answer.txt", kind="private")],
        files={
            "verify.sh": "#!/usr/bin/env bash\nexit 0\n",
            "tests/test_answer.py": "def test_answer(): assert True\n",
            "private/answer.txt": "secret\n",
        },
    )
    result = materialize_environment(spec, tmp_path)

    assert audit_environment_manifest(result.path)["ok"] is True

    (result.path / "tests" / "test_answer.py").write_text("def test_answer(): assert False\n")
    audit = audit_environment_manifest(result.path)

    assert audit["ok"] is False
    assert audit["tampered_paths"] == ["tests/test_answer.py"]
