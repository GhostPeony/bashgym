"""Clean-process checks for the lightweight public package surface."""

from __future__ import annotations

import subprocess
import sys


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_importing_package_does_not_load_optional_runtime_stacks():
    result = _run_python(
        "import sys, bashgym; "
        "assert bashgym.__version__ == '0.2.0'; "
        "assert 'bashgym.factory' not in sys.modules; "
        "assert 'bashgym.gym' not in sys.modules; "
        "assert 'data_designer' not in sys.modules; "
        "assert 'docker' not in sys.modules"
    )

    assert result.returncode == 0, result.stderr


def test_importing_cli_does_not_load_data_designer_or_docker():
    result = _run_python(
        "import sys, bashgym.cli; "
        "assert 'data_designer' not in sys.modules; "
        "assert 'docker' not in sys.modules"
    )

    assert result.returncode == 0, result.stderr


def test_legacy_convenience_exports_still_resolve_lazily():
    result = _run_python(
        "from bashgym import BashGymEnv, Settings, Trainer, Verifier; "
        "assert all(value is not None for value in (BashGymEnv, Settings, Trainer, Verifier))"
    )

    assert result.returncode == 0, result.stderr
