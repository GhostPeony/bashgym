"""The control service imports without training or report extras."""

import os
import subprocess
import sys
from pathlib import Path


def test_control_api_constructs_without_training_or_reporting_extras(tmp_path):
    source = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class WithoutExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'docx', 'reportlab', 'data_designer'}:
            raise ModuleNotFoundError('Optional extra deliberately absent: ' + fullname)
sys.meta_path.insert(0, WithoutExtras())
from bashgym.api.routes import create_app
assert create_app() is not None
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        cwd=tmp_path,
        env={
            **os.environ,
            "BASHGYM_DIR": str(tmp_path),
            "BASHGYM_MODE": "headless",
            "OLLAMA_ENABLED": "false",
        },
        capture_output=True,
        text=True,
        timeout=45,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr
