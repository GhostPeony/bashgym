"""
Tests for AI coding tool adapter detection and install/uninstall.

Tests cover:
  - detect_gemini_cli() directory and hook detection
  - detect_codex() directory and command detection
  - detect_copilot_cli() directory and hooks config detection
  - install/uninstall for gemini_cli adapter
"""

import json
import os
import platform
from pathlib import Path
from unittest.mock import patch

import pytest

from bashgym.trace_capture.detector import (
    ToolInfo,
    detect_gemini_cli,
    detect_codex,
    detect_copilot_cli,
    _get_home_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_home(tmp_path, monkeypatch):
    """Patch _get_home_dir and environment to use tmp_path as home."""
    monkeypatch.setattr(
        "bashgym.trace_capture.detector._get_home_dir",
        lambda: tmp_path,
    )
    # Also patch USERPROFILE for Windows-specific code paths
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


# ---------------------------------------------------------------------------
# 1. detect_gemini_cli()
# ---------------------------------------------------------------------------

class TestDetectGeminiCli:
    """Detection of Gemini CLI installation and hooks."""

    def test_not_installed_when_dir_missing(self, tmp_path, monkeypatch):
        """When ~/.gemini/ does not exist and gemini is not on PATH."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)

        info = detect_gemini_cli()
        assert info.installed is False
        assert info.hooks_installed is False

    def test_installed_when_dir_exists(self, tmp_path, monkeypatch):
        """When ~/.gemini/ exists, treat as installed even without binary."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)
        (tmp_path / ".gemini").mkdir()

        info = detect_gemini_cli()
        assert info.installed is True
        assert info.name == "Gemini CLI"
        assert info.adapter_type == "gemini_cli"

    def test_installed_when_command_found(self, tmp_path, monkeypatch):
        """When gemini binary is on PATH."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "bashgym.trace_capture.detector.shutil.which",
            lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None,
        )

        info = detect_gemini_cli()
        assert info.installed is True

    def test_hooks_installed_when_settings_has_bashgym(self, tmp_path, monkeypatch):
        """When settings.json has bashgym AfterTool entry."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)

        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        settings = {
            "hooks": {
                "AfterTool": [
                    {"name": "bashgym", "matcher": "", "hooks": [], "timeout": 5000}
                ]
            }
        }
        (gemini_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")

        info = detect_gemini_cli()
        assert info.installed is True
        assert info.hooks_installed is True

    def test_hooks_not_installed_with_empty_settings(self, tmp_path, monkeypatch):
        """When settings.json exists but has no bashgym hooks."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)

        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        (gemini_dir / "settings.json").write_text("{}", encoding="utf-8")

        info = detect_gemini_cli()
        assert info.installed is True
        assert info.hooks_installed is False

    def test_hooks_not_installed_with_other_hooks(self, tmp_path, monkeypatch):
        """When settings.json has hooks but not bashgym ones."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)

        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        settings = {
            "hooks": {
                "AfterTool": [
                    {"name": "other-tool", "matcher": "", "hooks": [], "timeout": 5000}
                ]
            }
        }
        (gemini_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")

        info = detect_gemini_cli()
        assert info.hooks_installed is False


# ---------------------------------------------------------------------------
# 2. detect_codex()
# ---------------------------------------------------------------------------

class TestDetectCodex:
    """Detection of Codex installation."""

    def test_not_installed_when_nothing_exists(self, tmp_path, monkeypatch):
        """When codex command not found and ~/.codex/ doesn't exist."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)

        info = detect_codex()
        assert info.installed is False
        assert info.hooks_installed is False

    def test_installed_when_dir_exists(self, tmp_path, monkeypatch):
        """When ~/.codex/ exists."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)
        (tmp_path / ".codex").mkdir()

        info = detect_codex()
        assert info.installed is True
        assert info.name == "Codex"
        assert info.adapter_type == "codex"

    def test_installed_when_command_found(self, tmp_path, monkeypatch):
        """When codex binary is on PATH."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "bashgym.trace_capture.detector.shutil.which",
            lambda cmd: "/usr/bin/codex" if cmd == "codex" else None,
        )

        info = detect_codex()
        assert info.installed is True

    def test_hooks_always_false(self, tmp_path, monkeypatch):
        """Codex has no live hooks, so hooks_installed is always False."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "bashgym.trace_capture.detector.shutil.which",
            lambda cmd: "/usr/bin/codex" if cmd == "codex" else None,
        )
        (tmp_path / ".codex").mkdir()

        info = detect_codex()
        assert info.hooks_installed is False


# ---------------------------------------------------------------------------
# 3. detect_copilot_cli()
# ---------------------------------------------------------------------------

class TestDetectCopilotCli:
    """Detection of GitHub Copilot CLI installation and hooks."""

    def test_not_installed_when_nothing_exists(self, tmp_path, monkeypatch):
        """When copilot command not found and ~/.copilot/ doesn't exist."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)

        info = detect_copilot_cli()
        assert info.installed is False
        assert info.hooks_installed is False

    def test_installed_when_dir_exists(self, tmp_path, monkeypatch):
        """When ~/.copilot/ exists."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)
        (tmp_path / ".copilot").mkdir()

        info = detect_copilot_cli()
        assert info.installed is True
        assert info.name == "Copilot CLI"
        assert info.adapter_type == "copilot_cli"

    def test_hooks_installed_when_config_exists(self, tmp_path, monkeypatch):
        """When bashgym-hooks.json exists in ~/.copilot/hooks/."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)

        copilot_dir = tmp_path / ".copilot"
        hooks_dir = copilot_dir / "hooks"
        hooks_dir.mkdir(parents=True)
        (hooks_dir / "bashgym-hooks.json").write_text("{}", encoding="utf-8")

        info = detect_copilot_cli()
        assert info.installed is True
        assert info.hooks_installed is True

    def test_hooks_not_installed_without_config(self, tmp_path, monkeypatch):
        """When ~/.copilot/hooks/ exists but no bashgym config."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr("bashgym.trace_capture.detector.shutil.which", lambda _: None)

        copilot_dir = tmp_path / ".copilot"
        hooks_dir = copilot_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        info = detect_copilot_cli()
        assert info.installed is True
        assert info.hooks_installed is False

    def test_installed_via_copilot_command(self, tmp_path, monkeypatch):
        """When 'copilot' command is found on PATH."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "bashgym.trace_capture.detector.shutil.which",
            lambda cmd: "/usr/bin/copilot" if cmd == "copilot" else None,
        )

        info = detect_copilot_cli()
        assert info.installed is True

    def test_installed_via_github_copilot_cli_command(self, tmp_path, monkeypatch):
        """When 'github-copilot-cli' command is found on PATH."""
        _patch_home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "bashgym.trace_capture.detector.shutil.which",
            lambda cmd: "/usr/bin/github-copilot-cli" if cmd == "github-copilot-cli" else None,
        )

        info = detect_copilot_cli()
        assert info.installed is True


# ---------------------------------------------------------------------------
# 4. Gemini CLI install/uninstall
# ---------------------------------------------------------------------------

class TestGeminiCliInstallUninstall:
    """Test install and uninstall of Gemini CLI hooks via adapter."""

    def test_install_creates_settings_json(self, tmp_path, monkeypatch):
        """install_gemini_cli_hooks creates settings.json with correct structure."""
        from bashgym.trace_capture.adapters.gemini_cli import (
            install_gemini_cli_hooks,
            _get_gemini_dir,
            _get_settings_path,
            HOOK_CONFIG,
        )

        # Patch home dir functions
        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_gemini_dir",
            lambda: tmp_path / ".gemini",
        )
        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_hooks_dir",
            lambda: tmp_path / ".gemini" / "hooks",
        )
        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_settings_path",
            lambda: tmp_path / ".gemini" / "settings.json",
        )

        # Create fake source hooks directory with hook files
        source_dir = tmp_path / "source_hooks"
        source_dir.mkdir()
        for hook_file in HOOK_CONFIG.keys():
            (source_dir / hook_file).write_text("# hook script", encoding="utf-8")

        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_source_hooks_dir",
            lambda: source_dir,
        )

        success, msg = install_gemini_cli_hooks()
        assert success is True

        # Verify settings.json was created
        settings_path = tmp_path / ".gemini" / "settings.json"
        assert settings_path.exists()

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "hooks" in settings

        # Should have entries for each hook config
        for settings_key in HOOK_CONFIG.values():
            assert settings_key in settings["hooks"]
            entries = settings["hooks"][settings_key]
            assert len(entries) == 1
            assert entries[0]["name"] == "bashgym"
            assert entries[0]["timeout"] == 5000

    def test_uninstall_removes_bashgym_preserves_others(self, tmp_path, monkeypatch):
        """uninstall_gemini_cli_hooks removes bashgym entries but preserves others."""
        from bashgym.trace_capture.adapters.gemini_cli import (
            uninstall_gemini_cli_hooks,
            HOOK_CONFIG,
        )

        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_gemini_dir",
            lambda: tmp_path / ".gemini",
        )
        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_hooks_dir",
            lambda: tmp_path / ".gemini" / "hooks",
        )
        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_settings_path",
            lambda: tmp_path / ".gemini" / "settings.json",
        )

        # Create settings with bashgym + other entries
        gemini_dir = tmp_path / ".gemini"
        hooks_dir = gemini_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        settings = {
            "hooks": {
                "AfterTool": [
                    {"name": "bashgym", "matcher": "", "hooks": [], "timeout": 5000},
                    {"name": "other-tool", "matcher": "", "hooks": [], "timeout": 3000},
                ],
                "SessionEnd": [
                    {"name": "bashgym", "matcher": "", "hooks": [], "timeout": 5000},
                ],
            },
            "other_setting": "preserved",
        }
        settings_path = gemini_dir / "settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        # Create hook files to remove
        for hook_file in HOOK_CONFIG.keys():
            (hooks_dir / hook_file).write_text("# hook", encoding="utf-8")

        success, msg = uninstall_gemini_cli_hooks()
        assert success is True

        # Verify settings.json still exists with other entries preserved
        updated = json.loads(settings_path.read_text(encoding="utf-8"))
        assert updated["other_setting"] == "preserved"

        # bashgym entries should be gone
        after_tool = updated["hooks"]["AfterTool"]
        assert len(after_tool) == 1
        assert after_tool[0]["name"] == "other-tool"

        # SessionEnd key should be removed entirely (no entries left)
        assert "SessionEnd" not in updated["hooks"]

    def test_uninstall_with_no_hooks(self, tmp_path, monkeypatch):
        """uninstall_gemini_cli_hooks is safe when no hooks exist."""
        from bashgym.trace_capture.adapters.gemini_cli import uninstall_gemini_cli_hooks

        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_gemini_dir",
            lambda: tmp_path / ".gemini",
        )
        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_hooks_dir",
            lambda: tmp_path / ".gemini" / "hooks",
        )
        monkeypatch.setattr(
            "bashgym.trace_capture.adapters.gemini_cli._get_settings_path",
            lambda: tmp_path / ".gemini" / "settings.json",
        )

        success, msg = uninstall_gemini_cli_hooks()
        assert success is True
        assert "No hooks to remove" in msg
