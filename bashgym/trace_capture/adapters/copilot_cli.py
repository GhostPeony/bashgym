"""
Copilot CLI Adapter

Installs hooks for GitHub Copilot CLI trace capture.
Uses a JSON config file in ~/.copilot/hooks/ that points
to our hook scripts.
"""

import os
import json
import shutil
import platform
from pathlib import Path
from typing import Tuple


# Hook config file that Copilot CLI reads
HOOKS_CONFIG_FILE = "bashgym-hooks.json"


def _get_copilot_dir() -> Path:
    """Get the Copilot CLI config directory."""
    if platform.system() == 'Windows':
        home = Path(os.environ.get("USERPROFILE", ""))
    else:
        home = Path.home()
    return home / ".copilot"


def _get_hooks_dir() -> Path:
    """Get the Copilot CLI hooks directory."""
    return _get_copilot_dir() / "hooks"


def _get_hooks_config_path() -> Path:
    """Get the path to our hooks config file."""
    return _get_hooks_dir() / HOOKS_CONFIG_FILE


def _get_source_hooks_dir() -> Path:
    """Get the source hooks directory from bashgym package."""
    current_dir = Path(__file__).parent.parent.parent
    return current_dir / "hooks"


def _build_hooks_config() -> dict:
    """
    Build the hooks configuration JSON that points to our hook scripts.

    Returns:
        Dict containing the hooks configuration.
    """
    source_dir = _get_source_hooks_dir()

    # Use the same hook scripts as Gemini CLI (they handle stdin/stdout JSON)
    after_tool_path = source_dir / "gemini_after_tool.py"
    session_end_path = source_dir / "gemini_session_end.py"

    config = {
        "name": "bashgym",
        "version": "1.0.0",
        "description": "Bash Gym trace capture hooks for Copilot CLI",
        "hooks": {
            "after_tool": {
                "command": f"python {after_tool_path}",
                "timeout": 5000
            },
            "session_end": {
                "command": f"python {session_end_path}",
                "timeout": 5000
            }
        }
    }

    return config


def install_copilot_cli_hooks() -> Tuple[bool, str]:
    """
    Install Bash Gym hooks for Copilot CLI.

    Creates ~/.copilot/hooks/bashgym-hooks.json config file
    pointing to our hook scripts.

    Returns:
        Tuple of (success, message)
    """
    hooks_dir = _get_hooks_dir()
    config_path = _get_hooks_config_path()

    # Create hooks directory if needed
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Build and write config
    config = _build_hooks_config()

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True, f"Installed Copilot CLI hooks config at {config_path}"
    except (IOError, OSError) as e:
        return False, f"Failed to install Copilot CLI hooks: {e}"


def uninstall_copilot_cli_hooks() -> Tuple[bool, str]:
    """
    Uninstall Bash Gym hooks from Copilot CLI.

    Removes the ~/.copilot/hooks/bashgym-hooks.json config file.

    Returns:
        Tuple of (success, message)
    """
    config_path = _get_hooks_config_path()

    if not config_path.exists():
        return True, "No Copilot CLI hooks to remove"

    try:
        config_path.unlink()
        return True, f"Removed Copilot CLI hooks config from {config_path}"
    except (IOError, OSError) as e:
        return False, f"Failed to remove Copilot CLI hooks: {e}"


def get_install_command() -> str:
    """Get manual install instructions for Copilot CLI hooks."""
    config_path = _get_hooks_config_path()
    return f"Create {config_path} with hook configuration"
