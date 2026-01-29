"""
Claude Code Adapter

Installs hooks for Claude Code trace capture.
Uses Claude Code's hook system (CLAUDE_HOOK_EVENT).
"""

import os
import json
import shutil
import platform
from pathlib import Path
from typing import Tuple, Dict, Any


# Map of hook files to their Claude Code settings key (PascalCase)
HOOK_CONFIG = {
    "pre_tool_use.py": "PreToolUse",
    "post_tool_use.py": "PostToolUse",
    "stop.py": "Stop",
    "notification.py": "Notification",
}


def _get_claude_dir() -> Path:
    """Get the Claude Code config directory."""
    if platform.system() == 'Windows':
        home = Path(os.environ.get("USERPROFILE", ""))
    else:
        home = Path.home()
    return home / ".claude"


def _get_hooks_dir() -> Path:
    """Get the Claude Code hooks directory."""
    return _get_claude_dir() / "hooks"


def _get_settings_path() -> Path:
    """Get the Claude Code settings.json path."""
    return _get_claude_dir() / "settings.json"


def _get_source_hooks_dir() -> Path:
    """Get the source hooks directory from bashgym package."""
    # Navigate from this file to the hooks directory
    current_dir = Path(__file__).parent.parent.parent
    return current_dir / "hooks"


def _update_settings(hooks_dir: Path) -> Tuple[bool, str]:
    """
    Update Claude Code settings.json with hook configuration.

    Uses the new Claude Code hook format with matcher and hooks arrays.

    Returns:
        Tuple of (success, message)
    """
    settings_path = _get_settings_path()

    # Load existing settings or create new
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        except (json.JSONDecodeError, IOError):
            settings = {}
    else:
        settings = {}

    # Ensure hooks section exists
    if "hooks" not in settings:
        settings["hooks"] = {}

    # Add each hook configuration using new format
    for hook_file, settings_key in HOOK_CONFIG.items():
        hook_path = hooks_dir / hook_file
        if hook_path.exists():
            # Use the path as-is - json.dump handles escaping automatically
            path_str = str(hook_path)

            # New Claude Code hook format with matcher and hooks arrays
            settings["hooks"][settings_key] = [
                {
                    "matcher": "",  # Empty string matcher = match all
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python {path_str}"
                        }
                    ]
                }
            ]

    # Write updated settings
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        return True, "Updated settings.json"
    except (IOError, OSError) as e:
        return False, f"Failed to update settings.json: {e}"


def install_claude_code_hooks() -> Tuple[bool, str]:
    """
    Install Bash Gym hooks for Claude Code.

    Returns:
        Tuple of (success, message)
    """
    hooks_dir = _get_hooks_dir()
    source_dir = _get_source_hooks_dir()

    # Check if source hooks exist
    if not source_dir.exists():
        return False, f"Source hooks not found at {source_dir}"

    # Create hooks directory if needed
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Files to install (all valid Claude Code hooks)
    hook_files = list(HOOK_CONFIG.keys())

    installed = []
    for hook_file in hook_files:
        source = source_dir / hook_file
        dest = hooks_dir / hook_file

        if not source.exists():
            # Skip missing hooks (not all may be implemented)
            continue

        try:
            shutil.copy2(source, dest)
            # Make executable on Unix
            if platform.system() != 'Windows':
                os.chmod(dest, 0o755)
            installed.append(hook_file)
        except (IOError, OSError) as e:
            return False, f"Failed to install {hook_file}: {e}"

    if not installed:
        return False, "No hook files found to install"

    # Update settings.json with hook configuration
    success, msg = _update_settings(hooks_dir)
    if not success:
        return False, msg

    return True, f"Installed Claude Code hooks: {', '.join(installed)}"


def uninstall_claude_code_hooks() -> Tuple[bool, str]:
    """
    Uninstall Bash Gym hooks from Claude Code.

    Returns:
        Tuple of (success, message)
    """
    hooks_dir = _get_hooks_dir()
    settings_path = _get_settings_path()

    removed = []

    # Remove hook files
    if hooks_dir.exists():
        for hook_file in HOOK_CONFIG.keys():
            hook_path = hooks_dir / hook_file
            if hook_path.exists():
                try:
                    hook_path.unlink()
                    removed.append(hook_file)
                except (IOError, OSError) as e:
                    return False, f"Failed to remove {hook_file}: {e}"

    # Remove hook configuration from settings.json
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)

            if "hooks" in settings:
                for settings_key in HOOK_CONFIG.values():
                    settings["hooks"].pop(settings_key, None)

                # Remove hooks section if empty
                if not settings["hooks"]:
                    del settings["hooks"]

                with open(settings_path, 'w') as f:
                    json.dump(settings, f, indent=2)

        except (json.JSONDecodeError, IOError) as e:
            return False, f"Failed to update settings.json: {e}"

    if removed:
        return True, f"Removed Claude Code hooks: {', '.join(removed)}"
    return True, "No hooks to remove"


def get_install_command() -> str:
    """Get the manual install command for Claude Code hooks."""
    if platform.system() == 'Windows':
        return 'xcopy /E /I bashgym\\hooks %USERPROFILE%\\.claude\\hooks'
    else:
        return 'cp -r bashgym/hooks/* ~/.claude/hooks/'
