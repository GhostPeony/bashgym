"""
Gemini CLI Adapter

Installs hooks for Gemini CLI trace capture.
Uses Gemini CLI's hook system (AfterTool, SessionEnd events).
"""

import json
import os
import platform
import shutil
from pathlib import Path

# Map of hook files to their Gemini CLI settings key (PascalCase)
HOOK_CONFIG = {
    "gemini_after_tool.py": "AfterTool",
    "gemini_session_end.py": "SessionEnd",
}


def _get_gemini_dir() -> Path:
    """Get the Gemini CLI config directory."""
    if platform.system() == "Windows":
        home = Path(os.environ.get("USERPROFILE", ""))
    else:
        home = Path.home()
    return home / ".gemini"


def _get_hooks_dir() -> Path:
    """Get the Gemini CLI hooks directory."""
    return _get_gemini_dir() / "hooks"


def _get_settings_path() -> Path:
    """Get the Gemini CLI settings.json path."""
    return _get_gemini_dir() / "settings.json"


def _get_source_hooks_dir() -> Path:
    """Get the source hooks directory from bashgym package."""
    # Navigate from this file to the hooks directory
    current_dir = Path(__file__).parent.parent.parent
    return current_dir / "hooks"


def _update_settings(hooks_dir: Path) -> tuple[bool, str]:
    """
    Update Gemini CLI settings.json with hook configuration.

    Uses the same format as Claude Code hooks: matcher + hooks arrays,
    with a 'name' field containing 'bashgym' and timeout of 5000ms.

    Returns:
        Tuple of (success, message)
    """
    settings_path = _get_settings_path()

    # Load existing settings or create new
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)
        except (OSError, json.JSONDecodeError):
            settings = {}
    else:
        settings = {}

    # Ensure hooks section exists
    if "hooks" not in settings:
        settings["hooks"] = {}

    # Add each hook configuration
    for hook_file, settings_key in HOOK_CONFIG.items():
        hook_path = hooks_dir / hook_file
        if hook_path.exists():
            path_str = str(hook_path)

            settings["hooks"][settings_key] = [
                {
                    "name": "bashgym",
                    "matcher": "",  # Empty string matcher = match all
                    "hooks": [{"type": "command", "command": f"python {path_str}"}],
                    "timeout": 5000,
                }
            ]

    # Write updated settings
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        return True, "Updated settings.json"
    except OSError as e:
        return False, f"Failed to update settings.json: {e}"


def install_gemini_cli_hooks() -> tuple[bool, str]:
    """
    Install Bash Gym hooks for Gemini CLI.

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

    # Files to install
    hook_files = list(HOOK_CONFIG.keys())

    installed = []
    for hook_file in hook_files:
        source = source_dir / hook_file
        dest = hooks_dir / hook_file

        if not source.exists():
            # Skip missing hooks
            continue

        try:
            shutil.copy2(source, dest)
            # Make executable on Unix
            if platform.system() != "Windows":
                os.chmod(dest, 0o755)
            installed.append(hook_file)
        except OSError as e:
            return False, f"Failed to install {hook_file}: {e}"

    if not installed:
        return False, "No hook files found to install"

    # Update settings.json with hook configuration
    success, msg = _update_settings(hooks_dir)
    if not success:
        return False, msg

    return True, f"Installed Gemini CLI hooks: {', '.join(installed)}"


def uninstall_gemini_cli_hooks() -> tuple[bool, str]:
    """
    Uninstall Bash Gym hooks from Gemini CLI.

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
                except OSError as e:
                    return False, f"Failed to remove {hook_file}: {e}"

    # Remove hook configuration from settings.json
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)

            if "hooks" in settings:
                for settings_key in HOOK_CONFIG.values():
                    if settings_key in settings["hooks"]:
                        # Remove only bashgym entries (filter by name)
                        settings["hooks"][settings_key] = [
                            entry
                            for entry in settings["hooks"][settings_key]
                            if entry.get("name") != "bashgym"
                        ]
                        # Remove the key if no entries remain
                        if not settings["hooks"][settings_key]:
                            del settings["hooks"][settings_key]

                # Remove hooks section if empty
                if not settings["hooks"]:
                    del settings["hooks"]

                with open(settings_path, "w") as f:
                    json.dump(settings, f, indent=2)

        except (OSError, json.JSONDecodeError) as e:
            return False, f"Failed to update settings.json: {e}"

    if removed:
        return True, f"Removed Gemini CLI hooks: {', '.join(removed)}"
    return True, "No hooks to remove"


def get_install_command() -> str:
    """Get the manual install command for Gemini CLI hooks."""
    if platform.system() == "Windows":
        return "xcopy /E /I bashgym\\hooks\\gemini_* %USERPROFILE%\\.gemini\\hooks"
    else:
        return "cp bashgym/hooks/gemini_* ~/.gemini/hooks/"
