"""
AI Coding Tool Detector

Auto-detects which AI coding assistants are installed on the system.
"""

import os
import shutil
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ToolInfo:
    """Information about a detected AI coding tool."""
    name: str
    installed: bool
    version: Optional[str] = None
    hooks_installed: bool = False
    hooks_path: Optional[Path] = None
    config_path: Optional[Path] = None
    adapter_type: str = "unknown"  # claude_code, opencode, aider, generic


def _get_home_dir() -> Path:
    """Get the user's home directory."""
    if platform.system() == 'Windows':
        return Path(os.environ.get("USERPROFILE", ""))
    return Path.home()


def detect_claude_code() -> ToolInfo:
    """Detect Claude Code installation."""
    home = _get_home_dir()
    hooks_path = home / ".claude" / "hooks"
    config_path = home / ".claude"

    # Check if claude command exists
    claude_cmd = shutil.which("claude")
    installed = claude_cmd is not None

    # Check for config directory as fallback
    if not installed and config_path.exists():
        installed = True

    # Check if our hooks are installed
    hooks_installed = False
    if hooks_path.exists():
        post_tool_hook = hooks_path / "post_tool_use.py"
        session_end_hook = hooks_path / "session_end.py"
        hooks_installed = post_tool_hook.exists() and session_end_hook.exists()

    return ToolInfo(
        name="Claude Code",
        installed=installed,
        hooks_installed=hooks_installed,
        hooks_path=hooks_path,
        config_path=config_path,
        adapter_type="claude_code"
    )


def detect_opencode() -> ToolInfo:
    """Detect OpenCode installation."""
    home = _get_home_dir()

    # OpenCode config locations - check multiple possible paths
    if platform.system() == 'Windows':
        # Windows: check both .config and AppData
        possible_configs = [
            home / ".config" / "opencode",
            Path(os.environ.get("APPDATA", "")) / "opencode",
            Path(os.environ.get("LOCALAPPDATA", "")) / "opencode",
        ]
    else:
        possible_configs = [
            home / ".config" / "opencode",
        ]

    # Find first existing config path
    config_path = None
    for path in possible_configs:
        if path and path.exists():
            config_path = path
            break

    # Default to first option if none exist
    if config_path is None:
        config_path = possible_configs[0] if possible_configs else home / ".config" / "opencode"

    plugins_path = config_path / "plugins"

    # Check if opencode command exists
    opencode_cmd = shutil.which("opencode")
    installed = opencode_cmd is not None

    # Also check for npx/npm installed opencode
    if not installed:
        npx_cmd = shutil.which("npx")
        if npx_cmd:
            # npx opencode might be available
            installed = True  # Assume available if npx exists

    # Check for config directory as fallback
    if not installed and config_path.exists():
        installed = True

    # Check if our plugin is installed
    hooks_installed = False
    if plugins_path.exists():
        bashgym_plugin = plugins_path / "bashgym-trace.ts"
        hooks_installed = bashgym_plugin.exists()

    return ToolInfo(
        name="OpenCode",
        installed=installed,
        hooks_installed=hooks_installed,
        hooks_path=plugins_path,
        config_path=config_path,
        adapter_type="opencode"
    )


def detect_aider() -> ToolInfo:
    """Detect Aider installation."""
    home = _get_home_dir()
    config_path = home / ".aider"

    # Check if aider command exists
    aider_cmd = shutil.which("aider")
    installed = aider_cmd is not None

    return ToolInfo(
        name="Aider",
        installed=installed,
        hooks_installed=False,  # Not yet supported
        hooks_path=None,
        config_path=config_path,
        adapter_type="aider"
    )


def detect_continue() -> ToolInfo:
    """Detect Continue.dev installation."""
    home = _get_home_dir()

    # Continue stores config in ~/.continue
    config_path = home / ".continue"
    installed = config_path.exists()

    return ToolInfo(
        name="Continue",
        installed=installed,
        hooks_installed=False,  # Not yet supported
        hooks_path=None,
        config_path=config_path,
        adapter_type="continue"
    )


def detect_cursor() -> ToolInfo:
    """Detect Cursor installation."""
    home = _get_home_dir()

    # Cursor is an app, check for config
    if platform.system() == 'Windows':
        config_path = home / "AppData" / "Roaming" / "Cursor"
    elif platform.system() == 'Darwin':
        config_path = home / "Library" / "Application Support" / "Cursor"
    else:
        config_path = home / ".config" / "Cursor"

    installed = config_path.exists()

    return ToolInfo(
        name="Cursor",
        installed=installed,
        hooks_installed=False,  # Not yet supported
        hooks_path=None,
        config_path=config_path,
        adapter_type="cursor"
    )


def detect_tools() -> List[ToolInfo]:
    """
    Detect all supported AI coding tools.

    Returns a list of ToolInfo for each detected tool.
    """
    detectors = [
        detect_claude_code,
        detect_opencode,
        detect_aider,
        detect_continue,
        detect_cursor,
    ]

    tools = []
    for detector in detectors:
        try:
            tool = detector()
            tools.append(tool)
        except Exception as e:
            print(f"Warning: Error detecting tool: {e}")

    return tools


def get_tool_status() -> Dict[str, Any]:
    """
    Get a summary of all detected tools and their status.

    Returns a dict suitable for API responses.
    """
    tools = detect_tools()

    installed_tools = [t for t in tools if t.installed]
    configured_tools = [t for t in tools if t.hooks_installed]

    return {
        "tools": [
            {
                "name": t.name,
                "installed": t.installed,
                "hooks_installed": t.hooks_installed,
                "adapter_type": t.adapter_type,
                "hooks_path": str(t.hooks_path) if t.hooks_path else None,
            }
            for t in tools
        ],
        "summary": {
            "installed_count": len(installed_tools),
            "configured_count": len(configured_tools),
            "installed_names": [t.name for t in installed_tools],
            "configured_names": [t.name for t in configured_tools],
        }
    }


def get_primary_tool() -> Optional[ToolInfo]:
    """
    Get the primary (preferred) AI coding tool.

    Priority: Claude Code > OpenCode > Aider > others
    """
    tools = detect_tools()

    # Priority order
    priority = ["claude_code", "opencode", "aider", "continue", "cursor"]

    for adapter_type in priority:
        for tool in tools:
            if tool.adapter_type == adapter_type and tool.installed:
                return tool

    return None
