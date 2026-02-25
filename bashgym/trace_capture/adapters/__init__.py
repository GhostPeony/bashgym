"""
AI Coding Tool Adapters

Each adapter provides tool-specific trace capture functionality.
"""

from .claude_code import install_claude_code_hooks, uninstall_claude_code_hooks
from .opencode import install_opencode_plugin, uninstall_opencode_plugin
from .gemini_cli import install_gemini_cli_hooks, uninstall_gemini_cli_hooks
from .codex import install_codex_hooks, uninstall_codex_hooks, import_codex_sessions
from .copilot_cli import install_copilot_cli_hooks, uninstall_copilot_cli_hooks

__all__ = [
    'install_claude_code_hooks',
    'uninstall_claude_code_hooks',
    'install_opencode_plugin',
    'uninstall_opencode_plugin',
    'install_gemini_cli_hooks',
    'uninstall_gemini_cli_hooks',
    'install_codex_hooks',
    'uninstall_codex_hooks',
    'import_codex_sessions',
    'install_copilot_cli_hooks',
    'uninstall_copilot_cli_hooks',
]
