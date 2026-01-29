"""
AI Coding Tool Adapters

Each adapter provides tool-specific trace capture functionality.
"""

from .claude_code import install_claude_code_hooks, uninstall_claude_code_hooks
from .opencode import install_opencode_plugin, uninstall_opencode_plugin

__all__ = [
    'install_claude_code_hooks',
    'uninstall_claude_code_hooks',
    'install_opencode_plugin',
    'uninstall_opencode_plugin',
]
