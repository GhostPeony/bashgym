"""
Trace Importers

Import traces from various sources into BashGym format.
"""

from .chatgpt import (
    ChatGPTImporter,
    import_chatgpt_sessions,
)
from .claude_history import (
    ClaudeSessionImporter,
    import_recent,
    import_session,
    import_today,
)
from .copilot_history import (
    CopilotSessionImporter,
    import_copilot_sessions,
)
from .gemini_history import (
    GeminiSessionImporter,
    import_gemini_sessions,
)
from .mcp_logs import (
    MCPLogImporter,
    import_mcp_logs,
)
from .opencode_history import (
    OpenCodeSessionImporter,
    import_opencode_sessions,
)

__all__ = [
    # Claude Code
    "import_today",
    "import_recent",
    "import_session",
    "ClaudeSessionImporter",
    # Gemini CLI
    "import_gemini_sessions",
    "GeminiSessionImporter",
    # Copilot CLI
    "import_copilot_sessions",
    "CopilotSessionImporter",
    # OpenCode
    "import_opencode_sessions",
    "OpenCodeSessionImporter",
    # ChatGPT
    "import_chatgpt_sessions",
    "ChatGPTImporter",
    # MCP Tool Logs
    "import_mcp_logs",
    "MCPLogImporter",
]
