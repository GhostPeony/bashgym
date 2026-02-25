"""
Trace Importers

Import traces from various sources into BashGym format.
"""

from .claude_history import (
    import_today,
    import_recent,
    import_session,
    ClaudeSessionImporter,
)
from .gemini_history import (
    import_gemini_sessions,
    GeminiSessionImporter,
)
from .copilot_history import (
    import_copilot_sessions,
    CopilotSessionImporter,
)
from .opencode_history import (
    import_opencode_sessions,
    OpenCodeSessionImporter,
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
]
