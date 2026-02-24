"""ToolRegistry — merges core, skill, memory, and awareness tools.

Provides a unified tool list for the Peony agent loop, with
self-awareness capabilities so the agent can introspect on its
own abilities.
"""

from typing import Dict, List, Optional


# ------------------------------------------------------------------
# Core gym tools
# ------------------------------------------------------------------

CORE_TOOLS: List[Dict] = [
    {
        "name": "import_traces",
        "description": "Import new Claude Code sessions from the local history directory into the trace store.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_trace_status",
        "description": "Get trace counts by tier (gold, pending, failed) and overall statistics.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "classify_pending_traces",
        "description": "Auto-classify pending traces into gold or failed tiers based on quality heuristics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, report what would be classified without making changes.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "start_training",
        "description": "Start a fine-tuning run on gold traces.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["sft", "dpo", "grpo"],
                    "description": "Training strategy to use.",
                },
                "model": {
                    "type": "string",
                    "description": "Base model identifier for fine-tuning.",
                },
            },
            "required": ["strategy", "model"],
        },
    },
    {
        "name": "get_training_status",
        "description": "Check the status of active and recent training jobs.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_shell_command",
        "description": "Execute a shell command. Use as an escape hatch for operations not covered by other tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this command needs to be run (logged for audit).",
                },
            },
            "required": ["command", "reason"],
        },
    },
]


# ------------------------------------------------------------------
# Memory tools
# ------------------------------------------------------------------

MEMORY_TOOLS: List[Dict] = [
    {
        "name": "remember_fact",
        "description": "Remember a fact in persistent memory, categorized for later recall.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["project", "preference", "context", "technical"],
                    "description": "Category for the fact.",
                },
                "content": {
                    "type": "string",
                    "description": "The fact content to remember.",
                },
            },
            "required": ["category", "content"],
        },
    },
    {
        "name": "recall_facts",
        "description": "Search persistent memory for previously remembered facts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (project, preference, context, technical).",
                },
                "keyword": {
                    "type": "string",
                    "description": "Filter by keyword (case-insensitive substring match).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "forget_fact",
        "description": "Remove a previously remembered fact from persistent memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fact_id": {
                    "type": "string",
                    "description": "The unique identifier of the fact to remove.",
                },
            },
            "required": ["fact_id"],
        },
    },
    {
        "name": "update_user_profile",
        "description": "Update a field in the persistent user profile.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": [
                        "hf_username",
                        "preferred_base_model",
                        "preferred_strategy",
                        "projects",
                        "notes",
                    ],
                    "description": "Profile field to update.",
                },
                "value": {
                    "type": "string",
                    "description": "New value for the field.",
                },
            },
            "required": ["field", "value"],
        },
    },
]


# ------------------------------------------------------------------
# Awareness tools
# ------------------------------------------------------------------

AWARENESS_TOOLS: List[Dict] = [
    {
        "name": "list_my_capabilities",
        "description": "List all available tools and skills, optionally filtered by category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter capabilities by category name (case-insensitive match against tool names and descriptions).",
                },
            },
            "required": [],
        },
    },
]


# ------------------------------------------------------------------
# ToolRegistry
# ------------------------------------------------------------------

# Category labels for capabilities_summary, in display order.
_CATEGORY_MAP = {
    "Core Gym": CORE_TOOLS,
    "HuggingFace CLI": None,  # populated from skill tools at runtime
    "HuggingFace Datasets": None,
    "HuggingFace Training": None,
    "HuggingFace Evaluation": None,
    "HuggingFace Inference": None,
    "HuggingFace Tracking": None,
    "HuggingFace Papers": None,
    "HuggingFace Tools": None,
    "Memory": MEMORY_TOOLS,
    "System": AWARENESS_TOOLS,
}


class ToolRegistry:
    """Merges core, skill, memory, and awareness tools into a dynamic tool list.

    Provides self-awareness capabilities so the agent can introspect on
    its own tool set at runtime.
    """

    def build_tools(
        self, skill_tools: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Build the unified tool list.

        Returns ``CORE_TOOLS`` + *skill_tools* (deduplicated) +
        ``MEMORY_TOOLS`` + ``AWARENESS_TOOLS``.  Deduplication is by
        ``name``; the first occurrence wins.
        """
        merged: List[Dict] = []
        seen: set[str] = set()

        for tool in CORE_TOOLS:
            if tool["name"] not in seen:
                merged.append(tool)
                seen.add(tool["name"])

        if skill_tools:
            for tool in skill_tools:
                if tool["name"] not in seen:
                    merged.append(tool)
                    seen.add(tool["name"])

        for tool in MEMORY_TOOLS:
            if tool["name"] not in seen:
                merged.append(tool)
                seen.add(tool["name"])

        for tool in AWARENESS_TOOLS:
            if tool["name"] not in seen:
                merged.append(tool)
                seen.add(tool["name"])

        return merged

    def capabilities_summary(self) -> str:
        """Return a formatted summary of all tool categories.

        Used to inject self-awareness into the agent's system prompt.
        """
        lines: List[str] = ["--- YOUR CAPABILITIES ---"]
        for category in _CATEGORY_MAP:
            lines.append(f"- {category}")
        lines.append("--- END CAPABILITIES ---")
        return "\n".join(lines)

    def list_capabilities(self, category: Optional[str] = None) -> str:
        """Return a formatted listing of tools.

        If *category* is provided, only tools whose name or description
        contains the category string (case-insensitive) are included.
        """
        all_tools = self.build_tools()

        if category is not None:
            cat_lower = category.lower()
            all_tools = [
                t
                for t in all_tools
                if cat_lower in t["name"].lower()
                or cat_lower in t["description"].lower()
            ]

        if not all_tools:
            return "No matching capabilities found."

        lines: List[str] = []
        for t in all_tools:
            lines.append(f"- **{t['name']}**: {t['description']}")
        return "\n".join(lines)
