"""ToolRegistry — merges core, skill, memory, and awareness tools.

Provides a unified tool list for the Peony agent loop, with
self-awareness capabilities so the agent can introspect on its
own abilities.
"""

from bashgym.agent.hf_context_tools import HF_CONTEXT_TOOLS
from bashgym.agent.skill_lab_tools import SKILL_LAB_TOOLS

# ------------------------------------------------------------------
# Core gym tools
# ------------------------------------------------------------------

CORE_TOOLS: list[dict] = [
    {
        "name": "import_traces",
        "description": "Import data from Claude Code sessions. Can import session traces, subagent conversations, file edits, plans, todos, prompts, environment data, and debug/API traffic metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "all",
                            "sessions",
                            "subagents",
                            "edits",
                            "plans",
                            "prompts",
                            "todos",
                            "environments",
                            "debug",
                        ],
                    },
                    "description": "Which data sources to import. Defaults to ['all'].",
                },
                "days": {
                    "type": "integer",
                    "description": "Only import data from the last N days. Defaults to 60.",
                },
                "project_filter": {
                    "type": "string",
                    "description": "Only import from projects matching this substring.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, scan and report what would be imported without actually importing.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "scan_claude_data",
        "description": "Scan ~/.claude and show what data is available but not yet imported. Shows counts per source type.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_collection_status",
        "description": "Show current collection stats per source type: total found, already collected, and available to collect.",
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
                    "enum": [
                        "sft",
                        "dpo",
                        "grpo",
                        "rlvr",
                        "distillation",
                        "session_distillation",
                    ],
                    "description": "Direct BashGym training strategy to use.",
                },
                "model": {
                    "type": "string",
                    "description": "Base model identifier for fine-tuning.",
                },
                "dataset_path": {
                    "type": "string",
                    "description": "Optional JSONL dataset path. Omit to use the default trace data.",
                },
                "compute_target": {
                    "type": "string",
                    "description": "Execution target label such as local, cloud, or ssh:<device>.",
                },
                "config": {
                    "type": "object",
                    "description": (
                        "Validated TrainingRequest overrides. Use this for method-specific and "
                        "storage settings; strategy, base_model, compute_target, correlation_id, "
                        "and origin must stay at the top level. The API rejects unknown fields."
                    ),
                    "properties": {
                        "num_epochs": {"type": "integer", "minimum": 1, "maximum": 100},
                        "batch_size": {"type": "integer", "minimum": 1, "maximum": 64},
                        "learning_rate": {"type": "number", "exclusiveMinimum": 0},
                        "warmup_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                        "gradient_accumulation_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 128,
                        },
                        "max_seq_length": {"type": "integer", "minimum": 1},
                        "save_steps": {"type": "integer", "minimum": 10},
                        "checkpoint_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "artifact_retention": {
                            "type": "string",
                            "enum": [
                                "adapter_only",
                                "adapter_checkpoint",
                                "deployable",
                                "full_run",
                            ],
                        },
                        "use_lora": {"type": "boolean"},
                        "lora_rank": {"type": "integer", "minimum": 1},
                        "lora_alpha": {"type": "integer", "minimum": 1},
                        "lora_dropout": {"type": "number", "minimum": 0, "maximum": 0.5},
                        "load_in_4bit": {"type": "boolean"},
                        "sft_backend": {
                            "type": "string",
                            "enum": ["auto", "unsloth", "plain"],
                        },
                        "dpo_backend": {
                            "type": "string",
                            "enum": ["auto", "unsloth", "plain"],
                        },
                        "dpo_beta": {"type": "number", "minimum": 0.01, "maximum": 1},
                        "training_profile": {"type": "string"},
                        "grpo_backend": {"type": "string"},
                        "grpo_loss_type": {"type": "string"},
                        "grpo_ratio_clip_min": {
                            "type": "number",
                            "minimum": 0,
                            "exclusiveMaximum": 1,
                        },
                        "grpo_ratio_clip_max": {"type": "number", "minimum": 0},
                        "grpo_reward_mode": {"type": "string"},
                        "grpo_group_size": {"type": "integer", "minimum": 2, "maximum": 64},
                        "grpo_num_generations": {
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 64,
                        },
                        "grpo_temperature": {
                            "type": "number",
                            "minimum": 0.1,
                            "maximum": 2,
                        },
                        "filter_zero_std_groups": {"type": "boolean"},
                        "active_sampling": {"type": "boolean"},
                        "token_level_loss": {"type": "boolean"},
                        "lm_head_fp32": {"type": "boolean"},
                        "teacher_model": {"type": "string"},
                        "teacher_temperature": {
                            "type": "number",
                            "minimum": 0.1,
                            "maximum": 10,
                        },
                        "distillation_alpha": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "session_distillation_alpha": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "session_distillation_temperature": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 10,
                        },
                        "session_distillation_min_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "session_distillation_mask_policy": {"type": "string"},
                        "session_distillation_context_mode": {"type": "string"},
                        "session_distillation_reader": {"type": "string"},
                        "auto_export_gguf": {"type": "boolean"},
                        "gguf_quantization": {"type": "string"},
                        "auto_push_hf": {"type": "boolean"},
                        "hf_repo_name": {"type": "string"},
                        "hf_private": {"type": "boolean"},
                        "hf_upload_artifact": {
                            "type": "string",
                            "enum": ["auto", "adapter", "merged"],
                        },
                        "use_remote_ssh": {"type": "boolean"},
                        "device_id": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
                "correlation_id": {
                    "type": "string",
                    "description": "Optional workflow id that links planning and training nodes.",
                },
                "tracking_context": {
                    "type": "object",
                    "description": (
                        "Exact reproducibility boundary for an official ledger run. Supply this "
                        "after verifying the project, experiment, model revision, dataset snapshot, "
                        "and compute environment. Omit only for an explicitly unassigned smoke/ad-hoc run."
                    ),
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "project_id": {"type": "string"},
                        "project_display_name": {"type": "string"},
                        "project_description": {"type": "string"},
                        "experiment_id": {"type": "string"},
                        "experiment_name": {"type": "string"},
                        "objective": {"type": "string"},
                        "task_type": {"type": "string"},
                        "model_id": {"type": "string"},
                        "model_version_id": {"type": "string"},
                        "model_source_uri": {"type": "string"},
                        "model_source_revision": {"type": "string"},
                        "model_config_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "dataset_id": {"type": "string"},
                        "dataset_version_id": {"type": "string"},
                        "dataset_source_uri": {"type": "string"},
                        "dataset_content_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "dataset_split_manifest": {"type": "object"},
                        "dataset_row_counts": {"type": "object"},
                        "environment_id": {"type": "string"},
                        "environment_runtime_digest": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "environment_hardware": {"type": "object"},
                        "campaign_id": {"type": "string"},
                        "study_id": {"type": "string"},
                        "action_id": {"type": "string"},
                        "owner_actor_id": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object"},
                    },
                    "required": [
                        "workspace_id",
                        "project_id",
                        "project_display_name",
                        "experiment_id",
                        "experiment_name",
                        "objective",
                        "task_type",
                        "model_id",
                        "model_version_id",
                        "model_source_uri",
                        "model_config_digest",
                        "dataset_id",
                        "dataset_version_id",
                        "dataset_source_uri",
                        "dataset_content_digest",
                        "environment_id",
                        "environment_runtime_digest",
                    ],
                    "additionalProperties": False,
                },
                "origin": {
                    "type": "object",
                    "description": "Source from workspace context. Include panel_id and terminal_id when available.",
                    "properties": {
                        "panel_id": {"type": "string"},
                        "terminal_id": {"type": "string"},
                        "agent": {"type": "string"},
                    },
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
        "name": "list_experiment_projects",
        "description": (
            "List project-isolated experiment namespaces from BashGym's authoritative local ledger. "
            "Use this before guessing which project a run belongs to."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Ledger workspace. Defaults to desktop-local.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_experiment_context",
        "description": (
            "Read one project's bounded status, health signals, run history, evaluations, "
            "decisions, and stable evidence references without crossing project boundaries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "project_id": {"type": "string"},
                "recent_limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["workspace_id", "project_id"],
        },
    },
    {
        "name": "get_experiment_run",
        "description": (
            "Inspect one official run with its attempt identities, evaluation results, and artifact references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "project_id": {"type": "string"},
                "run_id": {"type": "string"},
            },
            "required": ["workspace_id", "project_id", "run_id"],
        },
    },
    {
        "name": "start_data_designer",
        "description": "Start a Data Designer synthetic dataset generation job.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pipeline": {
                    "type": "string",
                    "description": "Data Designer pipeline name, such as coding_agent_sft.",
                },
                "num_records": {
                    "type": "integer",
                    "description": "Number of records to generate.",
                },
                "seed_source": {
                    "type": "string",
                    "description": "Optional seed file, directory, or Hugging Face dataset.",
                },
                "seed_type": {
                    "type": "string",
                    "enum": ["traces", "agent_rollouts", "huggingface", "file", "unstructured"],
                },
                "model": {
                    "type": "string",
                    "description": "Optional text model override.",
                },
                "provider": {
                    "type": "string",
                    "description": "Provider name. Defaults to nvidia.",
                },
                "provider_endpoint": {
                    "type": "string",
                    "description": "Optional OpenAI-compatible provider endpoint.",
                },
                "origin": {
                    "type": "object",
                    "description": "Source canvas panel or terminal from workspace context.",
                    "properties": {
                        "panel_id": {"type": "string"},
                        "terminal_id": {"type": "string"},
                        "agent": {"type": "string"},
                    },
                },
            },
            "required": ["pipeline"],
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
    *HF_CONTEXT_TOOLS,
    *SKILL_LAB_TOOLS,
]


# ------------------------------------------------------------------
# Memory tools
# ------------------------------------------------------------------

MEMORY_TOOLS: list[dict] = [
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

AWARENESS_TOOLS: list[dict] = [
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

    def build_tools(self, skill_tools: list[dict] | None = None) -> list[dict]:
        """Build the unified tool list.

        Returns ``CORE_TOOLS`` + *skill_tools* (deduplicated) +
        ``MEMORY_TOOLS`` + ``AWARENESS_TOOLS``.  Deduplication is by
        ``name``; the first occurrence wins.
        """
        merged: list[dict] = []
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
        lines: list[str] = ["--- YOUR CAPABILITIES ---"]
        for category in _CATEGORY_MAP:
            lines.append(f"- {category}")
        lines.append("--- END CAPABILITIES ---")
        return "\n".join(lines)

    def list_capabilities(self, category: str | None = None) -> str:
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
                if cat_lower in t["name"].lower() or cat_lower in t["description"].lower()
            ]

        if not all_tools:
            return "No matching capabilities found."

        lines: list[str] = []
        for t in all_tools:
            lines.append(f"- **{t['name']}**: {t['description']}")
        return "\n".join(lines)
