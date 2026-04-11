"""Scoring constants for the HF dataset research agent.

Tune the scorer here, not in scoring.py. The weights must sum to 1.0.

Rationale for the weight distribution (from WORKSTREAMS.md):
  task match: 30%  — strongest signal; a dataset tagged 'code-generation' is
                     far more likely to help than one that merely mentions code.
  schema:     20%  — the second strongest signal: a dataset we can load into
                     an existing bashgym contract without column surgery is
                     immediately usable.
  license:    15%  — hard-fail below a threshold (see BLOCKED_LICENSE_PREFIXES),
                     weighted as a positive bonus above.
  size:       15%  — we want datasets in the sweet spot for our compute budget.
  freshness:  10%  — recent updates correlate with maintenance and data hygiene.
  popularity: 10%  — weak quality signal; HF downloads loosely track usefulness.
"""

# Weights must sum to 1.0
WEIGHTS = {
    "task_match": 0.30,
    "license": 0.15,
    "size": 0.15,
    "schema": 0.20,
    "freshness": 0.10,
    "popularity": 0.10,
}

# Task-match tag bonuses. Keys are lowercase tag tokens (after stripping any
# "task_categories:" prefix). Value is the 0-1 credit awarded for a direct match.
TASK_TAGS = {
    "code-generation": 1.0,
    "code": 0.9,
    "agentic": 1.0,
    "tool-use": 1.0,
    "bash": 1.0,
    "shell": 0.9,
    "swe": 1.0,
    "software-engineering": 1.0,
    "coding": 0.8,
    "python": 0.6,
}

# License policy
PERMISSIVE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "cdla-permissive-2.0",
}
WARN_LICENSES = {
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "openrail",
    "openrail++",
    "bigscience-openrail-m",
}
# Any license starting with these prefixes is hard-rejected.
BLOCKED_LICENSE_PREFIXES = ("cc-by-nc", "cc-nc", "other")

# Size windows (row count)
SIZE_MIN_HARD = 10
SIZE_MAX_HARD = 1_000_000
SIZE_IDEAL_MIN = 100
SIZE_IDEAL_MAX = 50_000

# Freshness windows (days since last_modified)
FRESHNESS_FULL_DAYS = 365
FRESHNESS_STALE_DAYS = 730

# Popularity normalization ceiling — above this download count the popularity
# score saturates at 1.0. Keeps superstar datasets from dominating the ranking.
POPULARITY_SATURATION_DOWNLOADS = 10_000

# Schema-to-format mapping. Evaluated in order — first match wins.
# Each entry: (required_column_names, bashgym_format_name).
SCHEMA_PATTERNS: list[tuple[frozenset[str], str]] = [
    (frozenset(["messages"]), "sft"),
    (frozenset(["prompt", "chosen", "rejected"]), "dpo"),
    (frozenset(["prompt", "tests"]), "grpo"),
    (frozenset(["prompt", "test_cases"]), "grpo"),
    (frozenset(["prompt", "completion"]), "sft"),
    (frozenset(["prompt", "response"]), "sft"),
    (frozenset(["instruction", "output"]), "sft"),
    (frozenset(["instruction", "input", "output"]), "sft"),
    (frozenset(["question", "answer"]), "sft"),
]

# Column mapping hints for download_command generation.
# hf_col -> DataDesigner seed column name.
COLUMN_MAP_HINTS: dict[str, str] = {
    "instruction": "seed_task",
    "input": "seed_context",
    "prompt": "seed_task",
    "question": "seed_task",
    "output": "seed_response",
    "response": "seed_response",
    "completion": "seed_response",
    "answer": "seed_response",
}
