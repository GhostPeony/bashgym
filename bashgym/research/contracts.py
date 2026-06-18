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
# These are high-confidence matches — exact column names used by bashgym contracts.
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

# Heuristic fallback: if no exact SCHEMA_PATTERNS match, look for any column
# that looks prompt-like + any column that looks response-like. Lower confidence
# than SCHEMA_PATTERNS, but catches idiomatic dataset layouts (mbpp's text+code,
# SWE-bench's problem_statement+patch, humaneval-pro's raw_problem+raw_solution).
PROMPT_LIKE_COLS: frozenset[str] = frozenset(
    [
        "prompt",
        "instruction",
        "input",
        "question",
        "task",
        "text",
        "problem",
        "problem_statement",
        "query",
        "raw_problem",
        "new_problem",
        "source",
        "context",
    ]
)

RESPONSE_LIKE_COLS: frozenset[str] = frozenset(
    [
        "completion",
        "response",
        "output",
        "answer",
        "code",
        "solution",
        "patch",
        "raw_solution",
        "new_solution",
        "target",
        "label",
    ]
)

# Columns that indicate a GRPO-compatible test/verification column.
TEST_LIKE_COLS: frozenset[str] = frozenset(
    [
        "tests",
        "test_cases",
        "test_list",
        "test_code",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    ]
)

# Columns that indicate a DPO-compatible preference structure.
PREFERENCE_COLS: frozenset[str] = frozenset(["chosen", "rejected"])

# Column mapping hints for download_command generation.
# hf_col -> DataDesigner seed column name.
COLUMN_MAP_HINTS: dict[str, str] = {
    # Prompt-like → seed_task
    "instruction": "seed_task",
    "input": "seed_context",
    "prompt": "seed_task",
    "question": "seed_task",
    "task": "seed_task",
    "text": "seed_task",
    "problem": "seed_task",
    "problem_statement": "seed_task",
    "query": "seed_task",
    "raw_problem": "seed_task",
    "new_problem": "seed_task",
    "context": "seed_context",
    # Response-like → seed_response
    "output": "seed_response",
    "response": "seed_response",
    "completion": "seed_response",
    "answer": "seed_response",
    "code": "seed_response",
    "solution": "seed_response",
    "patch": "seed_response",
    "raw_solution": "seed_response",
    "new_solution": "seed_response",
    "target": "seed_response",
}
