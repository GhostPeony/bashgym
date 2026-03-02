# Training Data Guide

> How BashGym captures, structures, and exports coding agent traces for model fine-tuning.

This document is the canonical reference for the training data pipeline. It covers the data format spec, pipeline architecture, quality tiers, framework compatibility, and guidelines for agents working on improving the system.

---

## Architecture Overview

```
Claude Code Sessions (~/.claude/projects/)
    |
    v  [Import]  claude_history.py
TraceSession JSON (data/traces/)
    |
    v  [Classify]  trace_processor.py + quality_calculator.py
Tiered Traces (data/gold_traces/, data/silver_traces/, etc.)
    |
    v  [Segment]  example_generator.py
Training Examples (structured tool-call messages)
    |
    v  [Export]  NeMo JSONL / TRL Dataset
Fine-Tuning (SFT, DPO, GRPO)
```

---

## Data Format Specification

### Structured Tool-Call Format (v2 -- current)

Training examples use multi-turn `messages[]` with `tool_calls` and `tool` roles. This is compatible with NeMo, HuggingFace TRL, and OpenAI fine-tuning formats.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert software development agent..."
    },
    {
      "role": "user",
      "content": "Fix the authentication bug in src/auth.py"
    },
    {
      "role": "assistant",
      "content": "<thinking>\nThe user wants me to fix an auth bug. Let me read the file first.\n</thinking>\n\n<plan>\nFirst I will read src/auth.py to understand the current implementation.\n</plan>\n\nI'll investigate the authentication module.",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "Read",
            "arguments": "{\"file_path\": \"src/auth.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "Read",
      "content": "def authenticate(user, pwd):\n    return pwd == stored[user]"
    },
    {
      "role": "assistant",
      "content": "<reflection>\nI notice the password is compared as plaintext. This is a security vulnerability.\n</reflection>\n\nFound the bug. Passwords are compared as plaintext.",
      "tool_calls": [
        {
          "id": "call_2",
          "type": "function",
          "function": {
            "name": "Edit",
            "arguments": "{\"file_path\": \"src/auth.py\", \"old_string\": \"return pwd == stored[user]\", \"new_string\": \"return bcrypt.checkpw(pwd.encode(), stored[user])\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_2",
      "name": "Edit",
      "content": "File updated successfully."
    },
    {
      "role": "assistant",
      "content": "Fixed. Passwords now use bcrypt hash comparison. All tests pass."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "Read",
        "description": "Read file contents",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {"type": "string", "description": "Absolute path to file"}
          },
          "required": ["file_path"]
        }
      }
    }
  ],
  "metadata": {
    "source_trace": "imported_claude_abc123.json",
    "success_rate": 1.0,
    "quality_score": 0.85,
    "total_steps": 3,
    "per_step_success": [true, true, true],
    "repo_name": "ghostwork",
    "repo_path": "C:/Users/Cade/projects/ghostwork",
    "model": "claude-opus-4-5-20251101",
    "generated_at": "2026-02-25T12:00:00Z"
  }
}
```

### Key Format Rules

| Rule | Details |
|------|---------|
| **`arguments` is a JSON string** | Must be `"{\"key\": \"value\"}"`, not a parsed object |
| **Each tool call has a unique `id`** | Format: `"call_{md5_hash[:8]}"` |
| **Tool outputs truncated to 2000 chars** | Longer outputs get `\n... [truncated]` suffix |
| **Final message has no `tool_calls`** | Last assistant message is a text summary |
| **`tools` array included per example** | Lists all tool schemas the agent can use |
| **`per_step_success` preserved** | Boolean array matching step order |
| **Cognitive tags in assistant content** | `<thinking>`, `<plan>`, `<reflection>` tags wrap reasoning data |
| **`repo_name` / `repo_path` in metadata** | Identifies the source repository for repo-aware filtering |

### Legacy Format (v1 -- deprecated)

The old format used a flat 3-message structure with markdown prose. Do not generate new data in this format. Existing v1 data in `data/training_batches/` can be identified by the absence of `tool_calls` in assistant messages.

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "**Step 1: Execute command**\n```bash\ngrep..."}
  ]
}
```

---

## Tool Schema Registry

These are the tools the agent can call. Schemas follow JSON Schema format.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| **Bash** | Execute shell commands | `command: string`, `description?: string` |
| **Read** | Read file contents | `file_path: string`, `offset?: number`, `limit?: number` |
| **Write** | Write/create files | `file_path: string`, `content: string` |
| **Edit** | Targeted string replacement | `file_path: string`, `old_string: string`, `new_string: string` |
| **Grep** | Search file contents | `pattern: string`, `path?: string`, `glob?: string` |
| **Glob** | Find files by pattern | `pattern: string`, `path?: string` |

Full schemas are defined in `bashgym/factory/example_generator.py::TOOL_SCHEMAS`.

---

## Raw Trace Structure (from Claude Code)

Each imported session trace contains:

### Session-Level Fields

```python
{
    "session_id": "uuid",
    "timestamp": "ISO 8601",
    "source_tool": "claude_code",
    "primary_repo": {"path": "...", "name": "ghostwork"},
    "metadata": {
        "user_initial_prompt": "Fix the bug in...",
        "all_user_prompts": [{"text": "...", "timestamp": "..."}],
        "total_input_tokens": 5250,
        "total_output_tokens": 1840,
        "total_cache_creation_tokens": 0,
        "total_cache_read_tokens": 0,
        "api_equivalent_cost_usd": 0.025,
        "conversation_turns": 1,
        "thinking_block_count": 0,
        "total_tool_calls": 5,
        "subagent_count": 0,
        "subagent_total_tokens": 0,
        "subagent_total_duration_ms": 0,
        "models_used": ["claude-opus-4-5-20251101"],
        "tools_used": ["Bash", "Read", "Edit"],
        "git_branch": "feat/my-feature",
        "verification_passed": true
    },
    "summary": {
        "total_steps": 5,
        "successful_steps": 4,
        "failed_steps": 1,
        "success_rate": 0.8,
        "tool_breakdown": {"Bash": 2, "Read": 2, "Edit": 1}
    },
    "trace": [ ...steps... ]
}
```

### Per-Step Fields

```python
{
    "step_id": "session_uuid_tool_use_id",
    "timestamp": "ISO 8601",
    "tool_name": "Bash",
    "command": "{\"command\": \"ls -la\", \"description\": \"List files\"}",
    "output": "total 42\ndrwxr-xr-x ...",
    "exit_code": 0,
    "success": true,
    "cwd": "/home/user/project",
    "repo": {"path": "/home/user/project", "name": "project"},
    "cognitive": {
        "thinking": "I need to check the directory structure...",
        "plan": "First I will list the files to understand the layout.",
        "reflection": null,
        "decision_rationale": "Let me check the project structure."
    },
    "metadata": {
        "model": "claude-opus-4-5-20251101",
        "input_tokens": 2500,
        "output_tokens": 150,
        "thinking_content": "I need to check the directory...",
        "assistant_text": "Let me check the project structure.",
        "cognitive": { "...same as step.cognitive..." },
        "stop_reason": "tool_use"
    }
}
```

---

## Quality Tiers and Classification

### Quality Metrics (7 dimensions)

| Metric | Weight | Calculation |
|--------|--------|-------------|
| **Success Rate** | 25% | `successful_steps / total_steps` |
| **Verification** | 20% | 1.0 if passed, 0.0 if failed, 0.5 if unknown |
| **Cognitive Quality** | 15% | Presence of thinking/plan/reflection per step. Scores improve with better cognitive extraction. |
| **Complexity** | 15% | Tool diversity + command patterns + control flow |
| **Tool Diversity** | 10% | `unique_tools / max_tools (~6)` |
| **Efficiency** | 10% | Output length normalized |
| **Length** | 5% | Bell curve around ideal (10-30 steps) |

### Tier Thresholds

| Tier | Quality Score | Success Rate | Training Use |
|------|:------------:|:------------:|-------------|
| **Gold** | >= 0.75 | >= 90% | SFT training data |
| **Silver** | >= 0.55 | >= 75% | DPO chosen examples |
| **Bronze** | >= 0.40 | >= 60% | DPO rejected examples |
| **Rejected** | < 0.40 | < 60% | Excluded from training |

### Classification Commands

```bash
# Auto-classify all pending traces
POST /api/traces/auto-classify

# Manually promote/demote
POST /api/traces/{trace_id}/promote
POST /api/traces/{trace_id}/demote
```

---

## Pipeline Operations

### 1. Import Traces

```bash
# Import new sessions from Claude Code history
POST /api/traces/import

# Import sessions added since last import
GET /api/traces/import-since

# Sync from ~/.bashgym/traces/ to project data/
POST /api/traces/sync
```

### 2. Generate Training Examples

```bash
# Generate structured examples from a single trace
POST /api/traces/{trace_id}/generate-examples

# List generated examples
GET /api/training/examples

# Export to NeMo format with train/val split
POST /api/training/export
```

### Cognitive Data in Training Examples

When `include_cognitive=True` (the default), assistant messages before each tool call include structured reasoning from the original session. This teaches models to reason-then-act.

| Tag | Source | Description |
|-----|--------|-------------|
| `<thinking>` | `cognitive.thinking` or `metadata.thinking_content` | Raw extended thinking blocks from Claude |
| `<plan>` | `cognitive.plan` | Paragraphs matching plan patterns (e.g. "I'll first...", "step 1...") |
| `<reflection>` | `cognitive.reflection` | Paragraphs matching reflection patterns (e.g. "I notice...", "the error is...") |
| *(plain text)* | `cognitive.decision_rationale` or `metadata.assistant_text` | Remaining assistant text not matching plan/reflection |

Deduplication: if the same text appears in multiple cognitive fields (e.g. plan == thinking), it is only emitted once under the highest-priority tag. Priority order: thinking > plan > reflection > rationale.

**Import limits**: Thinking blocks are preserved up to 10,000 chars and text blocks up to 5,000 chars at import time. These are large enough to capture full extended thinking from Opus sessions.

### Repo-Aware Export

Training examples carry `repo_name` and `repo_path` in their metadata, sourced from the session's `primary_repo` field. This enables repo-specific training:

```python
# Export only examples from a specific repo
generator.export_for_nemo(
    examples, output_dir,
    repo_filter=["ghostwork"]
)
```

The full pipeline: API filters traces by repo -> examples carry repo metadata -> `export_for_nemo(repo_filter=...)` filters at export time.

### 3. Aggregate Analytics

```bash
# Cross-session analytics (tool stats, quality distribution, composition)
GET /api/traces/analytics

# Timeline stats (24h hourly buckets)
GET /api/traces/stats
```

---

## Framework Compatibility

### NVIDIA NeMo

The structured format is directly compatible with NeMo SFT. Key points:

- NeMo JSONL expects `messages` array with `role`/`content` fields
- `tool_calls` on assistant messages are supported -- loss is masked to only train on tool-call arguments
- RPO (Reward-aware Preference Optimization) supports `chosen_reward`/`rejected_reward` floats from our quality scores
- NeMo Curator can be used for GPU-accelerated deduplication and quality filtering

### HuggingFace TRL

TRL's `SFTTrainer` natively consumes the format:

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files="train.jsonl")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="output"),
    train_dataset=dataset,
)
```

Key TRL features we can use:
- **`tools` column** in dataset -- TRL applies tool-call chat templates automatically
- **`assistant_only_loss=True`** -- only train on assistant responses, not tool outputs
- **DPOTrainer** with tool-call preference pairs
- **PRMTrainer** (Process Reward Model) with `per_step_success` for stepwise supervision
- **GRPO** with decomposed rewards (format + correctness + efficiency)

### Qwen2.5-Coder (Target Model)

Qwen2.5-Coder uses ChatML with `<tool_call>` / `<tool_response>` XML tags. TRL handles the conversion automatically when the tokenizer has a tool-call chat template. No manual formatting needed.

### Unsloth

Unsloth wraps TRL with 2-5x speed improvements. Same dataset format works directly:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
# ... SFTTrainer with same dataset
```

---

## DPO Pair Generation

DPO training requires preference pairs: a chosen response (good) and a rejected response (bad) for the same prompt.

### Pairing Strategy

| Chosen Source | Rejected Source | Method |
|--------------|----------------|--------|
| Gold trace | Failed trace (same task type) | Semantic similarity matching on prompts |
| Gold trace | Guardrail-blocked response | From `/api/observability/guardrails/dpo-negatives` |
| Silver trace | Bronze trace | Quality score ordering |

### DPO Format (NeMo-compatible)

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Fix the auth bug"}
  ],
  "chosen": [
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "Fixed and verified."}
  ],
  "rejected": [
    {"role": "assistant", "content": "You should change line 15 to use bcrypt..."}
  ],
  "chosen_reward": 0.85,
  "rejected_reward": 0.25
}
```

---

## Data Quality Guidelines

### What Makes a Good Training Example

1. **Task clarity** -- the user prompt is specific and actionable
2. **Tool usage is purposeful** -- each step moves toward the goal
3. **Error recovery** -- if a step fails, the next step addresses it
4. **Verification** -- the trace ends with a test or confirmation step
5. **Reasonable length** -- 3-30 steps. Very short = trivial, very long = noisy

### What to Filter Out

- Sessions with < 2 tool calls (nothing to learn from)
- Sessions with > 50 steps (too long for context window, likely unfocused)
- Sessions where success_rate < 50% (unless specifically collecting for DPO rejected)
- Sessions with no `user_initial_prompt` (can't construct the task)
- Duplicate sessions (same SHA256 hash of prompt + command sequence)

### Data Curation Best Practices (from research)

| Practice | Source | Details |
|----------|--------|---------|
| Cap 3 trajectories per task | SWE-smith | Prevents sampling bias |
| Filter by model capability | ToolACE-R | Mid-difficulty examples are most effective |
| Include recovery trajectories | NAT, ToolACE-R | Teach the model to recover from errors |
| Quality >> Quantity | SWE-smith | 5k expert examples > 50k noisy examples |
| Curriculum learning | Fireworks AI | Start with simple tasks, increase complexity |

---

## Known Data Gaps

These are tracked issues where data is lost or never captured:

| Gap | Impact | Status |
|-----|--------|--------|
| **No per-step token counts in Claude Code exports** | Can't track cost per tool call | Claude Code JSONL has no `usage` fields |
| **No step duration tracking** | Can't optimize for latency | Timestamps exist but duration isn't derived |
| **Cache hit ratio not surfaced** | Can't optimize prompt caching strategy | `cache_creation_tokens` and `cache_read_tokens` captured but not displayed |
| **Sub-agent metrics not surfaced** | Can't analyze delegation efficiency | `subagent_count`, `subagent_total_tokens` captured but not displayed |
| ~~**Thinking blocks not in training data**~~ | **RESOLVED** | Thinking, plan, and reflection are now injected as `<thinking>`, `<plan>`, `<reflection>` tags in assistant messages. Deduplication prevents the same text appearing in multiple tags. |

---

## Agent Guidelines for Improving This System

### If you're working on the import pipeline (`trace_capture/`)

- Raw Claude Code sessions are JSONL in `~/.claude/projects/<slug>/<session-id>.jsonl`
- Each line is an event: `file-history-snapshot`, `progress`, `user`, `assistant`, `tool-use`
- The importer in `claude_history.py` parses these into `TraceSession` objects
- **Don't truncate outputs more aggressively** -- 10KB is the limit, training examples truncate further
- **Do preserve all metadata** -- even fields we don't use yet (sub-agent data, thinking blocks)

### If you're working on the example generator (`factory/`)

- **Always output structured tool-call format** (v2) -- never markdown prose
- **Parse `command` as JSON** when possible -- many commands are `{"command": "ls", "description": "..."}` or `{"file_path": "/path"}`
- **Keep `per_step_success` as a boolean array** in metadata
- **Include the `tools` array** with every example
- **Test with actual trace files** from `data/traces/` or `~/.bashgym/traces/`

### If you're working on quality scoring (`factory/trace_processor.py`)

- The 6 metrics and their weights are documented above
- **Don't change weights without A/B testing** on training outcomes
- Consider adding: token efficiency score (tokens used vs task complexity)
- Consider adding: thinking utilization score (does the model think before complex steps?)

### If you're working on the training pipeline (`trainer.py`)

- **SFT first, then DPO/GRPO** -- always start with supervised fine-tuning
- Training requires Python 3.12 with CUDA (not 3.14 -- no PyTorch wheels)
- Unsloth + PyTorch CUDA 13.0: `pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
- The trainer auto-detects Python 3.12 at `C:\Users\{user}\AppData\Local\Programs\Python\Python312\python.exe`
- Training scripts go to `~/.bashgym/models/{run_id}/train_sft.py`
- Logs stream via WebSocket (`training:log` message type)

### If you're working on the frontend

- The Trace Browser (`TraceBrowser.tsx`) is the main UI for trace management
- `tracesStore.ts` manages trace state (Zustand)
- Tool breakdown (`summary.tool_breakdown`) should be displayed per-trace
- Aggregate analytics should power a dashboard section
- Use Botanical Brutalist design tokens (see `globals.css` and `CLAUDE.md`)

### If you're adding new data to training examples

Before adding a new field to the training format:
1. Check if the raw trace captures it (see "Raw Trace Structure" above)
2. Check if it survives through `trace_processor.py` processing
3. Decide: should it go in `messages[]` (model sees it) or `metadata` (training infra only)?
4. Update the `TOOL_SCHEMAS` if adding a new tool
5. Update this document

---

## File Reference

| File | Layer | Purpose |
|------|-------|---------|
| `bashgym/trace_capture/importers/claude_history.py` | Import | Parses Claude Code session JSONL |
| `bashgym/trace_capture/core.py` | Import | TraceStep, TraceSession, TraceCapture classes |
| `bashgym/factory/trace_processor.py` | Process | Quality scoring, normalization, redaction |
| `bashgym/factory/quality_calculator.py` | Process | 6-metric quality calculation |
| `bashgym/factory/example_generator.py` | Generate | Session segmentation, structured tool-call format |
| `bashgym/factory/data_factory.py` | Generate | TrainingExample/DPOExample classes, augmentation |
| `bashgym/api/routes.py` | API | Trace CRUD, import, classify, export endpoints |
| `bashgym/trainer.py` | Train | SFT/DPO/GRPO training script generation |
| `frontend/src/components/traces/TraceBrowser.tsx` | UI | Trace management interface |
| `frontend/src/stores/tracesStore.ts` | UI | Trace state management |
