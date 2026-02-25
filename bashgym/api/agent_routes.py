# bashgym/api/agent_routes.py
"""API routes for Peony — the botanical assistant for Bash Gym."""

import json
import logging
import os
import secrets
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bashgym.agent.memory import PeonyMemory
from bashgym.agent.skills.registry import SkillRegistry
from bashgym.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_skill_registry = SkillRegistry()
_tool_registry = ToolRegistry()
_memory = PeonyMemory()

# ---------------------------------------------------------------------------
# In-memory pending actions (shell confirmation gate)
# ---------------------------------------------------------------------------

PENDING_ACTIONS: dict[str, dict] = {}  # token → {cmd, reason, messages, tool_use_id, expires_at, tools}


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


class PendingAction(BaseModel):
    type: str  # "shell_command"
    command: str
    reason: str
    token: str


class ChatResponse(BaseModel):
    response: str
    context_used: List[str] = []
    pending_action: Optional[PendingAction] = None


class ConfirmActionRequest(BaseModel):
    token: str
    approved: bool
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------

class SessionMeta(BaseModel):
    session_id: str
    name: str
    created_at: str
    updated_at: str
    message_count: int = 0


class SessionMessage(BaseModel):
    id: str
    role: str
    content: str
    timestamp: float
    context_used: list[str] = []


class SaveSessionRequest(BaseModel):
    session_id: str
    name: str
    messages: list[SessionMessage]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_peony_logs_dir() -> Path:
    from bashgym.config import get_bashgym_dir
    d = get_bashgym_dir() / "peony_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_sessions_index_path() -> Path:
    return _get_peony_logs_dir() / "_sessions_index.json"


def _read_sessions_index() -> list[dict]:
    path = _get_sessions_index_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_sessions_index(index: list[dict]) -> None:
    path = _get_sessions_index_path()
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# System context gathering
# ---------------------------------------------------------------------------

async def _gather_system_context() -> str:
    """Gather current system state to give Peony full awareness."""
    sections = []

    # 1. System stats (traces, models, training)
    try:
        from bashgym.config import get_settings, get_bashgym_dir
        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        gold = len(list((data_dir / "gold_traces").glob("*.json"))) if (data_dir / "gold_traces").exists() else 0
        silver = len(list((data_dir / "silver_traces").glob("*.json"))) if (data_dir / "silver_traces").exists() else 0
        bronze = len(list((data_dir / "bronze_traces").glob("*.json"))) if (data_dir / "bronze_traces").exists() else 0
        failed = len(list((data_dir / "failed_traces").glob("*.json"))) if (data_dir / "failed_traces").exists() else 0
        models = len(list((data_dir / "models").iterdir())) if (data_dir / "models").exists() else 0
        pending = 0
        for trace_dir in [get_bashgym_dir() / "traces", data_dir / "traces"]:
            if trace_dir.exists():
                pending += len(list(trace_dir.glob("session_*.json")))
                pending += len(list(trace_dir.glob("imported_*.json")))
        sections.append(
            f"**System Stats:**\n"
            f"- Gold traces: {gold}\n"
            f"- Silver traces: {silver}\n"
            f"- Bronze traces: {bronze}\n"
            f"- Failed traces: {failed}\n"
            f"- Pending traces: {pending}\n"
            f"- Trained models: {models}\n"
            f"- Base model: {settings.training.base_model}"
        )
    except Exception as e:
        logger.debug(f"Could not gather system stats: {e}")

    # 2. GPU / hardware info
    try:
        from bashgym.api.system_info import get_system_info_service
        sysinfo = get_system_info_service().get_system_info()
        gpu_lines = []
        for gpu in sysinfo.gpus:
            gpu_lines.append(
                f"  - {gpu.model} | "
                f"VRAM: {gpu.vram_used:.1f}/{gpu.vram:.1f} GB | Util: {gpu.utilization}%"
            )
        sections.append(
            f"**Hardware:**\n"
            f"- Platform: {sysinfo.platform_name} ({sysinfo.arch})\n"
            f"- RAM: {sysinfo.available_ram:.1f}/{sysinfo.total_ram:.1f} GB available\n"
            f"- CUDA: {'Yes (' + sysinfo.cuda_version + ')' if sysinfo.cuda_available else 'No'}\n"
            f"- GPUs:\n" + ('\n'.join(gpu_lines) if gpu_lines else '  - None detected')
        )
    except Exception as e:
        logger.debug(f"Could not gather hardware info: {e}")

    # 3. Trace repos
    try:
        from bashgym.trace_capture.importers.claude_history import ClaudeSessionImporter
        importer = ClaudeSessionImporter()
        projects_dir = importer.find_projects_dir()
        if projects_dir and projects_dir.exists():
            repos = [
                {"name": d.name, "trace_count": len(list(d.glob("*.jsonl")))}
                for d in projects_dir.iterdir()
                if d.is_dir()
            ]
            repos.sort(key=lambda r: r["trace_count"], reverse=True)
            if repos:
                repo_lines = [f"  - {r['name']}: {r.get('trace_count', '?')} sessions" for r in repos[:10]]
                sections.append(
                    f"**Trace Repositories ({len(repos)} total):**\n" + '\n'.join(repo_lines)
                )
    except Exception as e:
        logger.debug(f"Could not gather trace repos: {e}")

    # 4. Training configuration
    try:
        from bashgym.config import get_settings
        settings = get_settings()
        sections.append(
            f"**Training Config:**\n"
            f"- Base model: {settings.training.base_model}\n"
            f"- Default strategy: SFT\n"
            f"- Auto-export GGUF: {settings.training.auto_export_gguf}\n"
            f"- Max sequence length: {settings.training.max_seq_length}"
        )
    except Exception as e:
        logger.debug(f"Could not gather training config: {e}")

    # 5. Active orchestration jobs
    try:
        from bashgym.api.orchestrator_routes import _jobs
        active = [j for j in _jobs.values() if j["status"] in ("decomposing", "executing")]
        if active:
            job_lines = [f"  - {j['id']}: {j['status']} ({j.get('spec', {}).title if j.get('spec') else 'untitled'})" for j in active]
            sections.append(
                f"**Active Orchestration Jobs ({len(active)}):**\n" + '\n'.join(job_lines)
            )
    except Exception as e:
        logger.debug(f"Could not gather orchestration status: {e}")

    return '\n\n'.join(sections) if sections else 'System context unavailable.'


def _build_system_prompt(context: str, memory_prompt: str = "", skill_knowledge: str = "") -> str:
    sections = [
        # 1. Core identity
        "You are Peony — the botanical assistant for Bash Gym, a self-improving "
        "agentic development gym that captures coding sessions, transforms them into training "
        "data, and fine-tunes models.",

        # 2. Tool usage instructions
        "You have access to tools that let you take real actions in the system. Use them "
        "when the user asks you to do something actionable (import traces, search models, "
        "start training, etc.). For questions and analysis, respond directly.\n\n"
        "When you use a tool, briefly tell the user what you're doing before/after. "
        "Summarize tool results in natural language — don't dump raw JSON.\n\n"
        "For run_shell_command: only use this as a last resort when no structured tool covers "
        "the need. Always provide a clear reason.",

        # 3. Capabilities summary (always present)
        _tool_registry.capabilities_summary(),
    ]

    # 4. Memory prompt (if non-empty)
    if memory_prompt:
        sections.append(memory_prompt)

    # 5. Current system state
    sections.append(f"--- CURRENT SYSTEM STATE ---\n{context}\n--- END SYSTEM STATE ---")

    # 6. Skill knowledge (if non-empty)
    if skill_knowledge:
        sections.append(f"--- RELEVANT SKILL KNOWLEDGE ---\n{skill_knowledge}\n--- END SKILL KNOWLEDGE ---")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

async def _execute_tool(name: str, tool_input: dict) -> str:
    """Execute a Peony tool and return a string result for the tool_result block."""

    # ----- Memory tools (local, no HTTP needed) -----
    if name == "remember_fact":
        fact = _memory.remember_fact(tool_input["category"], tool_input["content"])
        return json.dumps({"status": "remembered", "fact": fact})

    if name == "recall_facts":
        facts = _memory.recall_facts(
            category=tool_input.get("category"),
            keyword=tool_input.get("keyword"),
        )
        return json.dumps({"facts": facts, "count": len(facts)})

    if name == "forget_fact":
        try:
            _memory.forget_fact(tool_input["fact_id"])
            return json.dumps({"status": "forgotten"})
        except ValueError as e:
            return json.dumps({"error": str(e)})

    if name == "update_user_profile":
        try:
            _memory.update_profile(tool_input["field"], tool_input["value"])
            profile = _memory.load_profile()
            return json.dumps({"status": "updated", "profile": profile})
        except ValueError as e:
            return json.dumps({"error": str(e)})

    # ----- Awareness tools (local, no HTTP needed) -----
    if name == "list_my_capabilities":
        result = _tool_registry.list_capabilities(category=tool_input.get("category"))
        return result

    # ----- Data collection tools (local, no HTTP needed) -----
    if name == "import_traces":
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        sources = tool_input.get("sources", ["all"])
        dry_run = tool_input.get("dry_run", False)
        project_filter = tool_input.get("project_filter")

        # Calculate 'since' from days parameter
        since = None
        days = tool_input.get("days")
        if days:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Resolve "all" or "sessions" — session import goes through existing HTTP endpoint
        source_list = None  # None means all in scanner
        if "all" not in sources:
            source_list = [s for s in sources if s != "sessions"]

        results = {}

        # Handle session import via existing endpoint if requested
        if source_list is None or "sessions" in sources:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post("http://localhost:8003/api/traces/import")
                    resp.raise_for_status()
                    results["sessions"] = resp.json()
            except Exception as e:
                results["sessions"] = {"error": str(e)}

        if dry_run:
            scan_results = scanner.scan_all(sources=source_list, since=since, project_filter=project_filter)
            for src, scan_result in scan_results.items():
                results[src] = {
                    "total_found": scan_result.total_found,
                    "already_collected": scan_result.already_collected,
                    "new_available": scan_result.new_available,
                }
        else:
            collect_results = scanner.collect_all(sources=source_list, since=since, project_filter=project_filter)
            for src, batch_result in collect_results.items():
                results[src] = {
                    "collected": batch_result.collected_count,
                    "skipped": batch_result.skipped_count,
                    "errors": batch_result.error_count,
                }

        return json.dumps(results)

    if name == "scan_claude_data":
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        scan_results = scanner.scan_all()
        results = {}
        for src, scan_result in scan_results.items():
            results[src] = {
                "total_found": scan_result.total_found,
                "already_collected": scan_result.already_collected,
                "new_available": scan_result.new_available,
            }
        return json.dumps(results)

    if name == "get_collection_status":
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        status = scanner.status()
        return json.dumps(status)

    # ----- HTTP-based tools -----
    import httpx

    base_url = "http://localhost:8003"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if name == "get_trace_status":
                resp = await client.get(f"{base_url}/api/stats")
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "classify_pending_traces":
                dry_run = tool_input.get("dry_run", True)
                resp = await client.post(
                    f"{base_url}/api/traces/auto-classify",
                    params={"dry_run": str(dry_run).lower(), "auto_promote": str(not dry_run).lower()},
                )
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "start_training":
                payload = {
                    "strategy": tool_input.get("strategy", "sft"),
                }
                if tool_input.get("model"):
                    payload["base_model"] = tool_input["model"]
                resp = await client.post(f"{base_url}/api/training/start", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "get_training_status":
                resp = await client.get(f"{base_url}/api/training")
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "hf_search_models":
                params = {}
                if tool_input.get("task"):
                    params["task"] = tool_input["task"]
                if tool_input.get("sort"):
                    params["sort"] = tool_input["sort"]
                if tool_input.get("limit"):
                    params["limit"] = tool_input["limit"]
                resp = await client.get(f"{base_url}/api/hf/models/search", params=params)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "hf_get_job_status":
                job_id = tool_input["job_id"]
                resp = await client.get(f"{base_url}/api/hf/jobs/{job_id}")
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "hf_test_inference":
                payload = {
                    "model": tool_input["model_id"],
                    "prompt": tool_input["prompt"],
                }
                resp = await client.post(f"{base_url}/api/hf/inference/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "hf_evaluate_model":
                payload = {
                    "model_id": tool_input["model_id"],
                    "metric": tool_input.get("metric", "accuracy"),
                }
                resp = await client.post(f"{base_url}/api/hf/evaluate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except httpx.HTTPStatusError as e:
            return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(request: ChatRequest):
    """Send a message to Peony, the botanical assistant."""
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured. Peony requires an API key."
        )

    # Load memory
    memory_prompt = _memory.build_memory_prompt()

    # Match skills to user message
    matched_skills = _skill_registry.match(request.message)
    skill_tools = _skill_registry.get_tools(matched_skills)
    skill_knowledge = _skill_registry.get_knowledge(matched_skills)

    # Build dynamic tool list
    tools = _tool_registry.build_tools(skill_tools=skill_tools)

    # Gather live system context
    context = await _gather_system_context()
    context_used = [s.split(':**')[0].replace('**', '') for s in context.split('\n\n') if ':**' in s]

    # Build system prompt with all sections
    system_prompt = _build_system_prompt(context, memory_prompt, skill_knowledge)

    # Build message list
    messages = []
    if request.history:
        for msg in request.history:
            if msg.role in ('user', 'assistant'):
                messages.append({"role": msg.role, "content": msg.content})
    else:
        messages.append({"role": "user", "content": request.message})

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # First Claude call with tools
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "tools": tools,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text[:200]}")
            raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.status_code}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="LLM request timed out")
        except Exception as e:
            logger.error(f"Peony chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    stop_reason = data.get("stop_reason")

    # --- Tool use path ---
    if stop_reason == "tool_use":
        tool_use_blocks = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
        tool_results = []
        pending_shell: Optional[dict] = None

        for block in tool_use_blocks:
            tool_name = block["name"]
            tool_id = block["id"]
            tool_input = block.get("input", {})

            if tool_name == "run_shell_command":
                # Gate: require user confirmation
                token = secrets.token_urlsafe(16)
                expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
                PENDING_ACTIONS[token] = {
                    "command": tool_input["command"],
                    "reason": tool_input.get("reason", ""),
                    "messages": messages + [{"role": "assistant", "content": data["content"]}],
                    "tool_use_id": tool_id,
                    "expires_at": expires_at,
                    "system_prompt": system_prompt,
                    "context_used": context_used,
                    "headers": headers,
                    "tools": tools,
                }
                pending_shell = {
                    "type": "shell_command",
                    "command": tool_input["command"],
                    "reason": tool_input.get("reason", ""),
                    "token": token,
                }
                # Return immediately for shell commands — user must confirm
                # Emit a brief text response so the UI shows something
                return ChatResponse(
                    response="",
                    context_used=context_used,
                    pending_action=PendingAction(**pending_shell),
                )
            else:
                # Execute structured tools immediately
                result_str = await _execute_tool(tool_name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_str,
                })

        if not tool_results:
            # Only shell command was requested; already returned above
            return ChatResponse(response="", context_used=context_used)

        # Second Claude call with tool results
        follow_up_messages = messages + [
            {"role": "assistant", "content": data["content"]},
            {"role": "user", "content": tool_results},
        ]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client2:
                resp2 = await client2.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": "claude-sonnet-4-5-20250929",
                        "max_tokens": 1024,
                        "system": system_prompt,
                        "tools": tools,
                        "messages": follow_up_messages,
                    },
                )
                resp2.raise_for_status()
                data2 = resp2.json()
        except Exception as e:
            logger.error(f"Peony second call error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        response_text = ""
        for block in data2.get("content", []):
            if block.get("type") == "text":
                response_text += block["text"]

        return ChatResponse(
            response=response_text or "Done.",
            context_used=context_used,
        )

    # --- Normal end_turn path ---
    response_text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            response_text += block["text"]

    if not response_text:
        response_text = "I couldn't generate a response. Please try again."

    return ChatResponse(response=response_text, context_used=context_used)


# ---------------------------------------------------------------------------
# Confirm action endpoint
# ---------------------------------------------------------------------------

@router.post("/confirm-action")
async def confirm_action(request: ConfirmActionRequest):
    """Approve or deny a pending shell command and resume the Claude conversation."""
    import httpx

    token = request.token
    action = PENDING_ACTIONS.pop(token, None)
    if not action:
        raise HTTPException(status_code=404, detail="Pending action not found or expired")

    # Check expiry
    if datetime.now(timezone.utc) > action["expires_at"]:
        raise HTTPException(status_code=410, detail="Pending action expired")

    tool_use_id = action["tool_use_id"]
    messages = action["messages"]
    system_prompt = action["system_prompt"]
    context_used = action["context_used"]
    headers = action["headers"]
    action_tools = action.get("tools", _tool_registry.build_tools())

    if request.approved:
        # Run the command
        try:
            proc = subprocess.run(
                action["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = proc.stdout or ""
            if proc.stderr:
                output += f"\n[stderr]: {proc.stderr}"
            tool_result_content = output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            tool_result_content = "[Error]: Command timed out after 30 seconds"
        except Exception as e:
            tool_result_content = f"[Error]: {e}"
    else:
        tool_result_content = "User declined to run this command."

    # Resume Claude conversation with tool result
    follow_up_messages = messages + [
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_result_content,
            }],
        }
    ]

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "tools": action_tools,
                    "messages": follow_up_messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response_text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            response_text += block["text"]

    return ChatResponse(
        response=response_text or "Done.",
        context_used=context_used,
    )


# ---------------------------------------------------------------------------
# Session CRUD endpoints
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_sessions():
    """List all Peony chat sessions."""
    return _read_sessions_index()


@router.get("/sessions/{session_id}")
async def load_session(session_id: str):
    """Load messages for a specific session from its JSONL log."""
    log_path = _get_peony_logs_dir() / f"{session_id}.jsonl"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    messages = []
    try:
        for line in log_path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            if record.get("type") == "message":
                messages.append(SessionMessage(
                    id=record["id"],
                    role=record["role"],
                    content=record["content"],
                    timestamp=record["timestamp"],
                    context_used=record.get("context_used", []),
                ))
    except Exception as e:
        logger.error(f"Error reading session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read session")

    return messages


@router.post("/sessions")
async def save_session(req: SaveSessionRequest):
    """Save a session — writes JSONL file and updates the index."""
    logs_dir = _get_peony_logs_dir()
    log_path = logs_dir / f"{req.session_id}.jsonl"

    # Write JSONL
    now = datetime.now(timezone.utc).isoformat()

    lines = []
    # Meta line
    lines.append(json.dumps({
        "type": "meta",
        "session_id": req.session_id,
        "name": req.name,
        "created_at": now,
        "updated_at": now,
    }))
    # Message lines
    for msg in req.messages:
        lines.append(json.dumps({
            "type": "message",
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "context_used": msg.context_used,
        }))

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update index
    index = _read_sessions_index()
    # Remove existing entry for this session
    index = [s for s in index if s.get("session_id") != req.session_id]
    index.append({
        "session_id": req.session_id,
        "name": req.name,
        "created_at": now,
        "updated_at": now,
        "message_count": len(req.messages),
    })
    _write_sessions_index(index)

    return {"status": "ok", "session_id": req.session_id}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session log and remove it from the index."""
    log_path = _get_peony_logs_dir() / f"{session_id}.jsonl"
    if log_path.exists():
        log_path.unlink()

    index = _read_sessions_index()
    index = [s for s in index if s.get("session_id") != session_id]
    _write_sessions_index(index)

    return {"status": "ok", "session_id": session_id}


# ---------------------------------------------------------------------------
# Session summarization
# ---------------------------------------------------------------------------

@router.post("/summarize-session/{session_id}")
async def summarize_session(session_id: str):
    """Generate and save an episode summary for a completed session."""
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")

    # Load session messages
    log_path = _get_peony_logs_dir() / f"{session_id}.jsonl"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    messages_text = []
    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "message":
            messages_text.append(f"{record['role']}: {record['content']}")

    if not messages_text:
        raise HTTPException(status_code=400, detail="Session has no messages")

    transcript = "\n".join(messages_text[-20:])  # Last 20 messages max

    # Generate summary via Claude Haiku (fast, cheap)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [{
                        "role": "user",
                        "content": (
                            "Summarize this conversation in 2-3 sentences. "
                            "Focus on what the user wanted, what was accomplished, "
                            "and any decisions made.\n\n" + transcript
                        ),
                    }],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"LLM API error: {e.response.status_code}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    summary_text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            summary_text += block["text"]

    if not summary_text:
        raise HTTPException(status_code=500, detail="Failed to generate summary")

    episode = _memory.save_episode(session_id, summary_text)
    return {"status": "ok", "episode": episode}
