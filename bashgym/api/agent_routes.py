# bashgym/api/agent_routes.py
"""API routes for the system-aware Gym Agent chatbot."""

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    response: str
    context_used: List[str] = []


async def _gather_system_context() -> str:
    """Gather current system state to give the agent full awareness."""
    sections = []

    # 1. System stats (traces, models, training)
    try:
        from pathlib import Path
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
        from bashgym.trace_capture.importers.claude_history import ClaudeHistoryImporter
        importer = ClaudeHistoryImporter()
        repos = importer.list_repos()
        if repos:
            repo_lines = [f"  - {r['name']}: {r.get('trace_count', '?')} traces" for r in repos[:10]]
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


def _build_system_prompt(context: str) -> str:
    return (
        "You are the Gym Agent — a system-aware assistant for Bash Gym, a self-improving "
        "agentic development gym that captures coding sessions, transforms them into training "
        "data, and fine-tunes models.\n\n"
        "You have access to the user's current system state. Use it to give specific, "
        "actionable advice. Be concise and direct.\n\n"
        "You can help with:\n"
        "- Planning training runs (choosing strategy, model, hyperparameters)\n"
        "- Scheduling data generation (when to run, how much to generate)\n"
        "- Assessing system bandwidth (GPU memory, available VRAM, concurrent capacity)\n"
        "- Analyzing trace quality and recommending next steps\n"
        "- Suggesting orchestration specs for multi-agent work\n"
        "- Troubleshooting training or data pipeline issues\n\n"
        "When giving recommendations, reference the actual numbers from the system context. "
        "Be specific about model names, trace counts, and resource constraints.\n\n"
        f"--- CURRENT SYSTEM STATE ---\n{context}\n--- END SYSTEM STATE ---"
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    """Send a message to the system-aware Gym Agent."""
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured. The Gym Agent requires an API key."
        )

    # Gather live system context
    context = await _gather_system_context()
    system_prompt = _build_system_prompt(context)
    context_used = [s.split(':**')[0].replace('**', '') for s in context.split('\n\n') if ':**' in s]

    # Build message list
    messages = []
    if request.history:
        for msg in request.history:
            if msg.role in ('user', 'assistant'):
                messages.append({"role": msg.role, "content": msg.content})
    else:
        messages.append({"role": "user", "content": request.message})

    # Call Claude API
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
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text from response
        response_text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                response_text += block["text"]

        if not response_text:
            response_text = "I couldn't generate a response. Please try again."

        return ChatResponse(response=response_text, context_used=context_used)

    except httpx.HTTPStatusError as e:
        logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text[:200]}")
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
