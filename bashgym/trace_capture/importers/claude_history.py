"""
Claude Code Session History Importer

Imports tool execution traces from Claude Code's session JSONL files
into BashGym's trace format.

Claude Code stores session transcripts at:
  ~/.claude/projects/<project-slug>/<session-id>.jsonl

Each line is a JSON object with types like:
  - "user" / "assistant" - conversation messages
  - "progress" - tool execution progress
  - "tool_use" embedded in assistant messages
  - "tool_result" embedded in progress messages

Instrumentation:
  - PII filtering on user prompts and tool outputs
  - Injection detection on user prompts
  - Profiling spans for import operations
"""

import json
import os
import platform
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict, field

import re as _re

from ..core import TraceStep, TraceSession, TraceCapture, CognitiveData, estimate_cost_usd

# Import instrumentation (optional - graceful degradation if not available)
try:
    from bashgym.core import get_instrumentation, Instrumentation
    HAS_INSTRUMENTATION = True
except ImportError:
    HAS_INSTRUMENTATION = False
    Instrumentation = None


@dataclass
class ImportResult:
    """Result of an import operation."""
    session_id: str
    source_file: Path
    steps_imported: int
    destination_file: Optional[Path] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    # Instrumentation stats
    pii_redactions: int = 0
    injection_blocked: bool = False
    sanitized_prompt: Optional[str] = None


class ClaudeSessionImporter:
    """
    Import traces from Claude Code session files.

    Claude Code stores session transcripts as JSONL files in:
    ~/.claude/projects/<project-slug>/<session-id>.jsonl
    """

    # Tools to exclude from trace steps (empty = capture everything)
    EXCLUDED_TOOLS: Set[str] = set()

    def __init__(self):
        self.trace_capture = TraceCapture()
        self.claude_dir = self._get_claude_dir()
        self.imported_sessions_file = self.trace_capture.bashgym_dir / "imported_sessions.json"
        self._imported_sessions: Optional[Set[str]] = None

    @staticmethod
    def _get_claude_dir() -> Path:
        """Get Claude Code's data directory."""
        if platform.system() == 'Windows':
            base = Path(os.environ.get("USERPROFILE", ""))
        else:
            base = Path.home()
        return base / ".claude"

    @property
    def imported_sessions(self) -> Set[str]:
        """Get set of already imported session IDs."""
        if self._imported_sessions is None:
            self._imported_sessions = self._load_imported_sessions()
        return self._imported_sessions

    def _load_imported_sessions(self) -> Set[str]:
        """Load the list of already imported session IDs."""
        if not self.imported_sessions_file.exists():
            return set()
        try:
            with open(self.imported_sessions_file, 'r') as f:
                data = json.load(f)
                return set(data.get("sessions", []))
        except (json.JSONDecodeError, IOError):
            return set()

    def _save_imported_session(self, session_id: str) -> None:
        """Mark a session as imported."""
        self.imported_sessions.add(session_id)
        try:
            with open(self.imported_sessions_file, 'w') as f:
                json.dump({"sessions": list(self.imported_sessions)}, f)
        except IOError as e:
            print(f"Warning: Could not save imported sessions list: {e}")

    def find_projects_dir(self) -> Optional[Path]:
        """Find Claude Code's projects directory."""
        projects_dir = self.claude_dir / "projects"
        return projects_dir if projects_dir.exists() else None

    def find_session_files(
        self,
        project_filter: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[Tuple[Path, datetime]]:
        """
        Find session JSONL files, optionally filtered by date and project.

        Args:
            project_filter: Only include sessions from projects matching this substring
            since: Only include sessions modified after this time
            until: Only include sessions modified before this time

        Returns:
            List of (file_path, modified_time) tuples sorted by modified time (newest first)
        """
        projects_dir = self.find_projects_dir()
        if not projects_dir:
            return []

        session_files = []

        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Apply project filter
            if project_filter and project_filter.lower() not in project_dir.name.lower():
                continue

            for jsonl_file in project_dir.glob("*.jsonl"):
                # Get file modification time
                mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime, tz=timezone.utc)

                # Apply date filters
                if since and mtime < since:
                    continue
                if until and mtime > until:
                    continue

                session_files.append((jsonl_file, mtime))

        # Sort by modification time, newest first
        session_files.sort(key=lambda x: x[1], reverse=True)
        return session_files

    @staticmethod
    def _extract_text_from_content(content) -> Optional[str]:
        """Extract text from message content (string or list of blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts) if texts else None
        return None

    # Patterns indicating agent reflection / self-correction
    _REFLECTION_PATTERNS = [
        _re.compile(r"(?:I notice|I see|that didn't work|that failed|let me reconsider|the error|looking at the error|the issue is|the problem is|instead,? I)", _re.IGNORECASE),
    ]

    # Patterns indicating an explicit plan
    _PLAN_PATTERNS = [
        _re.compile(r"(?:(?:my |the )?plan is|here's (?:my |the )?plan|I'll|let me|first,? I|step \d)", _re.IGNORECASE),
    ]

    @staticmethod
    def _extract_cognitive(
        thinking_parts: List[str],
        text_parts: List[str],
        thinking_max_chars: int = 10000,
        text_max_chars: int = 5000,
    ) -> CognitiveData:
        """Extract structured cognitive data from thinking and text blocks.

        Separates raw thinking, plans, reflections, and decision rationale
        into distinct fields for downstream use in training example generation
        and quality scoring.

        Extracts matching *paragraphs* rather than the entire text block, so
        plan, reflection, and decision_rationale are semantically distinct.
        """
        thinking_raw = "\n".join(thinking_parts)[:thinking_max_chars] if thinking_parts else None
        text_raw = "\n".join(text_parts)[:text_max_chars] if text_parts else None

        plan_paragraphs: List[str] = []
        reflection_paragraphs: List[str] = []
        remainder_paragraphs: List[str] = []

        if text_raw:
            paragraphs = [p.strip() for p in text_raw.split("\n\n") if p.strip()]

            for para in paragraphs:
                is_plan = any(pat.search(para) for pat in ClaudeSessionImporter._PLAN_PATTERNS)
                is_reflection = any(pat.search(para) for pat in ClaudeSessionImporter._REFLECTION_PATTERNS)

                if is_plan:
                    plan_paragraphs.append(para)
                if is_reflection:
                    reflection_paragraphs.append(para)
                if not is_plan and not is_reflection:
                    remainder_paragraphs.append(para)

        # Also check thinking blocks for structured patterns
        if thinking_raw:
            thinking_paragraphs = [p.strip() for p in thinking_raw.split("\n\n") if p.strip()]
            for para in thinking_paragraphs:
                if not plan_paragraphs and any(pat.search(para) for pat in ClaudeSessionImporter._PLAN_PATTERNS):
                    plan_paragraphs.append(para)
                if not reflection_paragraphs and any(pat.search(para) for pat in ClaudeSessionImporter._REFLECTION_PATTERNS):
                    reflection_paragraphs.append(para)

        plan = "\n\n".join(plan_paragraphs) if plan_paragraphs else None
        reflection = "\n\n".join(reflection_paragraphs) if reflection_paragraphs else None
        # Decision rationale = text that didn't match plan/reflection patterns
        decision_rationale = "\n\n".join(remainder_paragraphs) if remainder_paragraphs else text_raw

        return CognitiveData(
            thinking=thinking_raw,
            plan=plan,
            reflection=reflection,
            decision_rationale=decision_rationale,
        )

    @staticmethod
    def _apply_tool_result(steps: list, tool_use_id: str, result_content, is_error: bool) -> None:
        """Match a tool_result to its corresponding step and update output/success."""
        for step in steps:
            if step.metadata.get("tool_use_id") == tool_use_id:
                if isinstance(result_content, str):
                    step.output = result_content[:10000]
                elif isinstance(result_content, list):
                    text_parts = []
                    for block in result_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    step.output = "\n".join(text_parts)[:10000]
                step.success = not is_error
                step.exit_code = 1 if is_error else 0
                break

    def parse_session_file(
        self,
        session_file: Path,
        thinking_max_chars: int = 10000,
        text_max_chars: int = 5000,
    ) -> Tuple[List[TraceStep], Dict[str, Any]]:
        """
        Parse a Claude Code session JSONL file and extract all available data.

        Captures tool executions, model/token usage, thinking blocks, all user
        messages, subagent metadata, git branch, and session-level aggregates.

        Args:
            session_file: Path to the .jsonl session file
            thinking_max_chars: Max chars to keep from each thinking block
            text_max_chars: Max chars to keep from each assistant text block

        Returns:
            Tuple of (steps, session_metadata_dict)
        """
        steps: List[TraceStep] = []
        session_id = session_file.stem
        cwd: Optional[str] = None

        # Session-level accumulators
        models_used: Set[str] = set()
        tools_used: Set[str] = set()
        meta: Dict[str, Any] = {
            "user_initial_prompt": None,
            "all_user_prompts": [],
            "plan_contents": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_cache_read_tokens": 0,
            "conversation_turns": 0,
            "thinking_block_count": 0,
            "total_tool_calls": 0,
            "subagent_count": 0,
            "subagent_total_tokens": 0,
            "subagent_total_duration_ms": 0,
            "git_branch": None,
            "claude_version": None,
            "session_slug": None,
            "session_id_original": None,
        }

        try:
            # Thinking/text blocks often arrive in separate assistant events
            # from the tool_use that follows. Accumulate across events until
            # a tool_use consumes them.
            pending_thinking: List[str] = []
            pending_text: List[str] = []

            with open(session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # --- Session-level fields (first occurrence wins) ---
                    if "cwd" in event:
                        cwd = event["cwd"]
                    if meta["git_branch"] is None and "gitBranch" in event:
                        meta["git_branch"] = event["gitBranch"]
                    if meta["claude_version"] is None and "version" in event:
                        meta["claude_version"] = event["version"]
                    if meta["session_slug"] is None and "slug" in event:
                        meta["session_slug"] = event["slug"]
                    if meta["session_id_original"] is None and "sessionId" in event:
                        meta["session_id_original"] = event["sessionId"]

                    event_type = event.get("type")

                    # ===================================================
                    # ASSISTANT messages: model, usage, thinking, text, tool_use
                    # ===================================================
                    if event_type == "assistant":
                        msg = event.get("message", {})
                        model = msg.get("model", "")
                        usage = msg.get("usage", {})
                        timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())

                        # Accumulate model & token usage
                        if model:
                            models_used.add(model)
                        meta["total_input_tokens"] += usage.get("input_tokens", 0)
                        meta["total_output_tokens"] += usage.get("output_tokens", 0)
                        meta["total_cache_creation_tokens"] += usage.get("cache_creation_input_tokens", 0)
                        meta["total_cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)

                        # Walk content blocks: thinking/text precede tool_use
                        content = msg.get("content", [])
                        if not isinstance(content, list):
                            continue

                        for item in content:
                            block_type = item.get("type") if isinstance(item, dict) else None

                            if block_type == "thinking":
                                meta["thinking_block_count"] += 1
                                thinking_str = item.get("thinking", "")
                                if thinking_str:
                                    pending_thinking.append(thinking_str[:thinking_max_chars])

                            elif block_type == "text":
                                text_str = item.get("text", "")
                                if text_str:
                                    pending_text.append(text_str[:text_max_chars])

                            elif block_type == "tool_use":
                                tool_name = item.get("name", "")
                                tool_input = item.get("input", {})
                                tool_id = item.get("id", "")

                                meta["total_tool_calls"] += 1
                                tools_used.add(tool_name)

                                if tool_name not in self.EXCLUDED_TOOLS:
                                    # Extract structured cognitive data
                                    cognitive = self._extract_cognitive(
                                        pending_thinking, pending_text,
                                        thinking_max_chars, text_max_chars,
                                    )
                                    cognitive_dict = cognitive.to_dict() if not cognitive.is_empty() else None

                                    step = TraceStep(
                                        step_id=f"{session_id}_{tool_id}",
                                        timestamp=timestamp,
                                        tool_name=tool_name,
                                        command=json.dumps(tool_input),
                                        output="",
                                        exit_code=None,
                                        success=None,
                                        cwd=cwd or "",
                                        repo={"path": cwd, "name": Path(cwd).name if cwd else "unknown"},
                                        source_tool="claude_code",
                                        cognitive=cognitive_dict,
                                        metadata={
                                            "tool_use_id": tool_id,
                                            "session_id": session_id,
                                            "imported_from": str(session_file),
                                            "model": model,
                                            "input_tokens": usage.get("input_tokens", 0),
                                            "output_tokens": usage.get("output_tokens", 0),
                                            "thinking_content": "\n".join(pending_thinking) if pending_thinking else None,
                                            "assistant_text": "\n".join(pending_text) if pending_text else None,
                                            "cognitive": cognitive_dict,
                                            "stop_reason": msg.get("stop_reason"),
                                        }
                                    )
                                    steps.append(step)

                                # Reset pending blocks after associating with this tool_use
                                pending_thinking = []
                                pending_text = []

                    # ===================================================
                    # USER messages: prompts, tool_result, toolUseResult, planContent
                    # ===================================================
                    elif event_type == "user":
                        message = event.get("message", {})
                        content = message.get("content", [])

                        # Extract user prompt text
                        user_text = self._extract_text_from_content(content)
                        if user_text:
                            meta["all_user_prompts"].append({
                                "text": user_text[:2000],
                                "timestamp": event.get("timestamp"),
                            })
                            if meta["user_initial_prompt"] is None:
                                meta["user_initial_prompt"] = user_text[:500]
                            meta["conversation_turns"] += 1

                        # Extract planContent
                        plan = event.get("planContent")
                        if plan:
                            meta["plan_contents"].append(str(plan)[:2000])

                        # Extract subagent metadata from toolUseResult
                        tur = event.get("toolUseResult")
                        if isinstance(tur, dict) and "totalTokens" in tur:
                            meta["subagent_count"] += 1
                            meta["subagent_total_tokens"] += tur.get("totalTokens", 0)
                            meta["subagent_total_duration_ms"] += tur.get("totalDurationMs", 0)

                        # Match tool_result blocks back to steps
                        for item in content if isinstance(content, list) else []:
                            if isinstance(item, dict) and item.get("type") == "tool_result":
                                self._apply_tool_result(
                                    steps,
                                    item.get("tool_use_id", ""),
                                    item.get("content", ""),
                                    item.get("is_error", False),
                                )

                    # ===================================================
                    # PROGRESS events: agent_progress tool results
                    # ===================================================
                    elif event_type == "progress":
                        data = event.get("data", {})
                        if data.get("type") == "agent_progress":
                            msg_content = data.get("message", {}).get("message", {}).get("content", [])
                            for item in msg_content if isinstance(msg_content, list) else []:
                                if isinstance(item, dict) and item.get("type") == "tool_result":
                                    self._apply_tool_result(
                                        steps,
                                        item.get("tool_use_id", ""),
                                        item.get("content", ""),
                                        item.get("is_error", False),
                                    )

        except IOError as e:
            print(f"Error reading session file {session_file}: {e}")
            return [], {"user_initial_prompt": None}

        # Finalize session metadata
        meta["models_used"] = sorted(models_used)
        meta["tools_used"] = sorted(tools_used)
        meta["model"] = meta["models_used"][0] if meta["models_used"] else ""
        meta["api_equivalent_cost_usd"] = estimate_cost_usd(
            meta["model"],
            meta["total_input_tokens"],
            meta["total_output_tokens"],
            meta["total_cache_creation_tokens"],
            meta["total_cache_read_tokens"],
        )

        return steps, meta

    def import_session(
        self,
        session_file: Path,
        force: bool = False
    ) -> ImportResult:
        """
        Import a single session file into BashGym format.

        Args:
            session_file: Path to the .jsonl session file
            force: Import even if already imported

        Returns:
            ImportResult with import details
        """
        session_id = session_file.stem

        # Check if already imported
        if not force and session_id in self.imported_sessions:
            return ImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="Already imported"
            )

        # Parse the session
        steps, session_meta = self.parse_session_file(session_file)

        if not steps:
            return ImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="No relevant tool executions found"
            )

        # Extract user prompt, spread remaining metadata into session
        user_initial_prompt = session_meta.pop("user_initial_prompt", None) or "Imported session"
        session = TraceSession.from_steps(
            steps,
            source_tool="claude_code",
            verification_passed=None,  # Unknown until actually verified
            imported=True,
            import_source=str(session_file),
            user_initial_prompt=user_initial_prompt,
            **session_meta
        )

        # Save to traces directory
        mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
        timestamp = mtime.strftime("%Y%m%d_%H%M%S")
        filename = f"imported_claude_{session_id[:8]}_{timestamp}.json"
        destination = self.trace_capture.traces_dir / filename

        try:
            with open(destination, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
        except IOError as e:
            return ImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                error=f"Failed to write trace: {e}"
            )

        # Mark as imported
        self._save_imported_session(session_id)

        return ImportResult(
            session_id=session_id,
            source_file=session_file,
            steps_imported=len(steps),
            destination_file=destination
        )

    async def import_session_async(
        self,
        session_file: Path,
        force: bool = False,
        instrumentation: Optional["Instrumentation"] = None
    ) -> ImportResult:
        """
        Import a single session file with instrumentation (async).

        Applies PII filtering to user prompts and tool outputs,
        checks for injection attempts, and profiles the import.

        Args:
            session_file: Path to the .jsonl session file
            force: Import even if already imported
            instrumentation: Optional Instrumentation instance (uses global if None)

        Returns:
            ImportResult with import details and instrumentation stats
        """
        session_id = session_file.stem

        # Get instrumentation instance
        inst = instrumentation
        if inst is None and HAS_INSTRUMENTATION:
            inst = get_instrumentation()

        # Start profiling trace for this import
        trace_id = ""
        if inst and inst.profiler_enabled:
            trace_id = inst.start_trace(
                f"import:{session_id[:8]}",
                metadata={"source_file": str(session_file)}
            )

        pii_redactions = 0
        injection_blocked = False

        try:
            # Check if already imported
            if not force and session_id in self.imported_sessions:
                return ImportResult(
                    session_id=session_id,
                    source_file=session_file,
                    steps_imported=0,
                    skipped=True,
                    skip_reason="Already imported"
                )

            # Parse the session
            steps, session_meta = self.parse_session_file(session_file)

            if not steps:
                return ImportResult(
                    session_id=session_id,
                    source_file=session_file,
                    steps_imported=0,
                    skipped=True,
                    skip_reason="No relevant tool executions found"
                )

            # Extract user prompt for injection/PII checks
            user_initial_prompt = session_meta.pop("user_initial_prompt", None)
            sanitized_prompt = user_initial_prompt

            # Apply instrumentation if available
            if inst:
                # Check user prompt for injection
                if user_initial_prompt:
                    is_safe = await inst.check_injection(
                        user_initial_prompt,
                        location="import.user_prompt"
                    )
                    if not is_safe:
                        injection_blocked = True
                        sanitized_prompt = "[INJECTION_DETECTED] " + user_initial_prompt

                    # Filter PII from user prompt
                    original_prompt = sanitized_prompt
                    sanitized_prompt = await inst.filter_pii(
                        sanitized_prompt,
                        location="import.user_prompt"
                    )
                    if sanitized_prompt != original_prompt:
                        pii_redactions += 1

                # Filter PII from all_user_prompts entries
                for prompt_entry in session_meta.get("all_user_prompts", []):
                    original_text = prompt_entry["text"]
                    prompt_entry["text"] = await inst.filter_pii(
                        original_text,
                        location="import.user_prompt"
                    )
                    if prompt_entry["text"] != original_text:
                        pii_redactions += 1

                # Filter PII from tool outputs
                for step in steps:
                    if step.output:
                        original_output = step.output
                        step.output = await inst.filter_pii(
                            step.output,
                            location="import.tool_output"
                        )
                        if step.output != original_output:
                            pii_redactions += 1

                    # Also check command inputs for PII
                    if step.command:
                        original_command = step.command
                        step.command = await inst.filter_pii(
                            step.command,
                            location="import.tool_command"
                        )
                        if step.command != original_command:
                            pii_redactions += 1

            # Create trace session with all enriched metadata
            session = TraceSession.from_steps(
                steps,
                source_tool="claude_code",
                verification_passed=None,  # Unknown until actually verified
                imported=True,
                import_source=str(session_file),
                user_initial_prompt=sanitized_prompt or "Imported session",
                **session_meta
            )

            # Add instrumentation metadata
            session.metadata["pii_redactions"] = pii_redactions
            session.metadata["injection_blocked"] = injection_blocked

            # Save to traces directory
            mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
            timestamp = mtime.strftime("%Y%m%d_%H%M%S")
            filename = f"imported_claude_{session_id[:8]}_{timestamp}.json"
            destination = self.trace_capture.traces_dir / filename

            try:
                with open(destination, 'w', encoding='utf-8') as f:
                    json.dump(asdict(session), f, indent=2, ensure_ascii=False)
            except IOError as e:
                return ImportResult(
                    session_id=session_id,
                    source_file=session_file,
                    steps_imported=0,
                    error=f"Failed to write trace: {e}",
                    pii_redactions=pii_redactions,
                    injection_blocked=injection_blocked
                )

            # Mark as imported
            self._save_imported_session(session_id)

            return ImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=len(steps),
                destination_file=destination,
                pii_redactions=pii_redactions,
                injection_blocked=injection_blocked,
                sanitized_prompt=sanitized_prompt
            )

        finally:
            # End profiling trace
            if inst and trace_id:
                inst.end_trace(trace_id)


async def import_today_async(
    project_filter: Optional[str] = None,
    verbose: bool = True
) -> List[ImportResult]:
    """
    Import today's Claude Code sessions with instrumentation (async).

    Args:
        project_filter: Only import from projects matching this substring
        verbose: Print progress messages

    Returns:
        List of ImportResult objects
    """
    importer = ClaudeSessionImporter()

    # Get start of today (local time)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_utc = today.astimezone(timezone.utc)

    session_files = importer.find_session_files(
        project_filter=project_filter,
        since=today_utc
    )

    if verbose:
        print(f"[BashGym] Found {len(session_files)} session(s) from today")

    results = []
    total_pii = 0
    total_injections = 0

    for session_file, mtime in session_files:
        result = await importer.import_session_async(session_file)
        results.append(result)
        total_pii += result.pii_redactions
        if result.injection_blocked:
            total_injections += 1

        if verbose:
            if result.skipped:
                print(f"  [-] {session_file.name}: {result.skip_reason}")
            elif result.error:
                print(f"  [!] {session_file.name}: {result.error}")
            else:
                pii_note = f" (PII: {result.pii_redactions})" if result.pii_redactions else ""
                inj_note = " [INJECTION]" if result.injection_blocked else ""
                print(f"  [+] {session_file.name}: {result.steps_imported} steps{pii_note}{inj_note}")

    if verbose and (total_pii or total_injections):
        print(f"[BashGym] Instrumentation: {total_pii} PII redactions, {total_injections} injection flags")

    return results


def import_today(
    project_filter: Optional[str] = None,
    verbose: bool = True
) -> List[ImportResult]:
    """
    Import today's Claude Code sessions.

    Args:
        project_filter: Only import from projects matching this substring
        verbose: Print progress messages

    Returns:
        List of ImportResult objects
    """
    importer = ClaudeSessionImporter()

    # Get start of today (local time)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_utc = today.astimezone(timezone.utc)

    session_files = importer.find_session_files(
        project_filter=project_filter,
        since=today_utc
    )

    if verbose:
        print(f"[BashGym] Found {len(session_files)} session(s) from today")

    results = []
    for session_file, mtime in session_files:
        result = importer.import_session(session_file)
        results.append(result)

        if verbose:
            if result.skipped:
                print(f"  [-] {session_file.name}: {result.skip_reason}")
            elif result.error:
                print(f"  [!] {session_file.name}: {result.error}")
            else:
                print(f"  [+] {session_file.name}: {result.steps_imported} steps imported")

    return results


def import_recent(
    days: int = 60,
    project_filter: Optional[str] = None,
    verbose: bool = True,
    force: bool = False
) -> List[ImportResult]:
    """
    Import recent Claude Code sessions.

    Args:
        days: Number of days to look back (default 60)
        project_filter: Only import from projects matching this substring
        verbose: Print progress messages
        force: Import even if already imported (overwrites existing traces)

    Returns:
        List of ImportResult objects
    """
    importer = ClaudeSessionImporter()

    # Calculate cutoff date
    since = datetime.now(timezone.utc) - timedelta(days=days)

    session_files = importer.find_session_files(
        project_filter=project_filter,
        since=since
    )

    if verbose:
        print(f"[BashGym] Found {len(session_files)} session(s) from last {days} days")

    results = []
    for session_file, mtime in session_files:
        result = importer.import_session(session_file, force=force)
        results.append(result)

        if verbose:
            if result.skipped:
                print(f"  [-] {session_file.name}: {result.skip_reason}")
            elif result.error:
                print(f"  [!] {session_file.name}: {result.error}")
            else:
                print(f"  [+] {session_file.name}: {result.steps_imported} steps imported")

    return results


def import_session(
    session_path: str,
    force: bool = False,
    verbose: bool = True
) -> ImportResult:
    """
    Import a specific session file.

    Args:
        session_path: Path to the session .jsonl file
        force: Import even if already imported
        verbose: Print progress messages

    Returns:
        ImportResult object
    """
    importer = ClaudeSessionImporter()
    session_file = Path(session_path)

    if not session_file.exists():
        return ImportResult(
            session_id="unknown",
            source_file=session_file,
            steps_imported=0,
            error=f"File not found: {session_path}"
        )

    result = importer.import_session(session_file, force=force)

    if verbose:
        if result.skipped:
            print(f"[BashGym] [-] Skipped: {result.skip_reason}")
        elif result.error:
            print(f"[BashGym] [!] Error: {result.error}")
        else:
            print(f"[BashGym] [+] Imported {result.steps_imported} steps to {result.destination_file}")

    return result
