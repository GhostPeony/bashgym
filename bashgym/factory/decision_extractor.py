"""
Decision Extractor for The Factory Layer

Extracts structured decisions from trace steps and cognitive data.
Enables better DPO training pairs (step-level, not whole-trace-level)
and failure diagnosis by surfacing WHY an agent chose a particular action.

Module 3: Data Synthesis (The "Factory")
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime


@dataclass
class Decision:
    """A structured decision extracted from a trace step.

    Captures the intent, alternatives considered, the chosen action,
    reasoning behind it, and whether it succeeded -- enabling
    fine-grained DPO pair generation at the step level.
    """

    step_index: int
    intent: str                          # What the agent was trying to do
    options_considered: List[str]         # Alternative approaches mentioned in thinking
    chosen: str                          # What was actually done
    reasoning: str                       # Why this option (from <thinking> blocks)
    outcome: str                         # SUCCESS, FAILURE, PARTIAL
    tool_used: str                       # Which tool was called
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        d: Dict[str, Any] = {
            "step_index": self.step_index,
            "intent": self.intent,
            "options_considered": self.options_considered,
            "chosen": self.chosen,
            "reasoning": self.reasoning,
            "outcome": self.outcome,
            "tool_used": self.tool_used,
        }
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp.isoformat()
        return d


# ---------------------------------------------------------------------------
# Patterns used by extraction heuristics
# ---------------------------------------------------------------------------

# Phrases that signal the agent considered alternatives
_ALTERNATIVE_PHRASES = [
    r"(?:could|can|might|should)\s+(?:also|instead|alternatively)",
    r"(?:another|different|alternative)\s+(?:approach|option|way|method|strategy)",
    r"(?:option\s+\d|choice\s+\d)",
    r"(?:instead\s+of|rather\s+than)",
    r"let me (?:try|consider|think about)",
    r"(?:first|second|third) option",
    r"(?:plan [A-C]|approach [A-C])",
]
_ALTERNATIVES_RE = re.compile("|".join(_ALTERNATIVE_PHRASES), re.IGNORECASE)

# Phrases that signal a pivot in strategy
_PIVOT_PHRASES = [
    r"let me try a different",
    r"that didn't work",
    r"let me change approach",
    r"going to try another",
    r"instead,?\s+(?:I'll|let me|I will)",
    r"new approach",
    r"pivot(?:ing)? to",
    r"switching to",
    r"scratch that",
    r"actually,?\s+(?:let me|I'll|I should)",
]
_PIVOT_RE = re.compile("|".join(_PIVOT_PHRASES), re.IGNORECASE)

# Error / failure indicators in step output
_ERROR_INDICATORS = [
    r"(?:error|Error|ERROR)[\s:]",
    r"(?:traceback|Traceback)",
    r"(?:failed|FAILED|Failed)",
    r"(?:No such file|not found|command not found)",
    r"(?:Permission denied|permission denied)",
    r"(?:syntax error|SyntaxError)",
    r"(?:ModuleNotFoundError|ImportError)",
    r"(?:FileNotFoundError|OSError)",
    r"fatal:",
    r"ENOENT",
]
_ERROR_RE = re.compile("|".join(_ERROR_INDICATORS))


class DecisionExtractor:
    """Extract structured decisions from trace steps + cognitive data.

    Heuristics:
    1. Tool selection: when <thinking> mentions alternatives before a tool call
    2. Approach pivots: when agent changes strategy (file/dir change or explicit reasoning)
    3. Error recovery: when a step fails and next step shows adaptation
    4. Commit points: git commits represent task-completion decisions
    """

    def extract(
        self,
        steps: List[Dict[str, Any]],
        cognitive_data: Optional[Dict[str, Any]] = None,
    ) -> List[Decision]:
        """Main extraction method.

        Args:
            steps: Normalized trace steps (as produced by TraceProcessor._normalize_steps).
                   Each step has keys: tool, command, output, success, exit_code, cognitive, metadata.
            cognitive_data: Optional session-level cognitive data dict.

        Returns:
            List of Decision objects extracted from the steps.
        """
        if not steps:
            return []

        decisions: List[Decision] = []

        for i, step in enumerate(steps):
            # Gather cognitive text for this step
            thinking_text = self._gather_thinking(step, cognitive_data)

            # 1. Check if thinking mentions alternatives (deliberate tool selection)
            has_alternatives = bool(thinking_text and _ALTERNATIVES_RE.search(thinking_text))

            # 2. Approach pivot detection
            is_pivot = False
            if i > 0:
                is_pivot = self._is_approach_pivot(steps[i - 1], step)

            # 3. Error recovery detection
            is_recovery = False
            if i > 0:
                is_recovery = self._is_error_recovery(steps[i - 1], step)

            # 4. Commit point detection
            is_commit = self._is_commit_point(step)

            # Only produce a Decision when at least one heuristic fires
            if not (has_alternatives or is_pivot or is_recovery or is_commit):
                continue

            intent, options, reasoning = self._extract_from_thinking(thinking_text)

            # If we still have no intent, infer one from context
            if not intent:
                intent = self._infer_intent(step, is_pivot, is_recovery, is_commit)

            outcome = self._determine_outcome(i, steps)

            tool_used = step.get("tool", step.get("tool_name", "unknown"))
            command = step.get("command", step.get("input", ""))

            # Parse timestamp if available
            ts = self._parse_timestamp(step)

            decisions.append(Decision(
                step_index=i,
                intent=intent,
                options_considered=options,
                chosen=command[:500],  # Keep manageable size
                reasoning=reasoning,
                outcome=outcome,
                tool_used=tool_used,
                timestamp=ts,
            ))

        return decisions

    # ------------------------------------------------------------------
    # Thinking / cognitive extraction
    # ------------------------------------------------------------------

    def _gather_thinking(
        self,
        step: Dict[str, Any],
        session_cognitive: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Combine all available cognitive text for a step into one string."""
        parts: List[str] = []

        # Step-level cognitive data (set by _normalize_steps)
        cognitive = step.get("cognitive") or {}
        if isinstance(cognitive, dict):
            for key in ("thinking", "plan", "reflection", "decision_rationale"):
                val = cognitive.get(key)
                if val:
                    parts.append(str(val))

        # Metadata may also carry cognitive fields (legacy format)
        meta = step.get("metadata", {}) or {}
        if isinstance(meta, dict):
            inner_cog = meta.get("cognitive", {}) or {}
            if isinstance(inner_cog, dict):
                for key in ("thinking", "plan", "reflection", "decision_rationale"):
                    val = inner_cog.get(key)
                    if val and str(val) not in parts:
                        parts.append(str(val))

            # Legacy flat keys
            for key in ("thinking_content", "assistant_text"):
                val = meta.get(key)
                if val and str(val) not in parts:
                    parts.append(str(val))

        return "\n\n".join(parts)

    def _extract_from_thinking(self, thinking_text: str) -> Tuple[str, List[str], str]:
        """Parse thinking text to extract intent, options considered, and reasoning.

        Returns:
            (intent, options_considered, reasoning)
        """
        if not thinking_text:
            return ("", [], "")

        intent = ""
        options: List[str] = []
        reasoning = ""

        # --- Intent ---
        # Look for explicit intent markers: "I need to ...", "The goal is ...", "I want to ..."
        intent_patterns = [
            r"(?:I need to|I should|I want to|I'll|Let me|Going to|The goal is to)\s+(.+?)(?:\.|$)",
            r"(?:Task|Goal|Objective):\s*(.+?)(?:\n|$)",
        ]
        for pat in intent_patterns:
            m = re.search(pat, thinking_text, re.IGNORECASE | re.MULTILINE)
            if m:
                intent = m.group(1).strip()[:200]
                break

        # If no explicit intent, use the first sentence as a proxy
        if not intent:
            first_sentence = re.split(r'[.\n]', thinking_text.strip())[0].strip()
            if first_sentence:
                intent = first_sentence[:200]

        # --- Options considered ---
        # Extract sentences that mention alternatives
        sentences = re.split(r'(?<=[.!?])\s+', thinking_text)
        for sentence in sentences:
            if _ALTERNATIVES_RE.search(sentence):
                options.append(sentence.strip()[:300])

        # Also look for numbered options
        numbered = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-*]\s+)(.+)', thinking_text)
        if len(numbered) >= 2:
            # Looks like a list of options
            for item in numbered:
                clean = item.strip()[:300]
                if clean and clean not in options:
                    options.append(clean)

        # --- Reasoning ---
        # Look for "because", "since", "reason" clauses
        reason_patterns = [
            r"(?:because|since|the reason is|this is because)\s+(.+?)(?:\.|$)",
            r"(?:I chose|I picked|I went with|choosing)\s+.+?\s+(?:because|since|as)\s+(.+?)(?:\.|$)",
        ]
        for pat in reason_patterns:
            m = re.search(pat, thinking_text, re.IGNORECASE | re.MULTILINE)
            if m:
                reasoning = m.group(1).strip()[:500]
                break

        # Fallback: use the full thinking text (truncated) as reasoning
        if not reasoning:
            reasoning = thinking_text[:500]

        return (intent, options, reasoning)

    # ------------------------------------------------------------------
    # Outcome determination
    # ------------------------------------------------------------------

    def _determine_outcome(self, step_index: int, steps: List[Dict[str, Any]]) -> str:
        """Look at the current and subsequent steps to determine if this decision succeeded.

        - If this step itself failed (success=False or error in output) -> FAILURE
        - If next step is an error/retry of the same kind -> FAILURE
        - If the task completes soon after (git commit, final step, success run) -> SUCCESS
        - Otherwise -> PARTIAL
        """
        current = steps[step_index]

        # Check current step for explicit failure
        if current.get("success") is False:
            return "FAILURE"
        if current.get("exit_code") is not None and current.get("exit_code") != 0:
            return "FAILURE"

        output = str(current.get("output", ""))
        if _ERROR_RE.search(output):
            return "FAILURE"

        # Look ahead (up to 3 steps)
        lookahead = min(step_index + 4, len(steps))

        for j in range(step_index + 1, lookahead):
            future = steps[j]

            # If the next step is a retry / error recovery of the same tool -> this was a FAILURE
            if j == step_index + 1:
                future_output = str(future.get("output", ""))
                if future.get("success") is False or _ERROR_RE.search(future_output):
                    # The immediate next step also failed -- this is just cascading
                    pass
                elif self._is_error_recovery(current, future):
                    return "FAILURE"

            # If we see a git commit soon, the overall approach succeeded
            future_cmd = str(future.get("command", ""))
            if "git commit" in future_cmd or "git push" in future_cmd:
                return "SUCCESS"

        # If this is one of the last few steps and didn't fail, call it SUCCESS
        if step_index >= len(steps) - 3 and current.get("success") is not False:
            return "SUCCESS"

        return "PARTIAL"

    # ------------------------------------------------------------------
    # Pivot and recovery detection
    # ------------------------------------------------------------------

    def _is_approach_pivot(self, prev_step: Dict[str, Any], curr_step: Dict[str, Any]) -> bool:
        """Detect strategy changes between consecutive steps.

        Signals:
        - Different working directory (cwd change)
        - Different file being operated on (different file path in command)
        - Explicit pivot language in cognitive data
        - Switching tool types (e.g., from Read to Bash)
        """
        # Check cognitive text for pivot language
        thinking = self._gather_thinking(curr_step)
        if thinking and _PIVOT_RE.search(thinking):
            return True

        # Directory change heuristic
        prev_meta = prev_step.get("metadata", {}) or {}
        curr_meta = curr_step.get("metadata", {}) or {}
        prev_cwd = prev_meta.get("cwd", "")
        curr_cwd = curr_meta.get("cwd", "")
        if prev_cwd and curr_cwd and prev_cwd != curr_cwd:
            return True

        # File path change: extract leading file paths from commands
        prev_file = self._extract_file_path(prev_step.get("command", ""))
        curr_file = self._extract_file_path(curr_step.get("command", ""))
        prev_dir = self._get_directory(prev_file)
        curr_dir = self._get_directory(curr_file)
        if prev_dir and curr_dir and prev_dir != curr_dir:
            return True

        return False

    def _is_error_recovery(self, prev_step: Dict[str, Any], curr_step: Dict[str, Any]) -> bool:
        """Detect when previous step failed and current step adapts.

        True when:
        - prev_step has success=False or error output
        - curr_step is a different command (not just a repeat)
        """
        # Previous must have failed
        prev_failed = False
        if prev_step.get("success") is False:
            prev_failed = True
        elif prev_step.get("exit_code") is not None and prev_step.get("exit_code") != 0:
            prev_failed = True
        else:
            prev_output = str(prev_step.get("output", ""))
            if _ERROR_RE.search(prev_output):
                prev_failed = True

        if not prev_failed:
            return False

        # Current must be a different command (adaptation, not identical retry)
        prev_cmd = prev_step.get("command", "")
        curr_cmd = curr_step.get("command", "")
        if prev_cmd and curr_cmd and prev_cmd.strip() == curr_cmd.strip():
            return False  # exact retry, not adaptation

        return True

    def _is_commit_point(self, step: Dict[str, Any]) -> bool:
        """Detect git commit steps -- these represent task-completion decisions."""
        cmd = str(step.get("command", ""))
        tool = str(step.get("tool", step.get("tool_name", "")))
        return tool.lower() == "bash" and "git commit" in cmd

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_intent(
        self,
        step: Dict[str, Any],
        is_pivot: bool,
        is_recovery: bool,
        is_commit: bool,
    ) -> str:
        """Infer a human-readable intent when thinking blocks don't provide one."""
        tool = step.get("tool", step.get("tool_name", "unknown"))
        cmd = step.get("command", "")[:100]

        if is_commit:
            return f"Commit changes: {cmd}"
        if is_recovery:
            return f"Recover from error using {tool}: {cmd}"
        if is_pivot:
            return f"Change approach using {tool}: {cmd}"
        return f"Execute {tool}: {cmd}"

    @staticmethod
    def _extract_file_path(command: str) -> str:
        """Best-effort extraction of a file path from a command string."""
        if not command:
            return ""
        # Match paths that look like /foo/bar.py or foo/bar.py or C:\foo\bar
        m = re.search(r'(?:[A-Za-z]:\\|/)?[\w./\\-]+\.[\w]+', command)
        return m.group(0) if m else ""

    @staticmethod
    def _get_directory(path: str) -> str:
        """Get the directory portion of a path string."""
        if not path:
            return ""
        # Handle both / and \ separators
        for sep in ("/", "\\"):
            idx = path.rfind(sep)
            if idx > 0:
                return path[:idx]
        return ""

    @staticmethod
    def _parse_timestamp(step: Dict[str, Any]) -> Optional[datetime]:
        """Try to parse a timestamp from step metadata."""
        raw = step.get("timestamp") or (step.get("metadata", {}) or {}).get("timestamp")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except (ValueError, TypeError):
            return None
