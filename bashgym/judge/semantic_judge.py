"""
Semantic Judge — LLM-based quality evaluation for traces.

Goes beyond test pass/fail to evaluate solution QUALITY:
- Is the solution approach idiomatic or over-engineered?
- Are the trade-off decisions sound?
- Is the code clean and maintainable?
- Does the solution align with the original intent?

Uses a three-tier evaluation model (structural -> semantic -> goal-level).
This is the semantic tier.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bashgym.factory.trace_processor import ProcessedTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SemanticVerdict:
    """Result of a semantic quality evaluation."""

    trace_id: str
    score: float              # 0.0-1.0 overall quality
    confidence: float         # 0.0-1.0 how sure the judge is
    quality_flags: List[str]  # e.g. "clean", "over-engineered", "hacky", "elegant", "idiomatic"
    issues: List[str]         # specific problems found
    reasoning: str            # judge's explanation
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "score": self.score,
            "confidence": self.confidence,
            "quality_flags": self.quality_flags,
            "issues": self.issues,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# The judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a senior code reviewer evaluating an AI coding agent's trace.
You assess QUALITY, not just correctness. A solution can pass all tests yet still be hacky.

Evaluate on four axes:
1. Solution Approach — Is it idiomatic? Minimal? Over-engineered? Hacky?
2. Decision Quality — Did the agent make good trade-offs? Recover well from errors?
3. Code Quality — Is the written code clean, readable, maintainable?
4. Task Alignment — Does the work match the original intent, or did it drift?

Respond with ONLY a JSON object (no markdown fences, no commentary):
{
  "score": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "quality_flags": [<strings from: "clean", "elegant", "idiomatic", "minimal", "well-reasoned", "over-engineered", "hacky", "brittle", "verbose", "off-target", "incomplete">],
  "issues": [<specific problems, empty list if none>],
  "reasoning": "<1-3 sentence explanation>"
}"""

_JUDGE_USER_TEMPLATE = """\
## Task Description
{task_description}

## Agent Steps (last {step_count} significant)
{steps_text}

## Decision Log
{decision_log}

Rate the quality of this agent's work."""


# ---------------------------------------------------------------------------
# SemanticJudge
# ---------------------------------------------------------------------------

class SemanticJudge:
    """LLM-based quality evaluation for traces.

    Evaluates:
    1. Solution approach (idiomatic? over-engineered? hacky?)
    2. Decision quality (good trade-offs?)
    3. Code quality (clean code written?)
    4. Task alignment (matches original intent?)

    Uses Haiku by default for cost efficiency. Fully optional — when disabled
    or when the API call fails, the rest of the pipeline continues unaffected.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        enabled: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.enabled = enabled
        self._client = None  # lazy init

    def _get_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic()
            except ImportError:
                logger.warning("anthropic package not installed — semantic judge disabled")
                self._client = None
            except Exception as exc:
                logger.warning("Failed to initialize Anthropic client: %s", exc)
                self._client = None
        return self._client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def evaluate(self, trace: "ProcessedTrace") -> SemanticVerdict:
        """Evaluate a ProcessedTrace using LLM judge.

        On any failure (API error, malformed response, disabled judge),
        returns a neutral verdict (score=0.5, confidence=0.0) so the
        pipeline can continue with its existing classification.
        """
        if not self.enabled:
            return self._neutral_verdict(trace.trace_id, "Judge disabled")

        client = self._get_client()
        if client is None:
            return self._neutral_verdict(trace.trace_id, "No API client available")

        prompt = self._build_prompt(trace)

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text from response
            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            return self._parse_response(trace.trace_id, response_text)

        except Exception as exc:
            logger.warning(
                "Semantic judge failed for trace %s: %s", trace.trace_id, exc
            )
            return self._neutral_verdict(trace.trace_id, f"API error: {exc}")

    async def evaluate_batch(
        self,
        traces: List["ProcessedTrace"],
        concurrency: int = 5,
    ) -> List[SemanticVerdict]:
        """Batch evaluate with rate limiting via asyncio.Semaphore."""
        if not self.enabled or not traces:
            return [
                self._neutral_verdict(t.trace_id, "Judge disabled or empty batch")
                for t in traces
            ]

        semaphore = asyncio.Semaphore(concurrency)

        async def _eval_one(trace: "ProcessedTrace") -> SemanticVerdict:
            async with semaphore:
                return await self.evaluate(trace)

        return await asyncio.gather(*[_eval_one(t) for t in traces])

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #

    def _build_prompt(self, trace: "ProcessedTrace") -> str:
        """Build the judge prompt from trace data."""
        # Task description
        task_description = trace.task_prompt or "No task description available."
        if len(task_description) > 2000:
            task_description = task_description[:2000] + "... [truncated]"

        # Last N significant steps (skip trivial reads, keep tool calls with substance)
        significant_steps = self._select_significant_steps(
            trace.normalized_steps, max_steps=10
        )
        steps_text = self._format_steps(significant_steps)

        # Decision log
        if trace.decisions:
            decision_entries = []
            for d in trace.decisions[-8:]:  # last 8 decisions
                entry = f"- [{d.outcome}] {d.intent}"
                if d.reasoning:
                    entry += f" — {d.reasoning[:150]}"
                decision_entries.append(entry)
            decision_log = "\n".join(decision_entries)
        else:
            decision_log = "No decision log available."

        return _JUDGE_USER_TEMPLATE.format(
            task_description=task_description,
            step_count=len(significant_steps),
            steps_text=steps_text,
            decision_log=decision_log,
        )

    def _select_significant_steps(
        self,
        steps: List[Dict[str, Any]],
        max_steps: int = 10,
    ) -> List[Dict[str, Any]]:
        """Select the most significant steps for evaluation.

        Prioritizes:
        - Steps with errors (recovery is quality-relevant)
        - Write/Edit operations (code authorship)
        - Bash commands (actual execution)
        - Final steps (outcome)

        Deprioritizes:
        - Repeated Read calls
        - Glob/search calls
        """
        if len(steps) <= max_steps:
            return steps

        scored: List[tuple] = []
        for i, step in enumerate(steps):
            tool = str(step.get("tool", step.get("tool_name", ""))).lower()
            has_error = step.get("success") is False or step.get("exit_code", 0) != 0
            is_write = tool in ("write", "edit", "patch")
            is_bash = tool == "bash"
            is_read = tool in ("read", "glob", "grep", "search")

            # Position bonus: first and last steps matter more
            position_score = 0
            if i < 3:
                position_score = 2
            elif i >= len(steps) - 3:
                position_score = 3

            significance = position_score
            if has_error:
                significance += 4
            if is_write:
                significance += 3
            if is_bash:
                significance += 2
            if is_read:
                significance += 0  # least significant

            scored.append((significance, i, step))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:max_steps]
        # Restore original order
        selected.sort(key=lambda x: x[1])
        return [s[2] for s in selected]

    def _format_steps(self, steps: List[Dict[str, Any]]) -> str:
        """Format steps into a readable text block for the prompt."""
        if not steps:
            return "No steps available."

        lines = []
        for i, step in enumerate(steps):
            tool = step.get("tool", step.get("tool_name", "unknown"))
            command = step.get("command", step.get("input", ""))
            output = step.get("output", "")
            success = step.get("success", True)
            exit_code = step.get("exit_code", 0)

            # Truncate long values
            if isinstance(command, str) and len(command) > 500:
                command = command[:500] + "..."
            if isinstance(output, str) and len(output) > 300:
                output = output[:300] + "..."

            status = "OK" if success is not False and exit_code == 0 else "FAIL"
            lines.append(f"Step {i + 1} [{tool}] ({status})")
            if command:
                lines.append(f"  > {command}")
            if output:
                lines.append(f"  < {output}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    def _parse_response(self, trace_id: str, response_text: str) -> SemanticVerdict:
        """Parse LLM response into SemanticVerdict.

        Handles:
        - Clean JSON
        - JSON wrapped in markdown code fences
        - Malformed JSON (returns neutral verdict)
        """
        text = response_text.strip()

        # Strip markdown fences if present
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse judge response for trace %s: %.200s",
                trace_id,
                response_text,
            )
            return self._neutral_verdict(
                trace_id, f"Malformed response: {response_text[:200]}"
            )

        # Extract fields with validation
        score = self._clamp(float(data.get("score", 0.5)), 0.0, 1.0)
        confidence = self._clamp(float(data.get("confidence", 0.0)), 0.0, 1.0)

        quality_flags = data.get("quality_flags", [])
        if not isinstance(quality_flags, list):
            quality_flags = []
        quality_flags = [str(f) for f in quality_flags]

        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        issues = [str(i) for i in issues]

        reasoning = str(data.get("reasoning", ""))

        return SemanticVerdict(
            trace_id=trace_id,
            score=score,
            confidence=confidence,
            quality_flags=quality_flags,
            issues=issues,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _neutral_verdict(trace_id: str, reason: str) -> SemanticVerdict:
        """Return a neutral verdict that won't affect classification."""
        return SemanticVerdict(
            trace_id=trace_id,
            score=0.5,
            confidence=0.0,
            quality_flags=[],
            issues=[],
            reasoning=f"Neutral verdict: {reason}",
        )

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Clamp a value to [lo, hi]."""
        return max(lo, min(hi, value))
