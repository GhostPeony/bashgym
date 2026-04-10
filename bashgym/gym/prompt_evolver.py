"""
Dual-Loop Prompt Evolution — Fast-loop mutation of worker prompts.

The FAST loop in the dual-loop training architecture:
- FAST (this): Analyze recent traces -> extract failure patterns -> mutate prompts
                -> evaluate -> deploy best variant. Immediate effect, no training needed.
- SLOW (existing trainer): Accumulate traces -> generate examples -> fine-tune weights.

Applies graph-rewriting principles to BashGym's worker prompts.

The evolution cycle:
1. Load recent traces (gold + failed)
2. Extract failure/success patterns from decision logs (deterministic, no LLM)
3. Generate prompt variant that addresses failures while preserving successes (LLM)
4. Evaluate variant using semantic judge on held-out traces (LLM)
5. Keep if improved, discard if not
6. Repeat for N generations

Module: Gym (Training Layer)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class FailurePattern:
    """A recurring failure pattern extracted from traces.

    Identified deterministically from decision logs — no LLM call needed.
    """

    pattern_type: str  # "wrong_tool", "missing_context", "anti_pattern", "incomplete_output"
    description: str  # Human-readable description
    frequency: int  # How many traces exhibit this
    example_decisions: list[dict[str, Any]]  # Sample decisions showing the pattern
    suggested_fix: str  # What prompt change might help


@dataclass
class PromptVariant:
    """A candidate prompt mutation.

    Tracks lineage (parent_variant), generation depth, performance metrics,
    and which failure patterns it was designed to address.
    """

    variant_id: str
    system_prompt_patches: dict[str, str]  # section_name -> new content
    tool_config_patches: dict[str, Any]  # tool setting overrides
    parent_variant: str | None = None  # lineage tracking
    generation: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    failure_patterns_addressed: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON persistence."""
        return {
            "variant_id": self.variant_id,
            "system_prompt_patches": self.system_prompt_patches,
            "tool_config_patches": self.tool_config_patches,
            "parent_variant": self.parent_variant,
            "generation": self.generation,
            "metrics": self.metrics,
            "failure_patterns_addressed": self.failure_patterns_addressed,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVariant:
        """Deserialize from a plain dict."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                ts = datetime.utcnow()
        elif not isinstance(ts, datetime):
            ts = datetime.utcnow()

        return cls(
            variant_id=data.get("variant_id", ""),
            system_prompt_patches=data.get("system_prompt_patches", {}),
            tool_config_patches=data.get("tool_config_patches", {}),
            parent_variant=data.get("parent_variant"),
            generation=data.get("generation", 0),
            metrics=data.get("metrics", {}),
            failure_patterns_addressed=data.get("failure_patterns_addressed", []),
            timestamp=ts,
        )


# =============================================================================
# Constants
# =============================================================================

# Pattern type classification
PATTERN_WRONG_TOOL = "wrong_tool"
PATTERN_MISSING_CONTEXT = "missing_context"
PATTERN_ANTI_PATTERN = "anti_pattern"
PATTERN_INCOMPLETE_OUTPUT = "incomplete_output"

# Minimum frequency to surface a failure pattern
_MIN_PATTERN_FREQUENCY = 2

# Error indicators that suggest missing context
_MISSING_CONTEXT_INDICATORS = [
    r"(?:undefined|is not defined|NameError)",
    r"(?:No such file|FileNotFoundError|ENOENT)",
    r"(?:ModuleNotFoundError|ImportError)",
    r"(?:not found|command not found)",
    r"(?:wrong path|invalid path)",
]
_MISSING_CONTEXT_RE = re.compile("|".join(_MISSING_CONTEXT_INDICATORS), re.IGNORECASE)

# Anti-pattern indicators (over-engineering, unnecessary complexity)
_ANTI_PATTERN_INDICATORS = [
    r"(?:revert|undo|roll\s*back)",
    r"(?:actually,?\s+let me|scratch that|start over)",
    r"(?:too complex|over.?engineer|unnecessary)",
]
_ANTI_PATTERN_RE = re.compile("|".join(_ANTI_PATTERN_INDICATORS), re.IGNORECASE)


# =============================================================================
# LLM Prompts for Evolution
# =============================================================================

_EVOLVE_SYSTEM_PROMPT = """\
You are a prompt engineering specialist. Your job is to improve an AI coding agent's \
system prompt based on observed failure patterns while preserving behaviors that work well.

You will receive:
1. The current system prompt (baseline)
2. Failure patterns with concrete examples
3. Success patterns to preserve

Output ONLY a JSON object (no markdown fences, no commentary):
{
  "system_prompt_patches": {
    "<section_name>": "<text to append to that section>"
  },
  "tool_config_patches": {},
  "reasoning": "<1-3 sentences explaining the changes>"
}

Valid section names:
- "identity": patches the identity/role layer
- "narrative": patches the context/narrative layer
- "focus": patches the task-specific focus layer
- "rules": adds new rules/guidelines
- "anti_patterns": adds anti-pattern warnings

Keep patches concise (under 200 words each). Focus on the highest-frequency failures first. \
Do NOT remove or contradict existing prompt content — only ADD targeted guidance."""

_EVOLVE_USER_TEMPLATE = """\
## Current System Prompt
{current_prompt}

## Failure Patterns (sorted by frequency, most common first)
{failure_patterns}

## Success Patterns (DO NOT disrupt these)
{success_patterns}

Generate targeted prompt patches to address the top failure patterns while \
preserving the success patterns."""

_EVALUATE_SYSTEM_PROMPT = """\
You are evaluating whether a proposed prompt improvement would have helped an AI agent \
produce better solutions. Given the original trace and the proposed prompt additions, \
estimate the quality improvement.

Output ONLY a JSON object:
{
  "improvement_score": <float -1.0 to 1.0>,
  "reasoning": "<1-2 sentences>"
}

Positive score = the prompt change would have helped.
Negative score = the prompt change would have hurt.
Zero = no effect."""

_EVALUATE_USER_TEMPLATE = """\
## Proposed Prompt Additions
{prompt_patches}

## Trace Summary
Steps: {step_count}
Outcome: {outcome}
Decision Log:
{decision_log}

## Key Issues in This Trace
{issues}

Would the proposed prompt additions have improved this agent's performance?"""


# =============================================================================
# Training Trigger (Ouroboros Close)
# =============================================================================


@dataclass
class TrainingTriggerConfig:
    """Config for triggering micro-trains from prompt evolution."""

    enabled: bool = False
    gold_trace_threshold: int = 10  # Trigger after N new gold traces
    micro_train_steps: int = 50
    base_model: str = "unsloth/gemma-4-E4B-it"
    adapter_path: str = ""
    gold_traces_dir: str = ""


class TrainingTrigger:
    """Monitors gold trace count and triggers micro-trains.

    Closes the ouroboros loop: when the fast loop (prompt evolution) produces
    enough new gold traces, the slow loop (weight training) runs a micro-train
    and reports the delta loss back. Negative delta = model improved = the
    prompt evolution is working.
    """

    def __init__(self, config: TrainingTriggerConfig):
        self.config = config
        self._last_gold_count: int = 0
        self._last_loss: float | None = None
        self._initialized = False

    def _count_gold_traces(self) -> int:
        """Count gold trace files in the configured directory."""
        gold_dir = Path(self.config.gold_traces_dir)
        if not gold_dir.exists():
            return 0
        return len(list(gold_dir.glob("*.json")))

    def _initialize(self) -> None:
        """Record the initial gold trace count (first call only)."""
        if not self._initialized:
            self._last_gold_count = self._count_gold_traces()
            self._initialized = True
            logger.info(
                "[TrainingTrigger] Initialized: %d gold traces", self._last_gold_count
            )

    def check_and_train(self) -> float | None:
        """If enough new gold traces accumulated, run micro-train, return delta loss.

        Returns:
            None if no training was triggered.
            (new_loss - old_loss) if training happened. Negative = improvement.
        """
        self._initialize()

        current_count = self._count_gold_traces()
        new_traces = current_count - self._last_gold_count

        if new_traces < self.config.gold_trace_threshold:
            return None

        logger.info(
            "[TrainingTrigger] %d new gold traces (threshold=%d) — triggering micro-train",
            new_traces,
            self.config.gold_trace_threshold,
        )

        try:
            from bashgym.factory.example_generator import ExampleGenerator
            from bashgym.gym.trainer import Trainer, TrainerConfig

            gold_dir = Path(self.config.gold_traces_dir)

            # Generate examples from gold traces
            generator = ExampleGenerator()
            examples, _stats = generator.process_directory(gold_dir)

            if not examples:
                logger.warning("[TrainingTrigger] No examples generated")
                return None

            # Export train/val split to temp dir
            import tempfile

            with tempfile.TemporaryDirectory(prefix="ouroboros_") as tmpdir:
                result = generator.export_for_nemo(examples, Path(tmpdir), train_split=0.9)
                train_path = Path(result["train"])
                val_path = Path(result["validation"]) if result.get("validation") else None

                # Micro-train
                trainer_config = TrainerConfig(
                    max_steps=self.config.micro_train_steps,
                    eval_strategy="steps",
                    eval_steps=max(10, self.config.micro_train_steps // 3),
                    auto_export_gguf=False,
                    save_steps=999999,
                    logging_steps=10,
                )
                trainer = Trainer(trainer_config)
                run = trainer.train_sft(
                    dataset_path=train_path,
                    val_dataset_path=val_path,
                )

                new_loss = run.metrics.get("eval_loss") or run.metrics.get("final_loss", 5.0)
                new_loss = float(new_loss)

            delta = new_loss - self._last_loss if self._last_loss is not None else 0.0
            self._last_loss = new_loss
            self._last_gold_count = current_count

            logger.info(
                "[TrainingTrigger] Micro-train complete: loss=%.4f, delta=%.4f",
                new_loss,
                delta,
            )
            return delta

        except Exception as e:
            logger.error("[TrainingTrigger] Micro-train failed: %s", e)
            return None


# =============================================================================
# PromptEvolver
# =============================================================================


class PromptEvolver:
    """Fast-loop evolution: mutate prompts/configs based on failure patterns.

    The evolution loop:
    1. Load recent traces (gold + failed)
    2. Extract failure/success patterns from decisions (deterministic)
    3. Generate prompt variant that addresses failures while preserving successes (LLM)
    4. Evaluate variant using semantic judge on held-out traces (LLM)
    5. Keep if improved, discard if not
    6. Repeat for N generations
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5-20250929",
        storage_dir: Path | None = None,
        training_trigger_config: TrainingTriggerConfig | None = None,
    ):
        self.provider = provider
        self.model = model
        self.storage_dir = storage_dir or Path.home() / ".bashgym" / "prompt_evolution"
        self.variants: list[PromptVariant] = []
        self.best_variant: PromptVariant | None = None
        self._client = None
        self._training_trigger = (
            TrainingTrigger(training_trigger_config)
            if training_trigger_config and training_trigger_config.enabled
            else None
        )

    def _get_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic()
            except ImportError:
                logger.warning(
                    "anthropic package not installed — prompt evolution LLM calls disabled"
                )
                self._client = None
            except Exception as exc:
                logger.warning("Failed to initialize Anthropic client: %s", exc)
                self._client = None
        return self._client

    # =========================================================================
    # Analysis Phase (deterministic — no LLM calls)
    # =========================================================================

    async def analyze_failures(self, traces: list[dict[str, Any]]) -> list[FailurePattern]:
        """Extract common failure patterns from traces.

        Uses decision logs to identify:
        - Repeated wrong tool choices (same tool fails 3+ times in a trace)
        - Missing context patterns (errors about undefined vars, wrong paths)
        - Anti-patterns (over-engineering, unnecessary complexity, reverts)
        - Incomplete outputs (task partially done, final steps show failure)

        Groups similar failures and counts frequency.
        Returns patterns sorted by frequency (most common first).

        This is fully deterministic — no LLM calls. Analysis is based on
        decision logs from the DecisionExtractor.
        """
        # Accumulators keyed by (pattern_type, description_key)
        pattern_counts: Counter = Counter()
        pattern_examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for trace in traces:
            decisions = self._extract_decisions_from_trace(trace)
            if not decisions:
                continue

            # --- Wrong tool detection ---
            tool_failures = self._detect_wrong_tool(decisions)
            for desc, examples in tool_failures:
                key = (PATTERN_WRONG_TOOL, desc)
                pattern_counts[key] += 1
                pattern_examples[key].extend(examples[:2])

            # --- Missing context detection ---
            context_failures = self._detect_missing_context(decisions, trace)
            for desc, examples in context_failures:
                key = (PATTERN_MISSING_CONTEXT, desc)
                pattern_counts[key] += 1
                pattern_examples[key].extend(examples[:2])

            # --- Anti-pattern detection ---
            anti_patterns = self._detect_anti_patterns(decisions, trace)
            for desc, examples in anti_patterns:
                key = (PATTERN_ANTI_PATTERN, desc)
                pattern_counts[key] += 1
                pattern_examples[key].extend(examples[:2])

            # --- Incomplete output detection ---
            incomplete = self._detect_incomplete_output(decisions, trace)
            for desc, examples in incomplete:
                key = (PATTERN_INCOMPLETE_OUTPUT, desc)
                pattern_counts[key] += 1
                pattern_examples[key].extend(examples[:2])

        # Build FailurePattern objects, filter by minimum frequency
        patterns: list[FailurePattern] = []
        for (ptype, desc), freq in pattern_counts.items():
            if freq < _MIN_PATTERN_FREQUENCY:
                continue
            examples = pattern_examples[(ptype, desc)][:4]
            patterns.append(
                FailurePattern(
                    pattern_type=ptype,
                    description=desc,
                    frequency=freq,
                    example_decisions=examples,
                    suggested_fix=self._suggest_fix(ptype, desc),
                )
            )

        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns

    async def analyze_successes(self, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract successful patterns to preserve.

        Identifies decisions that consistently lead to SUCCESS outcomes.
        These patterns should NOT be disrupted by prompt mutations.

        Returns a list of dicts with keys: tool, intent_summary, frequency.
        """
        success_tool_intents: Counter = Counter()

        for trace in traces:
            decisions = self._extract_decisions_from_trace(trace)
            for dec in decisions:
                if dec.get("outcome") == "SUCCESS":
                    tool = dec.get("tool_used", "unknown")
                    intent = dec.get("intent", "")[:80]
                    key = f"{tool}: {intent}" if intent else tool
                    success_tool_intents[key] += 1

        # Return patterns that appear across multiple traces
        results: list[dict[str, Any]] = []
        for pattern_key, count in success_tool_intents.most_common(15):
            if count < 2:
                break
            parts = pattern_key.split(": ", 1)
            results.append(
                {
                    "tool": parts[0],
                    "intent_summary": parts[1] if len(parts) > 1 else "",
                    "frequency": count,
                }
            )

        return results

    # =========================================================================
    # Evolution Phase (uses LLM)
    # =========================================================================

    async def evolve(
        self,
        current_prompt: str,
        failure_patterns: list[FailurePattern],
        success_patterns: list[dict[str, Any]],
        parent_variant: PromptVariant | None = None,
    ) -> PromptVariant:
        """Generate a prompt variant that addresses failures while preserving successes.

        Uses LLM to:
        1. DIAGNOSE: Analyze failure patterns and their root causes
        2. PRESCRIBE: Generate specific prompt additions/modifications
        3. VALIDATE: Check that prescribed changes don't conflict with success patterns

        The LLM receives:
        - Current prompt (the baseline)
        - Failure patterns with examples
        - Success patterns to preserve
        - Instructions to output JSON with prompt patches

        Returns a PromptVariant with system_prompt_patches.
        """
        client = self._get_client()
        if client is None:
            logger.warning("No LLM client available — returning empty variant")
            return self._empty_variant(parent_variant)

        # Format failure patterns for the prompt
        failure_lines: list[str] = []
        for i, fp in enumerate(failure_patterns[:8]):  # Top 8
            failure_lines.append(
                f"{i + 1}. [{fp.pattern_type}] (frequency={fp.frequency}) {fp.description}"
            )
            if fp.example_decisions:
                ex = fp.example_decisions[0]
                failure_lines.append(
                    f"   Example: tool={ex.get('tool_used', '?')}, "
                    f"intent={ex.get('intent', '?')[:100]}, "
                    f"outcome={ex.get('outcome', '?')}"
                )
            failure_lines.append(f"   Suggested fix: {fp.suggested_fix}")
            failure_lines.append("")

        # Format success patterns
        success_lines: list[str] = []
        for sp in success_patterns[:10]:
            success_lines.append(
                f"- {sp.get('tool', '?')}: {sp.get('intent_summary', '')} "
                f"(frequency={sp.get('frequency', 0)})"
            )

        prompt_text = _EVOLVE_USER_TEMPLATE.format(
            current_prompt=current_prompt[:4000],
            failure_patterns="\n".join(failure_lines) or "No failure patterns detected.",
            success_patterns="\n".join(success_lines) or "No success patterns available.",
        )

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_EVOLVE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt_text}],
            )

            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            return self._parse_evolve_response(response_text, failure_patterns, parent_variant)

        except Exception as exc:
            logger.warning("Prompt evolution LLM call failed: %s", exc)
            return self._empty_variant(parent_variant)

    # =========================================================================
    # Evaluation Phase (uses LLM as proxy)
    # =========================================================================

    async def evaluate_variant(
        self,
        variant: PromptVariant,
        baseline_score: float,
        test_traces: list[dict[str, Any]],
    ) -> float:
        """Score a variant using semantic judge as proxy.

        Simulates: "If this trace had been generated with the evolved prompt,
        would the solution quality be higher?"

        Uses the semantic judge to re-evaluate traces with the prompt context
        that the variant would have provided. Not a perfect evaluation
        (can't actually re-run the agent), but a reasonable proxy.

        Returns improvement delta over baseline (positive = better).
        """
        client = self._get_client()
        if client is None:
            return 0.0

        if not test_traces:
            return 0.0

        # Format the patches for evaluation
        patches_text = json.dumps(variant.system_prompt_patches, indent=2)

        improvement_sum = 0.0
        evaluated_count = 0

        for trace in test_traces:
            decisions = self._extract_decisions_from_trace(trace)
            steps = trace.get("normalized_steps", trace.get("steps", []))

            # Build trace summary for evaluation
            outcome = "unknown"
            if decisions:
                outcomes = [d.get("outcome", "") for d in decisions]
                failures = outcomes.count("FAILURE")
                successes = outcomes.count("SUCCESS")
                if failures > successes:
                    outcome = "mostly failed"
                elif successes > failures:
                    outcome = "mostly succeeded"
                else:
                    outcome = "mixed"

            decision_log_lines = []
            for d in decisions[-6:]:
                decision_log_lines.append(
                    f"- [{d.get('outcome', '?')}] {d.get('intent', '?')[:120]}"
                )

            # Gather issues from the trace
            issues = trace.get("issues", [])
            if not issues:
                issues = [d.get("intent", "") for d in decisions if d.get("outcome") == "FAILURE"]

            prompt_text = _EVALUATE_USER_TEMPLATE.format(
                prompt_patches=patches_text[:2000],
                step_count=len(steps),
                outcome=outcome,
                decision_log="\n".join(decision_log_lines) or "No decisions logged.",
                issues="\n".join(f"- {i}" for i in issues[:5]) or "None identified.",
            )

            try:
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=_EVALUATE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt_text}],
                )

                response_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        response_text += block.text

                score = self._parse_evaluate_response(response_text)
                improvement_sum += score
                evaluated_count += 1

            except Exception as exc:
                logger.debug("Evaluation call failed for trace: %s", exc)
                continue

        if evaluated_count == 0:
            return 0.0

        return round(improvement_sum / evaluated_count, 4)

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def evolution_loop(
        self,
        trace_dir: Path,
        generations: int = 10,
        callback: Callable | None = None,
    ) -> PromptVariant:
        """Main evolution loop.

        1. Load traces from trace_dir (both gold and failed)
        2. Split into analysis set (80%) and evaluation set (20%)
        3. Extract failure + success patterns from analysis set
        4. For each generation:
            a. Evolve a new variant from best current variant
            b. Evaluate on held-out set
            c. Keep if improved
            d. Emit PromptEvolved event
            e. Call callback with progress
        5. Save best variant to storage_dir
        6. Return best variant
        """
        # Load traces
        traces = self._load_traces(trace_dir)
        if not traces:
            logger.warning("No traces found in %s — cannot evolve", trace_dir)
            return self._empty_variant()

        logger.info(
            "Starting prompt evolution: %d traces, %d generations",
            len(traces),
            generations,
        )

        # Split: 80% analysis, 20% evaluation
        split_idx = max(1, int(len(traces) * 0.8))
        analysis_traces = traces[:split_idx]
        eval_traces = traces[split_idx:] if split_idx < len(traces) else traces[-2:]

        # Extract patterns (deterministic)
        failure_patterns = await self.analyze_failures(analysis_traces)
        success_patterns = await self.analyze_successes(analysis_traces)

        logger.info(
            "Extracted %d failure patterns, %d success patterns",
            len(failure_patterns),
            len(success_patterns),
        )

        if not failure_patterns:
            logger.info("No recurring failure patterns found — nothing to evolve")
            return self._empty_variant()

        # Get a baseline prompt (from context_builder's identity layer or a default)
        current_prompt = self._get_baseline_prompt()
        baseline_score = 0.0

        # Evolve for N generations
        for gen in range(generations):
            parent = self.best_variant

            # Generate variant
            variant = await self.evolve(
                current_prompt=current_prompt,
                failure_patterns=failure_patterns,
                success_patterns=success_patterns,
                parent_variant=parent,
            )
            variant.generation = gen + 1

            # Evaluate on held-out traces
            improvement = await self.evaluate_variant(variant, baseline_score, eval_traces)
            variant.metrics["improvement_delta"] = improvement
            variant.metrics["generation"] = gen + 1

            self.variants.append(variant)

            # Keep if improved
            if improvement > 0:
                self.best_variant = variant
                baseline_score += improvement
                current_prompt = self.apply_to_prompt(current_prompt, variant)

                logger.info(
                    "Generation %d: improvement +%.4f (variant %s kept)",
                    gen + 1,
                    improvement,
                    variant.variant_id[:8],
                )
            else:
                logger.info(
                    "Generation %d: no improvement (%.4f), variant discarded",
                    gen + 1,
                    improvement,
                )

            # Ouroboros: check if enough new gold traces to trigger micro-train
            if self._training_trigger:
                try:
                    import asyncio

                    delta_loss = await asyncio.to_thread(
                        self._training_trigger.check_and_train
                    )
                    if delta_loss is not None:
                        variant.metrics["training_delta_loss"] = delta_loss
                        if delta_loss < 0:
                            # Prompt evolution is producing better training data
                            improvement += abs(delta_loss) * 0.5
                            variant.metrics["improvement_delta"] = improvement
                            logger.info(
                                "Generation %d: ouroboros bonus %.4f (model improved)",
                                gen + 1,
                                abs(delta_loss) * 0.5,
                            )
                except Exception as e:
                    logger.warning("TrainingTrigger error: %s", e)

            # Emit event
            self._emit_evolved_event(variant, improvement)

            # Callback
            if callback:
                try:
                    callback(
                        {
                            "generation": gen + 1,
                            "total_generations": generations,
                            "improvement": improvement,
                            "variant_id": variant.variant_id,
                            "kept": improvement > 0,
                            "best_variant_id": (
                                self.best_variant.variant_id[:8] if self.best_variant else None
                            ),
                        }
                    )
                except Exception:
                    pass  # Callback errors shouldn't stop evolution

        # Save best variant
        if self.best_variant:
            self.save_variant(self.best_variant)
            logger.info(
                "Evolution complete. Best variant: %s (generation %d)",
                self.best_variant.variant_id[:8],
                self.best_variant.generation,
            )
        else:
            logger.info("Evolution complete. No improvements found.")

        return self.best_variant or self._empty_variant()

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_variant(self, variant: PromptVariant) -> None:
        """Save variant to storage_dir as JSON."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        filepath = self.storage_dir / f"{variant.variant_id}.json"
        filepath.write_text(
            json.dumps(variant.to_dict(), indent=2),
            encoding="utf-8",
        )

        # Also update the "best" pointer
        best_path = self.storage_dir / "best_variant.json"
        best_path.write_text(
            json.dumps(variant.to_dict(), indent=2),
            encoding="utf-8",
        )

        logger.debug("Saved variant %s to %s", variant.variant_id[:8], filepath)

    def load_best_variant(self) -> PromptVariant | None:
        """Load the best variant from storage_dir."""
        best_path = self.storage_dir / "best_variant.json"
        if not best_path.exists():
            return None

        try:
            data = json.loads(best_path.read_text(encoding="utf-8"))
            variant = PromptVariant.from_dict(data)
            self.best_variant = variant
            return variant
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load best variant: %s", exc)
            return None

    def get_lineage(self, variant_id: str) -> list[PromptVariant]:
        """Get the evolution lineage of a variant (ancestor chain)."""
        lineage: list[PromptVariant] = []

        # Build lookup from in-memory variants
        by_id = {v.variant_id: v for v in self.variants}

        # Also try loading from disk
        if self.storage_dir.exists():
            for filepath in self.storage_dir.glob("*.json"):
                if filepath.name == "best_variant.json":
                    continue
                try:
                    data = json.loads(filepath.read_text(encoding="utf-8"))
                    v = PromptVariant.from_dict(data)
                    if v.variant_id not in by_id:
                        by_id[v.variant_id] = v
                except (json.JSONDecodeError, OSError):
                    continue

        # Walk the lineage chain
        current_id: str | None = variant_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            variant = by_id.get(current_id)
            if variant is None:
                break
            lineage.append(variant)
            current_id = variant.parent_variant

        return lineage

    # =========================================================================
    # Integration
    # =========================================================================

    def apply_to_prompt(self, base_prompt: str, variant: PromptVariant) -> str:
        """Apply a variant's patches to a base prompt.

        Patches are keyed by section name (matching the three-layer structure):
        - "identity": patches the identity layer
        - "narrative": patches the narrative layer
        - "focus": patches the focus layer
        - "rules": adds new rules/guidelines
        - "anti_patterns": adds anti-pattern warnings

        For each patch, appends to the corresponding section.
        """
        result = base_prompt

        for section, patch_text in variant.system_prompt_patches.items():
            if not patch_text:
                continue

            # Find the section header and append after it
            section_markers = {
                "identity": "## Identity",
                "narrative": "## Context",
                "focus": "## Task",
                "rules": "### Rules",
                "anti_patterns": "### Rules",  # Anti-patterns go near rules
            }

            marker = section_markers.get(section)
            if marker and marker in result:
                # Find the next section header (## level) after the marker
                marker_pos = result.index(marker)
                # Look for next ## header after marker
                next_section = result.find("\n## ", marker_pos + len(marker))

                if next_section != -1:
                    # Insert before the next section
                    insert_text = f"\n\n### Evolved Guidance ({section})\n{patch_text}\n"
                    result = result[:next_section] + insert_text + result[next_section:]
                else:
                    # Append at the end
                    result += f"\n\n### Evolved Guidance ({section})\n{patch_text}\n"
            else:
                # Section not found — append at the end
                result += f"\n\n### Evolved Guidance ({section})\n{patch_text}\n"

        return result

    # =========================================================================
    # Internal: Failure Pattern Detection (deterministic)
    # =========================================================================

    def _extract_decisions_from_trace(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract decision dicts from a trace.

        Supports both pre-extracted decisions and raw steps that need extraction.
        """
        # Pre-extracted decisions (from ProcessedTrace or similar)
        decisions = trace.get("decisions", [])
        if decisions:
            # Normalize: could be Decision objects or dicts
            result = []
            for d in decisions:
                if isinstance(d, dict):
                    result.append(d)
                elif hasattr(d, "to_dict"):
                    result.append(d.to_dict())
                else:
                    result.append({"outcome": "PARTIAL"})
            return result

        # Fall back: extract from raw steps using DecisionExtractor
        steps = trace.get("normalized_steps", trace.get("steps", []))
        if not steps:
            return []

        try:
            from bashgym.factory.decision_extractor import DecisionExtractor

            extractor = DecisionExtractor()
            cognitive_data = trace.get("cognitive_data")
            extracted = extractor.extract(steps, cognitive_data)
            return [d.to_dict() for d in extracted]
        except ImportError:
            logger.debug("DecisionExtractor not available for trace analysis")
            return []

    def _detect_wrong_tool(
        self, decisions: list[dict[str, Any]]
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """Detect repeated wrong tool choices.

        A tool is "wrong" if the same tool fails 3+ times in sequence
        or the same tool type fails frequently across the trace.
        """
        results: list[tuple[str, list[dict[str, Any]]]] = []

        # Count consecutive failures per tool
        tool_fail_streak: dict[str, int] = defaultdict(int)
        tool_fail_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

        prev_tool = None
        streak = 0

        for dec in decisions:
            tool = dec.get("tool_used", "unknown")
            outcome = dec.get("outcome", "")

            if outcome == "FAILURE":
                tool_fail_streak[tool] += 1
                tool_fail_examples[tool].append(dec)

                if tool == prev_tool:
                    streak += 1
                else:
                    streak = 1

                if streak >= 3:
                    desc = f"Repeated {tool} failures ({streak}x in sequence)"
                    results.append((desc, tool_fail_examples[tool][-3:]))
            else:
                streak = 0

            prev_tool = tool

        # Also check overall tool failure rates
        for tool, fail_count in tool_fail_streak.items():
            total_uses = sum(1 for d in decisions if d.get("tool_used") == tool)
            if total_uses >= 3 and fail_count / total_uses > 0.6:
                desc = (
                    f"Tool '{tool}' fails frequently "
                    f"({fail_count}/{total_uses} = {fail_count / total_uses:.0%})"
                )
                results.append((desc, tool_fail_examples[tool][:3]))

        return results

    def _detect_missing_context(
        self,
        decisions: list[dict[str, Any]],
        trace: dict[str, Any],
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """Detect failures caused by missing context (undefined vars, wrong paths, etc.)."""
        results: list[tuple[str, list[dict[str, Any]]]] = []

        steps = trace.get("normalized_steps", trace.get("steps", []))
        context_failures: list[dict[str, Any]] = []

        for dec in decisions:
            if dec.get("outcome") != "FAILURE":
                continue

            # Check the corresponding step output for context-related errors
            step_idx = dec.get("step_index", -1)
            if 0 <= step_idx < len(steps):
                output = str(steps[step_idx].get("output", ""))
                if _MISSING_CONTEXT_RE.search(output):
                    context_failures.append(dec)

        if context_failures:
            desc = (
                f"Missing context errors (undefined variables, wrong paths, "
                f"missing imports) in {len(context_failures)} decisions"
            )
            results.append((desc, context_failures[:3]))

        return results

    def _detect_anti_patterns(
        self,
        decisions: list[dict[str, Any]],
        trace: dict[str, Any],
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """Detect anti-patterns: over-engineering, reverts, unnecessary pivots."""
        results: list[tuple[str, list[dict[str, Any]]]] = []

        revert_decisions: list[dict[str, Any]] = []
        pivot_decisions: list[dict[str, Any]] = []

        for dec in decisions:
            reasoning = dec.get("reasoning", "")
            intent = dec.get("intent", "")
            chosen = dec.get("chosen", "")
            combined = f"{reasoning} {intent} {chosen}"

            if _ANTI_PATTERN_RE.search(combined):
                if re.search(r"(?:revert|undo|roll\s*back)", combined, re.IGNORECASE):
                    revert_decisions.append(dec)
                else:
                    pivot_decisions.append(dec)

        if revert_decisions:
            desc = (
                f"Unnecessary reverts/undos ({len(revert_decisions)} occurrences) "
                f"— agent changed approach then rolled back"
            )
            results.append((desc, revert_decisions[:3]))

        if len(pivot_decisions) >= 3:
            desc = (
                f"Excessive strategy pivots ({len(pivot_decisions)} times) "
                f"— agent keeps changing approach without committing"
            )
            results.append((desc, pivot_decisions[:3]))

        return results

    def _detect_incomplete_output(
        self,
        decisions: list[dict[str, Any]],
        trace: dict[str, Any],
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """Detect traces where the task was only partially completed."""
        results: list[tuple[str, list[dict[str, Any]]]] = []

        if not decisions:
            return results

        # Check if the last few decisions indicate incompleteness
        last_decisions = decisions[-3:]
        final_outcomes = [d.get("outcome", "") for d in last_decisions]

        # Task is incomplete if the last decisions are failures or partial
        failure_count = final_outcomes.count("FAILURE")
        partial_count = final_outcomes.count("PARTIAL")

        if failure_count >= 2:
            desc = "Task ended with consecutive failures — likely incomplete"
            results.append((desc, last_decisions))
        elif partial_count >= 2 and "SUCCESS" not in final_outcomes:
            desc = "Task ended without clear success — output may be incomplete"
            results.append((desc, last_decisions))

        # Also check if there's no git commit (commit = completion signal)
        has_commit = any("git commit" in d.get("chosen", "") for d in decisions)
        if not has_commit and len(decisions) > 5:
            # Long trace with no commit — possible incompletion
            desc = "No git commit found in a long trace — task may not have finished"
            results.append((desc, last_decisions))

        return results

    def _suggest_fix(self, pattern_type: str, description: str) -> str:
        """Generate a suggested fix for a failure pattern (static heuristic)."""
        suggestions = {
            PATTERN_WRONG_TOOL: (
                "Add guidance about when to use each tool. "
                "Include examples of correct tool selection for common scenarios."
            ),
            PATTERN_MISSING_CONTEXT: (
                "Add instructions to verify file paths and variable names before use. "
                "Encourage the agent to read files before modifying them."
            ),
            PATTERN_ANTI_PATTERN: (
                "Add rules against unnecessary complexity. "
                "Instruct the agent to commit to an approach before starting. "
                "Discourage premature pivots."
            ),
            PATTERN_INCOMPLETE_OUTPUT: (
                "Add explicit completion criteria. "
                "Require the agent to verify task completion before stopping. "
                "Add a checklist of deliverables."
            ),
        }
        return suggestions.get(pattern_type, "Review and refine the relevant prompt section.")

    # =========================================================================
    # Internal: Response Parsing
    # =========================================================================

    def _parse_evolve_response(
        self,
        response_text: str,
        failure_patterns: list[FailurePattern],
        parent_variant: PromptVariant | None,
    ) -> PromptVariant:
        """Parse the LLM's evolution response into a PromptVariant."""
        text = response_text.strip()

        # Strip markdown fences if present
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse evolve response: %.200s", response_text)
            return self._empty_variant(parent_variant)

        prompt_patches = data.get("system_prompt_patches", {})
        tool_patches = data.get("tool_config_patches", {})

        if not isinstance(prompt_patches, dict):
            prompt_patches = {}
        if not isinstance(tool_patches, dict):
            tool_patches = {}

        return PromptVariant(
            variant_id=uuid.uuid4().hex[:12],
            system_prompt_patches=prompt_patches,
            tool_config_patches=tool_patches,
            parent_variant=parent_variant.variant_id if parent_variant else None,
            failure_patterns_addressed=[fp.description for fp in failure_patterns[:5]],
        )

    def _parse_evaluate_response(self, response_text: str) -> float:
        """Parse the evaluation response into an improvement score."""
        text = response_text.strip()

        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
            score = float(data.get("improvement_score", 0.0))
            return max(-1.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("Failed to parse evaluate response: %.200s", response_text)
            return 0.0

    # =========================================================================
    # Internal: Helpers
    # =========================================================================

    def _empty_variant(self, parent: PromptVariant | None = None) -> PromptVariant:
        """Create an empty variant (no changes)."""
        return PromptVariant(
            variant_id=uuid.uuid4().hex[:12],
            system_prompt_patches={},
            tool_config_patches={},
            parent_variant=parent.variant_id if parent else None,
        )

    def _get_baseline_prompt(self) -> str:
        """Get the baseline system prompt for evolution."""
        try:
            from bashgym.orchestrator.context_builder import WorkerContextBuilder  # noqa: F401

            builder = WorkerContextBuilder(
                dag_nodes={},
                spec_title="Prompt Evolution Baseline",
            )
            return builder.build_identity_layer()
        except ImportError:
            return (
                "## Identity\n\n"
                "You are a BashGym worker — an AI coding agent.\n\n"
                "### Rules\n"
                "- Write clean, well-typed Python 3.10+ code\n"
                "- Keep changes minimal and focused\n"
                "- Run tests after making changes\n"
            )

    def _load_traces(self, trace_dir: Path) -> list[dict[str, Any]]:
        """Load trace files from a directory (both gold and failed)."""
        traces: list[dict[str, Any]] = []

        if not trace_dir.exists():
            return traces

        # Load from the given directory and subdirectories
        search_dirs = [trace_dir]

        # Also check gold_traces and failed_traces subdirectories
        for subdir_name in ("gold_traces", "failed_traces"):
            subdir = trace_dir / subdir_name
            if subdir.exists():
                search_dirs.append(subdir)

        # Check parent's sibling directories (data/gold_traces, data/failed_traces)
        parent = trace_dir.parent
        for sibling_name in ("gold_traces", "failed_traces"):
            sibling = parent / sibling_name
            if sibling.exists() and sibling not in search_dirs:
                search_dirs.append(sibling)

        for search_dir in search_dirs:
            for filepath in search_dir.glob("*.json"):
                try:
                    data = json.loads(filepath.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        # Tag with source for analysis
                        if "gold" in str(search_dir):
                            data["_source"] = "gold"
                        elif "failed" in str(search_dir):
                            data["_source"] = "failed"
                        else:
                            data["_source"] = "pending"
                        traces.append(data)
                except (json.JSONDecodeError, OSError):
                    continue

        return traces

    def _emit_evolved_event(self, variant: PromptVariant, improvement: float) -> None:
        """Emit a PromptEvolved event via the event bus."""
        try:
            from bashgym.events.bus import event_bus
            from bashgym.events.types import PromptEvolved

            event_bus.emit(
                PromptEvolved(
                    variant_id=variant.variant_id,
                    generation=variant.generation,
                    improvement_delta=improvement,
                    patterns_addressed=variant.failure_patterns_addressed,
                )
            )
        except ImportError:
            pass  # Events module not available
