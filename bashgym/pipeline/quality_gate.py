"""Quality gate for hybrid auto-classification of imported traces.

Supports optional semantic judging: when a SemanticJudge is configured,
GOLD-classified traces get a second-pass LLM evaluation. Low-scoring
traces are demoted to PENDING (benefit of the doubt, not FAILED).
"""

import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .config import PipelineConfig

if TYPE_CHECKING:
    from bashgym.factory.trace_processor import ProcessedTrace
    from bashgym.judge.semantic_judge import SemanticJudge, SemanticVerdict

logger = logging.getLogger(__name__)

# Semantic score below this threshold demotes GOLD -> PENDING
_SEMANTIC_DEMOTION_THRESHOLD = 0.6


class Classification(str, Enum):
    GOLD = "gold"
    PENDING = "pending"
    FAILED = "failed"


class QualityGate:
    """Classifies imported traces as gold, pending, or failed.

    When a semantic_judge is provided, GOLD traces undergo LLM evaluation.
    If the semantic score is below the demotion threshold, the trace is
    demoted to PENDING rather than FAILED (benefit of the doubt).
    """

    def __init__(
        self,
        config: PipelineConfig,
        semantic_judge: Optional["SemanticJudge"] = None,
    ):
        self.config = config
        self.semantic_judge = semantic_judge

    def classify(self, success_rate: float, step_count: int) -> Classification:
        """Structural classification based on success rate and step count.

        This is the synchronous, fast-path classification. It does NOT
        invoke the semantic judge. Use classify_with_semantics() for the
        full pipeline with LLM evaluation.
        """
        if not self.config.classify_enabled:
            return Classification.PENDING

        if success_rate <= self.config.classify_fail_max_success_rate:
            return Classification.FAILED

        if (
            success_rate >= self.config.classify_gold_min_success_rate
            and step_count >= self.config.classify_gold_min_steps
        ):
            return Classification.GOLD

        return Classification.PENDING

    async def classify_with_semantics(
        self,
        success_rate: float,
        step_count: int,
        trace: Optional["ProcessedTrace"] = None,
    ) -> tuple["Classification", Optional["SemanticVerdict"]]:
        """Classify with optional semantic judge pass.

        Returns (classification, verdict). The verdict is None when the
        semantic judge is not configured or was not invoked (e.g. trace
        classified as FAILED by structural rules — no need to judge quality).
        """
        base_classification = self.classify(success_rate, step_count)

        # Only run semantic evaluation on GOLD traces (the borderline that matters)
        if (
            base_classification != Classification.GOLD
            or self.semantic_judge is None
            or trace is None
        ):
            return base_classification, None

        try:
            verdict = await self.semantic_judge.evaluate(trace)
        except Exception as exc:
            logger.warning("Semantic judge error, keeping GOLD: %s", exc)
            return base_classification, None

        # If confidence is 0 (neutral/failed verdict), don't act on it
        if verdict.confidence == 0.0:
            logger.debug(
                "Semantic verdict has zero confidence for %s, keeping GOLD",
                trace.trace_id,
            )
            return base_classification, verdict

        # Demote to PENDING if score is below threshold
        if verdict.score < _SEMANTIC_DEMOTION_THRESHOLD:
            logger.info(
                "Semantic judge demoted trace %s from GOLD to PENDING "
                "(score=%.2f, threshold=%.2f): %s",
                trace.trace_id,
                verdict.score,
                _SEMANTIC_DEMOTION_THRESHOLD,
                verdict.reasoning[:100],
            )
            return Classification.PENDING, verdict

        return Classification.GOLD, verdict

    def route_trace(
        self,
        trace_file: Path,
        classification: Classification,
        gold_dir: Path,
        failed_dir: Path,
    ) -> Path:
        if classification == Classification.GOLD:
            dest = gold_dir / trace_file.name
            shutil.move(str(trace_file), str(dest))
            return dest
        elif classification == Classification.FAILED:
            dest = failed_dir / trace_file.name
            shutil.move(str(trace_file), str(dest))
            return dest
        return trace_file  # pending stays in place
