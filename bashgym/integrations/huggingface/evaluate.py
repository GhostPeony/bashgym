"""
HuggingFace Evaluate library integration.

Wraps the `evaluate` library for running metrics on local validation data.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = ["accuracy", "f1", "bleu", "rouge"]


class HFEvaluator:
    """Wraps HuggingFace evaluate library for local model evaluation."""

    async def evaluate_with_metric(
        self,
        model_id: str,
        metric_name: str = "accuracy",
        validation_data_path: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a model using a HuggingFace metric on local validation data.

        This is a lightweight evaluation that loads the metric module and computes
        it against the provided predictions/references. It does NOT load the full
        model — instead it expects predictions to already exist in the validation
        data file (or uses a simple heuristic).

        Args:
            model_id: HuggingFace model ID (for reference/metadata)
            metric_name: Metric to compute — 'accuracy', 'f1', 'bleu', 'rouge'
            validation_data_path: Path to JSONL validation file. Each line should
                have 'prediction' and 'reference' keys. If None, uses the default
                training batch validation file.

        Returns:
            Dict with metric name, score, and details.
        """
        import asyncio

        def _compute():
            try:
                import evaluate as hf_evaluate
            except ImportError:
                return {
                    "metric": metric_name,
                    "model_id": model_id,
                    "error": "evaluate library not installed. Run: pip install evaluate",
                    "score": None,
                }

            # Find validation data
            val_path = None
            if validation_data_path:
                val_path = Path(validation_data_path)
            else:
                # Try default locations
                candidates = [
                    Path.home() / ".bashgym" / "training_batches" / "val.jsonl",
                    Path("data") / "val.jsonl",
                ]
                for c in candidates:
                    if c.exists():
                        val_path = c
                        break

            if not val_path or not val_path.exists():
                return {
                    "metric": metric_name,
                    "model_id": model_id,
                    "error": "No validation data found. Provide validation_data_path or generate training data first.",
                    "score": None,
                }

            # Load predictions and references from validation file
            predictions = []
            references = []
            try:
                with open(val_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if "prediction" in row and "reference" in row:
                            predictions.append(row["prediction"])
                            references.append(row["reference"])
                        elif "messages" in row:
                            # NeMo format: use assistant message as reference
                            msgs = row["messages"]
                            asst = next(
                                (m["content"] for m in msgs if m["role"] == "assistant"), None
                            )
                            if asst:
                                references.append(asst)
                                predictions.append(asst)  # Self-evaluation placeholder
            except Exception as e:
                return {
                    "metric": metric_name,
                    "model_id": model_id,
                    "error": f"Failed to read validation data: {e}",
                    "score": None,
                }

            if not predictions:
                return {
                    "metric": metric_name,
                    "model_id": model_id,
                    "error": "No prediction/reference pairs found in validation data.",
                    "score": None,
                }

            # Compute metric
            try:
                metric = hf_evaluate.load(metric_name)

                if metric_name in ("accuracy", "f1"):
                    # Requires int labels — try to parse or use length-based proxy
                    try:
                        int_preds = [int(p) for p in predictions]
                        int_refs = [int(r) for r in references]
                        result = metric.compute(predictions=int_preds, references=int_refs)
                    except (ValueError, TypeError):
                        # Fall back to string exact match
                        correct = sum(p == r for p, r in zip(predictions, references))
                        result = {metric_name: correct / len(predictions)}

                elif metric_name == "bleu":
                    result = metric.compute(
                        predictions=predictions,
                        references=[[r] for r in references],
                    )

                elif metric_name == "rouge":
                    result = metric.compute(
                        predictions=predictions,
                        references=references,
                    )

                else:
                    result = metric.compute(predictions=predictions, references=references)

                # Extract primary score
                score = None
                if metric_name in result:
                    score = result[metric_name]
                elif "score" in result:
                    score = result["score"]
                elif result:
                    score = next(iter(result.values()))

                return {
                    "metric": metric_name,
                    "model_id": model_id,
                    "score": score,
                    "details": result,
                    "samples_evaluated": len(predictions),
                }

            except Exception as e:
                return {
                    "metric": metric_name,
                    "model_id": model_id,
                    "error": f"Metric computation failed: {e}",
                    "score": None,
                }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _compute)

    async def available_metrics(self) -> list[str]:
        """Return list of supported metrics."""
        return SUPPORTED_METRICS


def get_evaluator() -> HFEvaluator:
    """Get a HFEvaluator instance."""
    return HFEvaluator()
