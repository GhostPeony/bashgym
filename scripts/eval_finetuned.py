#!/usr/bin/env python3
"""
Comprehensive Fine-Tune Evaluation — Base vs Fine-Tuned Gemma 4 E4B

Three-tier evaluation:
  Tier 1: Perplexity + Next-Tool Prediction (val set)
  Tier 2: Task Completion with LLM-as-Judge (gold traces)
  Tier 3: HumanEval code benchmark (standard)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Add bashgym to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
BASE_MODEL = "unsloth/gemma-4-E4B-it"
ADAPTER_PATH = Path.home() / ".unsloth/studio/outputs/unsloth_gemma-4-E4B-it_1775370273"
TRAIN_PATH = Path.home() / "bashgym-training/data/train.jsonl"
VAL_PATH = Path.home() / "bashgym-training/data/val.jsonl"
GOLD_TRACES_DIR = Path.home() / "bashgym/data/gold_traces"
REPORT_DIR = Path.home() / "bashgym/data/evaluation_results"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# =========================================================================
# Tier 1: Perplexity + Tool Prediction
# =========================================================================


def tier1_perplexity(models: dict, val_data: list[dict]) -> dict:
    """Compute cross-entropy loss on validation examples for both models."""
    logger.info("=" * 50)
    logger.info("TIER 1: Perplexity + Tool Prediction")
    logger.info("=" * 50)

    base_loss_fn = models["base_loss"]
    ft_loss_fn = models["ft_loss"]

    base_losses = []
    ft_losses = []

    for i, ex in enumerate(val_data):
        msgs = ex.get("messages", [])
        if len(msgs) < 3:
            continue

        try:
            bl = base_loss_fn(msgs)
            fl = ft_loss_fn(msgs)
            base_losses.append(bl)
            ft_losses.append(fl)

            if (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(val_data)}] base_loss={bl:.4f} ft_loss={fl:.4f}")
        except Exception as e:
            logger.warning(f"  Example {i}: failed — {e}")

    base_avg = sum(base_losses) / len(base_losses) if base_losses else float("inf")
    ft_avg = sum(ft_losses) / len(ft_losses) if ft_losses else float("inf")
    base_ppl = math.exp(base_avg) if base_avg < 100 else float("inf")
    ft_ppl = math.exp(ft_avg) if ft_avg < 100 else float("inf")

    logger.info(f"\n  Base model: avg_loss={base_avg:.4f} perplexity={base_ppl:.2f}")
    logger.info(f"  Fine-tuned: avg_loss={ft_avg:.4f} perplexity={ft_ppl:.2f}")
    logger.info(
        f"  Improvement: {((base_avg - ft_avg) / base_avg * 100):.1f}% loss reduction"
        if base_avg > 0
        else ""
    )

    return {
        "examples_evaluated": len(base_losses),
        "base": {
            "avg_loss": round(base_avg, 4),
            "perplexity": round(base_ppl, 2) if base_ppl < 1e6 else "inf",
            "losses": [round(loss, 4) for loss in base_losses],
        },
        "finetuned": {
            "avg_loss": round(ft_avg, 4),
            "perplexity": round(ft_ppl, 2) if ft_ppl < 1e6 else "inf",
            "losses": [round(loss, 4) for loss in ft_losses],
        },
        "loss_reduction_pct": round((base_avg - ft_avg) / base_avg * 100, 1) if base_avg > 0 else 0,
        "winner": "finetuned" if ft_avg < base_avg else "base",
    }


def tier1_tool_prediction(models: dict, val_data: list[dict]) -> dict:
    """Test if models predict the correct next tool at conversation cutpoints."""
    logger.info("\n--- Tool Prediction ---")

    base_gen = models["base_generate"]
    ft_gen = models["ft_generate"]

    base_correct = 0
    ft_correct = 0
    total = 0

    for i, ex in enumerate(val_data[:30]):  # Cap at 30 for speed
        msgs = ex.get("messages", [])

        # Find assistant messages with tool_calls (these are our prediction targets)
        tool_call_indices = []
        for mi, msg in enumerate(msgs):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_indices.append(mi)

        if not tool_call_indices:
            continue

        # Pick a cutpoint — the first tool call after initial user message
        cut_idx = tool_call_indices[0]
        if cut_idx < 2:
            continue

        prefix = msgs[:cut_idx]
        gold_tool = msgs[cut_idx]["tool_calls"][0]["function"]["name"]

        try:
            base_response = base_gen(prefix, max_new_tokens=200)
            ft_response = ft_gen(prefix, max_new_tokens=200)

            # Check if the gold tool name appears in the response
            if gold_tool.lower() in base_response.lower():
                base_correct += 1
            if gold_tool.lower() in ft_response.lower():
                ft_correct += 1
            total += 1

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  [{i+1}] gold={gold_tool} "
                    f"base={'Y' if gold_tool.lower() in base_response.lower() else 'N'} "
                    f"ft={'Y' if gold_tool.lower() in ft_response.lower() else 'N'}"
                )
        except Exception as e:
            logger.warning(f"  Example {i}: generation failed — {e}")

    base_acc = base_correct / total if total > 0 else 0
    ft_acc = ft_correct / total if total > 0 else 0

    logger.info("\n  Tool prediction accuracy:")
    logger.info(f"    Base:      {base_correct}/{total} ({base_acc:.1%})")
    logger.info(f"    Fine-tuned: {ft_correct}/{total} ({ft_acc:.1%})")

    return {
        "total_predictions": total,
        "base": {"correct": base_correct, "accuracy": round(base_acc, 4)},
        "finetuned": {"correct": ft_correct, "accuracy": round(ft_acc, 4)},
        "winner": "finetuned" if ft_acc > base_acc else "base" if base_acc > ft_acc else "tie",
    }


# =========================================================================
# Tier 2: Task Completion with LLM-as-Judge
# =========================================================================


def load_gold_traces(traces_dir: Path, n: int = 25) -> list[dict]:
    """Load and sample gold traces with extractable user prompts."""
    traces = []
    for path in sorted(traces_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(errors="replace"))
            # Need a user prompt and some trace steps
            prompt = None
            metadata = data.get("metadata", {})
            if isinstance(metadata, dict):
                prompt = metadata.get("user_initial_prompt", "")

            trace_steps = data.get("trace", [])
            if prompt and len(trace_steps) >= 3:
                traces.append(
                    {
                        "trace_id": data.get("session_id", path.stem),
                        "user_prompt": prompt,
                        "trace": trace_steps,
                        "summary": data.get("summary", {}),
                        "primary_repo": data.get("primary_repo", {}),
                    }
                )
        except (json.JSONDecodeError, OSError):
            continue

        if len(traces) >= n * 3:  # Collect extras to sample from
            break

    random.seed(42)
    if len(traces) > n:
        traces = random.sample(traces, n)

    return traces


def get_system_prompt() -> str:
    """Extract the system prompt from training data."""
    train = load_jsonl(TRAIN_PATH)
    if train:
        for msg in train[0].get("messages", []):
            if msg.get("role") == "system":
                return msg.get("content", "")
    return "You are an expert software development agent."


def tier2_task_completion(models: dict, num_traces: int = 25) -> dict:
    """Run both models on gold trace prompts, judge with Claude."""
    logger.info("\n" + "=" * 50)
    logger.info("TIER 2: Task Completion (LLM-as-Judge)")
    logger.info("=" * 50)

    # Check for Anthropic API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("  ANTHROPIC_API_KEY not set — skipping LLM-as-Judge scoring")
        logger.info("  Will still compare raw generation quality")

    traces = load_gold_traces(GOLD_TRACES_DIR, n=num_traces)
    logger.info(f"  Loaded {len(traces)} gold traces for evaluation")

    if not traces:
        return {"error": "No gold traces with extractable prompts found"}

    system_prompt = get_system_prompt()
    base_gen = models["base_generate"]
    ft_gen = models["ft_generate"]

    results = []
    base_scores = []
    ft_scores = []

    for i, trace in enumerate(traces):
        prompt = trace["user_prompt"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            base_response = base_gen(messages, max_new_tokens=512)
            ft_response = ft_gen(messages, max_new_tokens=512)
        except Exception as e:
            logger.warning(f"  Trace {i}: generation failed — {e}")
            continue

        # Gold trace tool sequence for comparison
        gold_tools = [step.get("tool_name", "?") for step in trace["trace"][:10]]

        result = {
            "trace_id": trace["trace_id"],
            "prompt": prompt[:200],
            "gold_tools": gold_tools,
            "base_response": base_response[:500],
            "ft_response": ft_response[:500],
        }

        # LLM-as-Judge scoring if API key available
        if api_key:
            try:
                judge_result = _judge_responses(
                    api_key, prompt, base_response, ft_response, gold_tools
                )
                result["judge"] = judge_result
                base_scores.append(judge_result.get("base_score", 0))
                ft_scores.append(judge_result.get("ft_score", 0))
            except Exception as e:
                logger.warning(f"  Trace {i}: judging failed — {e}")

        results.append(result)

        if (i + 1) % 5 == 0:
            logger.info(f"  [{i+1}/{len(traces)}] completed")

    # Aggregate
    base_avg = sum(base_scores) / len(base_scores) if base_scores else 0
    ft_avg = sum(ft_scores) / len(ft_scores) if ft_scores else 0

    if base_scores:
        logger.info("\n  LLM-as-Judge scores:")
        logger.info(f"    Base:       {base_avg:.2f}/5.0")
        logger.info(f"    Fine-tuned: {ft_avg:.2f}/5.0")
        wins = sum(1 for b, f in zip(base_scores, ft_scores) if f > b)
        ties = sum(1 for b, f in zip(base_scores, ft_scores) if f == b)
        losses = sum(1 for b, f in zip(base_scores, ft_scores) if f < b)
        logger.info(f"    FT wins: {wins}, ties: {ties}, base wins: {losses}")

    return {
        "traces_evaluated": len(results),
        "base": {
            "avg_judge_score": round(base_avg, 2) if base_scores else None,
        },
        "finetuned": {
            "avg_judge_score": round(ft_avg, 2) if ft_scores else None,
        },
        "head_to_head": (
            {
                "ft_wins": sum(1 for b, f in zip(base_scores, ft_scores) if f > b),
                "ties": sum(1 for b, f in zip(base_scores, ft_scores) if f == b),
                "base_wins": sum(1 for b, f in zip(base_scores, ft_scores) if f < b),
            }
            if base_scores
            else None
        ),
        "winner": (
            ("finetuned" if ft_avg > base_avg else "base" if base_avg > ft_avg else "tie")
            if base_scores
            else "unknown"
        ),
        "results": results,
    }


def _judge_responses(
    api_key: str,
    task_prompt: str,
    base_response: str,
    ft_response: str,
    gold_tools: list[str],
) -> dict:
    """Use Claude to judge which response is better."""
    import httpx

    judge_prompt = f"""You are evaluating two AI coding assistant responses to the same task.

TASK: {task_prompt[:1000]}

The gold-standard solution used these tools in order: {', '.join(gold_tools[:10])}

RESPONSE A (Base Model):
{base_response[:1500]}

RESPONSE B (Fine-Tuned Model):
{ft_response[:1500]}

Score each response from 1-5 on:
1. Task Understanding: Does it correctly identify what needs to be done?
2. Tool Selection: Does it choose appropriate tools/commands?
3. Code Quality: Is the code/approach correct and clean?
4. Efficiency: Does it take a reasonable approach without unnecessary steps?

Respond in JSON format:
{{"response_a_score": <1-5 overall>, "response_b_score": <1-5 overall>, "reasoning": "<brief explanation>"}}"""

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": judge_prompt}],
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Claude API error: {response.status_code}")

    text = response.json()["content"][0]["text"]

    # Parse JSON from response
    try:
        # Find JSON in response
        start = text.index("{")
        end = text.rindex("}") + 1
        scores = json.loads(text[start:end])
        return {
            "base_score": scores.get("response_a_score", 0),
            "ft_score": scores.get("response_b_score", 0),
            "reasoning": scores.get("reasoning", ""),
        }
    except (ValueError, json.JSONDecodeError):
        return {"base_score": 0, "ft_score": 0, "reasoning": f"Parse error: {text[:200]}"}


# =========================================================================
# Tier 3: HumanEval Benchmark
# =========================================================================


def tier3_humaneval(models: dict) -> dict:
    """Run HumanEval benchmark on both models."""
    logger.info("\n" + "=" * 50)
    logger.info("TIER 3: HumanEval Code Benchmark")
    logger.info("=" * 50)

    try:
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split="test")
        logger.info(f"  Loaded {len(ds)} HumanEval problems")
    except Exception as e:
        logger.warning(f"  Failed to load HumanEval: {e}")
        return {"error": str(e)}

    base_gen = models["base_generate"]
    ft_gen = models["ft_generate"]

    base_passed = 0
    ft_passed = 0
    total = 0
    results = []

    for i, problem in enumerate(ds):
        prompt = problem["prompt"]
        test_code = problem["test"]
        entry_point = problem["entry_point"]
        task_id = problem["task_id"]

        # Format as code completion prompt
        code_prompt = f"Complete the following Python function. Only output the function body, no explanation.\n\n{prompt}"

        try:
            base_code = base_gen(code_prompt, max_new_tokens=512, temperature=0.1)
            ft_code = ft_gen(code_prompt, max_new_tokens=512, temperature=0.1)
        except Exception as e:
            logger.warning(f"  {task_id}: generation failed — {e}")
            continue

        # Test both solutions
        base_pass = _test_solution(prompt, base_code, test_code, entry_point)
        ft_pass = _test_solution(prompt, ft_code, test_code, entry_point)

        if base_pass:
            base_passed += 1
        if ft_pass:
            ft_passed += 1
        total += 1

        results.append(
            {
                "task_id": task_id,
                "base_passed": base_pass,
                "ft_passed": ft_pass,
            }
        )

        if (i + 1) % 20 == 0:
            logger.info(
                f"  [{i+1}/{len(ds)}] base={base_passed}/{total} " f"ft={ft_passed}/{total}"
            )

    base_rate = base_passed / total if total > 0 else 0
    ft_rate = ft_passed / total if total > 0 else 0

    logger.info("\n  HumanEval pass@1:")
    logger.info(f"    Base:       {base_passed}/{total} ({base_rate:.1%})")
    logger.info(f"    Fine-tuned: {ft_passed}/{total} ({ft_rate:.1%})")

    return {
        "total_problems": total,
        "base": {"passed": base_passed, "pass_at_1": round(base_rate, 4)},
        "finetuned": {"passed": ft_passed, "pass_at_1": round(ft_rate, 4)},
        "winner": (
            "finetuned" if ft_rate > base_rate else "base" if base_rate > ft_rate else "tie"
        ),
        "details": results,
    }


def _test_solution(prompt: str, generated: str, test_code: str, entry_point: str) -> bool:
    """Test a generated solution against HumanEval test cases."""
    import subprocess
    import tempfile

    # Combine prompt + generated code + tests
    full_code = prompt + generated + "\n\n" + test_code + f"\ncheck({entry_point})\n"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            f.flush()

            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(f.name)
        except OSError:
            pass


# =========================================================================
# Report Generation
# =========================================================================


def generate_report(
    tier1_ppl: dict,
    tier1_tool: dict,
    tier2: dict,
    tier3: dict,
    adapter_config: dict,
) -> dict:
    """Generate the final comparison report."""

    winners = {}
    if tier1_ppl:
        winners["perplexity"] = tier1_ppl.get("winner", "unknown")
    if tier1_tool:
        winners["tool_prediction"] = tier1_tool.get("winner", "unknown")
    if tier2:
        winners["task_completion"] = tier2.get("winner", "unknown")
    if tier3:
        winners["humaneval"] = tier3.get("winner", "unknown")

    ft_wins = sum(1 for v in winners.values() if v == "finetuned")
    base_wins = sum(1 for v in winners.values() if v == "base")

    report = {
        "metadata": {
            "base_model": BASE_MODEL,
            "adapter_path": str(ADAPTER_PATH),
            "adapter_config": adapter_config,
            "evaluation_date": datetime.now().isoformat(),
            "val_examples": 80,
        },
        "tier1_perplexity": tier1_ppl,
        "tier1_tool_prediction": tier1_tool,
        "tier2_task_completion": {k: v for k, v in (tier2 or {}).items() if k != "results"},
        "tier3_humaneval": {k: v for k, v in (tier3 or {}).items() if k != "details"},
        "verdict": {
            "winners_per_category": winners,
            "ft_wins": ft_wins,
            "base_wins": base_wins,
            "overall": (
                "finetuned" if ft_wins > base_wins else "base" if base_wins > ft_wins else "mixed"
            ),
            "recommendation": _recommendation(tier1_ppl, tier1_tool, tier2, tier3),
        },
    }

    return report


def _recommendation(ppl, tool, tier2, tier3) -> str:
    parts = []

    if ppl and ppl.get("winner") == "finetuned":
        parts.append(
            f"Fine-tuning reduced loss by {ppl.get('loss_reduction_pct', 0):.1f}% on held-out data"
        )
    elif ppl:
        parts.append("Fine-tuning did NOT improve perplexity on held-out data")

    if tool and tool.get("winner") == "finetuned":
        parts.append("Fine-tuned model better predicts correct tool usage")

    if tier3:
        base_p = tier3.get("base", {}).get("pass_at_1", 0)
        ft_p = tier3.get("finetuned", {}).get("pass_at_1", 0)
        if ft_p < base_p * 0.9:
            parts.append(f"WARNING: Fine-tuning degraded HumanEval ({ft_p:.1%} vs {base_p:.1%})")
        elif ft_p > base_p:
            parts.append("General coding ability maintained or improved")

    return ". ".join(parts) if parts else "Insufficient data for recommendation"


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma 4 E4B")
    parser.add_argument("--tier1-only", action="store_true", help="Only run Tier 1")
    parser.add_argument("--skip-tier1", action="store_true")
    parser.add_argument("--skip-tier2", action="store_true")
    parser.add_argument("--skip-tier3", action="store_true")
    parser.add_argument("--num-traces", type=int, default=25)
    parser.add_argument(
        "--adapter",
        type=str,
        default=str(ADAPTER_PATH),
        help="Path to LoRA adapter directory",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FINE-TUNE EVALUATION: Base vs Fine-Tuned Gemma 4 E4B")
    logger.info("=" * 60)
    logger.info(f"Base model:  {BASE_MODEL}")
    logger.info(f"Adapter:     {args.adapter}")
    logger.info(f"Val data:    {VAL_PATH}")
    logger.info("")

    # Load adapter config for report
    adapter_config = {}
    config_path = Path(args.adapter) / "adapter_config.json"
    if config_path.exists():
        adapter_config = json.loads(config_path.read_text())

    # Load models
    logger.info("Loading models...")
    start = time.time()
    from bashgym.models.gemma_loader import load_models

    models = load_models(
        base_model=BASE_MODEL,
        adapter_path=args.adapter,
    )
    logger.info(f"Models loaded in {time.time() - start:.1f}s")
    logger.info(f"Adapter loaded: {models['has_adapter']}")
    logger.info("")

    # Load val data
    val_data = load_jsonl(VAL_PATH)
    logger.info(f"Loaded {len(val_data)} validation examples")

    # Run tiers
    tier1_ppl = None
    tier1_tool = None
    tier2 = None
    tier3 = None

    if not args.skip_tier1:
        tier1_ppl = tier1_perplexity(models, val_data)
        tier1_tool = tier1_tool_prediction(models, val_data)
        if args.tier1_only:
            logger.info("\n--tier1-only specified, stopping here")
            report = generate_report(tier1_ppl, tier1_tool, None, None, adapter_config)
            _save_report(report)
            return

    if not args.skip_tier2:
        tier2 = tier2_task_completion(models, num_traces=args.num_traces)

    if not args.skip_tier3:
        tier3 = tier3_humaneval(models)

    # Generate and save report
    report = generate_report(tier1_ppl, tier1_tool, tier2, tier3, adapter_config)
    _save_report(report)

    # Print verdict
    logger.info("\n" + "=" * 60)
    logger.info("VERDICT")
    logger.info("=" * 60)
    v = report["verdict"]
    for cat, winner in v["winners_per_category"].items():
        logger.info(f"  {cat:25s} → {winner}")
    logger.info(f"\n  Overall: {v['overall']}")
    logger.info(f"  {v['recommendation']}")


def _save_report(report: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"eval_gemma4_e4b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved to: {path}")


if __name__ == "__main__":
    main()
