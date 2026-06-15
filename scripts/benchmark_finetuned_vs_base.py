#!/usr/bin/env python3
"""
Benchmark fine-tuned vs base Gemma 4 E4B via llama-server endpoints.

Tests:
  Tier 1: Speed comparison (tokens/sec on both)
  Tier 2: Generation quality on val prompts
  Tier 3: Tool prediction accuracy on val examples
  Tier 4: LLM-as-judge head-to-head (Anthropic Claude as judge)
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8898/v1"
FT_URL = "http://127.0.0.1:8899/v1"
VAL_PATH = Path.home() / "bashgym-training" / "data-pipeline-fixed" / "val.jsonl"
REPORT_DIR = Path.home() / "bashgym" / "data" / "evaluation_results"


def call_model(url: str, messages: list, max_tokens: int = 200, temp: float = 0.1) -> dict:
    """Call llama-server OpenAI-compatible endpoint."""
    start = time.time()
    try:
        response = httpx.post(
            f"{url}/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}", "elapsed": elapsed}

        data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "elapsed": elapsed,
            "tokens_per_sec": data["usage"]["completion_tokens"] / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - start}


def load_val_examples() -> list[dict]:
    with open(VAL_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_user_prompt(ex: dict) -> str:
    for msg in ex.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_first_tool_used(ex: dict) -> str | None:
    for msg in ex.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return msg["tool_calls"][0]["function"]["name"]
    return None


# =====================================================================
# Tier 1: Speed comparison
# =====================================================================


def tier1_speed(prompts: list[str]) -> dict:
    logger.info("=" * 60)
    logger.info("TIER 1: Speed Comparison")
    logger.info("=" * 60)

    results = {"base": [], "finetuned": []}

    for i, prompt in enumerate(prompts):
        msgs = [
            {"role": "system", "content": "You are an expert software development agent."},
            {"role": "user", "content": prompt},
        ]

        base_result = call_model(BASE_URL, msgs, max_tokens=300)
        ft_result = call_model(FT_URL, msgs, max_tokens=300)

        results["base"].append(base_result)
        results["finetuned"].append(ft_result)

        if "error" not in base_result and "error" not in ft_result:
            logger.info(
                f"  [{i+1}/{len(prompts)}] base={base_result['tokens_per_sec']:.1f}t/s "
                f"ft={ft_result['tokens_per_sec']:.1f}t/s"
            )

    base_avg = sum(r.get("tokens_per_sec", 0) for r in results["base"] if "error" not in r) / max(
        1, sum(1 for r in results["base"] if "error" not in r)
    )
    ft_avg = sum(r.get("tokens_per_sec", 0) for r in results["finetuned"] if "error" not in r) / max(
        1, sum(1 for r in results["finetuned"] if "error" not in r)
    )

    logger.info(f"\n  Base avg: {base_avg:.1f} tok/s")
    logger.info(f"  FT avg:   {ft_avg:.1f} tok/s")

    return {
        "base_avg_tps": round(base_avg, 1),
        "ft_avg_tps": round(ft_avg, 1),
        "samples": len(prompts),
    }


# =====================================================================
# Tier 2: Tool prediction accuracy
# =====================================================================


def tier2_tool_prediction(val_examples: list[dict], n: int = 30) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("TIER 2: Tool Prediction Accuracy")
    logger.info("=" * 60)

    # Pick examples that have a clear first tool
    examples_with_tool = [
        ex for ex in val_examples if get_first_tool_used(ex)
    ][:n]
    logger.info(f"  Testing {len(examples_with_tool)} examples")

    base_correct = 0
    ft_correct = 0
    results = []

    for i, ex in enumerate(examples_with_tool):
        gold_tool = get_first_tool_used(ex)
        user_prompt = get_user_prompt(ex)
        if not user_prompt:
            continue

        # Get system prompt from the example
        system = ex["messages"][0]["content"] if ex["messages"][0].get("role") == "system" else "You are an expert software development agent."

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt[:2000]},
        ]

        base_result = call_model(BASE_URL, msgs, max_tokens=200)
        ft_result = call_model(FT_URL, msgs, max_tokens=200)

        base_response = base_result.get("content", "")
        ft_response = ft_result.get("content", "")

        # Check if gold tool name appears in response (case insensitive)
        base_match = gold_tool.lower() in base_response.lower()
        ft_match = gold_tool.lower() in ft_response.lower()

        if base_match:
            base_correct += 1
        if ft_match:
            ft_correct += 1

        results.append({
            "gold_tool": gold_tool,
            "prompt": user_prompt[:200],
            "base_match": base_match,
            "ft_match": ft_match,
        })

        if (i + 1) % 5 == 0:
            logger.info(
                f"  [{i+1}/{len(examples_with_tool)}] gold={gold_tool} "
                f"base={'Y' if base_match else 'N'} ft={'Y' if ft_match else 'N'}"
            )

    total = len(results)
    logger.info(f"\n  Base: {base_correct}/{total} ({base_correct/total*100:.0f}%)")
    logger.info(f"  FT:   {ft_correct}/{total} ({ft_correct/total*100:.0f}%)")

    return {
        "total": total,
        "base_correct": base_correct,
        "ft_correct": ft_correct,
        "base_accuracy": round(base_correct / total, 3) if total else 0,
        "ft_accuracy": round(ft_correct / total, 3) if total else 0,
        "winner": "finetuned" if ft_correct > base_correct else "base" if base_correct > ft_correct else "tie",
    }


# =====================================================================
# Tier 3: LLM-as-Judge head-to-head
# =====================================================================


def tier3_llm_judge(val_examples: list[dict], n: int = 15) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: LLM-as-Judge (Claude scores responses)")
    logger.info("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("  ANTHROPIC_API_KEY not set, skipping LLM judge tier")
        return {"error": "no_api_key"}

    random.seed(42)
    test_examples = random.sample(val_examples, min(n, len(val_examples)))

    base_scores = []
    ft_scores = []
    judgments = []

    for i, ex in enumerate(test_examples):
        user_prompt = get_user_prompt(ex)
        if not user_prompt or len(user_prompt) < 30:
            continue

        system = ex["messages"][0]["content"] if ex["messages"][0].get("role") == "system" else "You are an expert software development agent."
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt[:2000]},
        ]

        base_result = call_model(BASE_URL, msgs, max_tokens=400)
        ft_result = call_model(FT_URL, msgs, max_tokens=400)

        base_resp = base_result.get("content", "")
        ft_resp = ft_result.get("content", "")

        # Have Claude judge
        judge_prompt = f"""Compare two AI coding agent responses to the same task.

TASK: {user_prompt[:1500]}

RESPONSE A (model 1):
{base_resp[:1500]}

RESPONSE B (model 2):
{ft_resp[:1500]}

Score each from 1-5 on overall quality (correctness, appropriate tool use, helpfulness).

Reply with JSON only: {{"a_score": <1-5>, "b_score": <1-5>, "reason": "<brief>"}}"""

        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": judge_prompt}],
                },
                timeout=30,
            )
            text = response.json()["content"][0]["text"]
            start = text.index("{")
            end = text.rindex("}") + 1
            scores = json.loads(text[start:end])
            base_scores.append(scores["a_score"])
            ft_scores.append(scores["b_score"])
            judgments.append({
                "prompt": user_prompt[:200],
                "base_score": scores["a_score"],
                "ft_score": scores["b_score"],
                "reason": scores.get("reason", ""),
            })
            if (i + 1) % 3 == 0:
                logger.info(f"  [{i+1}/{len(test_examples)}] base={scores['a_score']} ft={scores['b_score']}")
        except Exception as e:
            logger.warning(f"  Judge failed: {e}")

    if not base_scores:
        return {"error": "no_judgments"}

    base_avg = sum(base_scores) / len(base_scores)
    ft_avg = sum(ft_scores) / len(ft_scores)
    ft_wins = sum(1 for b, f in zip(base_scores, ft_scores) if f > b)
    ties = sum(1 for b, f in zip(base_scores, ft_scores) if f == b)
    base_wins = sum(1 for b, f in zip(base_scores, ft_scores) if f < b)

    logger.info(f"\n  Base avg score: {base_avg:.2f}/5")
    logger.info(f"  FT avg score:   {ft_avg:.2f}/5")
    logger.info(f"  FT wins: {ft_wins}, ties: {ties}, base wins: {base_wins}")

    return {
        "judged": len(base_scores),
        "base_avg_score": round(base_avg, 2),
        "ft_avg_score": round(ft_avg, 2),
        "ft_wins": ft_wins,
        "ties": ties,
        "base_wins": base_wins,
        "winner": "finetuned" if ft_avg > base_avg else "base" if base_avg > ft_avg else "tie",
        "judgments": judgments,
    }


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tool", type=int, default=30, help="Tool prediction sample size")
    parser.add_argument("--n-judge", type=int, default=15, help="LLM judge sample size")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    # Verify both servers
    logger.info("Verifying servers...")
    for name, url in [("base", BASE_URL), ("finetuned", FT_URL)]:
        try:
            r = httpx.get(f"{url.rsplit('/', 1)[0]}/health", timeout=5)
            logger.info(f"  {name} ({url}): {r.json()}")
        except Exception as e:
            logger.error(f"  {name} ({url}): FAILED - {e}")
            return

    val_examples = load_val_examples()
    logger.info(f"Loaded {len(val_examples)} val examples\n")

    # Tier 1: Speed
    speed_prompts = [
        "List the Python files in the current directory.",
        "Read the contents of README.md",
        "Search for the function 'main' in all Python files.",
        "Create a new file called test.py with a hello world function.",
        "Run the test suite with pytest.",
    ]
    tier1 = tier1_speed(speed_prompts)

    # Tier 2: Tool prediction
    tier2 = tier2_tool_prediction(val_examples, n=args.n_tool)

    # Tier 3: LLM judge
    tier3 = None if args.skip_judge else tier3_llm_judge(val_examples, n=args.n_judge)

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "base_model_url": BASE_URL,
        "ft_model_url": FT_URL,
        "tier1_speed": tier1,
        "tier2_tool_prediction": tier2,
        "tier3_llm_judge": tier3,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL VERDICT")
    logger.info("=" * 60)
    logger.info(f"  Tier 1 Speed:        base={tier1['base_avg_tps']}t/s ft={tier1['ft_avg_tps']}t/s")
    logger.info(f"  Tier 2 Tool acc:     base={tier2['base_accuracy']:.1%} ft={tier2['ft_accuracy']:.1%} → {tier2['winner']}")
    if tier3 and "error" not in tier3:
        logger.info(f"  Tier 3 LLM judge:    base={tier3['base_avg_score']}/5 ft={tier3['ft_avg_score']}/5 → {tier3['winner']}")
        logger.info(f"     FT wins: {tier3['ft_wins']}, ties: {tier3['ties']}, base wins: {tier3['base_wins']}")
    logger.info(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
