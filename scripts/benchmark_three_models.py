#!/usr/bin/env python3
"""Benchmark base vs SFT vs DPO on tool prediction + speed."""

import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = {
    "base": "http://127.0.0.1:8898/v1",
    "sft": "http://127.0.0.1:8899/v1",
    "dpo": "http://127.0.0.1:8900/v1",
}
VAL_PATH = Path.home() / "bashgym-training" / "data-pipeline-fixed" / "val.jsonl"


def call(url, messages, max_tokens=200):
    start = time.time()
    try:
        r = httpx.post(
            f"{url}/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.1},
            timeout=120,
        )
        elapsed = time.time() - start
        if r.status_code != 200:
            return {"error": r.status_code, "elapsed": elapsed}
        d = r.json()
        return {
            "content": d["choices"][0]["message"]["content"],
            "tokens": d["usage"]["completion_tokens"],
            "elapsed": elapsed,
            "tps": d["usage"]["completion_tokens"] / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - start}


def get_first_tool(ex):
    for m in ex.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return m["tool_calls"][0]["function"]["name"]
    return None


def get_user_prompt(ex):
    for m in ex.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


with open(VAL_PATH) as f:
    val = [json.loads(line) for line in f if line.strip()]

# Speed test
print("=" * 60)
print("SPEED TEST (5 prompts)")
print("=" * 60)
speed_prompts = [
    "List Python files in the current directory.",
    "Read README.md",
    "Search for 'main' in all files.",
    "Create test.py with hello world",
    "Run pytest",
]
for name, url in MODELS.items():
    tps_list = []
    for p in speed_prompts:
        r = call(url, [{"role": "user", "content": p}])
        if "tps" in r:
            tps_list.append(r["tps"])
    avg = sum(tps_list) / len(tps_list) if tps_list else 0
    print(f"  {name:5s}: {avg:.1f} tok/s")

# Tool prediction
print(f"\n{'=' * 60}")
print("TOOL PREDICTION (30 val examples)")
print("=" * 60)
examples = [ex for ex in val if get_first_tool(ex)][:30]
results = {name: {"correct": 0, "total": 0} for name in MODELS}

for i, ex in enumerate(examples):
    gold = get_first_tool(ex)
    prompt = get_user_prompt(ex)[:2000]
    if not prompt or not gold:
        continue
    system = (
        ex["messages"][0]["content"]
        if ex["messages"][0].get("role") == "system"
        else "You are an expert software development agent."
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    for name, url in MODELS.items():
        r = call(url, msgs)
        content = r.get("content", "")
        if gold.lower() in content.lower():
            results[name]["correct"] += 1
        results[name]["total"] += 1

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(examples)}] gold={gold}")

print()
for name in MODELS:
    c = results[name]["correct"]
    t = results[name]["total"]
    pct = c / t * 100 if t > 0 else 0
    print(f"  {name:5s}: {c}/{t} ({pct:.0f}%)")

# Summary
print(f"\n{'=' * 60}")
print("VERDICT")
print("=" * 60)
best = max(results, key=lambda n: results[n]["correct"])
print(f"  Best tool prediction: {best} ({results[best]['correct']}/{results[best]['total']})")

report = {"speed": {}, "tool_prediction": results, "best": best}
out = (
    Path.home()
    / "bashgym"
    / "data"
    / "evaluation_results"
    / f"benchmark_3way_{int(time.time())}.json"
)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved: {out}")
