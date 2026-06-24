"""Agent-friendly BashGym command line interface.

This CLI is deliberately small and dependency-free. It gives agents and humans a
stable surface for discovering training docs, planning training runs, summarizing
replay artifacts, and starting the existing API server.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bashgym.eval.dppo_replay import read_dppo_replay_jsonl, summarize_dppo_replay_records
from bashgym.gym.run_analysis import analyze_run_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DOCS_DIR = REPO_ROOT / "docs" / "training"


@dataclass(frozen=True)
class TrainingRecipe:
    strategy: str
    when_to_use: str
    starting_settings: dict[str, Any]
    watch: list[str]
    next_steps: list[str]
    docs: list[str]


DOCS = {
    "overview": {
        "path": "docs/training/overview.md",
        "summary": "Training mental model, data sources, strategy overview, and world-model placement.",
    },
    "strategy": {
        "path": "docs/training/strategy-guide.md",
        "summary": "Starting settings for SFT, DPO, GRPO/RLVR, distillation, cascade, and DPPO.",
    },
    "world-models": {
        "path": "docs/training/world-models.md",
        "summary": "ECHO/RWML contracts, replay payloads, metrics, and backend integration boundaries.",
    },
    "metrics": {
        "path": "docs/training/metrics-runbook.md",
        "summary": "Diagnosis guide for flat pass@k, zero reward variance, timeouts, tamper, and more.",
    },
    "glossary": {
        "path": "docs/training/glossary.md",
        "summary": "Compact definitions for training, terminal RL, and world-model terms.",
    },
    "agent-cli": {
        "path": "docs/training/agent-cli.md",
        "summary": "Machine-readable CLI commands agents can use for setup and replay analysis.",
    },
}


def _recipe(strategy: str, hardware: str, data: str) -> TrainingRecipe:
    strategy = strategy.lower()
    base: dict[str, TrainingRecipe] = {
        "sft": TrainingRecipe(
            strategy="sft",
            when_to_use="First student model from gold traces or curated JSONL.",
            starting_settings={
                "learning_rate": 2e-4 if hardware in {"local_12gb", "local_24gb"} else 2e-5,
                "epochs": 1 if data == "small" else 3,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 4096 if hardware != "local_12gb" else 2048,
                "use_lora": True,
                "lora_rank": 16,
                "lora_alpha": 32,
                "load_in_4bit": hardware != "dgx",
            },
            watch=["train_loss", "eval_loss", "grad_norm", "truncation", "heldout_pass@k"],
            next_steps=[
                "Inspect generated examples for truncation.",
                "Run heldout trace eval and executable environment pass@k before routing.",
            ],
            docs=["overview", "strategy", "metrics"],
        ),
        "dpo": TrainingRecipe(
            strategy="dpo",
            when_to_use="Preference refinement after SFT with chosen/rejected pairs for the same prompt.",
            starting_settings={
                "dpo_beta": 0.1,
                "learning_rate": 1e-5,
                "epochs": 1,
                "max_seq_length": 4096 if hardware != "local_12gb" else 2048,
                "use_lora": True,
                "load_in_4bit": hardware != "dgx",
            },
            watch=[
                "rewards/chosen",
                "rewards/rejected",
                "reward_margin",
                "preference_accuracy",
                "heldout_behavior",
            ],
            next_steps=[
                "Audit pairs for shared prompt identity.",
                "Compare against the SFT checkpoint on heldout eval.",
            ],
            docs=["strategy", "metrics", "glossary"],
        ),
        "grpo": TrainingRecipe(
            strategy="grpo",
            when_to_use="Verifier-backed RL when sampled attempts sometimes pass and sometimes fail.",
            starting_settings={
                "training_profile": "terminal_rl_tmax_like",
                "grpo_group_size": 32 if hardware in {"dgx", "cloud"} else 8,
                "grpo_loss_type": "dapo",
                "filter_zero_std_groups": True,
                "active_sampling": True,
                "token_level_loss": True,
                "lm_head_fp32": True,
                "prompts_per_rollout_batch": 8,
                "max_tool_calls_per_episode": 64,
            },
            watch=[
                "reward",
                "reward_std",
                "frac_reward_zero_std",
                "pass@1",
                "pass@k",
                "timeout_rate",
                "tamper_rate",
            ],
            next_steps=[
                "Confirm reward groups have non-zero variance.",
                "Export DPPO replay with behavior logprobs if using the DPPO path.",
            ],
            docs=["strategy", "metrics", "world-models"],
        ),
        "world-model": TrainingRecipe(
            strategy="world-model",
            when_to_use="Auxiliary terminal-dynamics learning around GRPO/DPPO replay.",
            starting_settings={
                "echo_enabled": True,
                "echo_aux_lambda": 0.05,
                "rwml_enabled": True,
                "rwml_distance_threshold": 0.2,
                "rwml_easy_pass_rate_threshold": 0.8,
                "rwml_easy_keep_probability": 0.1,
                "rwml_history_window": 4,
                "rwml_kl_beta": 0.0,
            },
            watch=[
                "world_model_records",
                "rwml_transitions",
                "echo_observation_chars",
                "echo_loss",
                "rwml_pass_rate",
                "heldout_pass@k",
            ],
            next_steps=[
                "Export replay with include_world_model_replay=true.",
                "Use bashgym replay summarize --json to inspect coverage.",
                "Run a tiny installed-backend smoke before real training.",
            ],
            docs=["world-models", "metrics", "glossary"],
        ),
    }
    if strategy in {"rlvr", "dppo"}:
        strategy = "grpo"
    if strategy not in base:
        raise SystemExit(f"unsupported strategy {strategy!r}; choose {', '.join(sorted(base))}")
    return base[strategy]


def _emit(payload: dict[str, Any], *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(payload.get("title", "BashGym"))
    for key, value in payload.items():
        if key == "title":
            continue
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, indent=2, sort_keys=True)}")
        else:
            print(f"{key}: {value}")
    return 0


def _doc_entries() -> list[dict[str, Any]]:
    entries = []
    for topic, meta in DOCS.items():
        path = REPO_ROOT / meta["path"]
        entries.append(
            {
                "topic": topic,
                "path": str(path),
                "exists": path.exists(),
                "summary": meta["summary"],
            }
        )
    return entries


def cmd_manifest(args: argparse.Namespace) -> int:
    payload = {
        "title": "BashGym Agent Manifest",
        "ok": True,
        "commands": {
            "manifest": "Show agent-readable command and docs map.",
            "training docs": "List or read training docs by topic.",
            "training plan": "Recommend starting settings and metrics for a strategy.",
            "training analyze": "Analyze training metrics, DPPO replay, and release evidence.",
            "replay summarize": "Summarize DPPO replay JSONL, including world-model coverage.",
            "serve": "Start the existing BashGym FastAPI backend.",
        },
        "docs": _doc_entries(),
        "next": [
            {
                "reason": "Read training overview before changing run settings.",
                "command": "bashgym training docs --topic overview --json",
            },
            {
                "reason": "Generate a starter run config.",
                "command": "bashgym training plan --strategy sft --json",
            },
        ],
    }
    return _emit(payload, as_json=args.json)


def cmd_training_docs(args: argparse.Namespace) -> int:
    if args.topic:
        if args.topic not in DOCS:
            raise SystemExit(f"unknown topic {args.topic!r}; choose {', '.join(DOCS)}")
        meta = DOCS[args.topic]
        path = REPO_ROOT / meta["path"]
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        payload = {
            "title": f"BashGym Training Docs: {args.topic}",
            "ok": path.exists(),
            "topic": args.topic,
            "path": str(path),
            "summary": meta["summary"],
            "content": text if args.include_content or args.json else "",
            "next": [{"reason": "Pick a strategy.", "command": "bashgym training plan --json"}],
        }
        return _emit(payload, as_json=args.json)

    payload = {
        "title": "BashGym Training Docs",
        "ok": True,
        "docs": _doc_entries(),
        "next": [
            {
                "reason": "Read the compact glossary.",
                "command": "bashgym training docs --topic glossary --json",
            }
        ],
    }
    return _emit(payload, as_json=args.json)


def cmd_training_plan(args: argparse.Namespace) -> int:
    recipe = _recipe(args.strategy, args.hardware, args.data)
    payload = {
        "title": f"BashGym Training Plan: {recipe.strategy}",
        "ok": True,
        "strategy": recipe.strategy,
        "hardware": args.hardware,
        "data": args.data,
        "when_to_use": recipe.when_to_use,
        "starting_settings": recipe.starting_settings,
        "watch": recipe.watch,
        "docs": [
            {"topic": topic, "path": str(REPO_ROOT / DOCS[topic]["path"])} for topic in recipe.docs
        ],
        "next": [{"reason": reason, "command": command} for reason, command in _plan_next(recipe)],
    }
    return _emit(payload, as_json=args.json)


def cmd_training_analyze(args: argparse.Namespace) -> int:
    if args.run_id is None and args.metrics is None:
        raise SystemExit("training analyze requires --run-id or --metrics")
    analysis = analyze_run_artifacts(
        run_id=args.run_id,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        metrics_path=args.metrics,
        replay_path=args.replay,
        release_evidence_path=args.release_evidence,
    )
    payload = {
        "title": "BashGym Training Analysis",
        **analysis,
    }
    return _emit(payload, as_json=args.json)


def _plan_next(recipe: TrainingRecipe) -> list[tuple[str, str]]:
    if recipe.strategy == "world-model":
        return [
            ("Inspect world-model docs.", "bashgym training docs --topic world-models --json"),
            (
                "Summarize an enriched replay.",
                "bashgym replay summarize data/dppo_replay/latest.jsonl --json",
            ),
        ]
    if recipe.strategy == "grpo":
        return [
            (
                "Check zero-std and pass@k diagnostics.",
                "bashgym training docs --topic metrics --json",
            ),
            (
                "Inspect world-model setup if using DPPO replay.",
                "bashgym training plan --strategy world-model --json",
            ),
        ]
    return [("Review strategy details.", f"bashgym training docs --topic {recipe.docs[0]} --json")]


def cmd_replay_summarize(args: argparse.Namespace) -> int:
    records = read_dppo_replay_jsonl(args.path)
    summary = summarize_dppo_replay_records(records)
    payload = {
        "title": "BashGym DPPO Replay Summary",
        "ok": True,
        "path": str(Path(args.path).resolve()),
        "summary": summary,
        "next": [
            {
                "reason": "Understand world-model replay metrics.",
                "command": "bashgym training docs --topic world-models --json",
            }
        ],
    }
    return _emit(payload, as_json=args.json)


def cmd_serve(args: argparse.Namespace) -> int:
    from bashgym.main import main as serve_main

    server_args: list[str] = []
    if args.host is not None:
        server_args.extend(["--host", args.host])
    if args.port is not None:
        server_args.extend(["--port", str(args.port)])
    if args.reload:
        server_args.append("--reload")
    if args.workers is not None:
        server_args.extend(["--workers", str(args.workers)])
    if args.log_level is not None:
        server_args.extend(["--log-level", args.log_level])
    if args.env_file is not None:
        server_args.extend(["--env-file", args.env_file])

    extra_args = list(args.server_args or [])
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    server_args.extend(extra_args)

    old_argv = sys.argv[:]
    try:
        sys.argv = ["bashgym serve", *server_args]
        serve_main()
    finally:
        sys.argv = old_argv
    return 0


def build_parser() -> argparse.ArgumentParser:
    json_parent = argparse.ArgumentParser(add_help=False)
    json_parent.add_argument("--json", action="store_true", help="Emit a single JSON object")

    parser = argparse.ArgumentParser(
        prog="bashgym",
        description="BashGym agent CLI",
        parents=[json_parent],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest",
        help="Show agent-readable command map",
        parents=[json_parent],
    )
    manifest.set_defaults(func=cmd_manifest)

    training = subparsers.add_parser(
        "training",
        help="Training docs and setup helpers",
        parents=[json_parent],
    )
    training_sub = training.add_subparsers(dest="training_command", required=True)

    docs = training_sub.add_parser(
        "docs",
        help="List or read training docs",
        parents=[json_parent],
    )
    docs.add_argument("--topic", choices=sorted(DOCS), help="Doc topic to read")
    docs.add_argument(
        "--include-content",
        action="store_true",
        help="Include full markdown in non-JSON output",
    )
    docs.set_defaults(func=cmd_training_docs)

    plan = training_sub.add_parser(
        "plan",
        help="Recommend starting training settings",
        parents=[json_parent],
    )
    plan.add_argument(
        "--strategy",
        default="sft",
        choices=["sft", "dpo", "grpo", "rlvr", "dppo", "world-model"],
    )
    plan.add_argument(
        "--hardware",
        default="local_12gb",
        choices=["local_12gb", "local_24gb", "dgx", "cloud"],
    )
    plan.add_argument(
        "--data",
        default="gold_traces",
        choices=["gold_traces", "small", "custom_jsonl", "terminal_envs", "security"],
    )
    plan.set_defaults(func=cmd_training_plan)

    analyze = training_sub.add_parser(
        "analyze",
        help="Analyze training run metrics and evidence",
        parents=[json_parent],
    )
    analyze.add_argument("--run-id", help="Run id under --models-dir, usually data/models/<run-id>")
    analyze.add_argument(
        "--models-dir", default="data/models", help="Directory containing training runs"
    )
    analyze.add_argument("--metrics", help="Direct path to a metrics.jsonl artifact")
    analyze.add_argument("--replay", help="Optional DPPO replay JSONL artifact")
    analyze.add_argument("--release-evidence", help="Optional release-gate evidence JSON artifact")
    analyze.set_defaults(func=cmd_training_analyze)

    replay = subparsers.add_parser(
        "replay",
        help="DPPO replay helpers",
        parents=[json_parent],
    )
    replay_sub = replay.add_subparsers(dest="replay_command", required=True)
    replay_summary = replay_sub.add_parser(
        "summarize",
        help="Summarize DPPO replay JSONL",
        parents=[json_parent],
    )
    replay_summary.add_argument("path")
    replay_summary.set_defaults(func=cmd_replay_summarize)

    serve = subparsers.add_parser(
        "serve",
        help="Start the BashGym API server",
        parents=[json_parent],
    )
    serve.add_argument("--host", help="Host to bind the server to")
    serve.add_argument("--port", type=int, help="Port to bind the server to")
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    serve.add_argument("--workers", type=int, help="Number of worker processes")
    serve.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Server log level",
    )
    serve.add_argument("--env-file", help="Path to a .env file for configuration")
    serve.add_argument("server_args", nargs=argparse.REMAINDER)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
