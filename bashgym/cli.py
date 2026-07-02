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

from bashgym.compute import (
    get_compute_target,
    launch_plan,
    list_compute_targets,
    preflight_compute_target,
)
from bashgym.eval.dppo_replay import read_dppo_replay_jsonl, summarize_dppo_replay_records
from bashgym.gym.backend_smoke_bundle import (
    BackendSmokeBundleConfig,
    prepare_backend_smoke_bundle,
)
from bashgym.gym.run_analysis import analyze_run_artifacts
from bashgym.preferences import (
    evaluate_reward_model_file,
    train_reward_model_fixture_file,
    validate_preference_pairs_file,
    validate_reward_examples_file,
)
from bashgym.run_cards import (
    attach_run_card_evidence,
    create_run_card,
    explain_run_card_promotion,
    parse_thresholds,
    read_run_card,
    validate_run_card_file,
    write_run_card,
)
from bashgym.sources import (
    DEFAULT_SOURCE_FETCH_LIMIT,
    SourceUse,
    fetch_source_records,
    get_source,
    list_sources,
    prepare_source_artifacts,
    prepare_source_manifest,
    recommend_sources,
)
from bashgym.sources.catalog import validate_catalog
from bashgym.trace_capture.status_protocol import scrub_trace_replay_file

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
    "capabilities": {
        "path": "docs/training/capability-map.md",
        "summary": "Full training/eval capability map with ready, backend-dependent, and diagnostic surfaces.",
    },
    "methods-reference": {
        "path": "docs/training/training-methods-reference.md",
        "summary": "Method-by-method reference for SFT, DPO, GRPO/RLVR, DPPO, ECHO/RWML, and related ecosystems.",
    },
    "external-review": {
        "path": "docs/training/external-review-packet.md",
        "summary": "Shareable AI/ML reviewer packet with capabilities, claims, risks, and feedback questions.",
    },
    "rlhf-handbook-comparison": {
        "path": "docs/training/rlhf-handbook-comparison.md",
        "summary": "RLHF Book comparison with BashGym strengths, gaps, reviewer answers, and action plan.",
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
    "terminal-rl-recipe": {
        "path": "docs/training/tmax-terminal-rl-recipe.md",
        "summary": "TMax-style terminal RL recipe from environment pool to release gate.",
    },
    "session-distillation": {
        "path": "docs/training/session-distillation.md",
        "summary": "Hint-injected self-distillation method, artifact contract, defaults, and feasibility sources.",
    },
    "private-compute-checklist": {
        "path": "docs/training/private-compute-eval-checklist.md",
        "summary": "Private compute-target backend-smoke and eval checklist for DPPO/ECHO/RWML handoff.",
    },
}

DOC_ALIASES = {
    "gx10-checklist": "private-compute-checklist",
}


SETTING_GUIDANCE: dict[str, dict[str, str]] = {
    "learning_rate": {
        "means": "Optimizer step size. Higher learns faster but can destabilize loss, format, or KL.",
        "start_here": "Use smaller LR for DPO/RL and larger LR for local LoRA SFT.",
        "adjust_when": "Lower it when loss spikes, KL jumps, or heldout behavior regresses.",
    },
    "epochs": {
        "means": "Full passes over the dataset.",
        "start_here": "Use one short smoke epoch first; increase only after heldout behavior improves.",
        "adjust_when": "Lower it when train loss improves but eval/heldout gets worse.",
    },
    "batch_size": {
        "means": "Examples processed per device before gradient accumulation.",
        "start_here": "Keep it small on local GPUs; use accumulation for effective batch.",
        "adjust_when": "Lower it when VRAM is tight or OOM events appear.",
    },
    "gradient_accumulation_steps": {
        "means": "Optimizer-step batching without increasing per-step VRAM.",
        "start_here": "Use it to reach a stable effective batch on small GPUs.",
        "adjust_when": "Increase when gradients are noisy and memory headroom is limited.",
    },
    "max_seq_length": {
        "means": "Token budget per example or rollout context.",
        "start_here": "Use the shortest length that keeps instructions, tool calls, verifier output, and final fix.",
        "adjust_when": "Increase if truncation cuts important context; decrease if memory or throughput is poor.",
    },
    "use_lora": {
        "means": "Train adapter weights instead of all model weights.",
        "start_here": "Use LoRA for most BashGym iteration.",
        "adjust_when": "Consider full fine-tune only with enough data, memory, and evaluation budget.",
    },
    "lora_rank": {
        "means": "Adapter capacity.",
        "start_here": "Rank 16-32 is a practical starting range.",
        "adjust_when": "Increase for underfit specialists; decrease if overfit or memory pressure appears.",
    },
    "lora_alpha": {
        "means": "Scale applied to LoRA updates.",
        "start_here": "Start near rank or 2x rank.",
        "adjust_when": "Lower with unstable updates; raise cautiously for underfit adapters.",
    },
    "load_in_4bit": {
        "means": "QLoRA-style low-memory base-model loading.",
        "start_here": "Use on constrained local GPUs.",
        "adjust_when": "Disable when full precision is required and hardware supports it.",
    },
    "dpo_beta": {
        "means": "Reference-policy pull in DPO.",
        "start_here": "Start around 0.1.",
        "adjust_when": "Lower when reward margin grows but heldout behavior regresses.",
    },
    "training_profile": {
        "means": "Bundle of terminal-RL defaults for loss type, sampling, and rollout assumptions.",
        "start_here": "Use terminal_rl_tmax_like for verifier-backed shell environments.",
        "adjust_when": "Switch only when the backend or environment contract requires different semantics.",
    },
    "grpo_group_size": {
        "means": "Attempts sampled for one prompt and compared against each other.",
        "start_here": "Use 8 locally and 16-32 when hardware allows.",
        "adjust_when": "Increase when reward groups lack contrast; decrease when throughput is too low.",
    },
    "grpo_loss_type": {
        "means": "Policy-gradient loss variant.",
        "start_here": "Use DAPO/Dr. GRPO-style settings for long terminal rollouts.",
        "adjust_when": "Change only after reward variance, pass@k, and KL telemetry are understood.",
    },
    "filter_zero_std_groups": {
        "means": "Drop reward groups where all attempts receive the same reward.",
        "start_here": "Enable for terminal RL updates.",
        "adjust_when": "Keep zero-std groups for curriculum/world-model diagnostics, not policy updates.",
    },
    "active_sampling": {
        "means": "Sample extra groups to replace zero-variance groups.",
        "start_here": "Enable when verifier rewards are sparse.",
        "adjust_when": "Increase sampling only after verifier errors and timeouts are low.",
    },
    "token_level_loss": {
        "means": "Apply policy loss at token granularity for long trajectories.",
        "start_here": "Enable for terminal RL replay and long command traces.",
        "adjust_when": "Disable only for a backend that cannot report stable token-level telemetry.",
    },
    "lm_head_fp32": {
        "means": "Keep output head math in float32 for stability.",
        "start_here": "Enable for RL when memory allows.",
        "adjust_when": "Disable only if memory pressure blocks the smoke run.",
    },
    "prompts_per_rollout_batch": {
        "means": "How many prompts/environments are sampled per rollout batch.",
        "start_here": "Start small enough to inspect raw failures.",
        "adjust_when": "Raise after reward, timeout, and verifier-error metrics are stable.",
    },
    "max_tool_calls_per_episode": {
        "means": "Hard cap on shell/tool actions in one environment attempt.",
        "start_here": "Use enough calls for the task family without allowing long loops.",
        "adjust_when": "Lower when timeout or looping rises; raise only if valid solutions need more steps.",
    },
    "echo_aux_lambda": {
        "means": "Weight for auxiliary terminal-observation prediction loss.",
        "start_here": "Start at 0.05.",
        "adjust_when": "Lower if pass@k regresses; raise only if ECHO loss helps behavior.",
    },
    "rwml_distance_threshold": {
        "means": "Embedding-distance cutoff for a correct next-state prediction.",
        "start_here": "Start around 0.2.",
        "adjust_when": "Tighten when RWML passes too easily without behavior improvement.",
    },
    "rwml_easy_pass_rate_threshold": {
        "means": "Threshold for marking RWML transitions as easy.",
        "start_here": "Start at 0.8.",
        "adjust_when": "Raise when easy samples dominate without curriculum value.",
    },
    "rwml_easy_keep_probability": {
        "means": "Probability of retaining easy RWML transitions.",
        "start_here": "Start at 0.1 so hard transitions dominate.",
        "adjust_when": "Raise only if the model forgets simple terminal dynamics.",
    },
    "rwml_history_window": {
        "means": "Number of prior command/observation pairs included in a prediction target.",
        "start_here": "Start at 4.",
        "adjust_when": "Increase if predictions miss context; decrease if memory or noise rises.",
    },
    "rwml_kl_beta": {
        "means": "Optional KL regularization weight around RWML objective use.",
        "start_here": "Keep at 0 until backend smoke metrics justify it.",
        "adjust_when": "Raise only if RWML shaping destabilizes the policy.",
    },
    "reward_artifact": {
        "means": "Validated JSONL examples used to train or evaluate a learned reward model.",
        "start_here": "Use reward_examples.jsonl and validate it with --strict before serious runs.",
        "adjust_when": "Rebuild it when provenance, split, reward scale, or label metadata is missing.",
    },
    "reward_type": {
        "means": "Reward target family: preference reward, outcome reward, or process reward.",
        "start_here": "Use preference/outcome rewards first; use process rewards only when step labels exist.",
        "adjust_when": "Switch to PRM/process rewards only when trajectories include reliable step-level labels.",
    },
    "reward_loss": {
        "means": "Training objective shape for the reward head or scorer.",
        "start_here": "Use pairwise loss for preference comparisons and regression/classification for scored outcomes.",
        "adjust_when": "Change only after heldout pair accuracy and calibration expose the mismatch.",
    },
    "reward_scale": {
        "means": "Declared score range or label schema that makes reward values comparable.",
        "start_here": "Keep the scale explicit in the artifact instead of inferring it from values.",
        "adjust_when": "Normalize only when sources use incompatible score ranges and provenance is preserved.",
    },
    "eval_split_required": {
        "means": "Whether reward examples must include a heldout split before training claims are trusted.",
        "start_here": "Require it for reward-model, ORM, and PRM runs.",
        "adjust_when": "Do not disable it for anything beyond local fixture experiments.",
    },
    "session_distillation_alpha": {
        "means": "Blend between hinted-context KL and hard-label CE on the same target action.",
        "start_here": "Start at 0.7 so the hint guides the target while CE anchors the exact action.",
        "adjust_when": "Lower it if hinted behavior overfits or ordinary action quality regresses.",
    },
    "session_distillation_temperature": {
        "means": "Softness for the hinted-context distribution used in KL.",
        "start_here": "Start at 1.0 until run-level calibration evidence exists.",
        "adjust_when": "Increase cautiously for softer targets; lower if the objective gets too diffuse.",
    },
    "session_distillation_min_confidence": {
        "means": "Reader-confidence floor for accepting a failed-span hint.",
        "start_here": "Start at 0.6 to keep obvious local mistakes and skip weak guesses.",
        "adjust_when": "Raise it when hints are noisy; lower only for audited fixture experiments.",
    },
    "session_distillation_mask_policy": {
        "means": "Which tokens receive loss in the generated trainer script.",
        "start_here": "Use target_span_only so unrelated transcript context is not updated.",
        "adjust_when": "Do not widen until heldout behavior shows target-only masking is too narrow.",
    },
    "session_distillation_context_mode": {
        "means": "How the corrective hint is inserted into the context.",
        "start_here": "Use hint_injected to match the one-rollout self-distillation mechanism.",
        "adjust_when": "Keep fixed until alternate context construction has explicit tests and evals.",
    },
    "session_distillation_reader": {
        "means": "The reader that proposes local hints for failed spans.",
        "start_here": "Use heuristic for deterministic, auditable first runs.",
        "adjust_when": "Use model only as an audited reader after comparing hint quality on heldout traces.",
    },
}


METRIC_GUIDANCE: dict[str, dict[str, str]] = {
    "train_loss": {
        "role": "training_health",
        "means": "How well the model fits training examples.",
        "watch_for": "Should generally fall during a short smoke, but does not prove behavior.",
        "action": "If it rises, inspect LR, template formatting, truncation, and data quality.",
    },
    "eval_loss": {
        "role": "training_health",
        "means": "Loss on held-out examples from the same training format.",
        "watch_for": "Rising eval loss while train loss falls is overfit or drift.",
        "action": "Lower epochs/LR, rebalance examples, or shorten the run.",
    },
    "grad_norm": {
        "role": "training_health",
        "means": "Gradient magnitude.",
        "watch_for": "Large spikes often precede unstable updates.",
        "action": "Lower LR, add warmup/clipping, or inspect malformed examples.",
    },
    "truncation": {
        "role": "setup_check",
        "means": "Examples cut by max sequence length.",
        "watch_for": "Verifier output, final fixes, or recovery steps missing from training.",
        "action": "Raise max_seq_length or shorten examples before training longer.",
    },
    "heldout_pass@k": {
        "role": "behavior_evidence",
        "means": "Fraction of heldout tasks solved by any of k attempts.",
        "watch_for": "Flat pass@k despite better loss means the model learned format more than outcomes.",
        "action": "Add verifier-backed data or easier curriculum before scaling RL.",
    },
    "reward_margin": {
        "role": "preference_health",
        "means": "Chosen-vs-rejected reward separation.",
        "watch_for": "Growing margin with worse behavior means DPO is over-optimizing labels.",
        "action": "Lower beta/LR and audit chosen/rejected prompt identity.",
    },
    "preference_accuracy": {
        "role": "preference_health",
        "means": "How often chosen is scored above rejected.",
        "watch_for": "Noisy or flat accuracy suggests weak pairs.",
        "action": "Remove unrelated, trivial, or duplicated preference pairs.",
    },
    "reward": {
        "role": "rl_signal_quality",
        "means": "Verifier or reward-model score assigned to sampled attempts.",
        "watch_for": "Reward can rise while behavior overfits to shortcuts.",
        "action": "Pair with pass@k, holdout, timeout, tamper, and spurious controls.",
    },
    "reward_std": {
        "role": "rl_signal_quality",
        "means": "Reward variation inside sampled prompt groups.",
        "watch_for": "Near zero means GRPO has little relative signal.",
        "action": "Use active sampling, group-size changes, easier tasks, or SFT warm start.",
    },
    "frac_reward_zero_std": {
        "role": "rl_signal_quality",
        "means": "Share of reward groups with no reward variation.",
        "watch_for": "High values mean most policy updates are uninformative.",
        "action": "Filter zero-std groups for RL and mine them for curriculum diagnostics.",
    },
    "pass@1": {
        "role": "behavior_evidence",
        "means": "First-attempt task success.",
        "watch_for": "A stricter signal than pass@k for routable reliability.",
        "action": "Use holdout gates before routing based on this metric.",
    },
    "pass@k": {
        "role": "behavior_evidence",
        "means": "Task success across multiple attempts.",
        "watch_for": "Improvement can hide exploration-only gains.",
        "action": "Pair with pass@1, timeout, tamper, and holdout comparison.",
    },
    "timeout_rate": {
        "role": "safety_release",
        "means": "Fraction of attempts that exceed time/tool budgets.",
        "watch_for": "High timeouts indicate loops, blocking commands, or bad prompt budgets.",
        "action": "Lower max tool calls, inspect traces, and add concise recovery examples.",
    },
    "tamper_rate": {
        "role": "safety_release",
        "means": "Attempts that modify protected verifier/test/fixture state.",
        "watch_for": "Any non-zero value blocks promotion.",
        "action": "Fix guardrails and remove training examples that reward tampering.",
    },
    "world_model_records": {
        "role": "setup_check",
        "means": "Replay records with ECHO/RWML payloads.",
        "watch_for": "Coverage is required before world-model backend work.",
        "action": "Re-export replay with world-model payloads enabled.",
    },
    "rwml_transitions": {
        "role": "world_model_diagnostic",
        "means": "Next-state transition targets for RWML.",
        "watch_for": "Low count means the objective has little data.",
        "action": "Add rollouts with observations and command history.",
    },
    "echo_observation_chars": {
        "role": "world_model_diagnostic",
        "means": "Terminal observation text available for ECHO tokenization.",
        "watch_for": "Low coverage means the auxiliary loss has little signal.",
        "action": "Preserve raw rollout observations and re-export world-model replay.",
    },
    "echo_loss": {
        "role": "world_model_diagnostic",
        "means": "Auxiliary observation-prediction loss.",
        "watch_for": "Improvement is useful only if pass@k/safety do not regress.",
        "action": "Keep diagnostic until correlated with behavior.",
    },
    "rwml_pass_rate": {
        "role": "world_model_diagnostic",
        "means": "Share of next-state predictions within the RWML threshold.",
        "watch_for": "High pass rate without behavior gain can mean the threshold is too easy.",
        "action": "Tighten threshold or mine harder transitions.",
    },
    "heldout_pair_accuracy": {
        "role": "reward_model_health",
        "means": "How often the reward model ranks heldout chosen/preferred examples above rejected ones.",
        "watch_for": "High train accuracy with flat heldout accuracy means label memorization or leakage.",
        "action": "Audit source splits, pair difficulty, and prompt identity before using the reward model.",
    },
    "calibration_error": {
        "role": "reward_model_health",
        "means": "How well reward scores match observed correctness or preference probabilities.",
        "watch_for": "Poor calibration makes best-of-N and rejection sampling overconfident.",
        "action": "Calibrate on heldout data or keep the model in audit-only mode.",
    },
    "length_bias": {
        "role": "reward_model_health",
        "means": "Reward correlation with answer, trajectory, or command length.",
        "watch_for": "Positive length bias can select verbose or looping traces instead of better traces.",
        "action": "Add length-stratified evals, normalize features, or rebuild labels.",
    },
    "task_family_breakdown": {
        "role": "reward_model_health",
        "means": "Reward-model performance grouped by domain or task family.",
        "watch_for": "Aggregate accuracy can hide failure on the exact domain you plan to route.",
        "action": "Gate use by task family and collect more labels for weak slices.",
    },
    "reward_variance": {
        "role": "reward_model_health",
        "means": "Spread of reward scores across comparable attempts or trajectories.",
        "watch_for": "Near-zero variance means the model cannot distinguish candidates.",
        "action": "Improve labels or use verifiers/rejection controls before policy-gradient training.",
    },
    "eval_only_leakage": {
        "role": "safety_release",
        "means": "Whether eval-only sources or benchmark labels appear in training reward artifacts.",
        "watch_for": "Any leakage invalidates public benchmark or broad model-performance claims.",
        "action": "Block training export, rebuild the artifact, and preserve the manifest evidence.",
    },
    "session_distillation_loss": {
        "role": "session_distillation_health",
        "means": "Combined masked hinted-context KL and hard-label CE.",
        "watch_for": "Loss should move only with nonzero masked target tokens and cleaner heldout decisions.",
        "action": "If it falls without behavior gain, inspect hints and rebuild records around clearer spans.",
    },
    "session_distillation_kl": {
        "role": "session_distillation_health",
        "means": "KL between original-context predictions and hinted-context predictions on the target span.",
        "watch_for": "Large or unstable KL can mean noisy hints or too much alpha pressure.",
        "action": "Audit hint quality, lower alpha, or raise min confidence.",
    },
    "session_distillation_ce": {
        "role": "session_distillation_health",
        "means": "Hard-label CE on the original target action tokens.",
        "watch_for": "CE guards the exact action; rising CE can mean the KL target is pulling too hard.",
        "action": "Lower alpha or filter records where the target action is not the desired repair.",
    },
    "session_distillation_masked_tokens": {
        "role": "setup_check",
        "means": "Number of target tokens that actually receive Session Distillation loss.",
        "watch_for": "Zero masked tokens means the run is not training the intended span.",
        "action": "Rebuild records so target_text and target_span align before trusting loss metrics.",
    },
    "reader_confidence": {
        "role": "data_quality",
        "means": "Confidence assigned by the heuristic or model reader to the proposed local hint.",
        "watch_for": "Low confidence means the reader may be guessing instead of identifying a real recovery span.",
        "action": "Raise the threshold, inspect hints, or require model-reader audit before training.",
    },
    "heldout_recovery_accuracy": {
        "role": "behavior_evidence",
        "means": "How often the trained model chooses the correct recovery action on heldout failed-span cases.",
        "watch_for": "This should improve before treating Session Distillation loss as useful.",
        "action": "If flat, rebuild records from clearer failure/recovery pairs or move broader failures to DPO/GRPO.",
    },
    "tool_call_validity": {
        "role": "behavior_evidence",
        "means": "Whether generated tool calls or shell commands remain syntactically and schema valid.",
        "watch_for": "A repair objective should not break basic command/tool formatting.",
        "action": "Block promotion and mix in clean SFT/format examples if validity regresses.",
    },
}


def _recipe(strategy: str, hardware: str, data: str) -> TrainingRecipe:
    strategy = strategy.lower().replace("_", "-")
    if strategy in {"rm", "preference-rm", "preference-reward-model", "orm", "prm"}:
        strategy = "reward-model"
    if strategy in {"session", "session-distill", "session-distillation"}:
        strategy = "session-distillation"
    hardware = hardware.lower().replace("-", "_")
    if hardware in {"dgx", "gx10", "remote", "remote_gpu"}:
        hardware = "private_compute"
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
                "load_in_4bit": hardware != "private_compute",
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
                "load_in_4bit": hardware != "private_compute",
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
                "grpo_group_size": 32 if hardware in {"private_compute", "cloud"} else 8,
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
        "reward-model": TrainingRecipe(
            strategy="reward-model",
            when_to_use=(
                "Preference RM, ORM, or PRM training before rejection sampling, "
                "trajectory scoring, reward audits, or RL with a learned reward."
            ),
            starting_settings={
                "reward_artifact": "reward_examples.jsonl",
                "reward_type": (
                    "preference_reward or outcome_reward; use process_reward only with step labels"
                ),
                "reward_loss": "pairwise_or_regression",
                "reward_scale": "declared_in_artifact",
                "learning_rate": 1e-5,
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 4096 if hardware != "local_12gb" else 2048,
                "use_lora": True,
                "load_in_4bit": hardware != "private_compute",
                "eval_split_required": True,
            },
            watch=[
                "heldout_pair_accuracy",
                "calibration_error",
                "reward_margin",
                "length_bias",
                "task_family_breakdown",
                "reward_variance",
                "eval_only_leakage",
            ],
            next_steps=[
                "Validate reward examples before training: bashgym training reward-examples validate reward_examples.jsonl --strict --json.",
                "Keep RewardBench/CUARewardBench-style sources eval-only unless a source card explicitly allows training.",
                "Use the reward model first for audits, best-of-N, or rejection sampling with matched controls.",
            ],
            docs=["methods-reference", "metrics", "strategy", "rlhf-handbook-comparison"],
        ),
        "session-distillation": TrainingRecipe(
            strategy="session-distillation",
            when_to_use=(
                "Failed trace spans contain local mistakes, retries, or recovery pivots "
                "that should be repaired without rewriting the full trajectory."
            ),
            starting_settings={
                "artifact": "session_distillation_records.jsonl",
                "session_distillation_alpha": 0.7,
                "session_distillation_temperature": 1.0,
                "session_distillation_min_confidence": 0.6,
                "session_distillation_mask_policy": "target_span_only",
                "session_distillation_context_mode": "hint_injected",
                "session_distillation_reader": "heuristic",
                "max_seq_length": 4096 if hardware != "local_12gb" else 2048,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "use_lora": True,
                "load_in_4bit": hardware != "private_compute",
            },
            watch=[
                "session_distillation_loss",
                "session_distillation_kl",
                "session_distillation_ce",
                "session_distillation_masked_tokens",
                "reader_confidence",
                "heldout_recovery_accuracy",
                "tool_call_validity",
            ],
            next_steps=[
                "Build and validate session_distillation_records.jsonl from failed or recovery-rich traces.",
                "Inspect reader hints before training; noisy hints should be filtered or regenerated.",
                "Compare heldout recovery decisions against SFT and teacher-distillation baselines.",
            ],
            docs=[
                "session-distillation",
                "strategy",
                "metrics",
                "methods-reference",
            ],
        ),
    }
    if strategy in {"rlvr", "dppo"}:
        strategy = "grpo"
    if strategy not in base:
        raise SystemExit(f"unsupported strategy {strategy!r}; choose {', '.join(sorted(base))}")
    return base[strategy]


def _readiness_ladder(recipe: TrainingRecipe) -> list[dict[str, str]]:
    if recipe.strategy == "sft":
        return [
            {
                "stage": "data_contract",
                "evidence": "Gold examples are verified, deduplicated, and fit the selected max sequence length.",
                "promote_when": "Dataset inspection shows low truncation and no failed sessions mislabeled as gold.",
            },
            {
                "stage": "training_smoke",
                "evidence": "A short SFT run writes metrics with stable loss and no loader/template errors.",
                "promote_when": "Train loss moves down without grad norm spikes or obvious overfit.",
            },
            {
                "stage": "behavior_gate",
                "evidence": "Heldout trace eval and executable environment pass@k are attached.",
                "promote_when": "Behavior is not worse than baseline and pass@k improves or meets the route threshold.",
            },
        ]
    if recipe.strategy == "dpo":
        return [
            {
                "stage": "pair_contract",
                "evidence": "Chosen/rejected examples share the same prompt and differ by meaningful decisions.",
                "promote_when": "Pair audit finds no unrelated prompts, duplicated winners, or trivial rejected answers.",
            },
            {
                "stage": "preference_smoke",
                "evidence": "Reward margin and preference accuracy improve in a short run.",
                "promote_when": "Margin grows without chosen/rejected logprob collapse.",
            },
            {
                "stage": "behavior_gate",
                "evidence": "Heldout trace and environment gates compare against the SFT checkpoint.",
                "promote_when": "Preference improvements do not regress task behavior.",
            },
        ]
    if recipe.strategy == "reward-model":
        return [
            {
                "stage": "reward_artifact_contract",
                "evidence": "reward_examples.jsonl validates in strict mode with provenance, split, scale, and label metadata.",
                "promote_when": "No eval-only leakage, missing label source, missing split, or weak reward schema remains.",
            },
            {
                "stage": "heldout_reward_eval",
                "evidence": "Heldout pair accuracy, calibration, reward margin, and length-bias reports are saved.",
                "promote_when": "Heldout accuracy and calibration improve without selecting longer or easier artifacts only.",
            },
            {
                "stage": "use_site_gate",
                "evidence": "Reward-model use is tested through audit, best-of-N, or rejection sampling with a matched random control.",
                "promote_when": "Selected traces beat random-control traces on heldout pass@k and safety checks.",
            },
        ]
    if recipe.strategy == "world-model":
        return [
            {
                "stage": "replay_coverage",
                "evidence": "Replay summary reports world_model_records, ECHO observations, and RWML transitions.",
                "promote_when": "Coverage is non-zero for every enabled ECHO/RWML objective.",
            },
            {
                "stage": "backend_probe",
                "evidence": "Smoke bundle builds ECHO masks, RWML targets, and DPPO launch env vars.",
                "promote_when": "contract_ready=true and the probe reports observation tokens plus transition targets.",
            },
            {
                "stage": "diagnostic_quality",
                "evidence": "Backend metrics include echo_loss, rwml_pass_rate, and embedding-distance stats.",
                "promote_when": "Quality improves on heldout transitions and correlates with pass@k or fewer failures.",
            },
        ]
    if recipe.strategy == "session-distillation":
        return [
            {
                "stage": "record_contract",
                "evidence": "session_distillation_records.jsonl validates with original/hinted contexts, target text, target_span_only mask, verifier outcome, and provenance.",
                "promote_when": "Reader hints are inspectable, confidence-filtered, and tied to real failed/recovery spans.",
            },
            {
                "stage": "masked_loss_smoke",
                "evidence": "A short run writes session_distillation_loss, KL, CE, and masked-token metrics.",
                "promote_when": "Masked loss moves without zero masked tokens, loader errors, or hint leakage into original contexts.",
            },
            {
                "stage": "behavior_gate",
                "evidence": "Heldout recovery-decision checks and executable task behavior compare against SFT or teacher-distillation baselines.",
                "promote_when": "Recovery decisions improve without tool-format, timeout, verifier, or pass@k regression.",
            },
        ]
    return [
        {
            "stage": "rollout_contrast",
            "evidence": "Served/model rollouts produce pass/fail variation for the same prompt groups.",
            "promote_when": "reward_std is non-zero and frac_reward_zero_std is not dominating sampled groups.",
        },
        {
            "stage": "replay_contract",
            "evidence": "DPPO replay summary shows behavior logprobs and train logprobs where optimizer updates need them.",
            "promote_when": "behavior_logprobs_ready_records and train_logprobs_ready_records cover the replay.",
        },
        {
            "stage": "backend_smoke",
            "evidence": "training smoke-bundle writes readiness artifacts and a one-step backend run starts on a private compute target/backend.",
            "promote_when": "contract_ready=true, optimizer_ready=true for DPPO, and backend logs metrics/output.",
        },
        {
            "stage": "release_gate",
            "evidence": "Heldout, pass@k, tamper, spurious-reward, and external evidence are attached.",
            "promote_when": "Candidate beats or matches baseline without verifier, timeout, safety, or tamper regressions.",
        },
    ]


def _adjustment_rules(recipe: TrainingRecipe) -> list[dict[str, str]]:
    rules = [
        {
            "signal": "loss improves but heldout pass@k is flat",
            "action": "Inspect truncation and data quality; add verifier-backed examples before increasing epochs.",
        },
        {
            "signal": "timeouts, verifier errors, or tamper attempts rise",
            "action": "Stop promotion work, inspect rollouts, fix the environment or reward before training longer.",
        },
    ]
    if recipe.strategy == "sft":
        return rules + [
            {
                "signal": "eval loss rises while train loss falls",
                "action": "Lower epochs or learning rate, add dropout, and re-balance weak examples out of gold.",
            },
            {
                "signal": "tool-call format remains unreliable",
                "action": "Add clean format-heavy gold traces before moving to DPO or RL.",
            },
        ]
    if recipe.strategy == "dpo":
        return rules + [
            {
                "signal": "reward margin grows but behavior worsens",
                "action": "Lower beta or LR and rebuild pairs from same-prompt decision mistakes.",
            },
            {
                "signal": "preference accuracy is noisy",
                "action": "Audit prompt identity and remove trivial or unrelated rejected examples.",
            },
        ]
    if recipe.strategy == "reward-model":
        return rules + [
            {
                "signal": "heldout pair accuracy is high but calibration is poor",
                "action": "Keep the reward model audit-only, calibrate scores, and avoid best-of-N selection.",
            },
            {
                "signal": "length bias or task-family skew appears",
                "action": "Add length/task-family slices and block use on weak domains until labels improve.",
            },
            {
                "signal": "reward variance is near zero",
                "action": "Rebuild labels or use verifier rewards before using the model for RL or rejection sampling.",
            },
        ]
    if recipe.strategy == "world-model":
        return rules + [
            {
                "signal": "world-model coverage exists but quality is unclear",
                "action": "Run a tiny backend smoke that logs ECHO/RWML quality; keep metrics diagnostic-only.",
            },
            {
                "signal": "RWML pass rate is high but pass@k does not move",
                "action": "Tighten the distance threshold or mine high-error transitions for curriculum data.",
            },
        ]
    if recipe.strategy == "session-distillation":
        return rules + [
            {
                "signal": "masked tokens are zero or missing",
                "action": "Rebuild records so target_text and target_span point to the exact command/tool tokens being trained.",
            },
            {
                "signal": "reader confidence is low or hints are noisy",
                "action": "Raise min confidence, inspect failed spans, or switch to a model-reader audit before training.",
            },
            {
                "signal": "masked loss improves but heldout recovery decisions are flat",
                "action": "Use clearer failure/recovery spans or move broader mistakes to DPO/GRPO instead of increasing alpha.",
            },
        ]
    return rules + [
        {
            "signal": "reward_std is zero or frac_reward_zero_std is near 1.0",
            "action": "Enable active sampling, increase group size if compute allows, or return to SFT/distillation.",
        },
        {
            "signal": "smoke bundle has contract_ready=false",
            "action": "Fix replay/logprob/world-model coverage locally before spending private compute time.",
        },
    ]


def _metric_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": "setup_contracts",
            "role": "setup_check",
            "metrics": [
                "dataset_size",
                "truncation",
                "contract_ready",
                "optimizer_ready",
                "world_model_records",
            ],
            "decision": "Fix before training or before spending private compute/backend time.",
        },
        {
            "id": "optimization_health",
            "role": "training_health",
            "metrics": ["train_loss", "eval_loss", "grad_norm", "learning_rate", "kl", "entropy"],
            "decision": "Tune LR, warmup, epochs, sequence length, and loss weights.",
        },
        {
            "id": "session_distillation_health",
            "role": "training_health",
            "metrics": [
                "session_distillation_loss",
                "session_distillation_kl",
                "session_distillation_ce",
                "session_distillation_masked_tokens",
                "reader_confidence",
            ],
            "decision": "Trust only when masked loss aligns with heldout recovery decisions and tool validity.",
        },
        {
            "id": "preference_health",
            "role": "preference_health",
            "metrics": [
                "rewards/chosen",
                "rewards/rejected",
                "reward_margin",
                "preference_accuracy",
            ],
            "decision": "Trust only when heldout behavior does not regress against the SFT base.",
        },
        {
            "id": "reward_model_health",
            "role": "reward_model_health",
            "metrics": [
                "heldout_pair_accuracy",
                "calibration_error",
                "reward_margin",
                "length_bias",
                "task_family_breakdown",
                "reward_variance",
            ],
            "decision": "Use learned rewards only after heldout reward quality and bias checks pass.",
        },
        {
            "id": "rl_signal_quality",
            "role": "rl_signal_quality",
            "metrics": [
                "reward",
                "reward_std",
                "frac_reward_zero_std",
                "verifier_error_rate",
                "timeout_rate",
            ],
            "decision": "Scale RL only when reward groups have contrast and verifier errors are low.",
        },
        {
            "id": "behavior_evidence",
            "role": "behavior_evidence",
            "metrics": [
                "pass@1",
                "pass@k",
                "heldout_trace_delta",
                "holdout_comparison_delta",
                "external_benchmark_score",
            ],
            "decision": "Use these to decide whether the model is actually better.",
        },
        {
            "id": "safety_release",
            "role": "safety_release",
            "metrics": [
                "tamper_rate",
                "spurious_control_pass@1",
                "canary_failures",
                "verifier_error_rate",
            ],
            "decision": "Any tamper or reward-hacking signal blocks promotion.",
        },
        {
            "id": "world_model_diagnostics",
            "role": "world_model_diagnostic",
            "metrics": [
                "echo_loss",
                "echo_loss_delta",
                "rwml_pass_rate",
                "embedding_distance_mean",
                "embedding_distance_p95",
                "exit_code_accuracy",
                "test_result_accuracy",
            ],
            "decision": "Use for curriculum and diagnosis until correlated with pass@k and safety.",
        },
        {
            "id": "hardware_efficiency",
            "role": "operational_health",
            "metrics": [
                "tokens_per_second",
                "gpu_memory_peak_gb",
                "oom_count",
                "backend_import_status",
            ],
            "decision": "Use to size batch, sequence length, backend choice, and compute-target readiness.",
        },
    ]


def _recipe_stages() -> list[dict[str, str]]:
    return [
        {
            "id": "orient",
            "operator_question": "What are we trying to teach: format, preference, verifier outcome, or dynamics?",
            "proceed_when": "A strategy and first evaluation gate are selected before training.",
        },
        {
            "id": "data_contract",
            "operator_question": "Is the data valid for the selected strategy?",
            "proceed_when": "Examples, pairs, environments, or replay pass schema and split checks.",
        },
        {
            "id": "local_smoke",
            "operator_question": "Can a tiny run write metrics and artifacts locally?",
            "proceed_when": "Metrics JSONL, logs, and expected artifacts exist without loader/template errors.",
        },
        {
            "id": "behavior_baseline",
            "operator_question": "What does the base/SFT model solve before new RL or DPPO work?",
            "proceed_when": "Heldout trace or environment pass@k baseline is saved.",
        },
        {
            "id": "training_run",
            "operator_question": "Did the run improve the intended signal without breaking operations?",
            "proceed_when": "Training health, signal quality, timeout, verifier, and OOM metrics are acceptable.",
        },
        {
            "id": "release_evidence",
            "operator_question": "Did behavior improve on heldout tasks and controls?",
            "proceed_when": "Heldout, pass@k, holdout comparison, spurious, tamper, and benchmark evidence are attached.",
        },
        {
            "id": "backend_smoke",
            "operator_question": "If using DPPO/ECHO/RWML, can the installed backend consume the handoff?",
            "proceed_when": "Smoke bundle is ready and one installed-backend smoke saves logs/artifacts.",
        },
        {
            "id": "route_or_iterate",
            "operator_question": "Where is the student proven good enough, and where should it fall back?",
            "proceed_when": "Routing scope, rollback path, and next-data plan are explicit.",
        },
    ]


def _education_path() -> list[dict[str, str]]:
    return [
        {
            "id": "mental_model",
            "read": "docs/training/overview.md",
            "goal": "Understand traces, environments, strategies, and why loss is not release evidence.",
        },
        {
            "id": "settings",
            "read": "bashgym training plan --strategy sft --json",
            "goal": "Read starting settings plus per-setting guidance before changing inputs.",
        },
        {
            "id": "metrics",
            "read": "docs/training/metrics-runbook.md",
            "goal": "Know which metrics are setup checks, health signals, behavior evidence, or blockers.",
        },
        {
            "id": "terminal_rl_recipe",
            "read": "docs/training/tmax-terminal-rl-recipe.md",
            "goal": "Follow the environment-to-replay-to-backend sequence without skipping eval gates.",
        },
        {
            "id": "session_distillation",
            "read": "docs/training/session-distillation.md",
            "goal": "Understand one-rollout hint insertion, target-span masking, and when to use the method.",
        },
        {
            "id": "world_models",
            "read": "docs/training/world-models.md",
            "goal": "Use ECHO/RWML as diagnostics until correlated with heldout pass@k and safety.",
        },
        {
            "id": "private_compute_finalization",
            "read": "docs/training/private-compute-eval-checklist.md",
            "goal": "Run private compute evals/backend smokes only after local smoke artifacts are complete.",
        },
    ]


def _capability_matrix() -> dict[str, Any]:
    return {
        "status_key": {
            "ready": "User-facing workflow has implementation plus docs/tests or smoke evidence.",
            "ready_with_evidence": "Usable when the user supplies the required data, verifier, or release evidence.",
            "backend_dependent": "BashGym has contracts/adapters/planning, but an installed trainer backend must prove execution.",
            "diagnostic": "Useful for investigation or curriculum; not enough to approve promotion by itself.",
        },
        "journey": [
            {
                "stage": "learn_and_plan",
                "can_do": "Inspect training docs, generate starter plans, and choose a strategy.",
                "surfaces": ["Training Guides UI", "bashgym manifest", "bashgym training plan"],
                "evidence": ["plan JSON", "chosen strategy rationale"],
            },
            {
                "stage": "build_data",
                "can_do": "Convert traces, custom JSONL, decision pairs, failed-span Session Distillation records, security data, and terminal environments into artifacts.",
                "surfaces": ["Trace import", "Data Designer", "Decision DPO", "Environment Lab"],
                "evidence": [
                    "dataset manifests",
                    "quality labels",
                    "contamination checks",
                    "verifier metadata",
                ],
            },
            {
                "stage": "train",
                "can_do": "Run or plan SFT, DPO, GRPO/RLVR, distillation, Session Distillation, cascade RL, DPPO, and ECHO/RWML objectives.",
                "surfaces": [
                    "Training Config",
                    "Training Dashboard",
                    "trainer API",
                    "generated scripts",
                ],
                "evidence": [
                    "config snapshot",
                    "logs",
                    "metrics JSONL",
                    "checkpoints",
                    "backend version",
                ],
            },
            {
                "stage": "evaluate",
                "can_do": "Run heldout traces, environment pass@k, holdout gates, comparisons, canaries, and benchmark ingest.",
                "surfaces": ["Evaluator", "Environment Lab", "/api/eval/*", "model registry"],
                "evidence": [
                    "release verdict JSON",
                    "pass@k reports",
                    "holdout manifests",
                    "benchmark manifests",
                ],
            },
            {
                "stage": "analyze_and_promote",
                "can_do": "Combine metrics, replay, smoke readiness, and release evidence before routing.",
                "surfaces": [
                    "bashgym training analyze",
                    "Training Monitor",
                    "model registry",
                    "router",
                ],
                "evidence": ["analysis JSON", "release gate", "rollback path"],
            },
        ],
        "platform_surfaces": [
            {
                "id": "agent_cli",
                "role": "Machine-readable inspection, planning, replay summaries, smoke bundles, and conservative run analysis.",
                "commands": [
                    "bashgym manifest --json",
                    "bashgym training capabilities --json",
                    "bashgym training docs --topic <topic> --json",
                    "bashgym training plan --strategy <strategy> --json",
                    "bashgym replay summarize <path> --json",
                    "bashgym training smoke-bundle --replay <path> --output-dir <dir> --json",
                    "bashgym training analyze --metrics <path> --replay <path> --smoke-bundle <path> --release-evidence <path> --json",
                    "bashgym serve --host 127.0.0.1 --port 8000",
                ],
                "best_for": [
                    "new-session handoff",
                    "agent automation",
                    "compute-target artifact preparation",
                    "post-run diagnosis",
                ],
            },
            {
                "id": "training_api",
                "role": "Start, monitor, pause/resume/stop, export, and inspect training runs through the existing FastAPI backend.",
                "endpoints": [
                    "POST /api/training/start",
                    "GET /api/training/{run_id}",
                    "POST /api/training/{run_id}/pause",
                    "POST /api/training/{run_id}/resume",
                    "POST /api/training/{run_id}/stop",
                    "GET /api/training/runs",
                    "GET /api/training/runs/{run_id}/metrics",
                    "POST /api/training/export",
                    "POST /api/training/managed/submit",
                ],
                "best_for": ["UI-backed training", "managed jobs", "metrics persistence"],
            },
            {
                "id": "environment_api",
                "role": "Import, normalize, decontaminate, materialize, roll out, and gate executable terminal environments.",
                "endpoints": [
                    "GET /api/environments/pipelines",
                    "POST /api/environments/normalize",
                    "POST /api/environments/import-jsonl",
                    "POST /api/environments/decontaminate",
                    "POST /api/environments/materialize",
                    "POST /api/eval/environments/passk",
                    "POST /api/eval/environments/local-rollout-passk",
                    "POST /api/eval/environments/model-rollout-passk",
                    "POST /api/eval/environments/holdout-gate",
                    "POST /api/eval/environments/holdout-comparison",
                    "POST /api/eval/environments/spurious-reward-control",
                    "POST /api/eval/environments/reward-hacking-canaries",
                ],
                "best_for": ["TMax-style tasks", "terminal RL", "release gates"],
            },
            {
                "id": "eval_api",
                "role": "Combine heldout trace, executable-environment, public benchmark, and registry evidence.",
                "endpoints": [
                    "POST /api/eval/heldout",
                    "GET /api/eval/heldout/{job_id}",
                    "GET /api/eval/verdict/{model_id}",
                    "GET /api/eval/benchmark-commands",
                    "POST /api/eval/benchmarks/ingest",
                    "POST /api/eval/benchmarks/external-ingest",
                    "POST /api/eval/environments/dppo-replay/enrich",
                    "POST /api/eval/environments/dppo-replay/smoke-plan",
                ],
                "best_for": ["release evidence", "public benchmark ingest", "DPPO handoff"],
            },
            {
                "id": "device_and_hardware_api",
                "role": "Discover, register, preflight, and select private compute targets before large runs.",
                "endpoints": [
                    "GET /api/devices",
                    "POST /api/devices",
                    "POST /api/devices/discover",
                    "POST /api/devices/{device_id}/set-default",
                    "POST /api/devices/{device_id}/preflight",
                    "GET /api/ssh/preflight",
                    "GET /api/system/info",
                    "GET /api/system/gpus",
                    "GET /api/system/recommendations",
                    "GET /api/models/discover",
                ],
                "best_for": [
                    "compute-target readiness",
                    "private compute training",
                    "model fit checks",
                ],
            },
            {
                "id": "ui_surfaces",
                "role": "Human-facing control rooms for configuring, teaching, evaluating, and comparing training work.",
                "screens": [
                    "Training Monitor",
                    "Training Configuration",
                    "Training Guides",
                    "World-Model Quality panel",
                    "Factory -> Environment Lab",
                    "Evaluator -> Held-out Gate",
                    "Evaluator -> External benchmark ingest",
                    "Models -> profile, leaderboard, comparison, trends",
                    "Settings/Devices for private compute targets and model providers",
                ],
                "best_for": [
                    "operator education",
                    "guided config",
                    "manual evidence attachment",
                    "model promotion review",
                ],
            },
        ],
        "metric_catalog": _metric_catalog(),
        "recipe_stages": _recipe_stages(),
        "education_path": _education_path(),
        "data_sources": [
            {
                "id": "gold_traces",
                "status": "ready",
                "produces": [
                    "structured tool-call messages",
                    "SFT examples",
                    "DPO chosen examples",
                ],
                "best_for": ["first SFT baseline", "format learning", "repo-specific specialists"],
                "quality_gate": "Gold traces should be verified, complete, deduplicated, and low-truncation.",
            },
            {
                "id": "silver_bronze_failed_traces",
                "status": "ready_with_evidence",
                "produces": ["DPO rejected examples", "failure analysis", "curriculum gaps"],
                "best_for": ["preference training", "decision-level contrast", "metrics debugging"],
                "quality_gate": "Do not mix failed traces into SFT as success examples.",
            },
            {
                "id": "custom_jsonl",
                "status": "ready",
                "produces": ["SFT/DPO-compatible message datasets"],
                "best_for": [
                    "curated external data",
                    "hand-authored examples",
                    "migration from other tools",
                ],
                "quality_gate": "Validate message schema, tool-call JSON strings, and source manifests.",
            },
            {
                "id": "security_datasets",
                "status": "ready_with_evidence",
                "produces": ["security-specialist examples", "classification/analysis traces"],
                "best_for": ["malware/phishing/security-specialist behavior"],
                "quality_gate": "Preserve source labels, enrichment mode, and domain-specific safety context.",
            },
            {
                "id": "synthetic_data_designer",
                "status": "ready_with_evidence",
                "produces": [
                    "synthetic SFT rows",
                    "DPO pairs",
                    "tool-use rows",
                    "terminal environment proposals",
                ],
                "best_for": ["coverage expansion", "schema evolution", "environment generation"],
                "quality_gate": "Keep seed/source manifest, validator status, and decontamination metadata.",
            },
            {
                "id": "session_distillation_records",
                "status": "ready_with_evidence",
                "produces": ["hint-injected contexts", "target spans", "masked loss records"],
                "best_for": ["local failed-action repair", "recovery pivots", "retry-loop cleanup"],
                "quality_gate": "Original context must stay hint-free, hinted context must contain the hint tag, and target_span_only masks must align to target_text.",
            },
            {
                "id": "terminal_environments",
                "status": "ready_with_evidence",
                "produces": [
                    "rollout attempts",
                    "verifier rewards",
                    "pass@k reports",
                    "DPPO replay",
                ],
                "best_for": ["GRPO/RLVR", "DPPO", "holdout gates", "world-model replay"],
                "quality_gate": "Materialization, verifier-only pass, protected-file manifest, and split metadata must exist.",
            },
            {
                "id": "public_preference_reward_sources",
                "status": "ready_with_evidence",
                "produces": [
                    "DPO pairs",
                    "reward examples",
                    "reward-model eval manifests",
                ],
                "best_for": ["DPO", "Preference RM", "ORM", "PRM", "reward audits"],
                "quality_gate": "Preserve source manifest, label source, split, reward scale, and eval-only policy.",
            },
        ],
        "artifact_contracts": [
            {
                "id": "training_examples_jsonl",
                "owner_stage": "build_data",
                "used_by": ["SFT", "DPO", "distillation", "external trainers"],
                "must_preserve": ["messages", "tools", "metadata", "source_trace", "quality_score"],
            },
            {
                "id": "dpo_pairs_jsonl",
                "owner_stage": "build_data",
                "used_by": ["DPO", "preference ecosystem export"],
                "must_preserve": [
                    "prompt identity",
                    "chosen",
                    "rejected",
                    "pair source",
                    "quality labels",
                ],
            },
            {
                "id": "reward_examples_jsonl",
                "owner_stage": "build_data",
                "used_by": ["Preference RM", "ORM", "PRM", "rejection sampling"],
                "must_preserve": [
                    "reward type",
                    "prompt or trajectory",
                    "reward value or step rewards",
                    "label source",
                    "reward scale",
                    "split/decontamination metadata",
                ],
            },
            {
                "id": "reward_eval_json",
                "owner_stage": "evaluate_reward_model",
                "used_by": ["Preference RM", "ORM", "PRM", "RunCard promotion evidence"],
                "must_preserve": [
                    "heldout pair accuracy",
                    "calibration error",
                    "reward margin",
                    "length bias",
                    "task-family breakdown",
                    "eval-only leakage checks",
                ],
            },
            {
                "id": "session_distillation_records_jsonl",
                "owner_stage": "build_data",
                "used_by": [
                    "Session Distillation",
                    "Data Designer reader audit",
                    "RunCard promotion evidence",
                ],
                "must_preserve": [
                    "original_context",
                    "hinted_context",
                    "hint_text",
                    "target_text",
                    "target_span",
                    "loss_mask",
                    "reader_confidence",
                    "verifier_outcome",
                    "source_metadata",
                ],
            },
            {
                "id": "environment_spec",
                "owner_stage": "build_data",
                "used_by": ["environment rollouts", "pass@k", "holdout gates", "terminal RL"],
                "must_preserve": [
                    "id",
                    "instruction",
                    "workspace/build hints",
                    "verifier",
                    "protected-file manifest",
                ],
            },
            {
                "id": "metrics_jsonl",
                "owner_stage": "train",
                "used_by": ["training analyze", "Training Monitor", "release review"],
                "must_preserve": [
                    "step",
                    "loss/reward",
                    "reward_std",
                    "pass@k",
                    "timeout/tamper",
                    "world-model metrics",
                ],
            },
            {
                "id": "dppo_replay_jsonl",
                "owner_stage": "train",
                "used_by": ["DPPO backend", "ECHO/RWML backend probes", "training smoke-bundle"],
                "must_preserve": [
                    "environment",
                    "trajectory",
                    "reward",
                    "behavior/train logprobs",
                    "world_model payload",
                ],
            },
            {
                "id": "backend_smoke_bundle",
                "owner_stage": "prepare_backend_smoke",
                "used_by": [
                    "compute-target preflight",
                    "installed-backend smoke",
                    "training analyze",
                ],
                "must_preserve": [
                    "backend_smoke_readiness.json",
                    "dppo_replay_summary.json",
                    "world_model_backend_probe.json",
                    "dppo_launch_env.json",
                ],
            },
            {
                "id": "release_evidence_json",
                "owner_stage": "evaluate",
                "used_by": ["release gate", "model registry", "router decision"],
                "must_preserve": [
                    "heldout verdict",
                    "environment gates",
                    "external benchmark evidence",
                    "diagnostic world_model_quality",
                ],
            },
        ],
        "model_family_support": [
            {
                "id": "gemma4",
                "status": "profiled",
                "tool_call_format": "gemma4_delimited",
                "training_notes": [
                    "thinking template",
                    "multimodal module excludes",
                    "Gemma-specific patch",
                ],
                "best_fit": [
                    "small local specialists",
                    "larger GPU runs",
                    "MoE targets when hardware allows",
                ],
            },
            {
                "id": "qwen3",
                "display_name": "Qwen3 / Qwen3.6 family",
                "status": "profiled",
                "tool_call_format": "qwen_xml",
                "training_notes": ["thinking template", "long-context coding/reasoning family"],
                "checkpoint_guidance": "Prefer the newest compatible Qwen3.6, Qwen3-Coder, or provider-hosted Qwen3 checkpoint for the target hardware/backend.",
                "best_fit": ["coding agents", "terminal RL", "long-context traces"],
            },
            {
                "id": "qwen2.5",
                "display_name": "Qwen2.5 family",
                "status": "profiled",
                "tool_call_format": "qwen_xml",
                "training_notes": ["stable coder/instruct fallback family"],
                "checkpoint_guidance": "Use as a stable fallback when the latest Qwen3/Qwen3.6 checkpoint is unavailable or too large.",
                "best_fit": ["coding traces", "small-to-mid open model baselines"],
            },
            {
                "id": "llama3",
                "status": "profiled",
                "tool_call_format": "openai_json",
                "training_notes": ["OpenAI-style tool-call JSON"],
                "best_fit": ["general instruct baselines", "portable adapter experiments"],
            },
            {
                "id": "generic_hf_causal_lm",
                "status": "fallback",
                "tool_call_format": "openai_json",
                "training_notes": [
                    "works when no family-specific profile matches",
                    "add a profile for production-quality support",
                ],
                "best_fit": ["new open-weight models", "quick compatibility tests"],
            },
        ],
        "hardware_profiles": [
            {
                "id": "local_12gb",
                "best_for": ["fast iteration", "smoke tests", "small LoRA/QLoRA specialists"],
                "recommended_settings": [
                    "batch_size=1",
                    "gradient_accumulation_steps=8",
                    "max_seq_length=2048",
                    "load_in_4bit=true",
                ],
                "watch": ["OOM", "truncation", "large-vocab loss memory"],
            },
            {
                "id": "local_24gb",
                "best_for": ["larger local adapters", "longer traces", "stronger SFT baselines"],
                "recommended_settings": [
                    "max_seq_length=4096",
                    "LoRA rank 16-32",
                    "QLoRA for dense models",
                ],
                "watch": ["eval loss", "adapter overfit", "checkpoint size"],
            },
            {
                "id": "private_compute_target",
                "best_for": [
                    "larger dense/MoE targets",
                    "full or longer-context fine-tunes",
                    "DPPO/ECHO/RWML backend smoke",
                ],
                "recommended_settings": [
                    "Python 3.12 CUDA env",
                    "plain Transformers fallback when Unsloth cannot load",
                    "one-step backend smoke before scaling",
                ],
                "watch": ["backend imports", "CUDA/Triton compatibility", "artifact sync"],
            },
            {
                "id": "cloud_backend",
                "best_for": [
                    "managed fine-tunes",
                    "large batch experiments",
                    "external trainer backends",
                ],
                "recommended_settings": [
                    "preserve BashGym manifests",
                    "ingest metrics/release evidence back locally",
                ],
                "watch": ["data egress", "benchmark leakage", "reproducibility"],
            },
        ],
        "config_axes": [
            {
                "id": "data_scope",
                "choices": ["generalist", "mixed", "specialist"],
                "decision_rule": "Start generalist; use specialist only when a repo/domain has enough verified traces.",
            },
            {
                "id": "adapter_mode",
                "choices": ["LoRA", "QLoRA", "full fine-tune"],
                "decision_rule": "Use QLoRA on constrained local GPUs; consider full fine-tune only with abundant remote memory.",
            },
            {
                "id": "sequence_length",
                "choices": ["2048", "4096", "8192+"],
                "decision_rule": "Use the shortest context that preserves verifier output, final edit, and recovery steps.",
            },
            {
                "id": "terminal_rl_sampling",
                "choices": [
                    "group size",
                    "prompts per rollout batch",
                    "max tool calls",
                    "active sampling",
                    "zero-std filtering",
                ],
                "decision_rule": "Increase sample volume only when reward groups have contrast and verifier errors are low.",
            },
            {
                "id": "world_model_objectives",
                "choices": ["ECHO", "RWML", "both", "off"],
                "decision_rule": "Enable only when replay carries observations/transitions; keep quality metrics diagnostic.",
            },
            {
                "id": "promotion_thresholds",
                "choices": [
                    "heldout pass@k",
                    "holdout comparison delta",
                    "timeout/tamper limits",
                    "external benchmark minimums",
                ],
                "decision_rule": "Set thresholds before training and preserve the release evidence used to decide.",
            },
        ],
        "training": [
            {
                "id": "sft",
                "name": "Supervised fine-tuning",
                "status": "ready",
                "use_when": "The model needs tool-call format, repo conventions, or a first local baseline.",
                "knobs": ["learning_rate", "epochs", "max_seq_length", "lora_rank", "load_in_4bit"],
                "evidence": ["eval_loss", "heldout trace behavior", "environment pass@k"],
                "plan_command": "bashgym training plan --strategy sft --json",
            },
            {
                "id": "dpo",
                "name": "Direct preference optimization",
                "status": "ready",
                "use_when": "You have chosen/rejected responses for the same prompt, usually after SFT.",
                "knobs": ["dpo_beta", "learning_rate", "epochs", "pair filtering"],
                "evidence": ["reward margin", "preference accuracy", "no heldout regression"],
                "plan_command": "bashgym training plan --strategy dpo --json",
            },
            {
                "id": "reward_modeling",
                "name": "Reward-model, ORM, and PRM lane",
                "status": "ready_with_evidence",
                "use_when": "You need a learned scorer for reward audits, best-of-N, rejection sampling, or later RL.",
                "knobs": [
                    "reward_artifact",
                    "reward_type",
                    "reward_loss",
                    "reward_scale",
                    "learning_rate",
                ],
                "evidence": [
                    "strict reward examples",
                    "fixture smoke artifacts",
                    "heldout pair accuracy",
                    "calibration",
                    "length/task-family bias checks",
                ],
                "plan_command": "bashgym training plan --strategy reward-model --json",
            },
            {
                "id": "grpo_rlvr",
                "name": "GRPO/RLVR terminal RL",
                "status": "ready_with_evidence",
                "use_when": "Executable verifiers exist and sampled attempts sometimes pass and sometimes fail.",
                "knobs": [
                    "training_profile",
                    "grpo_group_size",
                    "active_sampling",
                    "filter_zero_std_groups",
                ],
                "evidence": ["reward_std", "frac_reward_zero_std", "pass@k", "timeout/tamper rate"],
                "plan_command": "bashgym training plan --strategy grpo --data terminal_envs --json",
            },
            {
                "id": "distillation",
                "name": "Teacher distillation",
                "status": "ready",
                "use_when": "A smaller model is too weak for RL or should imitate a stronger teacher.",
                "knobs": ["teacher_model", "teacher_temperature", "distillation_alpha"],
                "evidence": [
                    "student pass@k",
                    "teacher comparison",
                    "tool-format regression check",
                ],
                "plan_command": "bashgym training docs --topic strategy --json",
            },
            {
                "id": "session_distillation",
                "name": "Session Distillation",
                "status": "ready_with_evidence",
                "use_when": "Failed trace spans show local mistakes that can be repaired with a short hint without replacing the trajectory.",
                "knobs": [
                    "session_distillation_alpha",
                    "session_distillation_temperature",
                    "session_distillation_min_confidence",
                    "target_span_only mask",
                    "reader",
                ],
                "evidence": [
                    "valid session_distillation_records.jsonl",
                    "masked KL/CE metrics",
                    "heldout recovery-decision accuracy",
                    "tool-call validity",
                ],
                "plan_command": "bashgym training plan --strategy session-distillation --json",
            },
            {
                "id": "cascade_rl",
                "name": "Cascade RL",
                "status": "ready_with_evidence",
                "use_when": "You need staged domain learning and final generalist behavior.",
                "knobs": [
                    "domain stages",
                    "stage steps",
                    "private compute target",
                    "MOPD settings",
                ],
                "evidence": ["per-domain holdouts", "stage forgetting", "final generalist holdout"],
                "plan_command": "bashgym training docs --topic capabilities --json",
            },
            {
                "id": "dppo_replay",
                "name": "DPPO replay/backend path",
                "status": "backend_dependent",
                "use_when": "You have terminal rollouts, behavior/train logprobs, and an external backend.",
                "knobs": [
                    "backend",
                    "Binary-TV/KL threshold",
                    "replay path",
                    "train-logprob enrichment",
                ],
                "evidence": [
                    "backend_smoke_readiness.json",
                    "mask telemetry",
                    "one-step backend logs",
                ],
                "plan_command": "bashgym training smoke-bundle --replay <path> --output-dir <dir> --json",
            },
            {
                "id": "echo_rwml",
                "name": "ECHO/RWML world-model objectives",
                "status": "backend_dependent",
                "use_when": "You want terminal-dynamics prediction for auxiliary loss, reward shaping, or curriculum.",
                "knobs": [
                    "echo_aux_lambda",
                    "rwml_distance_threshold",
                    "rwml_history_window",
                    "embedding model",
                ],
                "evidence": [
                    "ECHO loss",
                    "RWML pass rate",
                    "embedding distance",
                    "heldout correlation",
                ],
                "plan_command": "bashgym training plan --strategy world-model --json",
            },
        ],
        "evaluation": [
            {
                "id": "heldout_trace",
                "status": "ready",
                "proves": "Candidate behavior against baseline traces without train-set reuse.",
                "blocks_on": [
                    "negative trace delta",
                    "forgetting drops",
                    "weak bootstrap confidence",
                ],
            },
            {
                "id": "environment_passk",
                "status": "ready",
                "proves": "Whether the agent solves executable terminal tasks across attempts.",
                "blocks_on": ["broken verifier", "timeout surge", "tamper/invalid attempts"],
            },
            {
                "id": "environment_holdout_gate",
                "status": "ready",
                "proves": "Whether a candidate survives grouped unseen environments.",
                "blocks_on": [
                    "contamination",
                    "pass@1 below threshold",
                    "timeout/tamper above threshold",
                ],
            },
            {
                "id": "holdout_comparison",
                "status": "ready",
                "proves": "Whether candidate beats base on the same holdout environments.",
                "blocks_on": [
                    "delta too small",
                    "confidence interval includes zero",
                    "operational regressions",
                ],
            },
            {
                "id": "spurious_reward_and_tamper",
                "status": "ready",
                "proves": "Whether reward gains come from shortcuts, leakage, or protected-file edits.",
                "blocks_on": ["spurious reward success", "verifier/test/fixture tamper"],
            },
            {
                "id": "external_benchmark_ingest",
                "status": "ready_with_evidence",
                "proves": "Public harness evidence from Terminal-Bench, BFCL, SWE-bench, lm-eval, or generic JSON.",
                "blocks_on": ["missing manifest", "harness failure", "benchmark/train leakage"],
            },
            {
                "id": "world_model_quality",
                "status": "diagnostic",
                "proves": "Whether ECHO/RWML quality metrics exist and improve.",
                "blocks_on": ["must not ship by itself", "needs pass@k and safety correlation"],
            },
        ],
        "backend_stacks": [
            {"id": "trl", "role": "Reference SFT/DPO/GRPO semantics and generated-script shape."},
            {"id": "unsloth", "role": "Fast local LoRA/QLoRA SFT, DPO, and GRPO when supported."},
            {"id": "plain_transformers_peft", "role": "Hardware/model compatibility fallback."},
            {"id": "verl", "role": "External scale-out RL backend candidate for DPPO/GRPO smoke."},
            {"id": "skyrl", "role": "External multi-turn/tool/environment RL backend candidate."},
            {"id": "openrlhf", "role": "External Ray/vLLM RLHF/RLVR backend candidate."},
            {
                "id": "axolotl_torchtune_llamafactory",
                "role": "Optional recipe/export or baseline ecosystems.",
            },
        ],
        "recommended_paths": [
            {
                "id": "first_student",
                "steps": [
                    "gold traces",
                    "SFT",
                    "heldout trace eval",
                    "environment pass@k",
                    "conservative routing",
                ],
            },
            {
                "id": "terminal_rl",
                "steps": [
                    "environment pool",
                    "pass@k baseline",
                    "GRPO/RLVR",
                    "holdout/comparison/canary gates",
                ],
            },
            {
                "id": "reward_model_lane",
                "steps": [
                    "reward examples",
                    "strict reward validation",
                    "fixture reward-model smoke",
                    "reward-model training plan",
                    "heldout reward eval",
                    "best-of-N/rejection control",
                ],
            },
            {
                "id": "dppo_backend",
                "steps": [
                    "served rollouts",
                    "DPPO replay",
                    "train-logprob enrichment",
                    "smoke bundle",
                    "one-step backend smoke",
                ],
            },
            {
                "id": "jepa_world_model",
                "steps": [
                    "world_model replay",
                    "ECHO/RWML backend smoke",
                    "diagnostic quality",
                    "heldout correlation",
                ],
            },
        ],
        "ecosystem_methods": [
            {
                "id": "ppo",
                "support_level": "backend_candidate",
                "where": ["TRL", "verl", "OpenRLHF"],
                "bashgym_position": "Use only behind environment/replay/eval contracts; GRPO/DPPO remain the primary terminal-RL paths.",
            },
            {
                "id": "rloo_reinforce_family",
                "support_level": "backend_candidate",
                "where": ["OpenRLHF"],
                "bashgym_position": "Potential external backend algorithm family, not a first-class BashGym workflow yet.",
            },
            {
                "id": "orpo_kto_ipo_simpo",
                "support_level": "preference_ecosystem_reference",
                "where": ["Unsloth", "Axolotl"],
                "bashgym_position": "Useful recipe/export candidates after SFT; DPO is the first-class preference path today.",
            },
            {
                "id": "gdpo_ebft",
                "support_level": "experimental_ecosystem_reference",
                "where": ["Axolotl"],
                "bashgym_position": "Track for future config import/export; do not advertise as a stable BashGym training path.",
            },
            {
                "id": "multimodal_rl",
                "support_level": "ecosystem_reference",
                "where": ["SkyRL"],
                "bashgym_position": "Relevant to future multimodal agents; current BashGym terminal gym is text/shell centered.",
            },
        ],
        "minimum_promotion_evidence": [
            "heldout trace eval is not worse than baseline",
            "environment pass@k improves or meets threshold",
            "grouped holdout gate passes",
            "base-vs-candidate comparison passes when available",
            "spurious-reward controls and tamper canaries stay clear",
            "reward-model runs attach strict reward examples plus heldout accuracy, calibration, and bias checks",
            "Session Distillation runs attach valid records, masked-loss metrics, and heldout recovery behavior",
            "external benchmark evidence is attached for broad claims",
            "DPPO/ECHO/RWML runs preserve smoke readiness and backend logs",
            "world-model quality remains diagnostic until correlated with pass@k and safety",
        ],
        "source_refs": [
            {
                "id": "trl",
                "url": "https://huggingface.co/docs/trl/en/index",
                "claim": "SFT, GRPO, DPO, reward modeling, and related post-training trainers.",
            },
            {
                "id": "unsloth",
                "url": "https://unsloth.ai/docs",
                "claim": "Efficient open-model training with SFT, preference optimization, and RL/GRPO materials.",
            },
            {
                "id": "verl",
                "url": "https://verl.readthedocs.io/",
                "claim": "Flexible LLM post-training framework for PPO/GRPO-style RL dataflows.",
            },
            {
                "id": "skyrl",
                "url": "https://skyrl.readthedocs.io/",
                "claim": "Full-stack RL library for modular LLM training and agent workloads.",
            },
            {
                "id": "openrlhf",
                "url": "https://openrlhf.readthedocs.io/",
                "claim": "Distributed RLHF/RLVR backend with PPO, GRPO, RLOO, and REINFORCE-family algorithms.",
            },
            {
                "id": "axolotl",
                "url": "https://docs.axolotl.ai/docs/rlhf.html",
                "claim": "YAML-first RLHF/preference ecosystem with DPO, IPO, KTO, ORPO, GRPO, GDPO, and EBFT references.",
            },
            {
                "id": "qwen",
                "url": "https://huggingface.co/collections/Qwen/qwen36",
                "claim": "Qwen3.6 checkpoints are current Qwen-family candidates; BashGym's qwen3 profile is a family compatibility profile, not a single pinned checkpoint.",
            },
        ],
    }


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
            "training capabilities": "Show structured training/eval/backend capability matrix.",
            "training plan": "Recommend starting settings and metrics for a strategy.",
            "training analyze": "Analyze training metrics, DPPO replay, and release evidence.",
            "training smoke-bundle": "Prepare DPPO/ECHO/RWML backend-smoke readiness artifacts.",
            "training runcard": "Create, validate, or attach evidence to a reproducible run card.",
            "training dpo-pairs": "Validate DPO/preference pair artifacts before serious runs.",
            "training reward-examples": "Validate RM/ORM/PRM reward example artifacts.",
            "training reward-model": "Run a dependency-free reward-model fixture smoke.",
            "training reward-eval": "Evaluate reward-model predictions and emit evidence metrics.",
            "sources list": "List curated public training/eval sources.",
            "sources inspect": "Inspect one source card and its guardrails.",
            "sources recommend": "Recommend sources for a domain and training/eval goal.",
            "sources fetch": "Fetch Hugging Face-backed source records into local JSONL.",
            "sources prepare": "Write a source manifest or convert local/fetched records into artifacts.",
            "compute targets": "List local, private, and cloud GPU dry-run targets.",
            "compute preflight": "Run non-invasive compute target readiness checks.",
            "compute launch": "Generate a dry-run provider launch plan.",
            "replay summarize": "Summarize DPPO replay JSONL, including world-model coverage.",
            "replay scrub": "Redact secrets and summarize long stdout/stderr in trace replay files.",
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
            {
                "reason": "Inspect curated public data/eval sources.",
                "command": "bashgym sources list --json",
            },
            {
                "reason": "Check compute targets before remote or cloud training.",
                "command": "bashgym compute targets --json",
            },
        ],
    }
    return _emit(payload, as_json=args.json)


def cmd_training_capabilities(args: argparse.Namespace) -> int:
    payload = {
        "title": "BashGym Training Capability Matrix",
        "ok": True,
        **_capability_matrix(),
        "docs": [{"topic": "capabilities", "path": str(REPO_ROOT / DOCS["capabilities"]["path"])}],
        "next": [
            {
                "reason": "Generate settings for a selected strategy.",
                "command": "bashgym training plan --strategy sft --json",
            },
            {
                "reason": "Read the human capability map.",
                "command": "bashgym training docs --topic capabilities --json",
            },
        ],
    }
    return _emit(payload, as_json=args.json)


def cmd_training_docs(args: argparse.Namespace) -> int:
    if args.topic:
        topic = DOC_ALIASES.get(args.topic, args.topic)
        if topic not in DOCS:
            choices = sorted([*DOCS, *DOC_ALIASES])
            raise SystemExit(f"unknown topic {args.topic!r}; choose {', '.join(choices)}")
        meta = DOCS[topic]
        path = REPO_ROOT / meta["path"]
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        payload = {
            "title": f"BashGym Training Docs: {topic}",
            "ok": path.exists(),
            "topic": topic,
            "requested_topic": args.topic,
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


def _setting_help(recipe: TrainingRecipe) -> list[dict[str, str]]:
    help_rows: list[dict[str, str]] = []
    for setting in recipe.starting_settings:
        guide = SETTING_GUIDANCE.get(setting)
        if guide is None:
            continue
        help_rows.append({"setting": setting, **guide})
    return help_rows


def _metric_guide(recipe: TrainingRecipe) -> list[dict[str, str]]:
    aliases = {
        "heldout_behavior": "heldout_pass@k",
        "heldout_pass@k": "heldout_pass@k",
    }
    guide_rows: list[dict[str, str]] = []
    for metric in recipe.watch:
        guide = METRIC_GUIDANCE.get(metric) or METRIC_GUIDANCE.get(aliases.get(metric, ""))
        if guide is None:
            continue
        guide_rows.append({"metric": metric, **guide})
    return guide_rows


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
        "settings_help": _setting_help(recipe),
        "watch": recipe.watch,
        "metric_guide": _metric_guide(recipe),
        "recommended_next_steps": recipe.next_steps,
        "readiness_ladder": _readiness_ladder(recipe),
        "adjustment_rules": _adjustment_rules(recipe),
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
        smoke_bundle_path=args.smoke_bundle,
    )
    payload = {
        "title": "BashGym Training Analysis",
        **analysis,
    }
    return _emit(payload, as_json=args.json)


def cmd_training_smoke_bundle(args: argparse.Namespace) -> int:
    report = prepare_backend_smoke_bundle(
        BackendSmokeBundleConfig(
            replay_path=Path(args.replay),
            output_dir=Path(args.output_dir),
            base_model=args.base_model,
            backend=args.backend,
            max_steps=args.max_steps,
            n_gpus_per_node=args.n_gpus_per_node,
            echo_enabled=not args.disable_echo,
            rwml_enabled=not args.disable_rwml,
            rwml_embedding_model=args.rwml_embedding_model,
            command_template=args.command_template,
            write_script=not args.no_script,
        )
    )
    payload = {
        "title": "BashGym Backend Smoke Bundle",
        **report,
    }
    return _emit(payload, as_json=args.json)


def cmd_training_session_records_build(args: argparse.Namespace) -> int:
    from bashgym.factory.session_distillation import (
        build_session_distillation_records_from_traces,
        save_session_distillation_records,
        validate_session_distillation_records,
    )

    records = build_session_distillation_records_from_traces(
        args.traces_dir,
        min_confidence=args.min_confidence,
        source_split=args.split or "",
        limit=args.limit,
    )
    payloads = [record.to_dict() for record in records]
    validation_errors = validate_session_distillation_records(payloads)

    written = None
    if records and not validation_errors:
        written = str(save_session_distillation_records(records, args.out))

    payload = {
        "title": "BashGym Session Distillation Records",
        "ok": bool(records) and not validation_errors,
        "traces_dir": str(args.traces_dir),
        "record_count": len(records),
        "output_path": written,
        "min_confidence": args.min_confidence,
        "validation_errors": validation_errors,
    }
    return _emit(payload, as_json=args.json)


def cmd_training_runcard_create(args: argparse.Namespace) -> int:
    card = create_run_card(
        run_id=args.run_id,
        training_method=args.training_method,
        base_model=args.base_model,
        compute_target_id=args.compute_target,
        training_plan_path=args.training_plan,
        source_manifest_path=args.source_manifest,
        preference_pairs_path=args.preference_pairs,
        reward_examples_path=args.reward_examples,
        reward_eval_path=args.reward_eval,
        dataset_card_path=args.dataset_card,
        backend=args.backend,
        metrics_path=args.metrics,
        release_evidence_path=args.release_evidence,
        smoke_bundle_path=args.smoke_bundle,
        session_distillation_records_path=args.session_distillation_records,
        session_distillation_metrics_path=args.session_distillation_metrics,
        session_distillation_reader_model=args.reader_model,
        session_distillation_confidence_threshold=args.confidence_threshold,
        session_distillation_hint_policy=args.hint_policy,
        session_distillation_mask_policy=args.mask_policy,
        session_distillation_target_token_count=args.target_token_count,
        claim_tier=args.claim_tier,
        thresholds=parse_thresholds(args.threshold),
        outputs=args.output_artifact or [],
        known_limitations=args.known_limitation or [],
        decision=args.decision,
        include_git=not args.no_git,
    )
    written = write_run_card(card, args.output)
    validation = (
        validate_run_card_file(written["path"], promotion=True)
        if args.promotion
        else {
            "findings": card.validation_findings(),
            "artifact_status": [],
            "promotion_explanation": explain_run_card_promotion(
                card,
                card.validation_findings(),
            ),
        }
    )
    findings = validation["findings"]
    payload = {
        "title": "BashGym Run Card",
        "ok": not any(finding["level"] == "fail" for finding in findings),
        "schema_version": "bashgym.run_card_result.v1",
        **written,
        "findings": findings,
        "artifact_status": validation["artifact_status"],
        "promotion_explanation": validation["promotion_explanation"],
    }
    return _emit(payload, as_json=args.json)


def cmd_training_runcard_validate(args: argparse.Namespace) -> int:
    validation = validate_run_card_file(args.path, promotion=args.promotion)
    findings = validation["findings"]
    payload = {
        "title": "BashGym Run Card Validation",
        "ok": not any(finding["level"] == "fail" for finding in findings),
        "schema_version": "bashgym.run_card_validation.v1",
        "path": args.path,
        "run_card": validation["run_card"],
        "findings": findings,
        "artifact_status": validation["artifact_status"],
        "promotion_explanation": validation["promotion_explanation"],
    }
    _emit(payload, as_json=args.json)
    return 0 if payload["ok"] else 2


def cmd_training_runcard_attach_evidence(args: argparse.Namespace) -> int:
    written = attach_run_card_evidence(
        args.path,
        metrics_path=args.metrics,
        release_evidence_path=args.release_evidence,
        preference_pairs_path=args.preference_pairs,
        reward_examples_path=args.reward_examples,
        reward_eval_path=args.reward_eval,
        smoke_bundle_path=args.smoke_bundle,
        session_distillation_records_path=args.session_distillation_records,
        session_distillation_metrics_path=args.session_distillation_metrics,
        claim_tier=args.claim_tier,
        output_path=args.output,
    )
    card = read_run_card(written["path"])
    validation = (
        validate_run_card_file(written["path"], promotion=True)
        if args.promotion
        else {
            "findings": card.validation_findings(),
            "artifact_status": [],
            "promotion_explanation": explain_run_card_promotion(
                card,
                card.validation_findings(),
            ),
        }
    )
    findings = validation["findings"]
    payload = {
        "title": "BashGym Run Card Evidence",
        "ok": not any(finding["level"] == "fail" for finding in findings),
        "schema_version": "bashgym.run_card_result.v1",
        **written,
        "findings": findings,
        "artifact_status": validation["artifact_status"],
        "promotion_explanation": validation["promotion_explanation"],
    }
    return _emit(payload, as_json=args.json)


def cmd_training_dpo_pairs_validate(args: argparse.Namespace) -> int:
    validation = validate_preference_pairs_file(
        args.path,
        strict=args.strict,
        max_length_ratio=args.max_length_ratio,
    )
    payload = {
        "title": "BashGym DPO Pair Validation",
        **validation,
    }
    _emit(payload, as_json=args.json)
    return 0 if validation["ok"] else 2


def cmd_training_reward_examples_validate(args: argparse.Namespace) -> int:
    validation = validate_reward_examples_file(
        args.path,
        strict=args.strict,
    )
    payload = {
        "title": "BashGym Reward Example Validation",
        **validation,
    }
    _emit(payload, as_json=args.json)
    return 0 if validation["ok"] else 2


def cmd_training_reward_eval_evaluate(args: argparse.Namespace) -> int:
    evaluation = evaluate_reward_model_file(
        args.path,
        split=args.split,
        calibration_bins=args.calibration_bins,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
        evaluation["output_path"] = str(output_path)
    payload = {
        "title": "BashGym Reward Model Eval",
        **evaluation,
        "next": [
            {
                "reason": "Attach reward eval evidence to a RunCard.",
                "command": "bashgym training runcard attach-evidence <run_card.json> --reward-eval <reward_eval.json> --promotion --json",
            }
        ],
    }
    _emit(payload, as_json=args.json)
    return 0 if payload["ok"] else 2


def cmd_training_reward_model_smoke(args: argparse.Namespace) -> int:
    report = train_reward_model_fixture_file(
        args.path,
        output_dir=args.output_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        max_features=args.max_features,
        calibration_bins=args.calibration_bins,
        strict=not args.lightweight,
    )
    reward_eval_path = report.get("artifacts", {}).get("reward_eval_path")
    payload = {
        "title": "BashGym Reward Model Fixture Smoke",
        **report,
        "next": [
            {
                "reason": "Attach reward-model fixture evidence to a RunCard.",
                "command": (
                    "bashgym training runcard attach-evidence <run_card.json> "
                    f"--reward-eval {reward_eval_path or '<reward_eval.json>'} --promotion --json"
                ),
            },
            {
                "reason": "Keep this as a contract smoke before a real TRL/OpenRLHF reward backend.",
                "command": "bashgym training docs --topic methods-reference --json",
            },
        ],
    }
    _emit(payload, as_json=args.json)
    return 0 if payload["ok"] else 2


def cmd_compute_targets(args: argparse.Namespace) -> int:
    payload = {
        "title": "BashGym Compute Targets",
        "ok": True,
        "schema_version": "bashgym.compute_targets.v1",
        "targets": [target.to_dict() for target in list_compute_targets()],
        "next": [
            {
                "reason": "Run a non-invasive preflight before launching.",
                "command": "bashgym compute preflight --target private_gpu --json",
            }
        ],
    }
    return _emit(payload, as_json=args.json)


def cmd_compute_preflight(args: argparse.Namespace) -> int:
    try:
        target = get_compute_target(args.target)
    except KeyError as exc:
        raise SystemExit(f"unknown compute target {args.target!r}") from exc
    payload = {
        "title": "BashGym Compute Preflight",
        "ok": True,
        **preflight_compute_target(target),
    }
    return _emit(payload, as_json=args.json)


def cmd_compute_launch(args: argparse.Namespace) -> int:
    if not args.dry_run:
        raise SystemExit("compute launch currently supports --dry-run only")
    try:
        target = get_compute_target(args.target)
    except KeyError as exc:
        raise SystemExit(f"unknown compute target {args.target!r}") from exc
    payload = {
        "title": "BashGym Compute Launch Plan",
        "ok": True,
        **launch_plan(target, plan_path=args.plan),
    }
    return _emit(payload, as_json=args.json)


def cmd_sources_list(args: argparse.Namespace) -> int:
    errors = validate_catalog()
    payload = {
        "title": "BashGym Source Library",
        "ok": not errors,
        "schema_version": "bashgym.source_catalog.v1",
        "count": len(list_sources()),
        "sources": [card.to_dict() for card in list_sources()],
        "validation_errors": errors,
        "next": [
            {
                "reason": "Inspect one source before preparing artifacts.",
                "command": "bashgym sources inspect harbor_terminal_bench --json",
            },
            {
                "reason": "Recommend training-eligible preference sources.",
                "command": "bashgym sources recommend --goal dpo --json",
            },
        ],
    }
    return _emit(payload, as_json=args.json)


def cmd_sources_inspect(args: argparse.Namespace) -> int:
    try:
        card = get_source(args.source_id)
    except KeyError as exc:
        raise SystemExit(f"unknown source {args.source_id!r}") from exc
    payload = {
        "title": f"BashGym Source: {card.name}",
        "ok": not card.validation_errors(),
        "schema_version": "bashgym.source_card.v1",
        "source": card.to_dict(),
        "validation_errors": card.validation_errors(),
    }
    return _emit(payload, as_json=args.json)


def cmd_sources_recommend(args: argparse.Namespace) -> int:
    payload = {
        "title": "BashGym Source Recommendations",
        "ok": True,
        "schema_version": "bashgym.source_recommendations.v1",
        "domain": args.domain,
        "goal": args.goal,
        "recommendations": recommend_sources(
            domain=args.domain,
            goal=args.goal,
            include_eval_only=args.include_eval_only,
        ),
    }
    return _emit(payload, as_json=args.json)


def cmd_sources_fetch(args: argparse.Namespace) -> int:
    try:
        card = get_source(args.source_id)
    except KeyError as exc:
        raise SystemExit(f"unknown source {args.source_id!r}") from exc
    payload = {
        "title": "BashGym Source Fetch",
        **fetch_source_records(
            card,
            output_dir=args.output_dir,
            split=args.split,
            subset=args.subset,
            revision=args.revision,
            limit=args.limit,
            approval_reason=args.approval_reason,
            force_refresh=args.force_refresh,
        ),
    }
    if not payload["ok"]:
        payload["next"] = [
            {
                "reason": "Use local JSON/JSONL records when a source card has no Hugging Face dataset.",
                "command": "bashgym sources prepare SOURCE_ID --input records.jsonl --output-dir data/sources/out --json",
            }
        ]
    _emit(payload, as_json=args.json)
    return 0 if payload["ok"] else 2


def cmd_sources_prepare(args: argparse.Namespace) -> int:
    try:
        card = get_source(args.source_id)
    except KeyError as exc:
        raise SystemExit(f"unknown source {args.source_id!r}") from exc
    input_path = args.input
    fetch_report = None
    if args.fetch:
        if args.input:
            raise SystemExit("sources prepare --fetch cannot be combined with --input")
        if not args.output_dir:
            raise SystemExit("sources prepare --fetch requires --output-dir")
        fetch_report = fetch_source_records(
            card,
            output_dir=args.output_dir,
            split=args.split,
            subset=args.subset,
            revision=args.revision,
            limit=args.limit if args.limit is not None else DEFAULT_SOURCE_FETCH_LIMIT,
            approval_reason=args.fetch_approval_reason,
            force_refresh=args.force_refresh,
        )
        if not fetch_report["ok"]:
            payload = {"title": "BashGym Source Fetch", **fetch_report}
            _emit(payload, as_json=args.json)
            return 2
        input_path = fetch_report["records_path"]
    if input_path:
        if not args.output_dir:
            raise SystemExit("sources prepare --input requires --output-dir")
        payload = {
            "title": "BashGym Source Artifacts",
            **prepare_source_artifacts(
                card,
                goal=args.goal,
                input_path=input_path,
                output_dir=args.output_dir,
                allow_eval_only=args.allow_eval_only,
                override_reason=args.override_reason,
                limit=args.limit,
            ),
        }
        if fetch_report:
            payload["fetch_report"] = fetch_report
    else:
        manifest = prepare_source_manifest(
            card,
            goal=args.goal,
            output_dir=args.output_dir,
            allow_eval_only=args.allow_eval_only,
            override_reason=args.override_reason,
        )
        payload = {
            "title": "BashGym Source Manifest",
            "ok": manifest["use_verdict"]["ok"],
            **manifest,
        }
    if not payload["ok"]:
        payload["next"] = [
            {
                "reason": "Pick a training-eligible source or keep this benchmark eval-only.",
                "command": "bashgym sources recommend --goal evaluation --include-eval-only --json",
            }
        ]
    _emit(payload, as_json=args.json)
    return 0 if payload["ok"] else 2


def _plan_next(recipe: TrainingRecipe) -> list[tuple[str, str]]:
    if recipe.strategy == "reward-model":
        return [
            (
                "Validate reward-model artifacts.",
                "bashgym training reward-examples validate reward_examples.jsonl --strict --json",
            ),
            (
                "Run a fixture reward-model smoke.",
                "bashgym training reward-model smoke reward_examples.jsonl --output-dir data/reward-model-smokes/latest --json",
            ),
            (
                "Review reward-model method details.",
                "bashgym training docs --topic methods-reference --json",
            ),
        ]
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


def cmd_replay_scrub(args: argparse.Namespace) -> int:
    try:
        report = scrub_trace_replay_file(
            args.path,
            output_path=args.output,
            max_output_chars=args.max_output_chars,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "title": "BashGym Replay Scrub",
            "ok": False,
            "error": str(exc),
            "next": [
                {
                    "reason": "Pass a JSON or JSONL trace/replay file.",
                    "command": "bashgym replay scrub trace.json --output trace.scrubbed.json --json",
                }
            ],
        }
        _emit(payload, as_json=args.json)
        return 2

    payload = {
        "title": "BashGym Replay Scrub",
        **report,
        "next": [
            {
                "reason": "Use scrubbed traces for review, public examples, and training data QA.",
                "command": "bashgym training docs --topic overview --json",
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
    docs.add_argument("--topic", choices=sorted([*DOCS, *DOC_ALIASES]), help="Doc topic to read")
    docs.add_argument(
        "--include-content",
        action="store_true",
        help="Include full markdown in non-JSON output",
    )
    docs.set_defaults(func=cmd_training_docs)

    capabilities = training_sub.add_parser(
        "capabilities",
        help="Show structured training/eval/backend capability matrix",
        parents=[json_parent],
    )
    capabilities.set_defaults(func=cmd_training_capabilities)

    plan = training_sub.add_parser(
        "plan",
        help="Recommend starting training settings",
        parents=[json_parent],
    )
    plan.add_argument(
        "--strategy",
        default="sft",
        choices=[
            "sft",
            "dpo",
            "grpo",
            "rlvr",
            "dppo",
            "reward-model",
            "reward_model",
            "rm",
            "preference-rm",
            "orm",
            "prm",
            "world-model",
            "session",
            "session-distill",
            "session-distillation",
        ],
    )
    plan.add_argument(
        "--hardware",
        default="local_12gb",
        choices=[
            "local_12gb",
            "local_24gb",
            "private_compute",
            "private-compute",
            "remote",
            "remote_gpu",
            "dgx",
            "gx10",
            "cloud",
        ],
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
    analyze.add_argument("--smoke-bundle", help="Optional backend_smoke_readiness.json artifact")
    analyze.set_defaults(func=cmd_training_analyze)

    smoke_bundle = training_sub.add_parser(
        "smoke-bundle",
        help="Prepare DPPO/ECHO/RWML backend-smoke readiness artifacts",
        parents=[json_parent],
    )
    smoke_bundle.add_argument("--replay", required=True, help="DPPO replay JSONL artifact")
    smoke_bundle.add_argument(
        "--output-dir",
        required=True,
        help="Directory where readiness JSON, launch env, and optional script are written",
    )
    smoke_bundle.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model id to pass to the backend smoke launcher",
    )
    smoke_bundle.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "verl", "skyrl", "tmax_open_instruct", "grpo_fallback"],
    )
    smoke_bundle.add_argument("--max-steps", type=int, default=1)
    smoke_bundle.add_argument("--n-gpus-per-node", type=int, default=1)
    smoke_bundle.add_argument("--rwml-embedding-model", default="")
    smoke_bundle.add_argument("--command-template", help="Override backend command template")
    smoke_bundle.add_argument("--disable-echo", action="store_true")
    smoke_bundle.add_argument("--disable-rwml", action="store_true")
    smoke_bundle.add_argument("--no-script", action="store_true", help="Do not write launch script")
    smoke_bundle.set_defaults(func=cmd_training_smoke_bundle)

    runcard = training_sub.add_parser(
        "runcard",
        help="Create, validate, or attach evidence to run cards",
        parents=[json_parent],
    )
    runcard_sub = runcard.add_subparsers(dest="runcard_command", required=True)

    runcard_create = runcard_sub.add_parser(
        "create",
        help="Create a run card JSON artifact",
        parents=[json_parent],
    )
    runcard_create.add_argument("--run-id", required=True)
    runcard_create.add_argument("--training-method", required=True)
    runcard_create.add_argument("--base-model", required=True)
    runcard_create.add_argument("--compute-target", required=True)
    runcard_create.add_argument("--output", required=True)
    runcard_create.add_argument("--training-plan")
    runcard_create.add_argument("--source-manifest")
    runcard_create.add_argument("--preference-pairs")
    runcard_create.add_argument("--reward-examples")
    runcard_create.add_argument("--reward-eval")
    runcard_create.add_argument("--dataset-card")
    runcard_create.add_argument("--backend")
    runcard_create.add_argument("--metrics")
    runcard_create.add_argument("--release-evidence")
    runcard_create.add_argument("--smoke-bundle")
    runcard_create.add_argument("--session-distillation-records")
    runcard_create.add_argument("--session-distillation-metrics")
    runcard_create.add_argument("--reader-model")
    runcard_create.add_argument("--confidence-threshold", type=float)
    runcard_create.add_argument("--hint-policy")
    runcard_create.add_argument("--mask-policy")
    runcard_create.add_argument("--target-token-count", type=int)
    runcard_create.add_argument(
        "--claim-tier",
        default="local_smoke",
        choices=["local_smoke", "narrow_routing", "broad_public_claim"],
    )
    runcard_create.add_argument(
        "--threshold", action="append", help="Promotion threshold key=value"
    )
    runcard_create.add_argument("--output-artifact", action="append")
    runcard_create.add_argument("--known-limitation", action="append")
    runcard_create.add_argument(
        "--decision",
        default="pending",
        choices=["pending", "hold", "promote", "reject", "route_narrowly"],
    )
    runcard_create.add_argument("--promotion", action="store_true")
    runcard_create.add_argument("--no-git", action="store_true")
    runcard_create.set_defaults(func=cmd_training_runcard_create)

    runcard_validate = runcard_sub.add_parser(
        "validate",
        help="Validate a run card JSON artifact",
        parents=[json_parent],
    )
    runcard_validate.add_argument("path")
    runcard_validate.add_argument("--promotion", action="store_true")
    runcard_validate.set_defaults(func=cmd_training_runcard_validate)

    runcard_attach = runcard_sub.add_parser(
        "attach-evidence",
        help="Attach metrics/release/smoke evidence to a run card",
        parents=[json_parent],
    )
    runcard_attach.add_argument("path")
    runcard_attach.add_argument("--metrics")
    runcard_attach.add_argument("--release-evidence")
    runcard_attach.add_argument("--preference-pairs")
    runcard_attach.add_argument("--reward-examples")
    runcard_attach.add_argument("--reward-eval")
    runcard_attach.add_argument("--smoke-bundle")
    runcard_attach.add_argument("--session-distillation-records")
    runcard_attach.add_argument("--session-distillation-metrics")
    runcard_attach.add_argument(
        "--claim-tier",
        choices=["local_smoke", "narrow_routing", "broad_public_claim"],
    )
    runcard_attach.add_argument("--output")
    runcard_attach.add_argument("--promotion", action="store_true")
    runcard_attach.set_defaults(func=cmd_training_runcard_attach_evidence)

    session_records = training_sub.add_parser(
        "session-records",
        help="Build Session Distillation records from trace files",
        parents=[json_parent],
    )
    session_records_sub = session_records.add_subparsers(
        dest="session_records_command",
        required=True,
    )
    session_records_build = session_records_sub.add_parser(
        "build",
        help="Build session_distillation_records.jsonl from a traces directory",
        parents=[json_parent],
    )
    session_records_build.add_argument("traces_dir")
    session_records_build.add_argument(
        "--out",
        default="data/session_distillation/records.jsonl",
        help="Output JSONL path (default: data/session_distillation/records.jsonl)",
    )
    session_records_build.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Drop reader hints below this confidence (default: 0.6)",
    )
    session_records_build.add_argument(
        "--split",
        default="",
        help="Record a split label in source_metadata (e.g. train/val/heldout)",
    )
    session_records_build.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many records (useful for a small smoke set)",
    )
    session_records_build.set_defaults(func=cmd_training_session_records_build)

    dpo_pairs = training_sub.add_parser(
        "dpo-pairs",
        help="Validate DPO/preference pair artifacts",
        parents=[json_parent],
    )
    dpo_pairs_sub = dpo_pairs.add_subparsers(dest="dpo_pairs_command", required=True)

    dpo_pairs_validate = dpo_pairs_sub.add_parser(
        "validate",
        help="Validate a DPO/preference pair JSONL or JSON artifact",
        parents=[json_parent],
    )
    dpo_pairs_validate.add_argument("path")
    dpo_pairs_validate.add_argument(
        "--strict",
        action="store_true",
        help="Fail missing provenance, quality, split, and decontamination metadata",
    )
    dpo_pairs_validate.add_argument("--max-length-ratio", type=float, default=3.0)
    dpo_pairs_validate.set_defaults(func=cmd_training_dpo_pairs_validate)

    reward_examples = training_sub.add_parser(
        "reward-examples",
        help="Validate reward-model, ORM, or PRM example artifacts",
        parents=[json_parent],
    )
    reward_examples_sub = reward_examples.add_subparsers(
        dest="reward_examples_command",
        required=True,
    )

    reward_examples_validate = reward_examples_sub.add_parser(
        "validate",
        help="Validate a reward example JSONL or JSON artifact",
        parents=[json_parent],
    )
    reward_examples_validate.add_argument("path")
    reward_examples_validate.add_argument(
        "--strict",
        action="store_true",
        help="Fail missing reward scale, provenance, quality, split, and decontamination metadata",
    )
    reward_examples_validate.set_defaults(func=cmd_training_reward_examples_validate)

    reward_model = training_sub.add_parser(
        "reward-model",
        help="Run reward-model fixture training and smoke evidence",
        parents=[json_parent],
    )
    reward_model_sub = reward_model.add_subparsers(
        dest="reward_model_command",
        required=True,
    )

    reward_model_smoke = reward_model_sub.add_parser(
        "smoke",
        help="Train a tiny dependency-free reward scorer and write evidence artifacts",
        parents=[json_parent],
    )
    reward_model_smoke.add_argument("path", help="Reward examples JSONL or JSON artifact")
    reward_model_smoke.add_argument(
        "--output-dir",
        required=True,
        help="Directory for fixture model, predictions, metrics, and reward_eval.json",
    )
    reward_model_smoke.add_argument("--train-split", default="train")
    reward_model_smoke.add_argument("--eval-split", default="eval")
    reward_model_smoke.add_argument("--epochs", type=int, default=8)
    reward_model_smoke.add_argument("--learning-rate", type=float, default=0.5)
    reward_model_smoke.add_argument("--l2", type=float, default=0.001)
    reward_model_smoke.add_argument("--max-features", type=int, default=2048)
    reward_model_smoke.add_argument("--calibration-bins", type=int, default=10)
    reward_model_smoke.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight validation instead of strict serious-run validation",
    )
    reward_model_smoke.set_defaults(func=cmd_training_reward_model_smoke)

    reward_eval = training_sub.add_parser(
        "reward-eval",
        help="Evaluate reward-model predictions against reward examples",
        parents=[json_parent],
    )
    reward_eval_sub = reward_eval.add_subparsers(dest="reward_eval_command", required=True)

    reward_eval_evaluate = reward_eval_sub.add_parser(
        "evaluate",
        help="Compute reward-model heldout accuracy, calibration, bias, and leakage metrics",
        parents=[json_parent],
    )
    reward_eval_evaluate.add_argument("path")
    reward_eval_evaluate.add_argument(
        "--split",
        default="eval",
        help="Split to evaluate; use 'all' to include every record",
    )
    reward_eval_evaluate.add_argument("--calibration-bins", type=int, default=10)
    reward_eval_evaluate.add_argument("--output", help="Optional reward_eval.json output path")
    reward_eval_evaluate.set_defaults(func=cmd_training_reward_eval_evaluate)

    sources = subparsers.add_parser(
        "sources",
        help="Curated public training/evaluation source library",
        parents=[json_parent],
    )
    sources_sub = sources.add_subparsers(dest="sources_command", required=True)

    sources_list = sources_sub.add_parser(
        "list",
        help="List curated source cards",
        parents=[json_parent],
    )
    sources_list.set_defaults(func=cmd_sources_list)

    sources_inspect = sources_sub.add_parser(
        "inspect",
        help="Inspect one source card",
        parents=[json_parent],
    )
    sources_inspect.add_argument("source_id")
    sources_inspect.set_defaults(func=cmd_sources_inspect)

    sources_recommend = sources_sub.add_parser(
        "recommend",
        help="Recommend source cards for a domain and goal",
        parents=[json_parent],
    )
    sources_recommend.add_argument("--domain")
    sources_recommend.add_argument(
        "--goal",
        choices=[use.value for use in SourceUse],
        help="Training/eval goal to match against source artifacts",
    )
    sources_recommend.add_argument(
        "--include-eval-only",
        action="store_true",
        help="Include eval-only benchmark sources in recommendations",
    )
    sources_recommend.set_defaults(func=cmd_sources_recommend)

    sources_fetch = sources_sub.add_parser(
        "fetch",
        help="Fetch Hugging Face-backed source records into local JSONL",
        parents=[json_parent],
    )
    sources_fetch.add_argument("source_id")
    sources_fetch.add_argument(
        "--output-dir", required=True, help="Directory for source_records.jsonl"
    )
    sources_fetch.add_argument("--split", default="train", help="Dataset split to fetch")
    sources_fetch.add_argument("--subset", help="Optional Hugging Face dataset config/subset")
    sources_fetch.add_argument("--revision", help="Optional Hugging Face dataset revision")
    sources_fetch.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SOURCE_FETCH_LIMIT,
        help="Maximum number of source records to fetch",
    )
    sources_fetch.add_argument(
        "--approval-reason",
        help="Required when fetching more than the default capped source record limit",
    )
    sources_fetch.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore a matching cached source_fetch_report/source_records pair",
    )
    sources_fetch.set_defaults(func=cmd_sources_fetch)

    sources_prepare = sources_sub.add_parser(
        "prepare",
        help="Write a source manifest or convert local/fetched records into artifacts",
        parents=[json_parent],
    )
    sources_prepare.add_argument("source_id")
    sources_prepare.add_argument(
        "--goal",
        default=SourceUse.EVALUATION.value,
        choices=[use.value for use in SourceUse],
    )
    sources_prepare.add_argument("--output-dir", help="Directory for source_manifest.json")
    sources_prepare.add_argument(
        "--input",
        help="Local JSON/JSONL source records to convert into BashGym artifacts",
    )
    sources_prepare.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch Hugging Face source records before converting artifacts",
    )
    sources_prepare.add_argument("--split", default="train", help="Dataset split for --fetch")
    sources_prepare.add_argument("--subset", help="Optional Hugging Face dataset config/subset")
    sources_prepare.add_argument("--revision", help="Optional Hugging Face dataset revision")
    sources_prepare.add_argument(
        "--limit",
        type=int,
        help="Maximum number of local or fetched source records to convert",
    )
    sources_prepare.add_argument(
        "--fetch-approval-reason",
        help="Required with --fetch when fetching more than the default capped source record limit",
    )
    sources_prepare.add_argument(
        "--force-refresh",
        action="store_true",
        help="With --fetch, ignore a matching cached source_fetch_report/source_records pair",
    )
    sources_prepare.add_argument(
        "--allow-eval-only",
        action="store_true",
        help="Allow eval-only sources for a training goal and record an override warning",
    )
    sources_prepare.add_argument("--override-reason", help="Reason for eval-only override")
    sources_prepare.set_defaults(func=cmd_sources_prepare)

    compute = subparsers.add_parser(
        "compute",
        help="Local, private, and cloud GPU dry-run target helpers",
        parents=[json_parent],
    )
    compute_sub = compute.add_subparsers(dest="compute_command", required=True)

    compute_targets = compute_sub.add_parser(
        "targets",
        help="List configured compute target templates",
        parents=[json_parent],
    )
    compute_targets.set_defaults(func=cmd_compute_targets)

    compute_preflight = compute_sub.add_parser(
        "preflight",
        help="Run non-invasive compute target checks",
        parents=[json_parent],
    )
    compute_preflight.add_argument("--target", required=True)
    compute_preflight.set_defaults(func=cmd_compute_preflight)

    compute_launch = compute_sub.add_parser(
        "launch",
        help="Generate a dry-run launch plan",
        parents=[json_parent],
    )
    compute_launch.add_argument("--target", required=True)
    compute_launch.add_argument("--plan", help="Training plan/config path")
    compute_launch.add_argument("--dry-run", action="store_true", help="Required; do not execute")
    compute_launch.set_defaults(func=cmd_compute_launch)

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
    replay_scrub = replay_sub.add_parser(
        "scrub",
        help="Redact secrets and summarize long trace replay output",
        parents=[json_parent],
    )
    replay_scrub.add_argument("path")
    replay_scrub.add_argument("--output", help="Optional JSON/JSONL path for scrubbed payload")
    replay_scrub.add_argument("--max-output-chars", type=int, default=2000)
    replay_scrub.set_defaults(func=cmd_replay_scrub)

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
