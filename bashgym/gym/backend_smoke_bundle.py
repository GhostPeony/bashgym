"""Local readiness bundle for DPPO plus ECHO/RWML backend smoke runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bashgym.eval.dppo_replay import (
    DPPO_REPLAY_SCHEMA_VERSION,
    read_dppo_replay_jsonl,
    summarize_dppo_replay_records,
)
from bashgym.gym.dppo_backend import DPPOBackendCapability
from bashgym.gym.dppo_launcher import (
    DPPOSmokeLaunchConfig,
    build_dppo_launch_env,
    prepare_dppo_smoke_launch,
)
from bashgym.gym.echo import ECHO_DEFAULT_LAMBDA
from bashgym.gym.rwml import (
    RWML_DEFAULT_DISTANCE_THRESHOLD,
    RWML_DEFAULT_EASY_KEEP_PROBABILITY,
    RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD,
    RWML_DEFAULT_HISTORY_WINDOW,
)
from bashgym.gym.world_model_backend import (
    WorldModelTrainerAdapter,
    WorldModelTrainerSettings,
)

SMOKE_BUNDLE_SCHEMA_VERSION = "bashgym.backend_smoke_bundle.v1"
READINESS_FILENAME = "backend_smoke_readiness.json"
REPLAY_SUMMARY_FILENAME = "dppo_replay_summary.json"
WORLD_MODEL_PROBE_FILENAME = "world_model_backend_probe.json"
LAUNCH_ENV_FILENAME = "dppo_launch_env.json"

CONTRACT_BLOCKERS = {
    "missing_replay_path",
    "empty_replay",
    "invalid_schema_version",
    "missing_behavior_logprobs",
    "missing_world_model_payloads",
    "missing_rwml_transitions",
    "missing_echo_observations",
}


@dataclass(frozen=True)
class BackendSmokeBundleConfig:
    replay_path: Path
    output_dir: Path
    base_model: str
    backend: str = "auto"
    max_steps: int = 1
    n_gpus_per_node: int = 1
    echo_enabled: bool = True
    echo_aux_lambda: float = ECHO_DEFAULT_LAMBDA
    rwml_enabled: bool = True
    rwml_distance_threshold: float = RWML_DEFAULT_DISTANCE_THRESHOLD
    rwml_easy_pass_rate_threshold: float = RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD
    rwml_easy_keep_probability: float = RWML_DEFAULT_EASY_KEEP_PROBABILITY
    rwml_history_window: int = RWML_DEFAULT_HISTORY_WINDOW
    rwml_embedding_model: str = ""
    rwml_kl_beta: float = 0.0
    command_template: str | None = None
    write_script: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "replay_path", Path(self.replay_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def prepare_backend_smoke_bundle(
    config: BackendSmokeBundleConfig,
    *,
    capabilities: dict[str, DPPOBackendCapability] | None = None,
) -> dict[str, Any]:
    """Create local readiness artifacts for a DPPO/ECHO/RWML backend smoke.

    The bundle is intentionally local and cheap. It proves that BashGym can parse
    a replay, inspect DPPO logprob readiness, tokenize ECHO spans with a simple
    probe tokenizer, count RWML transitions, and emit the backend launch env. A
    real trainer run remains a separate step when verl/SkyRL/open-instruct is
    installed.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    replay_exists = config.replay_path.exists()
    records = read_dppo_replay_jsonl(config.replay_path) if replay_exists else []
    replay_summary = _replay_summary(records)

    launch_config = _launch_config(config)
    launch_env = build_dppo_launch_env(launch_config)
    launch_plan = prepare_dppo_smoke_launch(
        launch_config,
        capabilities=capabilities,
    )
    world_model_probe = _world_model_probe(records, config)
    checks = _readiness_checks(
        replay_exists=replay_exists,
        records=records,
        replay_summary=replay_summary,
        world_model_probe=world_model_probe,
        launch_runnable=launch_plan.runnable,
        launch_reason=launch_plan.reason,
        echo_enabled=config.echo_enabled,
        rwml_enabled=config.rwml_enabled,
    )
    readiness = _readiness_report(
        config=config,
        replay_summary=replay_summary,
        world_model_probe=world_model_probe,
        launch_plan=launch_plan.to_dict(),
        launch_env=launch_env,
        checks=checks,
    )

    replay_summary_path = _write_json(config.output_dir / REPLAY_SUMMARY_FILENAME, replay_summary)
    world_model_probe_path = _write_json(
        config.output_dir / WORLD_MODEL_PROBE_FILENAME,
        world_model_probe,
    )
    launch_env_path = _write_json(config.output_dir / LAUNCH_ENV_FILENAME, launch_env)
    readiness_path = _write_json(config.output_dir / READINESS_FILENAME, readiness)

    readiness["artifacts"] = {
        "readiness": str(readiness_path),
        "replay_summary": str(replay_summary_path),
        "world_model_probe": str(world_model_probe_path),
        "launch_env": str(launch_env_path),
        "launch_script": launch_plan.script_path,
    }
    _write_json(readiness_path, readiness)
    return readiness


def _launch_config(config: BackendSmokeBundleConfig) -> DPPOSmokeLaunchConfig:
    return DPPOSmokeLaunchConfig(
        replay_path=config.replay_path,
        output_dir=config.output_dir,
        base_model=config.base_model,
        backend=config.backend,
        max_steps=config.max_steps,
        n_gpus_per_node=config.n_gpus_per_node,
        write_script=config.write_script,
        command_template=config.command_template,
        echo_enabled=config.echo_enabled,
        echo_aux_lambda=config.echo_aux_lambda,
        rwml_enabled=config.rwml_enabled,
        rwml_distance_threshold=config.rwml_distance_threshold,
        rwml_easy_pass_rate_threshold=config.rwml_easy_pass_rate_threshold,
        rwml_easy_keep_probability=config.rwml_easy_keep_probability,
        rwml_history_window=config.rwml_history_window,
        rwml_embedding_model=config.rwml_embedding_model,
        rwml_kl_beta=config.rwml_kl_beta,
    )


def _replay_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "schema_version": DPPO_REPLAY_SCHEMA_VERSION,
            "records": 0,
            "environments": 0,
            "environment_ids": [],
            "behavior_logprobs_ready_records": 0,
            "train_logprobs_ready_records": 0,
            "train_logprob_replay_required_records": 0,
            "world_model_records": 0,
            "world_model": {
                "records": 0,
                "records_missing_world_model": 0,
                "rwml_transitions": 0,
                "echo_observation_chars": 0,
            },
        }
    return summarize_dppo_replay_records(records)


def _world_model_probe(
    records: Sequence[dict[str, Any]],
    config: BackendSmokeBundleConfig,
) -> dict[str, Any]:
    settings = WorldModelTrainerSettings(
        echo_enabled=config.echo_enabled,
        echo_aux_lambda=config.echo_aux_lambda,
        rwml_enabled=config.rwml_enabled,
        rwml_distance_threshold=config.rwml_distance_threshold,
        rwml_easy_pass_rate_threshold=config.rwml_easy_pass_rate_threshold,
        rwml_easy_keep_probability=config.rwml_easy_keep_probability,
        rwml_history_window=config.rwml_history_window,
        rwml_embedding_model=config.rwml_embedding_model,
        rwml_kl_beta=config.rwml_kl_beta,
    )
    adapter = WorldModelTrainerAdapter.from_records(
        records,
        _probe_tokenizer,
        settings=settings,
    )
    batch = adapter.batch.to_dict()
    return {
        "settings": settings.to_dict(),
        "batch": batch,
        "echo_masks_buildable": batch["echo_sequences"] > 0 if config.echo_enabled else True,
        "rwml_targets_buildable": batch["rwml_transitions"] > 0 if config.rwml_enabled else True,
        "probe_tokenizer": "unicode-codepoint-per-character",
        "note": (
            "This probe proves BashGym replay spans can become backend masks and targets. "
            "The real trainer must rebuild masks with the model tokenizer."
        ),
    }


def _probe_tokenizer(text: str) -> list[int]:
    return [ord(char) for char in text]


def _readiness_checks(
    *,
    replay_exists: bool,
    records: list[dict[str, Any]],
    replay_summary: dict[str, Any],
    world_model_probe: dict[str, Any],
    launch_runnable: bool,
    launch_reason: str,
    echo_enabled: bool,
    rwml_enabled: bool,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    records_count = len(records)
    world_model = replay_summary.get("world_model") or {}
    backend_batch = world_model_probe.get("batch") or {}

    _add_check(
        checks,
        "missing_replay_path",
        "pass" if replay_exists else "fail",
        "Replay artifact exists." if replay_exists else "Replay artifact is missing.",
        "Export DPPO replay JSONL before preparing a backend smoke.",
    )
    _add_check(
        checks,
        "empty_replay",
        "pass" if records_count > 0 else "fail",
        f"Replay contains {records_count} records.",
        "Run at least one served-model rollout and export replay JSONL.",
    )

    invalid_schema = [
        str(record.get("schema_version"))
        for record in records
        if record.get("schema_version") != DPPO_REPLAY_SCHEMA_VERSION
    ]
    _add_check(
        checks,
        "invalid_schema_version",
        "pass" if not invalid_schema and records_count else "fail",
        (
            "All replay records use bashgym.dppo_replay.v1."
            if not invalid_schema and records_count
            else f"{len(invalid_schema)} records do not use {DPPO_REPLAY_SCHEMA_VERSION}."
        ),
        "Regenerate the replay with BashGym's DPPO replay exporter.",
    )

    behavior_ready = int(replay_summary.get("behavior_logprobs_ready_records") or 0)
    _add_check(
        checks,
        "missing_behavior_logprobs",
        _all_some_none_status(behavior_ready, records_count),
        f"{behavior_ready}/{records_count} records include behavior logprobs.",
        "Enable response logprobs during model rollouts before DPPO replay.",
    )
    train_ready = int(replay_summary.get("train_logprobs_ready_records") or 0)
    _add_check(
        checks,
        "train_logprob_replay_needed",
        "pass" if records_count and train_ready == records_count else "warn",
        f"{train_ready}/{records_count} records include train-policy logprobs.",
        "Run train-logprob replay/enrichment before a real DPPO optimizer update.",
    )

    world_model_records = int(replay_summary.get("world_model_records") or 0)
    _add_check(
        checks,
        "missing_world_model_payloads",
        (
            _all_some_none_status(world_model_records, records_count)
            if echo_enabled or rwml_enabled
            else "pass"
        ),
        f"{world_model_records}/{records_count} records include world_model payloads.",
        "Export replay with include_world_model_replay=true for ECHO/RWML.",
    )

    rwml_transitions = int(world_model.get("rwml_transitions") or 0)
    _add_check(
        checks,
        "missing_rwml_transitions",
        "pass" if (not rwml_enabled or rwml_transitions > 0) else "fail",
        f"RWML transitions: {rwml_transitions}.",
        "Use rollouts with command observations so RWML can build action -> next-state targets.",
    )
    echo_tokens = int(backend_batch.get("echo_observation_tokens") or 0)
    echo_chars = int(world_model.get("echo_observation_chars") or 0)
    _add_check(
        checks,
        "missing_echo_observations",
        "pass" if (not echo_enabled or (echo_chars > 0 and echo_tokens > 0)) else "fail",
        f"ECHO observation coverage: {echo_chars} chars, {echo_tokens} probe tokens.",
        "Keep terminal stdout/stderr observations in replay for ECHO supervision.",
    )
    _add_check(
        checks,
        "backend_launch_plan",
        "pass" if launch_runnable else "warn",
        launch_reason,
        "Install/configure verl, SkyRL, or TMax/open-instruct, or set a backend command template.",
    )
    return checks


def _all_some_none_status(ready: int, total: int) -> str:
    if total <= 0 or ready == 0:
        return "fail"
    if ready == total:
        return "pass"
    return "warn"


def _add_check(
    checks: list[dict[str, str]],
    code: str,
    status: str,
    message: str,
    next_step: str,
) -> None:
    checks.append(
        {
            "code": code,
            "status": status,
            "message": message,
            "next_step": next_step,
        }
    )


def _readiness_report(
    *,
    config: BackendSmokeBundleConfig,
    replay_summary: dict[str, Any],
    world_model_probe: dict[str, Any],
    launch_plan: dict[str, Any],
    launch_env: dict[str, str],
    checks: list[dict[str, str]],
) -> dict[str, Any]:
    failed = {check["code"] for check in checks if check["status"] == "fail"}
    warnings = {check["code"] for check in checks if check["status"] == "warn"}
    contract_ready = not (failed & CONTRACT_BLOCKERS)
    records = int(replay_summary.get("records") or 0)
    optimizer_ready = (
        contract_ready
        and records > 0
        and int(replay_summary.get("train_logprobs_ready_records") or 0) == records
    )
    backend_launch_ready = contract_ready and bool(launch_plan.get("runnable"))
    verdict = _verdict(
        contract_ready=contract_ready,
        optimizer_ready=optimizer_ready,
        backend_launch_ready=backend_launch_ready,
        failed=failed,
        warnings=warnings,
    )
    return {
        "schema_version": SMOKE_BUNDLE_SCHEMA_VERSION,
        "ok": contract_ready,
        "verdict": verdict,
        "contract_ready": contract_ready,
        "optimizer_ready": optimizer_ready,
        "backend_launch_ready": backend_launch_ready,
        "replay_path": str(config.replay_path),
        "output_dir": str(config.output_dir),
        "base_model": config.base_model,
        "backend": config.backend,
        "checks": checks,
        "replay_summary": replay_summary,
        "world_model_probe": world_model_probe,
        "launch_plan": launch_plan,
        "launch_env": launch_env,
        "next_actions": _next_actions(checks, optimizer_ready, backend_launch_ready),
        "artifacts": {},
    }


def _verdict(
    *,
    contract_ready: bool,
    optimizer_ready: bool,
    backend_launch_ready: bool,
    failed: set[str],
    warnings: set[str],
) -> dict[str, Any]:
    if not contract_ready:
        return {
            "level": "blocked",
            "summary": "Replay is not ready for DPPO/ECHO/RWML backend handoff.",
            "blocking_codes": sorted(failed & CONTRACT_BLOCKERS),
        }
    if not optimizer_ready:
        return {
            "level": "needs_train_logprob_replay",
            "summary": "Replay contract is ready, but DPPO optimizer logprob enrichment is incomplete.",
            "blocking_codes": sorted(warnings),
        }
    if not backend_launch_ready:
        return {
            "level": "needs_backend",
            "summary": "Replay and optimizer contract are ready; install/configure a backend to run it.",
            "blocking_codes": sorted(warnings),
        }
    return {
        "level": "ready",
        "summary": "Replay, world-model payloads, optimizer logprobs, and launch plan are ready.",
        "blocking_codes": [],
    }


def _next_actions(
    checks: list[dict[str, str]],
    optimizer_ready: bool,
    backend_launch_ready: bool,
) -> list[str]:
    actions: list[str] = []
    for check in checks:
        if check["status"] in {"fail", "warn"} and check["next_step"] not in actions:
            actions.append(check["next_step"])
    if optimizer_ready and not backend_launch_ready:
        actions.append("Copy the launch env/script to the private compute target after installing the selected backend.")
    if optimizer_ready and backend_launch_ready:
        actions.append("Run the materialized launch script on the target GPU machine.")
    return actions


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
