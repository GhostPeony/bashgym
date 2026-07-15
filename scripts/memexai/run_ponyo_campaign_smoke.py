#!/usr/bin/env python3
"""Run one bounded MemexAI smoke through the durable Ponyo campaign worker.

This is an operator verification harness, not a training-results generator.  It
creates a fresh campaign, submits a mode-only remote proposal, launches exactly
one cached-MNRL step through a hash-pinned server profile, replaces the worker
while the remote process is active, and records restart/adoption evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.contracts import (
    AutonomyProfile,
    Campaign,
    CampaignKind,
    CampaignManifest,
    CampaignStatus,
    CampaignTrigger,
    Capability,
    CredentialKind,
    ManifestRevision,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyProposalSubmission,
    TargetModelContract,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.remote import (
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
    RemoteCapacityPolicy,
    RemoteTrainingAdapter,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.worker import CampaignWorker
from bashgym.gym.remote_trainer import SSHConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--grouped-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--key-path", type=Path, required=True)
    parser.add_argument("--host", default="192.168.50.173")
    parser.add_argument("--username", default="ponyo")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--remote-work-dir", default="~/bashgym-training/campaign-smokes")
    parser.add_argument(
        "--remote-python",
        default=(
            "/home/ponyo/bashgym-training/"
            "memexai-positive-aware-v1-20260712/.venv/bin/python"
        ),
    )
    parser.add_argument(
        "--base-model-path", default="/home/ponyo/models/embedding/qwen3-embedding-0.6b"
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify and emit evidence for an already completed proof database.",
    )
    return parser.parse_args()


def require_regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return candidate.resolve()


def target_model_contract() -> TargetModelContract:
    return TargetModelContract(
        target_contract_key="memexai-embedding-v1",
        base_model_ref="Qwen/Qwen3-Embedding-0.6B",
        task="text-retrieval",
        representation_contract={"pooling": "last-token", "normalize": True},
    )


def create_active_campaign(
    repository: CampaignRuntimeRepository, campaign_id: str
) -> Campaign:
    campaign = Campaign(
        campaign_id=campaign_id,
        workspace_id="memexai-automation-proof",
        title="MemexAI bounded Ponyo campaign smoke",
        kind=CampaignKind.EMBEDDING_RETRIEVAL,
        objective=(
            "Prove server-owned Ponyo execution and restart adoption without making "
            "a promotion or model-quality claim."
        ),
        target_model=target_model_contract(),
        owner_actor_id="codex-agent",
    )
    manifest = CampaignManifest(
        approved_data_scopes=("memexai-positive-aware-v1-20260712",),
        compute_profile_id="ponyo-private",
        budget_limits={"gpu_hours": 0.10, "study_count": 1.0},
        evaluation_plan={"purpose": "runtime_smoke_only", "protected_test": False},
        promotion_gates={"promotion_allowed": False},
        protected_artifact_refs=("frozen-test-36-v1",),
    )
    repository.create_campaign(
        campaign,
        ManifestRevision(
            workspace_id=campaign.workspace_id,
            campaign_id=campaign.campaign_id,
            revision=1,
            manifest=manifest,
            actor_id="codex-agent",
            correlation_id=f"{campaign_id}:create",
        ),
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id=f"{campaign_id}:create",
        idempotency_key=f"{campaign_id}:create",
    )
    version = 1
    for trigger in (
        CampaignTrigger.VALIDATE,
        CampaignTrigger.VALIDATION_PASSED,
        CampaignTrigger.START,
    ):
        result = repository.transition_campaign(
            campaign.workspace_id,
            campaign.campaign_id,
            trigger,
            expected_version=version,
            actor_id="campaign-controller",
            credential_kind=CredentialKind.CONTROLLER,
            correlation_id=f"{campaign_id}:{trigger.value}",
            idempotency_key=f"{campaign_id}:{trigger.value}",
        )
        version = result.campaign.version
    return repository.get_campaign(campaign.workspace_id, campaign.campaign_id)


def submit_smoke_proposal(
    repository: CampaignRuntimeRepository, campaign: Campaign
) -> None:
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=(campaign.workspace_id,),
    )
    principal = auth.authenticate_access(auth.exchange_refresh(refresh.raw_token).raw_token)
    proposal = StudyProposalSubmission(
        proposal_id=f"{campaign.campaign_id}-cached-mnrl-smoke",
        workspace_id=campaign.workspace_id,
        campaign_id=campaign.campaign_id,
        hypothesis="The pinned cached-MNRL recipe can execute and be adopted after worker restart.",
        evidence_references=("candidate-b-full-b128-realized17-mb4-e2-bf16-r1",),
        study_family="embedding-retrieval-runtime-proof",
        primary_variable="campaign_worker_remote_execution",
        controlled_variables=("dataset", "base_model", "seed", "loss", "batch_size"),
        expected_outcome="One remote step completes with sealed evidence and one launch identity.",
        falsification_criterion=(
            "Reject if launch duplicates, identity changes, outputs are unsealed, or budget is unsettled."
        ),
        estimated_cost=0.02,
        priority=100,
        dataset_recipe={
            "schema_version": "recipe.v1",
            "data_scope_id": "memexai-positive-aware-v1-20260712",
        },
        training_recipe={
            "schema_version": "recipe.v1",
            "runtime": {"executor_kind": "registered_training"},
        },
        evaluation_recipe={
            "schema_version": "recipe.v1",
            "purpose": "runtime_smoke_only",
        },
        required_capabilities=frozenset({Capability.COMPUTE_SMOKE}),
        stage_plan=StagePlan(
            items=(
                StagePlanItem(
                    stage=StageKind.SMOKE_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="One-step remote runtime, metrics, and restart-adoption proof.",
                    input_contract={"quality_claim": False, "protected_test": False},
                ),
            )
        ),
        rationale="Bounded smoke evidence for the production campaign executor boundary.",
    )
    submission = CampaignService(repository).submit_proposal(
        proposal,
        expected_version=campaign.version,
        principal=principal,
        correlation_id=f"{campaign.campaign_id}:submit",
        idempotency_key=f"{campaign.campaign_id}:submit",
    )
    if not submission.record.validation.valid:
        raise RuntimeError(
            "campaign proposal rejected: "
            + ",".join(submission.record.validation.reason_codes)
        )


def build_profile(args: argparse.Namespace) -> ApprovedRemoteExecutorProfile:
    script = require_regular_file(args.script, "training script")
    grouped = require_regular_file(args.grouped_jsonl, "grouped training data")
    corpus = require_regular_file(args.corpus_jsonl, "corpus")
    key = require_regular_file(args.key_path, "SSH key")
    script_args = (
        "--grouped-jsonl",
        grouped.name,
        "--corpus-jsonl",
        corpus.name,
        "--base-model-path",
        args.base_model_path,
        "--output-dir",
        ".",
        "--train-splits",
        "train",
        "--query-prefix-mode",
        "memexai_youtube",
        "--batch-size",
        "16",
        "--epochs",
        "1",
        "--max-steps",
        "1",
        "--learning-rate",
        "2e-6",
        "--warmup-steps",
        "0",
        "--weight-decay",
        "0.01",
        "--max-seq-length",
        "1024",
        "--truncate-dim",
        "768",
        "--seed",
        "20260708",
        "--device",
        "cuda",
        "--loss",
        "cached_mnrl",
        "--temperature",
        "0.02",
        "--mini-batch-size",
        "4",
        "--negatives-per-query",
        "3",
        "--logging-steps",
        "1",
        "--checkpoint-save-steps",
        "-1",
        "--use-bf16",
    )
    stage = PinnedRemoteStageProfile(
        stage=StageKind.SMOKE_TRAINING,
        script_path=script,
        script_sha256=sha256_file(script),
        input_files=(grouped, corpus),
        input_sha256={grouped.name: sha256_file(grouped), corpus.name: sha256_file(corpus)},
        script_args=script_args,
        output_paths=("final", "training_manifest.json", "training_metrics.jsonl"),
        capacity_policy=RemoteCapacityPolicy(
            minimum_available_memory_gib=48.0,
            minimum_available_disk_gib=20.0,
            maximum_external_gpu_processes=0,
        ),
        budget_unit="gpu_hours",
        budget_reservation=0.02,
        python_executable=args.remote_python,
    )
    return ApprovedRemoteExecutorProfile(
        profile_id="memexai-positive-aware-cached-mnrl-smoke-v1",
        profile_revision=1,
        compute_profile_id="ponyo-private",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=canonical_hash(target_model_contract().model_dump(mode="json")),
        host=args.host,
        username=args.username,
        port=args.port,
        key_path=str(key),
        remote_work_dir=args.remote_work_dir,
        stages=(stage,),
    )


def make_worker(
    args: argparse.Namespace,
    profile: ApprovedRemoteExecutorProfile,
    seal_key: bytes,
    worker_id: str,
    *,
    leader_ttl_seconds: float,
) -> CampaignWorker:
    repository = CampaignRuntimeRepository(args.database)
    repository.initialize()
    adapter = RemoteTrainingAdapter(
        SSHConfig(
            host=profile.host,
            username=profile.username,
            port=profile.port,
            key_path=profile.key_path,
            remote_work_dir=profile.remote_work_dir,
        ),
        compute_profile_id=profile.compute_profile_id,
    )
    return CampaignWorker(
        repository,
        args.artifact_root,
        ArtifactSealer(seal_key, key_version="ponyo-campaign-smoke-v1"),
        data_directory=args.data_directory,
        worker_id=worker_id,
        leader_ttl=timedelta(seconds=leader_ttl_seconds),
        action_ttl=timedelta(seconds=2),
        remote_adapters={profile.compute_profile_id: adapter},
        remote_executor_profiles={
            (profile.compute_profile_id, profile.target_contract_key): profile
        },
    )


def remote_identity_projection(identity: Any) -> dict[str, Any]:
    return {
        "compute_profile_id": identity.compute_profile_id,
        "run_id": identity.run_id,
        "remote_run_directory": identity.remote_run_directory,
        "remote_pid": identity.remote_pid,
        "process_group_id": identity.process_group_id,
        "process_start_ticks": identity.process_start_ticks,
        "boot_id": identity.boot_id,
        "command_hash": identity.command_hash,
        "launch_manifest_sha256": identity.launch_manifest_sha256,
        "launched_at": identity.launched_at.isoformat(),
    }


def verify_existing_campaign(
    args: argparse.Namespace, profile: ApprovedRemoteExecutorProfile
) -> int:
    repository = CampaignRuntimeRepository(args.database)
    repository.initialize()
    workspace_id = "memexai-automation-proof"
    campaign = repository.get_campaign(workspace_id, args.campaign_id)
    attempts = repository.list_attempts(workspace_id, args.campaign_id)
    if len(attempts) != 1 or attempts[0].status.value != "completed":
        raise RuntimeError("existing proof must contain exactly one completed attempt")
    attempt = attempts[0]
    remote = repository.get_remote_run(workspace_id, attempt.attempt_id)
    if remote is None or remote.state.value != "completed":
        raise RuntimeError("existing proof has no completed remote identity")
    if attempt.claim_generation < 2:
        raise RuntimeError("existing proof does not show successor adoption")
    if not attempt.sealed_result_uri:
        raise RuntimeError("existing proof has no sealed result URI")
    result_root = Path(attempt.sealed_result_uri)
    required_outputs = (
        result_root / "final" / "model.safetensors",
        result_root / "training_manifest.json",
        result_root / "training_metrics.jsonl",
        result_root / "launch_manifest.json",
        result_root / "sealed_action_result.v1.json",
    )
    missing = [str(path) for path in required_outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"sealed result is incomplete: {missing}")
    artifacts = repository.list_artifacts(workspace_id, args.campaign_id)
    if not artifacts or any(not item.sealed or not item.valid for item in artifacts):
        raise RuntimeError("existing proof includes unsealed or invalid artifact records")
    totals = repository.budget_totals(workspace_id, args.campaign_id, "gpu_hours")
    if totals["reserved"] != 0.0 or round(totals["actual"], 6) != 0.02:
        raise RuntimeError(f"campaign budget was not settled: {totals}")
    if campaign.champion_ref is not None or campaign.status != CampaignStatus.ACTIVE:
        raise RuntimeError("runtime smoke must not promote or replace the base champion")
    metrics = repository.get_metric_series(
        workspace_id,
        attempt.attempt_id,
        "loss",
        source="training_metrics.jsonl",
    )
    if not metrics:
        raise RuntimeError("training loss series was not persisted")
    events = repository.list_events(workspace_id, args.campaign_id)
    launched = [
        event
        for _cursor, event in events
        if event.event_type == "campaign:remote-run-registered"
    ]
    adopted = [
        event for _cursor, event in events if event.event_type == "campaign:remote-run-adopted"
    ]
    if len(launched) != 1 or not adopted:
        raise RuntimeError(
            f"expected one launch and successor adoption, found {len(launched)} / {len(adopted)}"
        )
    evidence = {
        "schema_version": "memexai_ponyo_campaign_smoke_evidence.v1",
        "created_at": utc_now().isoformat(),
        "workspace_id": workspace_id,
        "campaign_id": args.campaign_id,
        "campaign_status": campaign.status.value,
        "champion_ref": campaign.champion_ref,
        "profile_id": profile.profile_id,
        "profile_revision": profile.profile_revision,
        "profile_digest": profile.profile_digest,
        "script_sha256": profile.stages[0].script_sha256,
        "input_sha256": dict(sorted(profile.stages[0].input_sha256.items())),
        "attempt_id": attempt.attempt_id,
        "action_id": attempt.action_id,
        "attempt_status": attempt.status.value,
        "claim_generation": attempt.claim_generation,
        "successor_lease_owner": attempt.lease_owner,
        "remote_identity": remote_identity_projection(remote.identity),
        "single_remote_launch_event": True,
        "remote_adoption_event_count": len(adopted),
        "launch_correlation_id": launched[0].correlation_id,
        "adoption_correlation_ids": sorted({event.correlation_id for event in adopted}),
        "sealed_result_uri": str(result_root),
        "sealed_outputs": [str(path.relative_to(result_root)) for path in required_outputs],
        "sealed_artifact_count": len(artifacts),
        "budget": totals,
        "loss_points": [point.model_dump(mode="json") for point in metrics],
        "event_count": len(events),
        "event_types": [event.event_type for _cursor, event in events],
        "promotion_performed": False,
        "protected_test_opened": False,
    }
    args.evidence_json.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    profile = build_profile(args)
    if args.verify_existing:
        if not args.database.is_file():
            raise ValueError("proof database does not exist")
        args.evidence_json.parent.mkdir(parents=True, exist_ok=True)
        return verify_existing_campaign(args, profile)
    if args.database.exists():
        raise ValueError("proof database must not already exist")
    args.data_directory.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.evidence_json.parent.mkdir(parents=True, exist_ok=True)

    repository = CampaignRuntimeRepository(args.database)
    repository.initialize()
    active = create_active_campaign(repository, args.campaign_id)
    submit_smoke_proposal(repository, active)
    seal_key = secrets.token_bytes(32)

    first = make_worker(
        args,
        profile,
        seal_key,
        "ponyo-smoke-worker-before-restart",
        leader_ttl_seconds=2,
    )
    first_result = first.run_once(now=utc_now())
    if first_result != "remote_running":
        raise RuntimeError(f"expected remote_running, received {first_result}")
    before_attempt = first.repository.list_attempts(active.workspace_id, active.campaign_id)[0]
    before_remote = first.repository.get_remote_run(active.workspace_id, before_attempt.attempt_id)
    if before_remote is None:
        raise RuntimeError("remote identity was not persisted before worker replacement")

    time.sleep(3.0)
    second = make_worker(
        args,
        profile,
        seal_key,
        "ponyo-smoke-worker-after-restart",
        leader_ttl_seconds=max(15.0, args.poll_seconds * 4),
    )
    deadline = time.monotonic() + args.timeout_seconds
    poll_results: list[str] = []
    while time.monotonic() < deadline:
        result = second.run_once(now=utc_now())
        poll_results.append(result)
        attempt = second.repository.get_attempt(active.workspace_id, before_attempt.attempt_id)
        if attempt.status.value in {"completed", "failed", "cancelled"}:
            break
        time.sleep(args.poll_seconds)
    else:
        raise TimeoutError("bounded Ponyo campaign smoke did not reach a terminal state")

    attempt = second.repository.get_attempt(active.workspace_id, before_attempt.attempt_id)
    if attempt.status.value != "completed":
        raise RuntimeError(f"Ponyo smoke ended as {attempt.status.value}")
    after_remote = second.repository.get_remote_run(active.workspace_id, attempt.attempt_id)
    if after_remote is None or after_remote.identity != before_remote.identity:
        raise RuntimeError("successor did not preserve the exact remote identity")
    if attempt.claim_generation < 2 or attempt.lease_owner != second.worker_id:
        raise RuntimeError("successor worker did not adopt the remote attempt")
    if not attempt.sealed_result_uri:
        raise RuntimeError("completed attempt has no sealed result URI")
    result_root = Path(attempt.sealed_result_uri)
    required_outputs = (
        result_root / "final" / "model.safetensors",
        result_root / "training_manifest.json",
        result_root / "training_metrics.jsonl",
        result_root / "launch_manifest.json",
        result_root / "sealed_action_result.v1.json",
    )
    missing = [str(path) for path in required_outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"sealed result is incomplete: {missing}")

    totals = second.repository.budget_totals(
        active.workspace_id, active.campaign_id, "gpu_hours"
    )
    if totals["reserved"] != 0.0 or round(totals["actual"], 6) != 0.02:
        raise RuntimeError(f"campaign budget was not settled: {totals}")
    campaign = second.repository.get_campaign(active.workspace_id, active.campaign_id)
    if campaign.champion_ref is not None or campaign.status != CampaignStatus.ACTIVE:
        raise RuntimeError("runtime smoke must not promote or replace the base champion")
    metrics = second.repository.get_metric_series(
        active.workspace_id,
        attempt.attempt_id,
        "loss",
        source="training_metrics.jsonl",
    )
    if not metrics:
        raise RuntimeError("training loss series was not persisted")
    events = second.repository.list_events(active.workspace_id, active.campaign_id)
    launched = [
        event
        for _cursor, event in events
        if event.event_type == "campaign:remote-run-registered"
    ]
    if len(launched) != 1:
        raise RuntimeError(f"expected one remote launch event, found {len(launched)}")

    evidence = {
        "schema_version": "memexai_ponyo_campaign_smoke_evidence.v1",
        "created_at": utc_now().isoformat(),
        "workspace_id": active.workspace_id,
        "campaign_id": active.campaign_id,
        "campaign_status": campaign.status.value,
        "champion_ref": campaign.champion_ref,
        "profile_id": profile.profile_id,
        "profile_revision": profile.profile_revision,
        "profile_digest": profile.profile_digest,
        "script_sha256": profile.stages[0].script_sha256,
        "input_sha256": dict(sorted(profile.stages[0].input_sha256.items())),
        "attempt_id": attempt.attempt_id,
        "action_id": attempt.action_id,
        "attempt_status": attempt.status.value,
        "claim_generation": attempt.claim_generation,
        "worker_before": first.worker_id,
        "worker_after": second.worker_id,
        "first_result": first_result,
        "successor_poll_results": poll_results,
        "remote_identity_before": remote_identity_projection(before_remote.identity),
        "remote_identity_after": remote_identity_projection(after_remote.identity),
        "single_remote_launch_event": True,
        "sealed_result_uri": str(result_root),
        "sealed_outputs": [str(path.relative_to(result_root)) for path in required_outputs],
        "budget": totals,
        "loss_points": [point.model_dump(mode="json") for point in metrics],
        "event_count": len(events),
        "event_types": [event.event_type for _cursor, event in events],
        "promotion_performed": False,
        "protected_test_opened": False,
    }
    args.evidence_json.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
