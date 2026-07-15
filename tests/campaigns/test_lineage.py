"""Fail-closed Git lineage for code-mutating AutoResearch hypotheses."""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.contracts import (
    ActionAttempt,
    Capability,
    CodeLineageRecord,
    CodeLineageState,
    CodeMutationKind,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    canonical_hash,
)
from bashgym.campaigns.lineage import (
    ApprovedSourceRepositoryProfile,
    GitHypothesisLineageManager,
    GitLineageError,
    code_mutation_kind_for_variable,
)
from bashgym.campaigns.persistence import CampaignPersistenceError, CampaignRepository
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignControllerService, CampaignService
from bashgym.campaigns.worker import CampaignWorker
from tests.campaigns.test_persistence import campaign
from tests.campaigns.test_proposals import activate, principal, proposal

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def initialized_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "source"
    repository.mkdir()
    git(repository, "init", "-b", "main")
    git(repository, "config", "user.name", "Test Operator")
    git(repository, "config", "user.email", "operator@example.invalid")
    files = {
        "bashgym/gym/trainer.py": "LEARNING_RATE = 1e-4\n",
        "bashgym/gym/environment.py": "MAX_STEPS = 8\n",
        "bashgym/rewards/scorer.py": "WEIGHT = 1.0\n",
        "bashgym/eval/verifier.py": "THRESHOLD = 0.5\n",
        "README.md": "source fixture\n",
    }
    for relative, content in files.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "initial source")
    return repository, git(repository, "rev-parse", "HEAD")


def source_profile(repository: Path) -> ApprovedSourceRepositoryProfile:
    return ApprovedSourceRepositoryProfile(
        profile_id="bashgym-source-v1",
        repository_path=repository,
        allowed_mutation_paths={
            CodeMutationKind.TRAINER: ("bashgym/gym/trainer.py",),
            CodeMutationKind.GYM: ("bashgym/gym",),
            CodeMutationKind.REWARD: ("bashgym/rewards",),
            CodeMutationKind.EVALUATOR: ("bashgym/eval",),
        },
    )


def requirement(kind: CodeMutationKind = CodeMutationKind.TRAINER) -> CodeLineageRecord:
    return CodeLineageRecord(
        lineage_id="lineage-candidate-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        proposal_id="candidate-1",
        mutation_kind=kind,
        source_repository_profile_id="bashgym-source-v1",
        state=CodeLineageState.REQUIRED,
        created_at=NOW,
        updated_at=NOW,
    )


def approved_lineage_remote_profile(
    tmp_path: Path, binding: ApprovedCodeLineageExecutionBinding | None = None
) -> ApprovedRemoteExecutorProfile:
    script = tmp_path / "train.py"
    data = tmp_path / "train.jsonl"
    key = tmp_path / "campaign-key"
    script.write_text("print('training')\n", encoding="utf-8")
    data.write_text("{}\n", encoding="utf-8")
    key.write_text("test-only-key\n", encoding="utf-8")
    stage = PinnedRemoteStageProfile(
        stage=StageKind.FULL_TRAINING,
        script_path=script,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input_files=(data,),
        input_sha256={data.name: hashlib.sha256(data.read_bytes()).hexdigest()},
        budget_reservation=0.25,
        code_lineage_binding=binding,
    )
    return ApprovedRemoteExecutorProfile(
        profile_id="memexai-approved-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=canonical_hash(campaign().target_model.model_dump(mode="json")),
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(stage,),
    )


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        ("trainer.optimizer", CodeMutationKind.TRAINER),
        ("algorithm.policy_loss", CodeMutationKind.TRAINER),
        ("gym.max_steps", CodeMutationKind.GYM),
        ("environment.tool_timeout", CodeMutationKind.GYM),
        ("reward.correctness_weight", CodeMutationKind.REWARD),
        ("evaluator.pass_threshold", CodeMutationKind.EVALUATOR),
        ("verifier.safety_gate", CodeMutationKind.EVALUATOR),
        ("training.learning_rate", None),
        ("dataset.mix_ratio", None),
    ],
)
def test_only_code_variables_require_git_lineage(variable, expected) -> None:
    assert code_mutation_kind_for_variable(variable) == expected


def test_prepare_capture_and_replay_preserve_one_scoped_commit(tmp_path: Path) -> None:
    repository, base_commit = initialized_repository(tmp_path)
    manager = GitHypothesisLineageManager(tmp_path / "worktrees")
    profile = source_profile(repository)

    prepared = manager.prepare(profile, requirement())

    assert prepared.record.state == CodeLineageState.PREPARED
    assert prepared.record.base_commit == base_commit
    assert prepared.record.branch_name.startswith("bashgym/autoresearch/")
    assert prepared.worktree_path.is_dir()
    assert str(repository) not in prepared.record.model_dump_json()

    trainer = prepared.worktree_path / "bashgym" / "gym" / "trainer.py"
    trainer.write_text("LEARNING_RATE = 2e-4\n", encoding="utf-8")
    captured = manager.capture(profile, prepared.record)

    assert captured.state == CodeLineageState.CAPTURED
    assert captured.commit_sha is not None
    assert captured.commit_sha != base_commit
    assert captured.changed_paths == ("bashgym/gym/trainer.py",)
    assert captured.patch_sha256 is not None
    assert git(repository, "rev-parse", captured.branch_name) == captured.commit_sha
    assert git(repository, "rev-list", "--count", f"{base_commit}..{captured.commit_sha}") == "1"
    message = git(repository, "show", "-s", "--format=%B", captured.commit_sha)
    assert f"BashGym-Lineage: {captured.lineage_id}" in message

    replay = manager.capture(profile, captured)
    assert replay == captured
    recovered_after_unpersisted_commit = manager.capture(profile, prepared.record)
    assert recovered_after_unpersisted_commit == captured


def test_materialized_snapshot_is_exact_bounded_and_tamper_evident(tmp_path: Path) -> None:
    repository, _base_commit = initialized_repository(tmp_path)
    manager = GitHypothesisLineageManager(tmp_path / "worktrees")
    profile = source_profile(repository)
    prepared = manager.prepare(profile, requirement())
    (prepared.worktree_path / "bashgym" / "gym" / "trainer.py").write_text(
        "LEARNING_RATE = 3e-4\n", encoding="utf-8"
    )
    captured = manager.capture(profile, prepared.record)

    receipt = manager.materialize_snapshot(
        profile,
        captured,
        tmp_path / "snapshots",
        entrypoint_path="bashgym/gym/trainer.py",
        max_archive_bytes=1024 * 1024,
    )

    assert receipt.record_digest == captured.record_digest
    assert receipt.commit_sha == captured.commit_sha
    assert receipt.archive_path.parent == (tmp_path / "snapshots").resolve()
    assert str(repository) not in repr(receipt)
    with tarfile.open(receipt.archive_path, mode="r:") as archive:
        assert archive.extractfile("source/bashgym/gym/trainer.py").read() == (
            b"LEARNING_RATE = 3e-4\n"
        )
        assert all(not member.issym() and not member.islnk() for member in archive.getmembers())
    manager.verify_snapshot_receipt(receipt, captured, max_archive_bytes=1024 * 1024)

    receipt.archive_path.write_bytes(receipt.archive_path.read_bytes() + b"tampered")
    with pytest.raises(GitLineageError, match="campaign_git_lineage_snapshot_digest_mismatch"):
        manager.verify_snapshot_receipt(receipt, captured, max_archive_bytes=1024 * 1024)


def test_snapshot_requires_an_entrypoint_from_the_captured_commit(tmp_path: Path) -> None:
    repository, _base_commit = initialized_repository(tmp_path)
    manager = GitHypothesisLineageManager(tmp_path / "worktrees")
    profile = source_profile(repository)
    prepared = manager.prepare(profile, requirement())
    (prepared.worktree_path / "bashgym" / "gym" / "trainer.py").write_text(
        "LEARNING_RATE = 4e-4\n", encoding="utf-8"
    )
    captured = manager.capture(profile, prepared.record)

    with pytest.raises(GitLineageError, match="campaign_git_lineage_entrypoint_unavailable"):
        manager.materialize_snapshot(
            profile,
            captured,
            tmp_path / "snapshots",
            entrypoint_path="scripts/missing.py",
            max_archive_bytes=1024 * 1024,
        )


def test_git_lineage_ignores_inherited_git_repository_and_diff_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _base_commit = initialized_repository(tmp_path)
    manager = GitHypothesisLineageManager(tmp_path / "worktrees")
    profile = source_profile(repository)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "redirected.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "redirected-worktree"))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "command-that-must-not-run")

    prepared = manager.prepare(profile, requirement())
    trainer = prepared.worktree_path / "bashgym" / "gym" / "trainer.py"
    trainer.write_text("LEARNING_RATE = 2e-4\n", encoding="utf-8")
    captured = manager.capture(profile, prepared.record)

    assert captured.changed_paths == ("bashgym/gym/trainer.py",)


def test_capture_timestamp_never_precedes_prepared_timestamp(tmp_path: Path) -> None:
    repository, _base_commit = initialized_repository(tmp_path)
    manager = GitHypothesisLineageManager(tmp_path / "worktrees")
    profile = source_profile(repository)
    prepared = manager.prepare(profile, requirement())
    future_prepared_at = prepared.record.updated_at + timedelta(days=1)
    prepared_record = prepared.record.model_copy(update={"updated_at": future_prepared_at})
    trainer = prepared.worktree_path / "bashgym" / "gym" / "trainer.py"
    trainer.write_text("LEARNING_RATE = 2e-4\n", encoding="utf-8")

    captured = manager.capture(profile, prepared_record)

    assert captured.updated_at == future_prepared_at
    assert captured.captured_at == future_prepared_at


def test_generated_branch_normalizes_git_forbidden_double_dots(tmp_path: Path) -> None:
    repository, _base_commit = initialized_repository(tmp_path)
    manager = GitHypothesisLineageManager(tmp_path / "worktrees")
    unsafe_slug = requirement().model_copy(update={"proposal_id": "candidate..one"})

    prepared = manager.prepare(source_profile(repository), unsafe_slug)

    assert ".." not in prepared.record.branch_name


def test_capture_rejects_changes_outside_operator_approved_scope(tmp_path: Path) -> None:
    repository, _base_commit = initialized_repository(tmp_path)
    manager = GitHypothesisLineageManager(tmp_path / "worktrees")
    profile = source_profile(repository)
    prepared = manager.prepare(profile, requirement())
    (prepared.worktree_path / "README.md").write_text("unauthorized\n", encoding="utf-8")

    with pytest.raises(GitLineageError) as raised:
        manager.capture(profile, prepared.record)

    assert raised.value.code == "campaign_git_lineage_path_not_approved"
    assert git(repository, "rev-parse", prepared.record.branch_name) == prepared.record.base_commit


def test_capture_rejects_orphan_commit_that_is_not_based_on_approved_head(
    tmp_path: Path,
) -> None:
    repository, _base_commit = initialized_repository(tmp_path)
    manager = GitHypothesisLineageManager(tmp_path / "worktrees")
    profile = source_profile(repository)
    prepared = manager.prepare(profile, requirement())
    worktree = prepared.worktree_path
    assert prepared.record.branch_name is not None

    git(worktree, "checkout", "--orphan", "orphan-candidate")
    git(worktree, "rm", "-rf", ".")
    trainer = worktree / "bashgym" / "gym" / "trainer.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("LEARNING_RATE = 2e-4\n", encoding="utf-8")
    git(worktree, "add", "bashgym/gym/trainer.py")
    git(
        worktree,
        "commit",
        "-m",
        "malicious orphan",
        "-m",
        f"BashGym-Lineage: {prepared.record.lineage_id}",
    )
    git(worktree, "branch", "-f", prepared.record.branch_name, "HEAD")
    git(worktree, "checkout", prepared.record.branch_name)

    with pytest.raises(GitLineageError, match="campaign_git_lineage_commit_parent_invalid"):
        manager.capture(profile, prepared.record)


def test_repository_persists_monotonic_required_prepared_captured_lineage(
    tmp_path: Path,
) -> None:
    repository = CampaignRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    activate(repository)
    service = CampaignService(repository)
    actor = principal(repository)
    submitted = service.submit_proposal(
        proposal("candidate-1"),
        expected_version=4,
        principal=actor,
        correlation_id="submit-candidate",
        idempotency_key="submit-candidate",
    )
    required = requirement()
    assert repository.register_code_lineage_requirement(required) == required

    prepared = required.model_copy(
        update={
            "state": CodeLineageState.PREPARED,
            "base_commit": "a" * 40,
            "branch_name": "bashgym/autoresearch/candidate-1-deadbeef",
            "updated_at": NOW,
        }
    )
    assert repository.advance_code_lineage(prepared) == prepared
    captured = prepared.model_copy(
        update={
            "state": CodeLineageState.CAPTURED,
            "commit_sha": "b" * 40,
            "changed_paths": ("bashgym/gym/trainer.py",),
            "patch_sha256": "c" * 64,
            "captured_at": NOW,
            "updated_at": NOW,
        }
    )
    assert repository.advance_code_lineage(captured) == captured
    assert repository.advance_code_lineage(captured) == captured
    assert repository.get_code_lineage("workspace-a", "candidate-1") == captured
    assert repository.list_code_lineages("workspace-a", "campaign-1") == (captured,)

    with pytest.raises(ValueError, match="campaign_code_lineage_transition_invalid"):
        repository.advance_code_lineage(required)

    assert submitted.event.event_type == "campaign:proposal-submitted"


def test_action_spec_fails_closed_then_binds_captured_lineage(tmp_path: Path) -> None:
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    activate(repository)
    submitted = CampaignService(repository).submit_proposal(
        proposal("candidate-1"),
        expected_version=4,
        principal=principal(repository),
        correlation_id="submit-candidate",
        idempotency_key="submit-candidate",
    )
    required = requirement()
    repository.register_code_lineage_requirement(required)
    selected = CampaignControllerService(
        repository, controller_id="campaign-controller"
    ).select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=submitted.campaign.version,
        correlation_id="select-candidate",
        idempotency_key="select-candidate",
    )
    assert selected is not None

    with pytest.raises(CampaignPersistenceError, match="campaign_code_lineage_not_captured"):
        repository.next_action_spec("workspace-a", "campaign-1", selected.study.study_id)

    prepared = CodeLineageRecord.model_validate(
        {
            **required.model_dump(mode="python"),
            "state": CodeLineageState.PREPARED,
            "base_commit": "a" * 40,
            "branch_name": "bashgym/autoresearch/candidate-1-deadbeef",
        }
    )
    repository.advance_code_lineage(prepared)
    captured = CodeLineageRecord.model_validate(
        {
            **prepared.model_dump(mode="python"),
            "state": CodeLineageState.CAPTURED,
            "commit_sha": "b" * 40,
            "changed_paths": ("bashgym/gym/trainer.py",),
            "patch_sha256": "c" * 64,
            "captured_at": NOW,
        }
    )
    repository.advance_code_lineage(captured)

    action = repository.next_action_spec("workspace-a", "campaign-1", selected.study.study_id)
    evidence = action.input_contract["code_lineage"]
    assert evidence["record_digest"] == captured.record_digest
    assert evidence["commit_sha"] == "b" * 40
    assert evidence["changed_paths"] == ["bashgym/gym/trainer.py"]


def test_code_mutation_cannot_schedule_before_lineage_registration(
    tmp_path: Path,
) -> None:
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    activate(repository)
    submitted = CampaignService(repository).submit_proposal(
        proposal("candidate-1").model_copy(update={"primary_variable": "trainer.optimizer"}),
        expected_version=4,
        principal=principal(repository),
        correlation_id="submit-candidate",
        idempotency_key="submit-candidate",
    )
    selected = CampaignControllerService(
        repository, controller_id="campaign-controller"
    ).select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=submitted.campaign.version,
        correlation_id="select-candidate",
        idempotency_key="select-candidate",
    )
    assert selected is not None

    with pytest.raises(
        CampaignPersistenceError,
        match="campaign_code_lineage_not_registered",
    ):
        repository.next_action_spec("workspace-a", "campaign-1", selected.study.study_id)


def test_code_mutation_rejects_lineage_for_a_different_code_surface(
    tmp_path: Path,
) -> None:
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    activate(repository)
    submitted = CampaignService(repository).submit_proposal(
        proposal("candidate-1").model_copy(update={"primary_variable": "trainer.optimizer"}),
        expected_version=4,
        principal=principal(repository),
        correlation_id="submit-candidate",
        idempotency_key="submit-candidate",
    )
    required = requirement(CodeMutationKind.REWARD)
    repository.register_code_lineage_requirement(required)
    prepared = required.model_copy(
        update={
            "state": CodeLineageState.PREPARED,
            "base_commit": "a" * 40,
            "branch_name": "bashgym/autoresearch/candidate-1-deadbeef",
        }
    )
    repository.advance_code_lineage(prepared)
    repository.advance_code_lineage(
        prepared.model_copy(
            update={
                "state": CodeLineageState.CAPTURED,
                "commit_sha": "b" * 40,
                "changed_paths": ("bashgym/rewards/scorer.py",),
                "patch_sha256": "c" * 64,
                "captured_at": NOW,
            }
        )
    )
    selected = CampaignControllerService(
        repository, controller_id="campaign-controller"
    ).select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=submitted.campaign.version,
        correlation_id="select-candidate",
        idempotency_key="select-candidate",
    )
    assert selected is not None

    with pytest.raises(
        CampaignPersistenceError,
        match="campaign_code_lineage_mutation_kind_mismatch",
    ):
        repository.next_action_spec("workspace-a", "campaign-1", selected.study.study_id)


def test_captured_lineage_requires_then_uses_registered_executor_binding(
    tmp_path: Path,
) -> None:
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    activate(repository)
    training_proposal = proposal("candidate-1").model_copy(
        update={
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
            },
            "required_capabilities": frozenset({Capability.COMPUTE_TRAIN_WITHIN_BUDGET}),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Run the captured code hypothesis.",
                    ),
                )
            ),
        }
    )
    submitted = CampaignService(repository).submit_proposal(
        training_proposal,
        expected_version=4,
        principal=principal(repository),
        correlation_id="submit-candidate",
        idempotency_key="submit-candidate",
    )
    required = requirement()
    repository.register_code_lineage_requirement(required)
    prepared = CodeLineageRecord.model_validate(
        {
            **required.model_dump(mode="python"),
            "state": CodeLineageState.PREPARED,
            "base_commit": "a" * 40,
            "branch_name": "bashgym/autoresearch/candidate-1-deadbeef",
        }
    )
    repository.advance_code_lineage(prepared)
    captured = CodeLineageRecord.model_validate(
        {
            **prepared.model_dump(mode="python"),
            "state": CodeLineageState.CAPTURED,
            "commit_sha": "b" * 40,
            "changed_paths": ("bashgym/gym/trainer.py",),
            "patch_sha256": "c" * 64,
            "captured_at": NOW,
        }
    )
    repository.advance_code_lineage(captured)
    selected = CampaignControllerService(
        repository, controller_id="campaign-controller"
    ).select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=submitted.campaign.version,
        correlation_id="select-candidate",
        idempotency_key="select-candidate",
    )
    assert selected is not None

    with pytest.raises(
        CampaignPersistenceError,
        match="campaign_code_lineage_execution_binding_required",
    ):
        unbound_profile = approved_lineage_remote_profile(tmp_path)
        repository.next_action_spec(
            "workspace-a",
            "campaign-1",
            selected.study.study_id,
            executor_profiles={
                (
                    unbound_profile.compute_profile_id,
                    unbound_profile.target_contract_key,
                ): unbound_profile
            },
        )

    binding = ApprovedCodeLineageExecutionBinding(
        binding_id="bashgym-trainer-entrypoint-v1",
        binding_revision=1,
        source_repository_profile_id="bashgym-source-v1",
        entrypoint_path="bashgym/gym/trainer.py",
        working_directory="source",
        max_archive_bytes=1024 * 1024,
    )
    profile = approved_lineage_remote_profile(tmp_path, binding)
    action = repository.next_action_spec(
        "workspace-a",
        "campaign-1",
        selected.study.study_id,
        executor_profiles={(profile.compute_profile_id, profile.target_contract_key): profile},
    )

    assert action.executor_kind == "ssh_remote"
    execution = action.executor_config["code_lineage_execution"]
    assert execution["binding_digest"] == binding.binding_digest
    assert execution["record_digest"] == captured.record_digest
    assert execution["commit_sha"] == captured.commit_sha


def test_worker_materializes_captured_commit_into_remote_launch_request(
    tmp_path: Path,
) -> None:
    source_repository, _base_commit = initialized_repository(tmp_path)
    source = source_profile(source_repository)
    manager = GitHypothesisLineageManager(tmp_path / "data" / "campaigns" / "source-worktrees")
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    activate(repository)
    training_proposal = proposal("candidate-1").model_copy(
        update={
            "primary_variable": "trainer.optimizer",
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
            },
            "required_capabilities": frozenset({Capability.COMPUTE_TRAIN_WITHIN_BUDGET}),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Run the captured code hypothesis.",
                    ),
                )
            ),
        }
    )
    submitted = CampaignService(repository).submit_proposal(
        training_proposal,
        expected_version=4,
        principal=principal(repository),
        correlation_id="submit-worker-candidate",
        idempotency_key="submit-worker-candidate",
    )
    required = requirement()
    repository.register_code_lineage_requirement(required)
    prepared = manager.prepare(source, required)
    repository.advance_code_lineage(prepared.record)
    (prepared.worktree_path / "bashgym" / "gym" / "trainer.py").write_text(
        "LEARNING_RATE = 5e-4\n", encoding="utf-8"
    )
    captured = manager.capture(source, prepared.record)
    repository.advance_code_lineage(captured)
    selected = CampaignControllerService(
        repository, controller_id="campaign-controller"
    ).select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=submitted.campaign.version,
        correlation_id="select-worker-candidate",
        idempotency_key="select-worker-candidate",
    )
    assert selected is not None
    binding = ApprovedCodeLineageExecutionBinding(
        binding_id="bashgym-trainer-entrypoint-v1",
        binding_revision=1,
        source_repository_profile_id=source.profile_id,
        entrypoint_path="bashgym/gym/trainer.py",
        max_archive_bytes=1024 * 1024,
    )
    remote_profile = approved_lineage_remote_profile(tmp_path, binding)
    profiles = {
        (remote_profile.compute_profile_id, remote_profile.target_contract_key): remote_profile
    }
    action = repository.next_action_spec(
        "workspace-a",
        "campaign-1",
        selected.study.study_id,
        executor_profiles=profiles,
    )
    attempt = ActionAttempt(
        attempt_id="attempt-lineage-worker-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id=selected.study.study_id,
        action_id="action-lineage-worker-1",
        attempt_number=1,
        input_digest=action.input_digest,
        candidate_digest=action.candidate_digest,
        manifest_revision=action.manifest_revision,
        stage=action.stage,
        executor={"kind": action.executor_kind, **action.executor_config},
    )
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"l" * 32, key_version="lineage-worker-v1"),
        data_directory=tmp_path / "data",
        remote_executor_profiles=profiles,
        source_repository_profiles={source.profile_id: source},
        lineage_manager=manager,
    )

    request = worker._remote_request(attempt)

    assert request.source_snapshot is not None
    assert request.source_snapshot.commit_sha == captured.commit_sha
    assert request.source_snapshot.record_digest == captured.record_digest
    assert request.source_snapshot.archive_path.is_file()
    assert worker._remote_request(attempt).source_snapshot == request.source_snapshot

    stripped_executor = dict(attempt.executor)
    stripped_executor.pop("code_lineage_execution")
    with pytest.raises(RuntimeError, match="campaign_remote_executor_profile_mismatch"):
        worker._remote_request(attempt.model_copy(update={"executor": stripped_executor}))

    request.source_snapshot.archive_path.write_bytes(
        request.source_snapshot.archive_path.read_bytes() + b"tampered"
    )
    with pytest.raises(RuntimeError, match="campaign_code_lineage_snapshot_invalid"):
        worker._remote_request(attempt)
