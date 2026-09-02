"""Content-only result keys and the reuse policy."""

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.result_reuse import reuse_enabled, stage_result_key
from bashgym.campaigns.runtime import ActionSpec


def _key(**overrides):
    base = dict(
        stage=StageKind.DATA_BUILD,
        executor_kind="ssh_remote",
        manifest_digest="m" * 64,
        stage_input={"rows": 1000},
        recipe_digest="r" * 64,
        executor_config={
            "profile_digest": "p" * 64,
            "expected_script_sha256": "s" * 64,
            "script_args": ["--rows", "1000"],
            "remote_resident_dataset": {"attempt_id": "attempt-a", "content_digest": "c" * 64},
            "profile_id": "lab",
            "profile_revision": 3,
            "budget_reservation": 0.25,
        },
        upstream_outputs=(),
    )
    base.update(overrides)
    return stage_result_key(**base)


def test_identity_bearing_executor_fields_do_not_change_the_key() -> None:
    other = _key(
        executor_config={
            "profile_digest": "p" * 64,
            "expected_script_sha256": "s" * 64,
            "script_args": ["--rows", "1000"],
            "remote_resident_dataset": {"attempt_id": "attempt-b", "content_digest": "c" * 64},
            "profile_id": "lab",
            "profile_revision": 4,
            "budget_reservation": 0.5,
        }
    )

    assert _key() == other


def test_content_fields_change_the_key() -> None:
    assert _key() != _key(recipe_digest="x" * 64)
    assert _key() != _key(stage_input={"rows": 2000})
    assert _key() != _key(manifest_digest="n" * 64)
    assert _key() != _key(
        upstream_outputs=(("full_training", "final/adapter.safetensors", "a" * 64),)
    )
    assert _key() != _key(
        executor_config={"profile_digest": "q" * 64, "expected_script_sha256": "s" * 64}
    )


def test_reuse_policy_excludes_training_and_opt_in_fake() -> None:
    assert reuse_enabled(stage=StageKind.DATA_BUILD, executor_kind="ssh_remote", runtime={})
    assert reuse_enabled(
        stage=StageKind.DEVELOPMENT_EVALUATION, executor_kind="development_evaluation", runtime={}
    )
    assert not reuse_enabled(stage=StageKind.FULL_TRAINING, executor_kind="ssh_remote", runtime={})
    assert not reuse_enabled(
        stage=StageKind.CONTRACT_EVALUATION, executor_kind="ssh_remote", runtime={}
    )
    assert not reuse_enabled(stage=StageKind.DATA_BUILD, executor_kind="fake", runtime={})
    assert reuse_enabled(
        stage=StageKind.DATA_BUILD, executor_kind="fake", runtime={"memoize": True}
    )


def test_result_key_does_not_change_action_key() -> None:
    fields = dict(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        stage_index=0,
        stage=StageKind.DATA_BUILD,
        input_contract={"stage_input": {}},
        candidate_digest="c" * 64,
        manifest_revision=1,
        budget_unit="gpu_hours",
        budget_reservation=0.25,
    )

    plain = ActionSpec(**fields)
    keyed = ActionSpec(**fields, result_key="d" * 64)

    assert plain.action_key == keyed.action_key
    assert plain.input_digest == keyed.input_digest
