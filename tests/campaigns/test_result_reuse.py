"""Content-only result keys and the reuse policy."""

from bashgym.campaigns.contracts import CampaignManifest, StageKind
from bashgym.campaigns.executor_registry import ExecutorRegistry
from bashgym.campaigns.result_reuse import (
    manifest_content_digest,
    reuse_enabled,
    stage_result_key,
)
from bashgym.campaigns.runtime import ActionSpec


def _key(**overrides):
    base = dict(
        stage=StageKind.DATA_BUILD,
        executor_kind="ssh_remote",
        manifest_content_digest="m" * 64,
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
    assert _key() != _key(manifest_content_digest="n" * 64)
    assert _key() != _key(
        upstream_outputs=(("full_training", "final/adapter.safetensors", "a" * 64),)
    )
    assert _key() != _key(
        executor_config={"profile_digest": "q" * 64, "expected_script_sha256": "s" * 64}
    )


def test_reuse_policy_excludes_training_and_opt_in_fake() -> None:
    assert reuse_enabled(stage=StageKind.DATA_BUILD, executor_kind="ssh_remote", runtime={})
    assert not reuse_enabled(stage=StageKind.FULL_TRAINING, executor_kind="ssh_remote", runtime={})
    assert not reuse_enabled(
        stage=StageKind.CONTRACT_EVALUATION, executor_kind="ssh_remote", runtime={}
    )
    assert not reuse_enabled(
        stage=StageKind.DEVELOPMENT_EVALUATION, executor_kind="development_evaluation", runtime={}
    )
    assert not reuse_enabled(
        stage=StageKind.DATA_BUILD, executor_kind="development_evaluation", runtime={}
    )
    assert not reuse_enabled(
        stage=StageKind.DATA_BUILD,
        executor_kind="development_evaluation",
        runtime={"memoize": True},
    )
    assert not reuse_enabled(stage=StageKind.DATA_BUILD, executor_kind="fake", runtime={})
    assert reuse_enabled(
        stage=StageKind.DATA_BUILD, executor_kind="fake", runtime={"memoize": True}
    )


class _PluginAdapter:
    """Third-party adapter double whose reuse capability is declared, not inferred."""

    def __init__(self, kind: str, *, reuses: bool) -> None:
        self.kind = kind
        self.reuses_completed_results = reuses
        self.allowed_stages = frozenset({StageKind.DATA_BUILD})

    def tick(self, worker, attempt, *, now):
        return "plugin_ticked"

    def reconcile(self, worker, attempt, *, now):
        return None

    def repair_allowed(self):
        return True


def _plugin_registry() -> ExecutorRegistry:
    registry = ExecutorRegistry()
    registry.register(_PluginAdapter("plugin_remote", reuses=True))
    registry.register(_PluginAdapter("plugin_local", reuses=False))
    registry.freeze()
    return registry


def test_reuse_policy_reads_the_declared_capability_of_any_registered_kind() -> None:
    registry = _plugin_registry()

    assert reuse_enabled(
        stage=StageKind.DATA_BUILD,
        executor_kind="plugin_remote",
        runtime={},
        registry=registry,
    )
    assert not reuse_enabled(
        stage=StageKind.DATA_BUILD,
        executor_kind="plugin_local",
        runtime={},
        registry=registry,
    )
    assert reuse_enabled(
        stage=StageKind.DATA_BUILD,
        executor_kind="plugin_local",
        runtime={"memoize": True},
        registry=registry,
    )
    assert not reuse_enabled(
        stage=StageKind.FULL_TRAINING,
        executor_kind="plugin_remote",
        runtime={},
        registry=registry,
    )


def test_reuse_policy_refuses_an_unregistered_kind() -> None:
    registry = _plugin_registry()

    assert not reuse_enabled(
        stage=StageKind.DATA_BUILD,
        executor_kind="mystery",
        runtime={"memoize": True},
        registry=registry,
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


def _manifest(**overrides) -> CampaignManifest:
    base = dict(
        approved_data_scopes=("memexai-approved-training",),
        compute_profile_id="ssh-gpu-lab",
        budget_limits={"gpu_hours": 12.0},
        evaluation_plan={"development_query_set": "dev-18-v1"},
        promotion_gates={"mrr_at_10_delta_min": 0.0},
    )
    base.update(overrides)
    return CampaignManifest(**base)


def test_manifest_projection_covers_only_what_a_stage_launch_reads() -> None:
    """Budgets, evaluation plan, gates, and retention never reach a stage script."""

    assert manifest_content_digest(_manifest()) == manifest_content_digest(
        _manifest(
            budget_limits={"gpu_hours": 400.0, "study_count": 9.0},
            evaluation_plan={"development_query_set": "dev-99-v9"},
            promotion_gates={"mrr_at_10_delta_min": 0.5},
            protected_artifact_refs=("frozen-test-36-v1",),
            max_proposal_rounds=9,
            retention_days_failed=7,
            allow_hf_publication=True,
            allow_external_handoff=True,
            allow_memexai_handoff=True,
        )
    )


def test_manifest_projection_covers_the_scope_and_compute_boundary() -> None:
    assert manifest_content_digest(_manifest()) != manifest_content_digest(
        _manifest(approved_data_scopes=("desktop-local",))
    )
    assert manifest_content_digest(_manifest()) != manifest_content_digest(
        _manifest(compute_profile_id="ssh-gpu-other")
    )
