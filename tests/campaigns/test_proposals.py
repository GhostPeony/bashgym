"""Durable proposal, controller selection, and bounded evidence tests."""

import json
import sqlite3

import pytest

from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.contracts import (
    AutonomyProfile,
    CampaignTrigger,
    Capability,
    CredentialKind,
    ProposalStatus,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyProposalSubmission,
    StudyStatus,
)
from bashgym.campaigns.executor_adapters import build_default_registry
from bashgym.campaigns.persistence import (
    CampaignRepository,
    IdempotencyConflictError,
    InvalidProposalTransitionError,
)
from bashgym.campaigns.proposals import validate_proposal_submission
from bashgym.campaigns.service import CampaignControllerService, CampaignService
from bashgym.research.acquisition import (
    CompetingHypothesis,
    ExperimentAcquisition,
    PredictedOutcome,
    ResearchContextBundle,
    ResearchContextSource,
)
from tests.campaigns.test_persistence import campaign, create, manifest


def principal(repository, profile=AutonomyProfile.CODEX_TRUSTED):
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id=("hermes-agent" if profile == AutonomyProfile.HERMES_BOUNDED else "codex-agent"),
        autonomy_profile=profile,
        workspace_ids=("workspace-a",),
    )
    return auth.authenticate_access(auth.exchange_refresh(refresh.raw_token).raw_token)


def activate(repository: CampaignRepository) -> None:
    create(repository)
    version = 1
    for trigger, key in (
        (CampaignTrigger.VALIDATE, "validate"),
        (CampaignTrigger.VALIDATION_PASSED, "ready"),
        (CampaignTrigger.START, "start"),
    ):
        result = repository.transition_campaign(
            "workspace-a",
            "campaign-1",
            trigger,
            expected_version=version,
            actor_id="campaign-controller",
            credential_kind=CredentialKind.CONTROLLER,
            correlation_id=key,
            idempotency_key=key,
        )
        version = result.campaign.version


def proposal(
    proposal_id: str,
    *,
    priority: int = 50,
    estimated_cost: float = 1.0,
    recipe_schema: bool = True,
) -> StudyProposalSubmission:
    recipe = {
        "data_scope_id": "memexai-approved-training",
        **({"schema_version": "recipe.v1"} if recipe_schema else {}),
    }
    return StudyProposalSubmission(
        proposal_id=proposal_id,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        hypothesis=f"Hypothesis for {proposal_id}",
        study_family="embedding-retrieval",
        primary_variable="learning_rate",
        controlled_variables=("batch_size",),
        expected_outcome="Improve development MRR.",
        falsification_criterion="Reject if development MRR regresses.",
        estimated_cost=estimated_cost,
        priority=priority,
        dataset_recipe=recipe,
        training_recipe={"schema_version": "recipe.v1"},
        evaluation_recipe={"schema_version": "recipe.v1"},
        stage_plan=StagePlan(
            items=(
                StagePlanItem(
                    stage=StageKind.DEVELOPMENT_EVALUATION,
                    disposition=StageDisposition.REQUIRED,
                    reason="Development evidence is required.",
                ),
            )
        ),
        rationale="A bounded change with a falsifiable development gate.",
    )


def diagnostic_proposal(proposal_id: str) -> StudyProposalSubmission:
    return proposal(proposal_id, estimated_cost=0.05).model_copy(
        update={
            "primary_variable": "diagnostic.loss_landscape",
            "controlled_variables": (),
            "evaluation_recipe": {
                "schema_version": "bashgym.autoresearch_diagnostic_recipe.v1",
                "probe_family": "loss_landscape",
                "question": "Is the retained checkpoint already beyond the useful region?",
                "hypothesis": "Held-out loss rises after the retained checkpoint.",
                "informs_methods": ["sft", "data_redesign"],
                "measurements": [
                    {
                        "name": "heldout_loss_slope",
                        "interpretation": "minimize",
                        "unit": "loss_per_step",
                    }
                ],
                "sample_limit": 64,
                "seed": 17,
                "data_scope_ids": ["memexai-approved-training"],
                "parameters": {"checkpoint_steps": [20, 40, 80]},
                "runtime": {"executor_kind": "registered_compute"},
            },
            "required_capabilities": frozenset({Capability.EVAL_DEVELOPMENT}),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.CONTRACT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Measure the agent-authored diagnostic on approved data.",
                    ),
                )
            ),
        }
    )


def test_registered_diagnostic_accepts_agent_authored_probe_without_design_registry(repository):
    value = diagnostic_proposal("diagnostic-open-probe")
    campaign_value = repository.get_campaign("workspace-a", "campaign-1")
    manifest_value = repository.get_manifest_revision(
        "workspace-a", "campaign-1", campaign_value.manifest_revision
    ).manifest

    validation = validate_proposal_submission(
        value,
        manifest_value,
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert validation.valid is True
    assert validation.reason_codes == ()


@pytest.fixture
def repository(tmp_path):
    value = CampaignRepository(tmp_path / "campaigns.sqlite3")
    value.initialize()
    activate(value)
    return value


def submit(service, value, actor, version, key):
    return service.submit_proposal(
        value,
        expected_version=version,
        principal=actor,
        correlation_id=f"correlation-{key}",
        idempotency_key=key,
    )


def test_hermes_priority_is_normalized_and_replay_is_exact(repository):
    service = CampaignService(repository)
    actor = principal(repository, AutonomyProfile.HERMES_BOUNDED)
    first = submit(service, proposal("proposal-1", priority=99), actor, 4, "submit-1")
    replay = submit(service, proposal("proposal-1", priority=99), actor, 4, "submit-1")

    assert first.record.proposal.priority == 50
    assert first.record.proposal.planner_actor_id == "hermes-agent"
    assert replay.replayed is True
    assert replay.record == first.record
    with pytest.raises(IdempotencyConflictError):
        submit(service, proposal("proposal-2"), actor, 4, "submit-1")


def test_invalid_proposal_is_rejected_without_creating_study(repository):
    service = CampaignService(repository)
    result = submit(
        service,
        proposal("proposal-invalid", recipe_schema=False),
        principal(repository),
        4,
        "invalid",
    )

    assert result.record.proposal.status == ProposalStatus.REJECTED
    assert result.record.validation.reason_codes == ("proposal_recipe_schema_missing",)
    assert result.event.event_type == "campaign:proposal-rejected"
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_studies").fetchone()[0] == 0


def test_research_context_and_acquisition_bind_to_exact_proposal(repository):
    base = proposal("proposal-cited")
    research_context = ResearchContextBundle(
        workspace_id=base.workspace_id,
        campaign_id=base.campaign_id,
        proposal_id=base.proposal_id,
        query="information gain experiment design",
        categories=("research",),
        status="available",
        sources=(
            ResearchContextSource(
                title="Model Discovery Agent",
                url="https://arxiv.org/abs/2608.09696",
                source_type="research",
            ),
        ),
    )
    acquisition = ExperimentAcquisition(
        workspace_id=base.workspace_id,
        campaign_id=base.campaign_id,
        proposal_id=base.proposal_id,
        selection_mode="information_gain",
        hypotheses=(
            CompetingHypothesis(
                hypothesis_id="h1", statement="Optimization", prior_probability=0.5
            ),
            CompetingHypothesis(hypothesis_id="h2", statement="Data", prior_probability=0.5),
        ),
        outcomes=(
            PredictedOutcome(outcome_id="up", label="Improves"),
            PredictedOutcome(outcome_id="flat", label="Flat"),
        ),
        conditional_outcome_probabilities={
            "h1": {"up": 0.8, "flat": 0.2},
            "h2": {"up": 0.2, "flat": 0.8},
        },
        expected_cost=1.0,
    )
    cited = base.model_copy(
        update={"research_context": research_context, "acquisition": acquisition}
    )
    valid = validate_proposal_submission(
        cited,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )
    assert valid.valid is True
    persisted = submit(
        CampaignService(repository),
        cited,
        principal(repository),
        4,
        "cited-proposal",
    )
    assert persisted.record.proposal.research_context == research_context
    assert persisted.record.proposal.acquisition == acquisition

    wrong_context = research_context.model_copy(update={"proposal_id": "proposal-other"})
    invalid = validate_proposal_submission(
        cited.model_copy(update={"research_context": wrong_context}),
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )
    assert invalid.reason_codes == ("proposal_research_context_binding_mismatch",)


def _live_training_proposal(proposal_id: str) -> StudyProposalSubmission:
    return proposal(proposal_id).model_copy(
        update={
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
            },
            "evaluation_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.SMOKE_TRAINING,
                        disposition=StageDisposition.NOT_APPLICABLE,
                        reason="Optionally prove the pinned recipe before the iteration.",
                    ),
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Run the pinned recipe inside the approved budget.",
                    ),
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Compare the candidate on the fixed development evaluation.",
                    ),
                )
            ),
        }
    )


def test_live_training_requires_declared_compute_capabilities(repository):
    service = CampaignService(repository)
    result = submit(
        service,
        _live_training_proposal("proposal-live-missing-caps"),
        principal(repository),
        4,
        "live-missing-caps",
    )

    assert result.record.validation.reason_codes == (
        "proposal_compute_training_capability_missing",
        "proposal_development_evaluation_capability_missing",
    )


def test_registered_data_build_recipe_is_typed_and_requires_data_capability(repository):
    value = proposal("proposal-data-build").model_copy(
        update={
            "dataset_recipe": {
                "schema_version": "bashgym.terminal_data_recipe.v1",
                "data_scope_id": "memexai-approved-training",
                "runtime": {"executor_kind": "registered_compute"},
                "script_args": ["--pipeline", "terminal_env_generation", "--rows", "64"],
            },
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
            },
            "evaluation_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "required_capabilities": frozenset(
                {
                    Capability.DATA_BUILD,
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.DATA_BUILD,
                        disposition=StageDisposition.REQUIRED,
                        reason="Generate and validate one remote-resident training dataset.",
                    ),
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Train against the exact generated dataset.",
                    ),
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Evaluate the candidate on the fixed held-out tasks.",
                    ),
                )
            ),
        }
    )

    valid = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )
    missing_capability = validate_proposal_submission(
        value.model_copy(
            update={
                "required_capabilities": frozenset(
                    {
                        Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                        Capability.EVAL_DEVELOPMENT,
                    }
                )
            }
        ),
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert valid.valid is True
    assert missing_capability.reason_codes == ("proposal_data_build_capability_missing",)


def test_autoresearch_data_design_recipe_rejects_actor_execution_material(repository):
    value = proposal("proposal-data-design").model_copy(
        update={
            "dataset_recipe": {
                "schema_version": "bashgym.autoresearch_data_design_recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
                "hypothesis": "Target stateful debugging failures.",
                "pipeline": "terminal_env_generation",
                "generation_brief": "Generate stateful debugging and recovery tasks.",
                "target_rows": 64,
                "train_fraction": 0.8,
                "seed": 17,
            },
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
            },
            "evaluation_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "required_capabilities": frozenset(
                {
                    Capability.DATA_BUILD,
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.DATA_BUILD,
                        disposition=StageDisposition.REQUIRED,
                        reason="Build the selected approved dataset design.",
                    ),
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Train with fixed settings.",
                    ),
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Evaluate on the fixed suite.",
                    ),
                )
            ),
        }
    )

    valid = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )
    invalid = validate_proposal_submission(
        value.model_copy(
            update={
                "dataset_recipe": {
                    **value.dataset_recipe,
                    "provider_endpoint": "http://unreviewed.invalid/v1",
                }
            }
        ),
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert valid.valid is True
    assert invalid.reason_codes == ("proposal_data_design_recipe_invalid",)


def test_tmax_training_recipe_rejects_unavailable_dppo_backend(repository):
    value = _live_training_proposal("proposal-tmax-dppo").model_copy(
        update={
            "training_recipe": {
                "schema_version": "bashgym.tmax_composite_training_recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
                "algorithm": "dppo",
            },
            "required_capabilities": frozenset(
                {
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
        }
    )

    result = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert result.reason_codes == ("proposal_tmax_training_recipe_invalid",)


def test_external_handoff_uses_generic_opt_in_and_keeps_legacy_capability_read_only(
    repository,
):
    actor = principal(repository)
    generic = proposal("proposal-generic-handoff").model_copy(
        update={"required_capabilities": frozenset({Capability.HANDOFF_EXTERNAL_PREPARE})}
    )

    denied = validate_proposal_submission(
        generic,
        manifest(),
        actor,
        existing_prerequisite_ids=frozenset(),
    )
    assert denied.reason_codes == ("proposal_external_handoff_not_approved",)

    allowed = validate_proposal_submission(
        generic,
        manifest().model_copy(update={"allow_external_handoff": True}),
        actor,
        existing_prerequisite_ids=frozenset(),
    )
    assert allowed.valid is True

    legacy_actor = actor.model_copy(
        update={"capabilities": actor.capabilities | {Capability.HANDOFF_MEMEXAI_PREPARE}}
    )
    legacy = generic.model_copy(
        update={
            "proposal_id": "proposal-legacy-handoff",
            "required_capabilities": frozenset({Capability.HANDOFF_MEMEXAI_PREPARE}),
        }
    )
    legacy_result = validate_proposal_submission(
        legacy,
        manifest().model_copy(update={"allow_memexai_handoff": True}),
        legacy_actor,
        existing_prerequisite_ids=frozenset(),
    )
    assert legacy_result.reason_codes == ("proposal_legacy_handoff_read_only",)


def test_live_training_rejects_actor_supplied_execution_material(repository):
    service = CampaignService(repository)
    value = _live_training_proposal("proposal-live-command").model_copy(
        update={
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {
                    "executor_kind": "registered_training",
                    "script_path": "/tmp/actor.py",
                },
            },
            "required_capabilities": frozenset(
                {
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
        }
    )
    result = submit(service, value, principal(repository), 4, "live-command")

    assert result.record.validation.reason_codes == (
        "proposal_executable_material_forbidden",
        "proposal_runtime_keys_not_allowed",
    )


def test_live_training_accepts_mode_only_with_declared_capabilities(repository):
    service = CampaignService(repository)
    value = _live_training_proposal("proposal-live-valid").model_copy(
        update={
            "required_capabilities": frozenset(
                {
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            )
        }
    )
    result = submit(service, value, principal(repository), 4, "live-valid")

    assert result.record.validation.valid is True
    assert result.record.proposal.status == ProposalStatus.SUBMITTED


def test_registered_compute_evaluation_requires_declared_capability(repository):
    service = CampaignService(repository)
    value = proposal("proposal-live-evaluation").model_copy(
        update={
            "evaluation_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Evaluate the immutable base on approved private compute.",
                    ),
                )
            ),
        }
    )
    missing = submit(service, value, principal(repository), 4, "live-eval-missing")
    assert missing.record.validation.reason_codes == (
        "proposal_development_evaluation_capability_missing",
    )

    accepted = submit(
        service,
        value.model_copy(
            update={
                "proposal_id": "proposal-live-evaluation-valid",
                "required_capabilities": frozenset({Capability.EVAL_DEVELOPMENT}),
            }
        ),
        principal(repository),
        missing.campaign.version,
        "live-eval-valid",
    )
    assert accepted.record.validation.valid is True


@pytest.mark.parametrize(
    ("items", "required_capabilities"),
    (
        (
            (
                StagePlanItem(
                    stage=StageKind.FULL_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="A candidate cannot skip its fixed evaluation.",
                ),
            ),
            frozenset(
                {
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
        ),
        (
            (
                StagePlanItem(
                    stage=StageKind.SMOKE_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="Smoke must not be required every iteration.",
                ),
                StagePlanItem(
                    stage=StageKind.FULL_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="Train the candidate.",
                ),
                StagePlanItem(
                    stage=StageKind.DEVELOPMENT_EVALUATION,
                    disposition=StageDisposition.REQUIRED,
                    reason="Evaluate the candidate.",
                ),
            ),
            frozenset(
                {
                    Capability.COMPUTE_SMOKE,
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
        ),
    ),
)
def test_live_proposal_rejects_non_autoresearch_required_stage_shape(
    repository,
    items,
    required_capabilities,
):
    value = _live_training_proposal("proposal-invalid-stage-shape").model_copy(
        update={
            "required_capabilities": required_capabilities,
            "stage_plan": StagePlan(items=items),
        }
    )

    result = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert result.reason_codes == ("proposal_autoresearch_stage_plan_invalid",)


def test_live_baseline_can_carry_registered_training_configuration(repository):
    value = _live_training_proposal("proposal-baseline-with-training-config").model_copy(
        update={
            "required_capabilities": frozenset({Capability.EVAL_DEVELOPMENT}),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Evaluate the immutable base without training it.",
                    ),
                )
            ),
        }
    )

    result = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert result.valid is True


@pytest.mark.parametrize(
    ("recipe_update", "items", "required_capabilities"),
    (
        (
            {
                "dataset_recipe": {
                    "schema_version": "recipe.v1",
                    "data_scope_id": "memexai-approved-training",
                }
            },
            (
                StagePlanItem(
                    stage=StageKind.DATA_BUILD,
                    disposition=StageDisposition.REQUIRED,
                    reason="Build the candidate dataset on registered compute.",
                ),
                StagePlanItem(
                    stage=StageKind.FULL_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="Train the candidate on registered compute.",
                ),
                StagePlanItem(
                    stage=StageKind.DEVELOPMENT_EVALUATION,
                    disposition=StageDisposition.REQUIRED,
                    reason="Evaluate the candidate on registered compute.",
                ),
            ),
            frozenset(
                {
                    Capability.DATA_BUILD,
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
        ),
        (
            {
                "training_recipe": {
                    "schema_version": "recipe.v1",
                    "runtime": {"executor_kind": "fake"},
                }
            },
            (
                StagePlanItem(
                    stage=StageKind.FULL_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="Train the candidate on registered compute.",
                ),
                StagePlanItem(
                    stage=StageKind.DEVELOPMENT_EVALUATION,
                    disposition=StageDisposition.REQUIRED,
                    reason="Evaluate the candidate on registered compute.",
                ),
            ),
            frozenset(
                {
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
        ),
        (
            {
                "evaluation_recipe": {
                    "schema_version": "recipe.v1",
                    "runtime": {"executor_kind": "fake"},
                }
            },
            (
                StagePlanItem(
                    stage=StageKind.DEVELOPMENT_EVALUATION,
                    disposition=StageDisposition.REQUIRED,
                    reason="Evaluate the immutable base on registered compute.",
                ),
            ),
            frozenset({Capability.EVAL_DEVELOPMENT}),
        ),
    ),
)
def test_live_proposal_requires_registered_runtime_for_each_required_stage(
    repository,
    recipe_update,
    items,
    required_capabilities,
):
    value = _live_training_proposal("proposal-required-stage-runtime").model_copy(
        update={
            **recipe_update,
            "required_capabilities": required_capabilities,
            "stage_plan": StagePlan(items=items),
        }
    )

    result = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert result.reason_codes == ("proposal_required_stage_runtime_not_registered",)


def test_live_baseline_rejects_extra_required_stage(repository):
    value = proposal("proposal-invalid-baseline-shape").model_copy(
        update={
            "evaluation_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "required_capabilities": frozenset({Capability.EVAL_DEVELOPMENT}),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.SMOKE_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="A baseline cannot require smoke training.",
                    ),
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Evaluate the immutable base.",
                    ),
                )
            ),
        }
    )

    result = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert result.reason_codes == ("proposal_autoresearch_stage_plan_invalid",)


def test_live_data_build_requires_exact_candidate_stage_order(repository):
    value = _live_training_proposal("proposal-invalid-data-order").model_copy(
        update={
            "dataset_recipe": {
                "schema_version": "recipe.v1",
                "data_scope_id": "memexai-approved-training",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "required_capabilities": frozenset(
                {
                    Capability.DATA_BUILD,
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Incorrectly train before building data.",
                    ),
                    StagePlanItem(
                        stage=StageKind.DATA_BUILD,
                        disposition=StageDisposition.REQUIRED,
                        reason="Incorrectly build data after training.",
                    ),
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Evaluate the candidate.",
                    ),
                )
            ),
        }
    )

    result = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert result.reason_codes == ("proposal_autoresearch_stage_plan_invalid",)


def test_fake_proposal_keeps_generic_stage_plan_behavior(repository):
    value = proposal("proposal-fake-generic").model_copy(
        update={
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.SMOKE_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Generic fake proposal shape remains unconstrained.",
                    ),
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Generic fake proposal shape remains unconstrained.",
                    ),
                )
            )
        }
    )

    result = validate_proposal_submission(
        value,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert result.valid is True


def test_withdraw_requires_submitted_status_and_expected_version(repository):
    service = CampaignService(repository)
    actor = principal(repository)
    submitted = submit(service, proposal("proposal-1"), actor, 4, "submit")
    withdrawn = service.withdraw_proposal(
        "workspace-a",
        "campaign-1",
        "proposal-1",
        expected_version=submitted.campaign.version,
        principal=actor,
        correlation_id="withdraw",
        idempotency_key="withdraw",
    )
    replay = service.withdraw_proposal(
        "workspace-a",
        "campaign-1",
        "proposal-1",
        expected_version=submitted.campaign.version,
        principal=actor,
        correlation_id="withdraw",
        idempotency_key="withdraw",
    )

    assert withdrawn.record.proposal.status == ProposalStatus.WITHDRAWN
    assert replay.replayed is True
    with pytest.raises(InvalidProposalTransitionError):
        service.withdraw_proposal(
            "workspace-a",
            "campaign-1",
            "proposal-1",
            expected_version=withdrawn.campaign.version,
            principal=actor,
            correlation_id="withdraw-again",
            idempotency_key="withdraw-again",
        )


def test_controller_selection_is_deterministic_and_creates_exactly_one_study(repository):
    service = CampaignService(repository)
    actor = principal(repository)
    first = submit(service, proposal("proposal-low", priority=20), actor, 4, "low")
    second = submit(
        service,
        proposal("proposal-costly", priority=90, estimated_cost=3),
        actor,
        first.campaign.version,
        "costly",
    )
    third = submit(
        service,
        proposal("proposal-cheap", priority=90, estimated_cost=1),
        actor,
        second.campaign.version,
        "cheap",
    )
    controller = CampaignControllerService(repository, controller_id="campaign-controller")
    selected = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=third.campaign.version,
        correlation_id="select",
        idempotency_key="select",
    )
    replay = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=third.campaign.version,
        correlation_id="select",
        idempotency_key="select",
    )

    assert selected is not None
    assert selected.record.proposal.proposal_id == "proposal-cheap"
    assert selected.record.proposal.status == ProposalStatus.ACCEPTED
    assert selected.study.status.value == "validated"
    assert replay == selected.__class__(
        selected.campaign,
        selected.event,
        selected.record,
        selected.study,
        replayed=True,
    )
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_studies").fetchone()[0] == 1


def test_controller_skips_higher_priority_proposal_with_unready_prerequisite(repository):
    service = CampaignService(repository)
    actor = principal(repository)
    initial = submit(service, proposal("proposal-prior"), actor, 4, "prior")
    controller = CampaignControllerService(repository, controller_id="campaign-controller")
    prior = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=initial.campaign.version,
        correlation_id="select-prior",
        idempotency_key="select-prior",
    )
    assert prior is not None
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaigns SET active_study_id = NULL
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            ("workspace-a", "campaign-1"),
        )
    blocked_submission = proposal("proposal-blocked", priority=100).model_copy(
        update={"prerequisite_study_ids": (prior.study.study_id,)}
    )
    blocked = submit(
        service,
        blocked_submission,
        actor,
        prior.campaign.version,
        "blocked",
    )
    ready = submit(
        service,
        proposal("proposal-ready", priority=50),
        actor,
        blocked.campaign.version,
        "ready-independent",
    )
    selected = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=ready.campaign.version,
        correlation_id="select-ready",
        idempotency_key="select-ready",
    )

    assert selected is not None
    assert selected.record.proposal.proposal_id == "proposal-ready"

    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_studies SET status = ?
            WHERE workspace_id = ? AND study_id = ?
            """,
            (StudyStatus.COMPLETED.value, "workspace-a", prior.study.study_id),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_study_id = NULL
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            ("workspace-a", "campaign-1"),
        )
    unblocked = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=selected.campaign.version,
        correlation_id="select-unblocked",
        idempotency_key="select-unblocked",
    )
    assert unblocked is not None
    assert unblocked.record.proposal.proposal_id == "proposal-blocked"


def test_advance_request_never_accepts_proposal(repository):
    service = CampaignService(repository)
    actor = principal(repository)
    submitted = submit(service, proposal("proposal-1"), actor, 4, "submit")
    advanced = service.request_advance(
        "workspace-a",
        "campaign-1",
        expected_version=submitted.campaign.version,
        principal=actor,
        correlation_id="advance",
        idempotency_key="advance",
    )

    assert advanced.event.event_type == "campaign:advance-requested"
    assert advanced.campaign.active_study_id is None
    assert service.proposals("workspace-a", "campaign-1", actor)[0].proposal.status == (
        ProposalStatus.SUBMITTED
    )
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_studies").fetchone()[0] == 0


def test_evidence_snapshot_is_bounded_and_excludes_rows_and_uris(repository, tmp_path):
    service = CampaignService(repository)
    actor = principal(repository)
    rejected = submit(
        service,
        proposal("proposal-invalid", recipe_schema=False),
        actor,
        4,
        "reject",
    )
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_evaluations(
                workspace_id, campaign_id, evaluation_id, evaluation_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "protected-eval",
                json.dumps({"protected_row": "NEVER_SURFACE_THIS"}),
                campaign().created_at.isoformat(),
            ),
        )
        nemo_reference = {
            "artifact_id": "artifact-nemo",
            "artifact_sha256": "b" * 64,
            "bundle_digest": "c" * 64,
            "environment_id": "star-count-v1",
            "environment_digest": "d" * 64,
            "rollout_batch_digest": "e" * 64,
            "token_evidence_digest": "f" * 64,
            "refit_receipt_digest": "1" * 64,
            "rollout_count": 2,
            "mean_total_reward": 0.75,
            "training_step": 4,
            "policy_revision": 4,
        }
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id, uri,
                sha256, size_bytes, schema_name, sealed, valid, metadata_json, created_at
            ) VALUES (?, ?, ?, NULL, ?, ?, 10, ?, 1, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "artifact-nemo",
                str(tmp_path / "private" / "nemo_gym_campaign_evidence.json"),
                "b" * 64,
                "nemo_gym_campaign_evidence.v1",
                json.dumps(
                    {
                        "nemo_gym": nemo_reference,
                        "raw_rollout": "NEVER_SURFACE_THIS",
                    }
                ),
                campaign().created_at.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id, uri,
                sha256, size_bytes, schema_name, sealed, valid, metadata_json, created_at
            ) VALUES (?, ?, ?, NULL, ?, ?, 10, ?, 1, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "artifact-safe",
                str(tmp_path / "private" / "model.bin"),
                "a" * 64,
                "model.v1",
                json.dumps({"private_excerpt": "NEVER_SURFACE_THIS"}),
                campaign().created_at.isoformat(),
            ),
        )

    snapshot = service.evidence("workspace-a", "campaign-1", actor)
    serialized = snapshot.model_dump_json()
    assert snapshot.campaign_version == rejected.campaign.version
    assert snapshot.proposal_counts[ProposalStatus.REJECTED] == 1
    assert snapshot.artifact_references[0].artifact_id == "artifact-safe"
    assert snapshot.nemo_gym_evidence_references[0].artifact_id == "artifact-nemo"
    assert snapshot.nemo_gym_evidence_references[0].rollout_count == 2
    assert snapshot.available_executors == build_default_registry().kinds()
    assert "NEVER_SURFACE_THIS" not in serialized
    assert "model.bin" not in serialized
    assert "uri" not in serialized.casefold()
    assert service.evidence("workspace-a", "campaign-1", actor).snapshot_digest == (
        snapshot.snapshot_digest
    )


def test_credential_shaped_values_and_placeholders_are_rejected(repository) -> None:
    leaked = proposal("proposal-leak").model_copy(
        update={
            "rationale": "Fetch the corpus with ghp_" + "a" * 36 + " before training.",
            "training_recipe": {
                "schema_version": "recipe.v1",
                "hub_token_name": "<ASK_USER: which secret holds the token>",
            },
        }
    )

    validation = validate_proposal_submission(
        leaked, manifest(), principal(repository), existing_prerequisite_ids=frozenset()
    )

    assert validation.valid is False
    assert "proposal_credential_shaped_value" in validation.reason_codes
    assert "proposal_unresolved_placeholder" in validation.reason_codes


def test_deeply_nested_recipe_is_reported_unscannable(repository) -> None:
    nested: dict = {"schema_version": "recipe.v1"}
    for _ in range(40):
        nested = {"schema_version": "recipe.v1", "child": nested}

    deep = proposal("proposal-deep-recipe").model_copy(update={"training_recipe": nested})

    validation = validate_proposal_submission(
        deep, manifest(), principal(repository), existing_prerequisite_ids=frozenset()
    )

    assert "proposal_content_unscannable" in validation.reason_codes


def test_clean_proposal_has_no_scan_reasons(repository) -> None:
    validation = validate_proposal_submission(
        proposal("proposal-clean"),
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )

    assert not {
        "proposal_credential_shaped_value",
        "proposal_unresolved_placeholder",
        "proposal_content_unscannable",
    } & set(validation.reason_codes)
