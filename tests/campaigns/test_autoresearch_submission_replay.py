"""Lost submission responses must replay before changing loop readiness checks."""

import pytest

from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignCore,
    AutoResearchConflictError,
    AutoResearchInvariantError,
    AutoResearchNextAction,
    AutoResearchRepository,
    ExperimentRole,
    ResultDecision,
)
from bashgym.campaigns.persistence import IdempotencyConflictError
from tests.campaigns.test_autoresearch_campaign import (
    _authoritative_outcome,
    _insert_authoritative_outcome,
    _recipe_proposal,
    activate,
    fresh_core,
    select_and_finish,
)
from tests.campaigns.test_proposals import principal


@pytest.fixture(params=["baseline", "candidate"])
def submitted(tmp_path, request):
    path, repository, core = fresh_core(tmp_path, target=None)
    activate(core)
    actor = principal(repository)
    submission = _recipe_proposal("baseline", learning_rate=0.001, seed=17)
    control = {}
    method = "submit_baseline"
    if request.param == "candidate":
        core.submit_baseline(
            submission,
            expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
            principal=actor,
            correlation_id="baseline",
            idempotency_key="baseline",
        )
        study, attempt = select_and_finish(repository, "baseline")
        _insert_authoritative_outcome(
            repository,
            _authoritative_outcome(
                "baseline",
                study,
                attempt,
                0.5,
                role=ExperimentRole.BASELINE,
                decision=ResultDecision.BASELINE,
                eligible_for_best=True,
            ),
        )
        submission = _recipe_proposal("candidate", learning_rate=0.002, seed=17).model_copy(
            update={
                "primary_variable": "training_recipe.learning_rate",
                "prerequisite_study_ids": (study,),
            }
        )
        control = {
            "parent_proposal_id": "baseline",
            "changed_variable": "training_recipe.learning_rate",
        }
        method = "submit_controlled_candidate"
    kwargs = {
        **control,
        "expected_version": repository.get_campaign("workspace-a", "campaign-1").version,
        "principal": actor,
        "correlation_id": "lost-response",
        "idempotency_key": "lost-response",
    }
    original = getattr(core, method)(submission, **kwargs)
    assert (
        core.state("workspace-a", "campaign-1").next_action
        == AutoResearchNextAction.WAIT_FOR_RESULT
    )
    reopened = AutoResearchRepository(path)
    reopened.initialize()
    return AutoResearchCampaignCore(reopened), submission, method, kwargs, original


def test_identical_submission_replays_after_restart_without_new_writes(submitted):
    core, submission, method, kwargs, original = submitted
    before = core.repository.get_campaign("workspace-a", "campaign-1")
    proposals = core.repository.list_proposals("workspace-a", "campaign-1")
    replay = getattr(core, method)(submission, **{**kwargs, "correlation_id": "retry"})
    assert replay.replayed is True
    assert replay.campaign == original.campaign
    assert replay.record == original.record
    assert replay.event == original.event
    assert core.repository.get_campaign("workspace-a", "campaign-1") == before
    assert core.repository.list_proposals("workspace-a", "campaign-1") == proposals


def test_duplicate_key_with_changed_submission_conflicts(submitted):
    core, submission, method, kwargs, _ = submitted
    changed = submission.model_copy(update={"hypothesis": "Conflicting retry"})
    with pytest.raises(IdempotencyConflictError):
        getattr(core, method)(changed, **kwargs)


def test_new_submission_still_requires_loop_readiness(submitted):
    core, submission, method, kwargs, _ = submitted
    changed = submission.model_copy(update={"proposal_id": "another-proposal"})
    with pytest.raises(AutoResearchInvariantError, match="autoresearch_proposal_not_ready"):
        getattr(core, method)(changed, **{**kwargs, "idempotency_key": "new-submission"})


def test_duplicate_cannot_change_control_metadata(submitted):
    core, submission, method, kwargs, _ = submitted
    if method == "submit_baseline":
        method = "submit_diagnostic"
        changed_kwargs = {**kwargs, "parent_proposal_id": "another-parent"}
    else:
        changed_kwargs = {**kwargs, "changed_variable": "training_recipe.seed"}
    with pytest.raises(AutoResearchConflictError, match="autoresearch_proposal_control_conflict"):
        getattr(core, method)(submission, **changed_kwargs)


def test_replay_still_requires_current_propose_capability(submitted):
    core, submission, method, kwargs, _ = submitted
    actor = kwargs["principal"].model_copy(update={"capabilities": frozenset()})
    with pytest.raises(PermissionError, match="campaign_capability_required"):
        getattr(core, method)(submission, **{**kwargs, "principal": actor})


def test_duplicate_cannot_change_expected_version(submitted):
    core, submission, method, kwargs, _ = submitted
    with pytest.raises(IdempotencyConflictError):
        getattr(core, method)(
            submission, **{**kwargs, "expected_version": kwargs["expected_version"] + 1}
        )
