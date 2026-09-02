"""Clone a persisted proposal into a new submission with an explicit diff."""

import pytest

from bashgym.campaigns.clone_study import (
    CloneStudyError,
    clone_diff,
    clone_proposal_submission,
)
from bashgym.campaigns.contracts import Capability, StudyProposalSubmission
from tests.campaigns.test_decision_packet import _proposal


def test_clone_copies_scientific_fields_and_applies_changes() -> None:
    source = _proposal()

    submission = clone_proposal_submission(
        source,
        proposal_id="candidate-2",
        changes={"training_recipe": {"learning_rate": 0.0002, "seed": 23}},
    )

    assert isinstance(submission, StudyProposalSubmission)
    assert submission.proposal_id == "candidate-2"
    assert submission.workspace_id == source.workspace_id
    assert submission.campaign_id == source.campaign_id
    assert submission.hypothesis == source.hypothesis
    assert submission.stage_plan == source.stage_plan
    assert submission.training_recipe == {"learning_rate": 0.0002, "seed": 23}
    assert clone_diff(source, submission) == {
        "training_recipe": {
            "from": {"learning_rate": 0.0001},
            "to": {"learning_rate": 0.0002, "seed": 23},
        }
    }


def test_clone_without_changes_is_a_verbatim_copy_with_an_empty_diff() -> None:
    source = _proposal()

    submission = clone_proposal_submission(source, proposal_id="candidate-2", changes={})

    assert clone_diff(source, submission) == {}
    assert submission.dataset_recipe == source.dataset_recipe


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"planner_actor_id": "someone"}, "clone_change_not_allowed"),
        ({"workspace_id": "workspace-b"}, "clone_change_not_allowed"),
        ({"unknown_field": 1}, "clone_change_not_allowed"),
    ],
)
def test_server_owned_and_unknown_changes_are_rejected(changes, code) -> None:
    with pytest.raises(CloneStudyError) as excinfo:
        clone_proposal_submission(_proposal(), proposal_id="candidate-2", changes=changes)

    assert excinfo.value.code == code


def test_clone_must_use_a_new_proposal_id() -> None:
    with pytest.raises(CloneStudyError) as excinfo:
        clone_proposal_submission(_proposal(), proposal_id="candidate-1", changes={})

    assert excinfo.value.code == "clone_proposal_id_reused"


def test_clone_with_non_empty_required_capabilities() -> None:
    source = _proposal().model_copy(
        update={
            "required_capabilities": frozenset(
                {
                    Capability.CAMPAIGN_READ,
                    Capability.STUDY_PROPOSE,
                    Capability.DATA_BUILD,
                }
            )
        }
    )

    submission = clone_proposal_submission(source, proposal_id="candidate-2", changes={})

    assert clone_diff(source, submission) == {}

    submission_with_change = clone_proposal_submission(
        source,
        proposal_id="candidate-3",
        changes={"hypothesis": "new hypothesis"},
    )

    diff = clone_diff(source, submission_with_change)
    assert set(diff.keys()) == {"hypothesis"}
