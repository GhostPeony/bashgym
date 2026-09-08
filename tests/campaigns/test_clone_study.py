"""Clone a persisted proposal into a new submission with an explicit diff."""

import pytest

from bashgym.campaigns.clone_study import (
    CLONEABLE_FIELDS,
    CloneStudyError,
    clone_diff,
    clone_proposal_submission,
)
from bashgym.campaigns.contracts import (
    Capability,
    ProposalStatus,
    StudyProposal,
    StudyProposalSubmission,
)
from bashgym.campaigns.persistence import CampaignRepository
from bashgym.campaigns.proposals import validate_proposal_submission
from bashgym.research.acquisition import (
    CompetingHypothesis,
    ExperimentAcquisition,
    PredictedOutcome,
    ResearchContextBundle,
    ResearchContextSource,
)
from tests.campaigns.test_persistence import manifest
from tests.campaigns.test_proposals import activate, principal
from tests.campaigns.test_proposals import proposal as proposal_submission


def source_proposal(proposal_id: str = "candidate-1", **updates) -> StudyProposal:
    """A persisted proposal built from the shared submission helper."""

    submission = proposal_submission(proposal_id)
    value = StudyProposal(
        **submission.model_dump(exclude={"schema_version"}),
        planner_actor_id="codex-agent",
        status=ProposalStatus.ACCEPTED,
        creation_sequence=2,
    )
    return value.model_copy(update=updates) if updates else value


def research_context(proposal_id: str) -> ResearchContextBundle:
    return ResearchContextBundle(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        proposal_id=proposal_id,
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


def acquisition(proposal_id: str) -> ExperimentAcquisition:
    return ExperimentAcquisition(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        proposal_id=proposal_id,
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


@pytest.fixture
def repository(tmp_path) -> CampaignRepository:
    value = CampaignRepository(tmp_path / "campaigns.sqlite3")
    value.initialize()
    activate(value)
    return value


def test_clone_copies_scientific_fields_and_applies_changes() -> None:
    source = source_proposal()

    submission = clone_proposal_submission(
        source,
        proposal_id="candidate-2",
        changes={"training_recipe": {"schema_version": "recipe.v1", "seed": 23}},
    )

    assert isinstance(submission, StudyProposalSubmission)
    assert submission.proposal_id == "candidate-2"
    assert submission.workspace_id == source.workspace_id
    assert submission.campaign_id == source.campaign_id
    assert submission.hypothesis == source.hypothesis
    assert submission.stage_plan == source.stage_plan
    assert submission.training_recipe == {"schema_version": "recipe.v1", "seed": 23}
    assert clone_diff(source, submission) == {
        "training_recipe": {
            "from": {"schema_version": "recipe.v1"},
            "to": {"schema_version": "recipe.v1", "seed": 23},
        }
    }


def test_clone_without_changes_is_a_verbatim_copy_with_an_empty_diff() -> None:
    source = source_proposal()

    submission = clone_proposal_submission(source, proposal_id="candidate-2", changes={})

    assert clone_diff(source, submission) == {}
    assert submission.dataset_recipe == source.dataset_recipe


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"planner_actor_id": "someone"}, "clone_change_not_allowed"),
        ({"workspace_id": "workspace-b"}, "clone_change_not_allowed"),
        ({"unknown_field": 1}, "clone_change_not_allowed"),
        ({"research_context": None}, "clone_change_not_allowed"),
    ],
)
def test_server_owned_and_unknown_changes_are_rejected(changes, code) -> None:
    with pytest.raises(CloneStudyError) as excinfo:
        clone_proposal_submission(source_proposal(), proposal_id="candidate-2", changes=changes)

    assert excinfo.value.code == code


def test_clone_must_use_a_new_proposal_id() -> None:
    with pytest.raises(CloneStudyError) as excinfo:
        clone_proposal_submission(source_proposal(), proposal_id="candidate-1", changes={})

    assert excinfo.value.code == "clone_proposal_id_reused"


def test_recipe_change_merges_shallowly_and_keeps_other_keys() -> None:
    source = source_proposal(
        training_recipe={
            "schema_version": "recipe.v1",
            "learning_rate": 0.0001,
            "seed": 7,
            "max_steps": 500,
        }
    )

    submission = clone_proposal_submission(
        source, proposal_id="candidate-2", changes={"training_recipe": {"seed": 23}}
    )

    assert submission.training_recipe == {
        "schema_version": "recipe.v1",
        "learning_rate": 0.0001,
        "seed": 23,
        "max_steps": 500,
    }
    assert clone_diff(source, submission)["training_recipe"]["to"] == submission.training_recipe


def test_recipe_change_with_a_none_value_removes_the_key() -> None:
    source = source_proposal(
        training_recipe={"schema_version": "recipe.v1", "learning_rate": 0.0001, "seed": 7}
    )

    submission = clone_proposal_submission(
        source, proposal_id="candidate-2", changes={"training_recipe": {"seed": None}}
    )

    assert submission.training_recipe == {"schema_version": "recipe.v1", "learning_rate": 0.0001}
    assert clone_diff(source, submission) == {
        "training_recipe": {
            "from": {"schema_version": "recipe.v1", "learning_rate": 0.0001, "seed": 7},
            "to": {"schema_version": "recipe.v1", "learning_rate": 0.0001},
        }
    }


def test_a_stored_null_recipe_value_survives_an_unrelated_change() -> None:
    source = source_proposal(
        training_recipe={"schema_version": "recipe.v1", "resume_from": None, "seed": 7}
    )

    submission = clone_proposal_submission(
        source, proposal_id="candidate-2", changes={"training_recipe": {"seed": 23}}
    )

    assert submission.training_recipe == {
        "schema_version": "recipe.v1",
        "resume_from": None,
        "seed": 23,
    }


@pytest.mark.parametrize("field", ["dataset_recipe", "training_recipe", "evaluation_recipe"])
def test_a_non_mapping_recipe_change_is_rejected(field) -> None:
    with pytest.raises(CloneStudyError) as excinfo:
        clone_proposal_submission(
            source_proposal(), proposal_id="candidate-2", changes={field: ["not", "a", "mapping"]}
        )

    assert excinfo.value.code == "clone_change_not_allowed"


def test_clone_with_non_empty_required_capabilities() -> None:
    source = source_proposal(
        required_capabilities=frozenset(
            {
                Capability.CAMPAIGN_READ,
                Capability.STUDY_PROPOSE,
                Capability.DATA_BUILD,
            }
        )
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
    assert diff["hypothesis"] == {"from": source.hypothesis, "to": "new hypothesis"}


def test_ordered_tuple_fields_render_in_stored_order() -> None:
    source = source_proposal(controlled_variables=("seed", "batch_size"))

    submission = clone_proposal_submission(
        source,
        proposal_id="candidate-2",
        changes={"controlled_variables": ["max_steps", "batch_size"]},
    )

    assert clone_diff(source, submission) == {
        "controlled_variables": {
            "from": ["seed", "batch_size"],
            "to": ["max_steps", "batch_size"],
        }
    }


def test_required_capabilities_render_sorted() -> None:
    source = source_proposal(
        required_capabilities=frozenset({Capability.STUDY_PROPOSE, Capability.CAMPAIGN_READ})
    )

    submission = clone_proposal_submission(
        source,
        proposal_id="candidate-2",
        changes={"required_capabilities": [Capability.DATA_BUILD.value]},
    )

    diff = clone_diff(source, submission)
    assert diff["required_capabilities"]["from"] == sorted(
        [Capability.STUDY_PROPOSE.value, Capability.CAMPAIGN_READ.value]
    )
    assert diff["required_capabilities"]["to"] == [Capability.DATA_BUILD.value]


def test_clone_of_a_proposal_bound_research_bundle_is_submittable(repository) -> None:
    source = source_proposal(
        research_context=research_context("candidate-1"),
        acquisition=acquisition("candidate-1"),
    )

    submission = clone_proposal_submission(source, proposal_id="candidate-2", changes={})

    assert submission.research_context is None
    assert submission.acquisition is not None
    assert submission.acquisition.proposal_id == "candidate-2"
    assert clone_diff(source, submission) == {}
    validation = validate_proposal_submission(
        submission,
        manifest(),
        principal(repository),
        existing_prerequisite_ids=frozenset(),
    )
    assert validation.reason_codes == ()
    assert validation.valid is True


def test_research_context_is_not_cloneable() -> None:
    assert "research_context" not in CLONEABLE_FIELDS
    assert "acquisition" in CLONEABLE_FIELDS
