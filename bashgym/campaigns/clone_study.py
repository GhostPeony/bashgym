"""Turn a persisted study proposal into a new submission with an explicit diff."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bashgym.campaigns.contracts import StudyProposal, StudyProposalSubmission
from bashgym.research.acquisition import ExperimentAcquisition

CLONEABLE_FIELDS: tuple[str, ...] = (
    "hypothesis",
    "evidence_references",
    "study_family",
    "primary_variable",
    "controlled_variables",
    "expected_outcome",
    "falsification_criterion",
    "estimated_cost",
    "priority",
    "prerequisite_study_ids",
    "dataset_recipe",
    "training_recipe",
    "evaluation_recipe",
    "required_capabilities",
    "stage_plan",
    "rationale",
    "acquisition",
)

_RECIPE_FIELDS = frozenset({"dataset_recipe", "training_recipe", "evaluation_recipe"})
_SET_FIELDS = frozenset({"required_capabilities"})


class CloneStudyError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def clone_proposal_submission(
    source: StudyProposal,
    *,
    proposal_id: str,
    changes: Mapping[str, Any],
) -> StudyProposalSubmission:
    """Copy the scientific fields of one proposal and apply bounded changes.

    A change to `dataset_recipe`, `training_recipe`, or `evaluation_recipe` merges
    shallowly into the stored recipe and removes only the keys the change itself
    names with a `None` value; every other cloneable field is replaced outright.
    `research_context` is not cloneable because its `retrieval_digest` covers the
    proposal id it was collected for, so the clone leaves it unset. `acquisition`
    carries no digest and is rebound to the new proposal id.
    """

    if proposal_id == source.proposal_id:
        raise CloneStudyError("clone_proposal_id_reused")
    disallowed = set(changes) - set(CLONEABLE_FIELDS)
    if disallowed:
        raise CloneStudyError("clone_change_not_allowed")
    payload: dict[str, Any] = {field: getattr(source, field) for field in CLONEABLE_FIELDS}
    acquisition = payload["acquisition"]
    if isinstance(acquisition, ExperimentAcquisition):
        payload["acquisition"] = acquisition.model_copy(update={"proposal_id": proposal_id})
    for field, value in changes.items():
        if field not in _RECIPE_FIELDS:
            payload[field] = value
            continue
        if not isinstance(value, Mapping):
            raise CloneStudyError("clone_change_not_allowed")
        merged = dict(payload[field])
        for key, item in value.items():
            if item is None:
                merged.pop(key, None)
            else:
                merged[key] = item
        payload[field] = merged
    return StudyProposalSubmission(
        proposal_id=proposal_id,
        workspace_id=source.workspace_id,
        campaign_id=source.campaign_id,
        **payload,
    )


def _ordered(field: str, value: Any) -> Any:
    if field in _SET_FIELDS and isinstance(value, list):
        return sorted(value)
    return value


def _rebound(value: Any, proposal_id: str) -> Any:
    """Neutralize the clone's mechanical acquisition rebind before comparison."""

    if isinstance(value, ExperimentAcquisition) and value.proposal_id != proposal_id:
        return value.model_copy(update={"proposal_id": proposal_id})
    return value


def clone_diff(
    source: StudyProposal, submission: StudyProposalSubmission
) -> dict[str, dict[str, Any]]:
    """Changed scientific fields, in JSON mode, for the agent to declare."""

    before = source.model_dump(mode="json")
    after = submission.model_dump(mode="json")
    return {
        field: {
            "from": _ordered(field, before[field]),
            "to": _ordered(field, after[field]),
        }
        for field in CLONEABLE_FIELDS
        if _rebound(getattr(source, field), submission.proposal_id) != getattr(submission, field)
    }


__all__ = ["CLONEABLE_FIELDS", "CloneStudyError", "clone_diff", "clone_proposal_submission"]
