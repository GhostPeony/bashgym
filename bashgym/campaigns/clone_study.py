"""Turn a persisted study proposal into a new submission with an explicit diff."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bashgym.campaigns.contracts import StudyProposal, StudyProposalSubmission

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
    "research_context",
    "acquisition",
)

_RECIPE_FIELDS = frozenset({"dataset_recipe", "training_recipe", "evaluation_recipe"})


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
    shallowly into the stored recipe and drops any key whose change value is
    `None`; every other cloneable field is replaced outright.
    """

    if proposal_id == source.proposal_id:
        raise CloneStudyError("clone_proposal_id_reused")
    disallowed = set(changes) - set(CLONEABLE_FIELDS)
    if disallowed:
        raise CloneStudyError("clone_change_not_allowed")
    payload: dict[str, Any] = {field: getattr(source, field) for field in CLONEABLE_FIELDS}
    for field, value in changes.items():
        if field not in _RECIPE_FIELDS:
            payload[field] = value
            continue
        if not isinstance(value, Mapping):
            raise CloneStudyError("clone_change_not_allowed")
        merged = {**payload[field], **value}
        payload[field] = {key: item for key, item in merged.items() if item is not None}
    return StudyProposalSubmission(
        proposal_id=proposal_id,
        workspace_id=source.workspace_id,
        campaign_id=source.campaign_id,
        **payload,
    )


def _ordered(value: Any) -> Any:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return sorted(value)
    return value


def clone_diff(
    source: StudyProposal, submission: StudyProposalSubmission
) -> dict[str, dict[str, Any]]:
    """Changed scientific fields, in JSON mode, for the agent to declare."""

    before = source.model_dump(mode="json")
    after = submission.model_dump(mode="json")
    return {
        field: {"from": _ordered(before[field]), "to": _ordered(after[field])}
        for field in CLONEABLE_FIELDS
        if getattr(source, field) != getattr(submission, field)
    }


__all__ = ["CLONEABLE_FIELDS", "CloneStudyError", "clone_diff", "clone_proposal_submission"]
