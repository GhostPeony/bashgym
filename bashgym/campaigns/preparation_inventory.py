"""Read-only, workspace-scoped preparation candidates from the experiment ledger.

Inventory never turns metadata into an execution binding. The existing setup
validator remains the authority for a selected recipe and execution target.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bashgym.campaigns.contracts import canonical_hash
from bashgym.ledger.persistence import ExperimentLedgerRepository, LedgerPersistenceError

MAX_PROJECTS = 50
MAX_CANDIDATES_PER_KIND = 100


def _public_label(value: Any, kind: str) -> str:
    fallback = {
        "models": "Registered model",
        "data": "Registered dataset",
        "evaluation": "Registered evaluation",
        "compute": "Registered execution environment",
    }[kind]
    # Compute labels often contain machine topology. Never publish them. Other
    # display names are accepted only as short prose, not filesystem/URL material.
    if kind == "compute" or not isinstance(value, str) or len(value) > 160:
        return fallback
    if re.search(r"(?:ghp|sk-proj|sk_live|xox[baprs]|AKIA|AIza)[_-]?[A-Za-z0-9]", value, re.I):
        return fallback
    if re.search(
        r"[\\/\x00-\x1f]|\b(?:\d{1,3}\.){3}\d{1,3}\b|\.(?:local|internal|lan|invalid)\b", value
    ):
        return fallback
    return value.strip() or fallback


def preparation_inventory(root: Path, workspace_id: str) -> dict[str, Any]:
    candidates: dict[str, list[dict[str, Any]]] = {
        "models": [],
        "data": [],
        "evaluation": [],
        "compute": [],
    }
    result: dict[str, Any] = {
        "schema_version": "bashgym.preparation_inventory.v1",
        "workspace_id": workspace_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates,
        "reason_codes": [],
        "truncated": False,
        "training_started": False,
        "execution_verified": False,
    }
    try:
        ledger = ExperimentLedgerRepository.open_existing(root / "campaigns" / "campaigns.sqlite3")
    except LedgerPersistenceError:
        result["reason_codes"] = ["registered_assets_unavailable"]
        return result
    projects = ledger.list_projects(workspace_id)
    result["truncated"] = len(projects) > MAX_PROJECTS
    sources = (
        ("models", ledger.list_model_versions, "model_version_id", "model_display_name"),
        ("data", ledger.list_dataset_versions, "dataset_version_id", "dataset_display_name"),
        ("evaluation", ledger.list_evaluation_suites, "evaluation_suite_id", "name"),
        ("compute", ledger.list_environments, "environment_id", "name"),
    )
    for project in projects[:MAX_PROJECTS]:
        for kind, load, identifier_key, label_key in sources:
            rows = load(workspace_id, project["project_id"])
            remaining = MAX_CANDIDATES_PER_KIND - len(candidates[kind])
            result["truncated"] |= len(rows) > remaining
            for row in rows[:remaining]:
                # Never copy source URIs, paths, hardware, arbitrary metadata or
                # transport fields into the browser inventory.
                identity = {
                    "workspace_id": workspace_id,
                    "project_id": project["project_id"],
                    "kind": kind,
                    "record_id": row[identifier_key],
                }
                candidates[kind].append(
                    {
                        "candidate_id": "asset_" + canonical_hash(identity)[:32],
                        "project_id": project["project_id"],
                        "record_id": row[identifier_key],
                        "label": _public_label(row.get(label_key), kind),
                        "record_digest": canonical_hash(row),
                        "source_registry": "experiment_ledger",
                        "evidence": "registered_metadata",
                        "execution_verified": False,
                        "next_action": "validate_registered_binding",
                    }
                )
    result["reason_codes"] = [f"{kind}_selection_required" for kind in candidates]
    return result
