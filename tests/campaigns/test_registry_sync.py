"""Plan-first synchronization of activation evidence into guided setup."""

from __future__ import annotations

import json
import sqlite3

import pytest

from bashgym.campaigns.activation import activate_autoresearch
from bashgym.campaigns.registry_sync import (
    AutoResearchRegistrySyncError,
    apply_autoresearch_registry_sync,
    plan_autoresearch_registry_sync,
    resolve_autoresearch_installation_id,
)
from bashgym.campaigns.worker_service import read_worker_config
from bashgym.ledger.persistence import ExperimentLedgerRepository
from tests.campaigns.test_autoresearch_activation import (
    _activation_fixture,
    _memory_secrets,
)


def _activated(tmp_path):
    definition, request = _activation_fixture(tmp_path)
    root = tmp_path / "installation"
    _secrets, resolve, write = _memory_secrets()
    activate_autoresearch(
        definition,
        request,
        data_directory=root,
        apply=True,
        secret_resolver=resolve,
        secret_writer=write,
    )
    database_path = root / "campaigns" / "campaigns.sqlite3"
    worker_config_path = root / "campaigns" / "worker-config.v1.json"
    return (
        definition,
        database_path,
        ExperimentLedgerRepository.open_existing(database_path),
        read_worker_config(worker_config_path),
    )


def test_plan_projects_only_logical_reachable_bindings_without_writes_or_private_data(tmp_path):
    definition, database_path, ledger, worker_config = _activated(tmp_path)
    with sqlite3.connect(database_path) as connection:
        before_tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

    plan = plan_autoresearch_registry_sync(
        definitions=(definition,),
        workspace_id="workspace-a",
        installation_id="ins_0123456789abcdef0123456789abcdef",
        ledger=ledger,
        worker_config=worker_config,
    )

    assert [(item.kind, item.availability) for item in plan.bindings] == [
        ("compute", "reachable"),
        ("data", "reachable"),
        ("evaluator", "reachable"),
        ("model", "reachable"),
    ]
    assert plan.ready is True
    wire = plan.model_dump_json().casefold()
    for private in ("192.0.2.10", "trainer", "launch-material", ".key", "source"):
        assert private not in wire
    assert "model_ref" not in wire
    with sqlite3.connect(database_path) as connection:
        after_tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    assert after_tables == before_tables
    assert "campaign_recovery_installations" not in {row[0] for row in after_tables}


def test_plan_marks_model_and_compute_unknown_without_exact_registered_executor(tmp_path):
    definition, _database_path, ledger, worker_config = _activated(tmp_path)
    worker_config = worker_config.model_copy(update={"approved_remote_profiles": ()})

    plan = plan_autoresearch_registry_sync(
        definitions=(definition,),
        workspace_id="workspace-a",
        installation_id="ins_0123456789abcdef0123456789abcdef",
        ledger=ledger,
        worker_config=worker_config,
    )

    bindings = {item.kind: item for item in plan.bindings}
    assert bindings["model"].availability == "unknown"
    assert bindings["compute"].availability == "unknown"
    assert bindings["model"].reason_codes == ("exact_training_executor_not_registered",)
    assert bindings["data"].availability == "reachable"
    assert bindings["evaluator"].availability == "reachable"
    assert plan.ready is False


def test_plan_can_generate_an_installation_identity_but_apply_requires_the_reviewed_value():
    generated = resolve_autoresearch_installation_id(None, apply=False)

    assert generated.startswith("ins_")
    assert len(generated) == 36
    assert resolve_autoresearch_installation_id(generated, apply=True) == generated
    with pytest.raises(AutoResearchRegistrySyncError, match="identity_required"):
        resolve_autoresearch_installation_id(None, apply=True)
    with pytest.raises(AutoResearchRegistrySyncError, match="identity_invalid"):
        resolve_autoresearch_installation_id("ins_not-reviewed", apply=False)


def test_apply_is_atomic_idempotent_and_never_returns_authority(tmp_path):
    definition, database_path, ledger, worker_config = _activated(tmp_path)
    plan = plan_autoresearch_registry_sync(
        definitions=(definition,),
        workspace_id="workspace-a",
        installation_id="ins_0123456789abcdef0123456789abcdef",
        ledger=ledger,
        worker_config=worker_config,
    )

    created = apply_autoresearch_registry_sync(
        plan,
        database_path=database_path,
        controller_owner_id="installation-controller",
        controller_lease_key="private-installation-lease",
    )
    replayed = apply_autoresearch_registry_sync(
        plan,
        database_path=database_path,
        controller_owner_id="installation-controller",
        controller_lease_key="private-installation-lease",
    )

    assert created.disposition == "created"
    assert created.created_bindings == 4
    assert replayed.disposition == "replayed"
    assert replayed.replayed_bindings == 4
    wire = json.dumps(replayed.model_dump(mode="json"), sort_keys=True)
    assert "installation-controller" not in wire
    assert "private-installation-lease" not in wire


def test_apply_rejects_conflicting_installation_authority_without_partial_changes(tmp_path):
    definition, database_path, ledger, worker_config = _activated(tmp_path)
    plan = plan_autoresearch_registry_sync(
        definitions=(definition,),
        workspace_id="workspace-a",
        installation_id="ins_0123456789abcdef0123456789abcdef",
        ledger=ledger,
        worker_config=worker_config,
    )
    apply_autoresearch_registry_sync(
        plan,
        database_path=database_path,
        controller_owner_id="installation-controller",
        controller_lease_key="private-installation-lease",
    )

    with pytest.raises(AutoResearchRegistrySyncError, match="authority_conflict"):
        apply_autoresearch_registry_sync(
            plan,
            database_path=database_path,
            controller_owner_id="other-controller",
            controller_lease_key="other-private-lease",
        )
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM campaign_recovery_bindings WHERE installation_id=?",
                (plan.installation_id,),
            ).fetchone()[0]
            == 4
        )
