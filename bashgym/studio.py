"""Local entry point for the existing durable research service and agent skills."""

from __future__ import annotations

import hmac
import importlib.util
import json
import re
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5


def _write_profile(root: Path, profile: dict[str, Any]) -> None:
    temporary = root / "studio.v1.json.tmp"
    temporary.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "studio.v1.json")


def _bootstrap_setup(root: Path, profile: dict[str, Any], db_path: Path) -> None:
    """Create preparation authority and an unconfigured installation, without leases."""
    from bashgym.campaigns.artifacts import ArtifactSealer
    from bashgym.campaigns.campaign_recovery import CampaignRecoveryRepository
    from bashgym.campaigns.worker import scheduler_lease_key
    from bashgym.secrets import get_secret, set_secret

    seal_ref = "BASHGYM_CAMPAIGN_SEAL_KEY"
    key = get_secret(seal_ref)
    if not key:
        # Never silently replace authority after setup or campaign state exists.
        with sqlite3.connect(db_path) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            has_state = any(
                table in tables
                and connection.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone()
                for table in (
                    "campaigns",
                    "campaign_guided_setup_sessions",
                    "campaign_recovery_receipts",
                )
            )
        if profile.get("installation_id") or has_state:
            raise ValueError("studio_seal_authority_unavailable")
        key = secrets.token_hex(32)
        set_secret(seal_ref, key)
        if get_secret(seal_ref) != key:
            raise ValueError("studio_seal_authority_unavailable")
    sealer = ArtifactSealer(key.encode("utf-8"), key_version="campaign-seal-v1")
    recovery = CampaignRecoveryRepository(db_path, sealer=sealer)
    recovery.initialize()
    # Derivation is stable across interrupted initialization; no new installation
    # is created on a retry before the profile update has reached disk.
    installation_id = (
        profile.get("installation_id")
        or "ins_" + uuid5(NAMESPACE_URL, "bashgym:studio:" + profile["human_credential_id"]).hex
    )
    if not re.fullmatch(r"ins_[0-9a-f]{32}", installation_id):
        raise ValueError("studio_profile_invalid")
    with sqlite3.connect(db_path) as connection:
        registered = connection.execute(
            "SELECT 1 FROM campaign_recovery_installations WHERE installation_id=?",
            (installation_id,),
        ).fetchone()
    if not registered:
        recovery.register_installation(
            installation_id=installation_id,
            controller_owner_id=f"unconfigured:{installation_id}",
            controller_lease_key=scheduler_lease_key(root),
        )
    if profile.get("installation_id") != installation_id:
        profile["installation_id"] = installation_id
        _write_profile(root, profile)


def read_profile(root: Path) -> dict[str, Any] | None:
    path = root / "studio.v1.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "bashgym.studio.v1"
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", value.get("workspace_id", ""))
        or not value.get("credential_ref")
        or not value.get("human_credential_id")
    ):
        raise ValueError("studio_profile_invalid")
    return value


def validate_installation_transition(
    connection,
    *,
    installation_id: str,
    controller_owner_id: str,
    workspace_id: str,
    expected_key: str,
    sealer,
) -> bool:
    """Read-only finalization preflight; true means an exact sealed transition exists."""
    from bashgym.campaigns.contracts import canonical_hash

    provisional_owner = "unconfigured:" + installation_id
    if controller_owner_id.startswith("unconfigured:"):
        raise ValueError("studio_controller_owner_invalid")
    row = connection.execute(
        "SELECT controller_owner_id, controller_lease_key FROM campaign_recovery_installations "
        "WHERE installation_id=?",
        (installation_id,),
    ).fetchone()
    if row is None or row[1] != expected_key:
        raise ValueError("studio_installation_authority_conflict")
    if row[0] not in {provisional_owner, controller_owner_id}:
        raise ValueError("studio_installation_authority_conflict")
    transition_table_exists = (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='studio_installation_transitions'"
        ).fetchone()
        is not None
    )
    previous = (
        connection.execute(
            "SELECT payload_json, seal FROM studio_installation_transitions WHERE installation_id=?",
            (installation_id,),
        ).fetchone()
        if transition_table_exists
        else None
    )
    if previous:
        payload = json.loads(previous[0])
        if not isinstance(payload, dict):
            raise ValueError("studio_installation_transition_conflict")
        if (
            not hmac.compare_digest(
                previous[1],
                sealer.sign_canonical_payload(payload, domain="studio-installation-transition-v1"),
            )
            or payload.get("schema_version") != "studio_installation_transition.v1"
            or payload.get("installation_id") != installation_id
            or payload.get("old_owner") != provisional_owner
            or payload.get("lease_key_digest") != canonical_hash(expected_key)
            or payload.get("new_owner") != controller_owner_id
            or payload.get("workspace_id") != workspace_id
            or row[0] != controller_owner_id
        ):
            raise ValueError("studio_installation_transition_conflict")
        return True
    if row[0] == controller_owner_id:
        raise ValueError("studio_installation_transition_missing")
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in (
        "campaign_recovery_bindings",
        "campaign_recovery_targets",
        "campaign_guided_setup_receipts",
        "campaign_guided_setup_bindings",
    ):
        if (
            table in tables
            and connection.execute(
                f'SELECT 1 FROM "{table}" WHERE installation_id=? LIMIT 1',
                (installation_id,),
            ).fetchone()
        ):
            raise ValueError("studio_installation_already_in_use")
    return False


def finalize_installation(contract: Any) -> None:
    """Bind this studio's unused provisional installation during explicit preparation.

    Established installation ownership is immutable here. A sealed transaction
    records the one allowed transition before activation can start any service.
    """
    from bashgym.campaigns.artifacts import ArtifactSealer
    from bashgym.campaigns.auth import CampaignAuthService
    from bashgym.campaigns.autoresearch import AutoResearchRepository
    from bashgym.campaigns.contracts import Capability, canonical_hash
    from bashgym.campaigns.worker import scheduler_lease_key
    from bashgym.secrets import get_secret

    root = contract.data_directory.expanduser().resolve()
    profile = read_profile(root)
    if profile is None or profile.get("installation_id") != contract.installation_id:
        return
    if (
        profile["workspace_id"] != contract.workspace_id
        or profile["credential_ref"] != contract.credential_ref
    ):
        raise ValueError("studio_preparation_scope_conflict")
    worker_path = root / "campaigns" / "worker-config.v1.json"
    if worker_path.exists():
        from bashgym.campaigns.worker_service import read_worker_config

        owner = read_worker_config(worker_path).controller_owner_id
        if owner is not None and owner != contract.controller_owner_id:
            raise ValueError("studio_existing_controller_conflict")
    raw = get_secret(profile["credential_ref"])
    key = get_secret("BASHGYM_CAMPAIGN_SEAL_KEY")
    if not raw or not key:
        raise ValueError("studio_preparation_authority_unavailable")
    repository = AutoResearchRepository(root / "campaigns" / "campaigns.sqlite3")
    repository.initialize()
    auth = CampaignAuthService(repository)
    principal = auth.authenticate_access(auth.exchange_refresh(raw).raw_token)
    principal.require(contract.workspace_id, Capability.CAMPAIGN_CREATE_FROM_TEMPLATE)
    sealer = ArtifactSealer(key.encode(), key_version="campaign-seal-v1")
    expected_key = scheduler_lease_key(root)
    provisional_owner = f"unconfigured:{contract.installation_id}"
    if contract.controller_owner_id.startswith("unconfigured:"):
        raise ValueError("studio_controller_owner_invalid")
    with sqlite3.connect(repository.db_path, timeout=10) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS studio_installation_transitions "
            "(installation_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, seal TEXT NOT NULL)"
        )
        if validate_installation_transition(
            connection,
            installation_id=contract.installation_id,
            controller_owner_id=contract.controller_owner_id,
            workspace_id=contract.workspace_id,
            expected_key=expected_key,
            sealer=sealer,
        ):
            return
        payload = {
            "schema_version": "studio_installation_transition.v1",
            "installation_id": contract.installation_id,
            "workspace_id": contract.workspace_id,
            "actor_id": principal.actor_id,
            "old_owner": provisional_owner,
            "new_owner": contract.controller_owner_id,
            "lease_key_digest": canonical_hash(expected_key),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        connection.execute(
            "UPDATE campaign_recovery_installations SET controller_owner_id=? "
            "WHERE installation_id=? AND controller_owner_id=? AND controller_lease_key=?",
            (
                contract.controller_owner_id,
                contract.installation_id,
                provisional_owner,
                expected_key,
            ),
        )
        connection.execute(
            "INSERT INTO studio_installation_transitions VALUES (?, ?, ?)",
            (
                contract.installation_id,
                json.dumps(payload, sort_keys=True),
                sealer.sign_canonical_payload(payload, domain="studio-installation-transition-v1"),
            ),
        )


def _health(root: Path) -> dict[str, Any]:
    from bashgym.campaigns.worker_service import probe_api_health

    return probe_api_health(expected_state_root=root)


def _service_action(health: dict[str, Any]) -> str | None:
    if not health.get("healthy"):
        return "start_headless_service"
    if health.get("state_root_match") is False:
        return "connect_matching_state_root"
    if health.get("studio_compatible") is not True:
        return "restart_updated_headless_service"
    return None


def _wait_for_service(root: Path, *, timeout_seconds: float = 15) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        health = _health(root)
        action = _service_action(health)
        if action is None:
            return health
        if action == "connect_matching_state_root":
            raise ValueError("studio_api_state_root_mismatch")
        if action == "restart_updated_headless_service":
            raise ValueError("studio_api_incompatible")
        if time.monotonic() >= deadline:
            raise ValueError("studio_api_unavailable")
        time.sleep(0.2)


def _setup_context(profile: dict[str, Any]) -> dict[str, Any]:
    from bashgym.campaigns.client import CampaignApiClient

    client = CampaignApiClient(
        api_base=profile["api_base"], credential_ref=profile["credential_ref"], timeout=5
    )
    return client.request_json(
        "GET", "/campaigns/setup/context", query={"workspace_id": profile["workspace_id"]}
    )


def doctor(root: Path) -> dict[str, Any]:
    """Read-only facts; package presence and API liveness are not recipe proof."""
    from bashgym.campaigns.preparation_inventory import preparation_inventory

    profile = read_profile(root)
    checks = {
        name: importlib.util.find_spec(name) is not None
        for name in ("fastapi", "uvicorn", "asyncssh", "torch", "data_designer")
    }
    result: dict[str, Any] = {
        "schema_version": "bashgym.studio_doctor.v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "initialized": profile is not None,
        "packages": checks,
        "api": _health(root),
        "recipe_verified": False,
        "ready_for_preparation": False,
        "next_action": "research_prepare" if profile else "initialize",
    }
    result["service_compatible"] = result["api"].get("studio_compatible") is True
    if profile:
        result["preparation_inventory"] = preparation_inventory(root, profile["workspace_id"])
    service_action = _service_action(result["api"])
    if service_action is not None and (profile or result["api"].get("healthy")):
        result["next_action"] = service_action
    result["recipe_readiness"] = {
        "status": "not_checked",
        "execution_verified": False,
        "checked_at": result["checked_at"],
        "reason_codes": ["setup_context_unavailable"],
    }
    if profile and service_action is None:
        from bashgym.campaigns.client import CampaignClientError

        try:
            context = _setup_context(profile)
            result["setup_context"] = context
            result["ready_for_preparation"] = True
            session = context.get("session")
            result["recipe_readiness"].update(
                status=(
                    "ready_for_validation"
                    if session and session.get("ready_for_validation")
                    else "blocked" if session else "not_selected"
                ),
                reason_codes=(session or context).get("reason_codes", []),
            )
        except CampaignClientError as exc:
            result["setup_error"] = exc.code
            result["next_action"] = (
                "repair_studio_credentials"
                if exc.status_code in {401, 403} or "auth" in exc.code or "credential" in exc.code
                else "inspect_setup_service"
            )
    return result


def initialize(
    root: Path,
    *,
    workspace_id: str | None = None,
    agent_host: str | None = None,
    start_service: bool = True,
) -> dict[str, Any]:
    """Resume local setup without selecting a learner, data, budget or starting training."""
    from bashgym.api import database
    from bashgym.campaigns.auth import CampaignAuthService
    from bashgym.campaigns.autoresearch import AutoResearchRepository
    from bashgym.campaigns.contracts import AutonomyProfile
    from bashgym.campaigns.preparation_inventory import preparation_inventory
    from bashgym.operator_skills import install_skills
    from bashgym.secrets import set_secret

    profile = read_profile(root)
    workspace_id = workspace_id or (profile or {}).get("workspace_id", "personal")
    agent_host = agent_host or (profile or {}).get("agent_host", "codex")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", workspace_id):
        raise ValueError("studio_workspace_invalid")
    if agent_host not in {"codex", "hermes", "claude"}:
        raise ValueError("studio_agent_host_invalid")
    root.mkdir(parents=True, exist_ok=True)
    replayed = profile is not None
    if profile and (profile["workspace_id"] != workspace_id or profile["agent_host"] != agent_host):
        raise ValueError("studio_profile_conflict")
    repository = AutoResearchRepository(root / "campaigns" / "campaigns.sqlite3")
    repository.initialize()
    auth = CampaignAuthService(repository)
    if profile is None:
        human = auth.issue_refresh_credential(
            actor_id="local-operator",
            autonomy_profile=AutonomyProfile.DESKTOP_USER,
            workspace_ids=(workspace_id,),
        )
        agent = auth.issue_refresh_credential(
            actor_id=f"studio-{agent_host}",
            autonomy_profile=(
                AutonomyProfile.HERMES_BOUNDED
                if agent_host == "hermes"
                else AutonomyProfile.CODEX_TRUSTED
            ),
            workspace_ids=(workspace_id,),
        )
        ref = f"BASHGYM_STUDIO_{agent.credential_id.replace('-', '_').upper()}"
        set_secret(ref, agent.raw_token)
        profile = {
            "schema_version": "bashgym.studio.v1",
            "workspace_id": workspace_id,
            "agent_host": agent_host,
            "credential_ref": ref,
            "human_credential_id": human.credential_id,
            "api_base": "http://127.0.0.1:8003/api",
        }
        _write_profile(root, profile)
    human = repository.get_actor_credential(profile["human_credential_id"])
    if (
        human is None
        or human.revoked_at is not None
        or human.expires_at <= datetime.now(timezone.utc)
    ):
        raise ValueError("studio_authority_expired")
    _bootstrap_setup(root, profile, repository.db_path)
    skills = install_skills(host=agent_host)
    service_health = None
    if start_service:
        from bashgym.campaigns.worker_service import ApiServiceManager, build_api_service_definition

        observed = _health(root)
        if observed.get("healthy") and _service_action(observed) is not None:
            # An existing responder is never replaced merely to complete init.
            _wait_for_service(root, timeout_seconds=0)
        if not observed.get("healthy"):
            definition = build_api_service_definition(data_directory=root)
            manager = ApiServiceManager()
            if definition.definition_path.exists():
                if definition.definition_path.read_bytes() != definition.definition_payload:
                    raise ValueError("studio_service_definition_conflict")
                if manager.status(definition)["supervisor_state"] != "available":
                    manager.start(definition)
            else:
                manager.install(definition)
        service_health = _wait_for_service(root)
        _setup_context(profile)
    database.set_db_path(root / "api" / "bashgym.db")
    database.init_db()
    code = database.issue_local_pairing(human.credential_id, human.authorization_revision)
    return {
        "schema_version": "bashgym.studio_init.v1",
        "replayed": replayed,
        "workspace_id": workspace_id,
        "agent_host": agent_host,
        "skills": skills,
        "browser_url": "http://127.0.0.1:8003",
        "pairing_code": code,
        "pairing_expires_in_seconds": 300,
        "next_action": "research_prepare" if service_health else "start_headless_service",
        "service_verified": service_health is not None,
        "service_health": service_health,
        "training_started": False,
        "learner": "select_during_preparation",
        "preparation_inventory": preparation_inventory(root, workspace_id),
    }
