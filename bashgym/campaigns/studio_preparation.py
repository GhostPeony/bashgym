"""Compile saved, approved installation records into the existing onboarding inputs.

This is a local deterministic preparation boundary. No acquisition, service,
remote preflight, profile registration, credential issuance, or Start occurs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bashgym.campaigns.activation import (
    AutoResearchActivationRequest,
    _validate_definition_bindings,
)
from bashgym.campaigns.autoresearch import AutoResearchStopRules
from bashgym.campaigns.contracts import AutonomyProfile, CredentialKind, canonical_hash, utc_now
from bashgym.campaigns.installation import _read_definition, autoresearch_binding_plan
from bashgym.campaigns.onboarding import (
    _MAX_RECEIPT_BYTES,
    AutoResearchOnboardingConflict,
    AutoResearchOnboardingContract,
    AutoResearchOnboardingCoordinator,
    _api_origin,
    _read_receipt,
    _receipt_path,
    guided_setup_snapshot_digest,
    validate_guided_setup_resume,
    validate_onboarding_secret_references,
)
from bashgym.campaigns.remote import RemoteModelRegistrationRequest


class PreparationInputsRequired(ValueError):  # noqa: N818
    """An exact existing preparation input is absent, never implicitly acquired."""

    def __init__(self, field: str, code: str):
        super().__init__(code)
        self.field = field
        self.code = code


def _require(value, field: str, code: str):
    if value is None or value == "":
        raise PreparationInputsRequired(field, code)
    return value


def _existing_agent(root, profile, secret_resolver):
    # Verify the existing refresh credential without minting an access token.
    from bashgym.campaigns.auth import _parse, _verify

    raw = _require(secret_resolver(profile["credential_ref"]), "agent", "agent_credential_required")
    credential_id, secret = _parse(raw, CredentialKind.REFRESH)
    path = root / "campaigns" / "campaigns.sqlite3"
    _require(path if path.is_file() else None, "installation", "campaign_registry_required")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM campaign_actor_credentials WHERE credential_id=?", (credential_id,)
        ).fetchone()
    expected = (
        AutonomyProfile.HERMES_BOUNDED.value
        if profile["agent_host"] == "hermes"
        else AutonomyProfile.CODEX_TRUSTED.value
    )
    if (
        row is None
        or row["actor_id"] != "studio-" + profile["agent_host"]
        or row["autonomy_profile"] != expected
        or row["credential_kind"] != CredentialKind.REFRESH.value
        or row["revoked_at"] is not None
        or datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00")) <= utc_now()
        or datetime.fromisoformat(row["token_not_before"].replace("Z", "+00:00")) > utc_now()
        or profile["workspace_id"] not in json.loads(row["workspace_ids_json"])
        or not _verify(secret, row["token_salt"], row["token_hash"])
    ):
        raise AutoResearchOnboardingConflict("studio_agent_authority_conflict")
    return row["actor_id"]


def _spec(model, record):
    # Storage adds bookkeeping columns; every actual spec field is preserved.
    return model.model_validate(
        {name: record[name] for name in model.model_fields if name in record}
    )


def _publish_private_input(path: Path, content: bytes) -> None:
    """Publish a complete private file atomically, without replacing prior inputs."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".preparation-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != content:
                raise AutoResearchOnboardingConflict("preparation_input_file_conflict")
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RegisteredPreparation:
    contract: AutoResearchOnboardingContract
    inputs: dict[str, bytes]
    session_version: int
    session_digest: str
    agent_host: str

    @property
    def contract_file(self) -> Path:
        return self.contract.definition_file.parent / "onboarding.json"

    def summary(self) -> dict[str, Any]:
        from bashgym.campaigns.autoresearch import AutoResearchTemplateDefinition

        definition = AutoResearchTemplateDefinition.model_validate_json(self.inputs["definition"])
        return {
            "schema_version": "bashgym.registered_preparation.v1",
            "onboarding_id": self.contract.onboarding_id,
            "campaign_id": self.contract.campaign_id,
            "campaign_title": self.contract.campaign_title,
            "agent_host": self.agent_host,
            "binding_plan": autoresearch_binding_plan(definition).model_dump(mode="json"),
            "session_id": self.contract.guided_setup_session_id,
            "session_version": self.session_version,
            "session_digest": self.session_digest,
            "stop_rules": self.contract.stop_rules.model_dump(mode="json"),
            "input_sha256": self.contract.expected_input_sha256,
            "contract_file": str(self.contract_file),
            "compute_started": False,
            "next_action": "write_preparation_inputs",
        }

    def write_inputs(self) -> dict[str, Any]:
        """Persist only already validated content-addressed local input files."""
        if {
            name: hashlib.sha256(content).hexdigest() for name, content in self.inputs.items()
        } != self.contract.expected_input_sha256:
            raise AutoResearchOnboardingConflict("preparation_input_digest_conflict")
        paths = {
            "definition": self.contract.definition_file,
            "activation": self.contract.activation_file,
            "model_request": self.contract.model_request_file,
        }
        payloads = {paths[name]: content for name, content in self.inputs.items()}
        payloads[self.contract_file] = (self.contract.model_dump_json(indent=2) + "\n").encode()
        # Check all conflicts before writing any part of the bundle.
        for path, content in payloads.items():
            if not path.resolve().is_relative_to(self.contract.data_directory.resolve()):
                raise AutoResearchOnboardingConflict("preparation_input_path_conflict")
            if path.is_symlink() or (path.exists() and path.read_bytes() != content):
                raise AutoResearchOnboardingConflict("preparation_input_file_conflict")
        self.contract_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path, content in payloads.items():
            _publish_private_input(path, content)
        plan = AutoResearchOnboardingCoordinator.plan(self.contract)
        return {
            **self.summary(),
            "next_action": "review_then_apply_onboarding",
            "onboarding": plan.model_dump(mode="json"),
        }


def build_registered_preparation(
    root: Path,
    *,
    workspace_id: str,
    template_id: str,
    expected_definition_digest: str,
    session_id: str | None,
    expected_version: int,
    onboarding_id: str,
    campaign_id: str,
    campaign_title: str,
    stop_rules: AutoResearchStopRules,
    controller_lease_key_ref: str,
    secret_resolver=None,
) -> RegisteredPreparation:
    """Resolve one installed template and exact saved draft, with no writes."""
    from bashgym.campaigns.artifacts import ArtifactSealer
    from bashgym.campaigns.guided_setup import GuidedSetupRepository
    from bashgym.campaigns.worker import scheduler_lease_key
    from bashgym.campaigns.worker_service import (
        load_approved_remote_profiles,
        load_approved_source_profiles,
        read_worker_config,
    )
    from bashgym.ledger.contracts import (
        DatasetSpec,
        DatasetVersionSpec,
        EvaluationSuiteSpec,
        ProjectSpec,
    )
    from bashgym.ledger.persistence import (
        ExperimentLedgerRepository,
        LedgerPersistenceError,
        RecordNotFoundError,
    )
    from bashgym.secrets import get_secret
    from bashgym.studio import read_profile, validate_installation_transition

    required_limits = {"max_attempts", "budget_unit", "max_total_cost", "minimum_improvement"}
    if not required_limits <= stop_rules.model_fields_set:
        raise PreparationInputsRequired("stop_rules", "explicit_stop_limits_required")
    root = root.expanduser().resolve()
    secret_resolver = secret_resolver or get_secret
    profile = _require(read_profile(root), "installation", "studio_initialization_required")
    validate_onboarding_secret_references(profile["credential_ref"], controller_lease_key_ref)
    if profile["workspace_id"] != workspace_id:
        raise AutoResearchOnboardingConflict("studio_workspace_conflict")
    actor = _existing_agent(root, profile, secret_resolver)
    # Reject path-like IDs before constructing an installed-template path.
    from pydantic import TypeAdapter

    from bashgym.campaigns.contracts import Identifier

    template_id = TypeAdapter(Identifier).validate_python(template_id)
    path = root / "campaigns" / "autoresearch-templates" / (template_id + ".json")
    _require(path if path.is_file() else None, "template", "installed_template_required")
    definition = _read_definition(path)
    if definition.template_id != template_id:
        raise AutoResearchOnboardingConflict("installed_template_identity_conflict")
    if definition.definition_digest != expected_definition_digest:
        raise AutoResearchOnboardingConflict("installed_template_digest_conflict")
    definition.validate_campaign_stop_rules(stop_rules)
    binding = autoresearch_binding_plan(definition)
    database = root / "campaigns" / "campaigns.sqlite3"
    seal_key = _require(
        secret_resolver("BASHGYM_CAMPAIGN_SEAL_KEY"), "installation", "existing_setup_seal_required"
    )
    setup = GuidedSetupRepository.open_binding_registry(
        database, sealer=ArtifactSealer(seal_key.encode(), key_version="campaign-seal-v1")
    )
    context = setup.context(
        workspace_id=workspace_id,
        actor_id=actor,
        definitions={template_id: definition},
        session_id=session_id,
        workspace_shared=True,
    )
    session = _require(context.get("session"), "session", "shared_setup_session_required")
    if type(expected_version) is not int or session["version"] != expected_version:
        raise AutoResearchOnboardingConflict("guided_setup_version_conflict")
    installation_id = _require(
        session["selections"].get("installation_id"),
        "installation",
        "installation_selection_required",
    )
    worker_path = root / "campaigns" / "worker-config.v1.json"
    _require(
        worker_path if worker_path.is_file() else None, "compute", "approved_worker_config_required"
    )
    worker = read_worker_config(worker_path)
    database = root / "campaigns" / "campaigns.sqlite3"
    if worker.data_directory != root or worker.database_path != database:
        raise AutoResearchOnboardingConflict("worker_state_root_conflict")
    owner = _require(worker.controller_owner_id, "compute", "configured_controller_owner_required")
    if secret_resolver(controller_lease_key_ref) != scheduler_lease_key(root):
        raise PreparationInputsRequired(
            "controller_lease_key_ref", "existing_controller_lease_reference_required"
        )
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        installation = connection.execute(
            "SELECT controller_owner_id, controller_lease_key FROM campaign_recovery_installations WHERE installation_id=?",
            (installation_id,),
        ).fetchone()
        if installation_id == profile.get("installation_id"):
            validate_installation_transition(
                connection,
                installation_id=installation_id,
                controller_owner_id=owner,
                workspace_id=workspace_id,
                expected_key=scheduler_lease_key(root),
                sealer=setup.sealer,
            )
    allowed_owners = {owner}
    if installation_id == profile.get("installation_id"):
        allowed_owners.add("unconfigured:" + installation_id)
    if (
        installation is None
        or installation[0] not in allowed_owners
        or installation[1] != scheduler_lease_key(root)
    ):
        raise AutoResearchOnboardingConflict("registered_installation_authority_conflict")
    executor = _require(
        load_approved_remote_profiles(worker).get(
            (binding.compute_profile_id, binding.target_contract_key)
        ),
        "compute",
        "approved_exact_executor_required",
    )
    source = _require(
        load_approved_source_profiles(worker).get(binding.source_repository_profile_id),
        "source",
        "approved_exact_source_required",
    )
    try:
        repository = ExperimentLedgerRepository.open_existing(database)
    except LedgerPersistenceError as exc:
        raise PreparationInputsRequired("ledger", "existing_ledger_schema_required") from exc
    try:
        project = _spec(
            ProjectSpec, repository.get_project(workspace_id, binding.ledger_project_id)
        )
        data_version = _spec(
            DatasetVersionSpec,
            repository.get_dataset_version(
                workspace_id, project.project_id, binding.dataset_version_id
            ),
        )
        dataset = _spec(
            DatasetSpec,
            repository.get_dataset(workspace_id, project.project_id, data_version.dataset_id),
        )
        evaluation = _spec(
            EvaluationSuiteSpec,
            repository.get_evaluation_suite(
                workspace_id, project.project_id, binding.evaluation_suite_id
            ),
        )
    except RecordNotFoundError as exc:
        raise PreparationInputsRequired("data_evaluation", "exact_ledger_records_required") from exc
    activation = AutoResearchActivationRequest(
        workspace_id=workspace_id,
        project=project,
        dataset=dataset,
        dataset_version=data_version,
        evaluation_suite=evaluation,
        source_profile=source,
        executor_profile=executor,
    )
    registered = _require(
        executor.registered_base_model, "model", "registered_exact_learner_required"
    )
    _require(registered.artifact_receipt, "model", "validated_model_artifact_receipt_required")
    _require(
        executor.registered_evaluation_dataset, "evaluation", "registered_heldout_dataset_required"
    )
    _validate_definition_bindings(definition, activation)
    assert registered.artifact_receipt is not None
    if (registered.compute_profile_id, registered.target_contract_key, registered.model_digest) != (
        binding.compute_profile_id,
        binding.target_contract_key,
        binding.target_model_digest,
    ):
        raise AutoResearchOnboardingConflict("registered_model_binding_conflict")
    receipt = registered.artifact_receipt
    model_request = RemoteModelRegistrationRequest(
        operation="register",
        source_id=registered.source_id,
        compute_profile_id=registered.compute_profile_id,
        target_contract_key=registered.target_contract_key,
        target_model_digest=registered.model_digest,
        model_id=receipt.model_id,
        revision=receipt.revision,
        remote_model_path=registered.remote_model_path,
    )
    encoded = {
        name: (value.model_dump_json(indent=2) + "\n").encode()
        for name, value in (
            ("definition", definition),
            ("activation", activation),
            ("model_request", model_request),
        )
    }
    if any(len(content) > _MAX_RECEIPT_BYTES for content in encoded.values()):
        raise AutoResearchOnboardingConflict("preparation_input_too_large")
    digests = {name: hashlib.sha256(content).hexdigest() for name, content in encoded.items()}
    # Include the scientific choices and draft identity so one directory names one contract.
    bundle_id = canonical_hash(
        {
            "inputs": digests,
            "onboarding_id": onboarding_id,
            "campaign_id": campaign_id,
            "campaign_title": campaign_title,
            "stop_rules": stop_rules.model_dump(mode="json"),
            "session_digest": guided_setup_snapshot_digest(session),
            "controller_lease_key_ref": controller_lease_key_ref,
        }
    )
    directory = root / "campaigns" / "onboarding" / "private" / "prepared" / bundle_id
    contract = AutoResearchOnboardingContract(
        onboarding_id=onboarding_id,
        data_directory=root,
        definition_file=directory / "definition.json",
        activation_file=directory / "activation.json",
        model_request_file=directory / "model-request.json",
        workspace_id=workspace_id,
        installation_id=installation_id,
        controller_owner_id=owner,
        controller_lease_key_ref=controller_lease_key_ref,
        api_base=profile["api_base"],
        credential_ref=profile["credential_ref"],
        campaign_id=campaign_id,
        campaign_title=campaign_title,
        guided_setup_session_id=session["session_id"],
        guided_setup_expected_version=session["version"],
        guided_setup_session_digest=guided_setup_snapshot_digest(session),
        expected_input_sha256=digests,
        stop_rules=stop_rules,
    )
    _api_origin(contract)
    validate_guided_setup_resume(contract, definition, session)
    prior_receipt = _read_receipt(_receipt_path(contract))
    if prior_receipt is not None:
        expected_digest = canonical_hash(
            {**contract.model_dump(mode="json"), "input_sha256": digests}
        )
        if prior_receipt.contract_digest != expected_digest:
            raise AutoResearchOnboardingConflict("onboarding_contract_changed_resume_saved_inputs")
    return RegisteredPreparation(
        contract,
        encoded,
        session["version"],
        guided_setup_snapshot_digest(session),
        profile["agent_host"],
    )


def validate_current_preparation_approvals(contract, definition, activation) -> None:
    """Recheck current approval records before any bridge-owned physical step."""
    from bashgym.campaigns.artifacts import ArtifactSealer
    from bashgym.campaigns.worker import scheduler_lease_key
    from bashgym.campaigns.worker_service import (
        load_approved_remote_profiles,
        load_approved_source_profiles,
        read_worker_config,
    )
    from bashgym.ledger.persistence import ExperimentLedgerRepository
    from bashgym.secrets import get_secret
    from bashgym.studio import read_profile, validate_installation_transition

    root = contract.data_directory.expanduser().resolve()
    profile = _require(read_profile(root), "installation", "studio_initialization_required")
    if (
        profile["workspace_id"] != contract.workspace_id
        or profile["credential_ref"] != contract.credential_ref
        or str(profile["api_base"]).rstrip("/") != str(contract.api_base).rstrip("/")
    ):
        raise AutoResearchOnboardingConflict("studio_preparation_scope_conflict")
    _existing_agent(root, profile, get_secret)
    current_definition = _read_definition(
        root / "campaigns" / "autoresearch-templates" / (definition.template_id + ".json")
    )
    if current_definition.definition_digest != definition.definition_digest:
        raise AutoResearchOnboardingConflict("installed_template_digest_conflict")
    worker = read_worker_config(root / "campaigns" / "worker-config.v1.json")
    database = root / "campaigns" / "campaigns.sqlite3"
    if (
        worker.data_directory != root
        or worker.database_path != database
        or worker.controller_owner_id != contract.controller_owner_id
    ):
        raise AutoResearchOnboardingConflict("current_worker_authority_conflict")
    binding = autoresearch_binding_plan(definition)
    current_executor = load_approved_remote_profiles(worker).get(
        (binding.compute_profile_id, binding.target_contract_key)
    )
    current_source = load_approved_source_profiles(worker).get(binding.source_repository_profile_id)
    if (
        current_executor != activation.executor_profile
        or current_source != activation.source_profile
    ):
        raise AutoResearchOnboardingConflict("current_preparation_approval_conflict")
    expected_key = scheduler_lease_key(root)
    if get_secret(contract.controller_lease_key_ref) != expected_key:
        raise AutoResearchOnboardingConflict("current_controller_lease_conflict")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        if contract.installation_id == profile.get("installation_id"):
            key = _require(
                get_secret("BASHGYM_CAMPAIGN_SEAL_KEY"),
                "installation",
                "existing_setup_seal_required",
            )
            validate_installation_transition(
                connection,
                installation_id=contract.installation_id,
                controller_owner_id=contract.controller_owner_id,
                workspace_id=contract.workspace_id,
                expected_key=expected_key,
                sealer=ArtifactSealer(key.encode(), key_version="campaign-seal-v1"),
            )
        else:
            row = connection.execute(
                "SELECT controller_owner_id, controller_lease_key FROM campaign_recovery_installations WHERE installation_id=?",
                (contract.installation_id,),
            ).fetchone()
            if row != (contract.controller_owner_id, expected_key):
                raise AutoResearchOnboardingConflict("registered_installation_authority_conflict")
    ledger = ExperimentLedgerRepository.open_existing(database)
    workspace = contract.workspace_id
    project = activation.project.project_id
    current_records = (
        (activation.project, ledger.get_project(workspace, project)),
        (activation.dataset, ledger.get_dataset(workspace, project, activation.dataset.dataset_id)),
        (
            activation.dataset_version,
            ledger.get_dataset_version(
                workspace, project, activation.dataset_version.dataset_version_id
            ),
        ),
        (
            activation.evaluation_suite,
            ledger.get_evaluation_suite(
                workspace, project, activation.evaluation_suite.evaluation_suite_id
            ),
        ),
    )
    if any(_spec(type(expected), current) != expected for expected, current in current_records):
        raise AutoResearchOnboardingConflict("current_ledger_approval_conflict")
