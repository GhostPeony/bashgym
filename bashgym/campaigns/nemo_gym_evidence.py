"""Campaign-native, sealed evidence for optional NeMo Gym rollouts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import (
    ActionAttempt,
    FrozenContractModel,
    GitObjectId,
    HexDigest,
    Identifier,
    canonical_hash,
)
from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.nemo_gym import validate_nemo_gym_rollout_batch

NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME = "nemo_gym_campaign_evidence.json"
NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA = "nemo_gym_campaign_evidence.v1"
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
TokenId = Annotated[int, Field(strict=True, ge=0)]


class NemoGymBundleFile(FrozenContractModel):
    path: str = Field(min_length=1, max_length=4096)
    sha256: HexDigest
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("NeMo Gym bundle file path must be a safe relative path")
        return value


class NemoGymBundleIdentity(FrozenContractModel):
    schema_version: Literal["bashgym_nemo_gym_bundle.v1"] = (
        "bashgym_nemo_gym_bundle.v1"
    )
    bashgym_source_revision: GitObjectId
    dataset_digest: HexDigest
    dataset_license: str = Field(min_length=1, max_length=128)
    environment_digest: HexDigest
    environment_id: Identifier
    files: tuple[NemoGymBundleFile, ...] = Field(min_length=1, max_length=10000)
    nemo_gym_source_revision: GitObjectId
    resources_server_id: Identifier
    verified: Literal[False] = False
    bundle_digest: HexDigest

    @field_validator("dataset_license")
    @classmethod
    def single_line_license(cls, value: str) -> str:
        if value != value.strip() or any(character in "\r\n\x00" for character in value):
            raise ValueError("NeMo Gym dataset license must be one explicit line")
        return value

    @field_validator("files")
    @classmethod
    def canonical_files(
        cls, value: tuple[NemoGymBundleFile, ...]
    ) -> tuple[NemoGymBundleFile, ...]:
        paths = tuple(item.path for item in value)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("NeMo Gym bundle files must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_bundle_digest(self) -> NemoGymBundleIdentity:
        identity = self.model_dump(mode="json", exclude={"bundle_digest"})
        if canonical_hash(identity) != self.bundle_digest:
            raise ValueError("NeMo Gym bundle digest mismatch")
        return self


class NemoGymMessageTokenReceipt(FrozenContractModel):
    item_id: Identifier
    prompt_token_ids: tuple[TokenId, ...] = Field(min_length=1, max_length=131072)
    generation_token_ids: tuple[TokenId, ...] = Field(min_length=1, max_length=131072)
    generation_log_probs: tuple[float, ...] = Field(min_length=1, max_length=131072)

    @field_validator("generation_log_probs")
    @classmethod
    def finite_log_probs(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("NeMo Gym generation logprobs must be finite")
        return value

    @model_validator(mode="after")
    def aligned_generation_evidence(self) -> NemoGymMessageTokenReceipt:
        if len(self.generation_token_ids) != len(self.generation_log_probs):
            raise ValueError("NeMo Gym generation token IDs and logprobs must align")
        return self


class NemoGymRefitEvidence(FrozenContractModel):
    refit_id: Identifier
    training_step: int = Field(ge=0)
    source_checkpoint_sha256: HexDigest
    policy_revision: int = Field(ge=0)
    generation_revision: int = Field(ge=0)
    synchronized: Literal[True] = True

    @model_validator(mode="after")
    def exact_generation_revision(self) -> NemoGymRefitEvidence:
        if self.generation_revision != self.policy_revision:
            raise ValueError("NeMo Gym generation revision does not match policy revision")
        return self


class NemoGymRolloutReceipt(FrozenContractModel):
    session_id: Identifier
    example_index: int = Field(ge=0)
    environment_id: Identifier
    environment_digest: HexDigest
    message_tokens: tuple[NemoGymMessageTokenReceipt, ...] = Field(
        min_length=1, max_length=4096
    )
    reward_components: dict[Identifier, float] = Field(min_length=1, max_length=100)
    total_reward: float
    refit: NemoGymRefitEvidence

    @field_validator("message_tokens")
    @classmethod
    def unique_message_ids(
        cls, value: tuple[NemoGymMessageTokenReceipt, ...]
    ) -> tuple[NemoGymMessageTokenReceipt, ...]:
        ids = tuple(item.item_id for item in value)
        if len(set(ids)) != len(ids):
            raise ValueError("NeMo Gym message item IDs must be unique within a rollout")
        return value

    @field_validator("reward_components")
    @classmethod
    def finite_sorted_components(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("NeMo Gym reward components must be finite")
        return dict(sorted(value.items()))

    @field_validator("total_reward")
    @classmethod
    def finite_total_reward(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("NeMo Gym total reward must be finite")
        return value


class NemoGymCampaignEvidence(FrozenContractModel):
    """Replay-complete Gym evidence bound to one exact campaign attempt."""

    schema_version: Literal["nemo_gym_campaign_evidence.v1"] = (
        NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA
    )
    workspace_id: Identifier
    campaign_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    manifest_revision: int = Field(ge=1)
    claim_generation: int = Field(ge=1)
    candidate_digest: HexDigest
    token_source: Literal["nemo_gym_model_server_message_fields"] = (
        "nemo_gym_model_server_message_fields"
    )
    environment_contract: dict[str, Any]
    bundle: NemoGymBundleIdentity
    rollouts: tuple[NemoGymRolloutReceipt, ...] = Field(min_length=1, max_length=4096)
    rollout_batch_digest: HexDigest = ""
    token_evidence_digest: HexDigest = ""
    refit_receipt_digest: HexDigest = ""
    mean_total_reward: float
    refit_synchronization_verified: Literal[True] = True
    evidence_digest: HexDigest = ""

    @model_validator(mode="after")
    def validate_complete_evidence(self) -> NemoGymCampaignEvidence:
        environment = EnvironmentSpec.from_dict(self.environment_contract)
        errors = environment.validation_errors()
        if errors:
            raise ValueError("invalid NeMo Gym environment contract: " + "; ".join(errors))
        environment_digest = canonical_hash(self.environment_contract)
        if (
            environment.id != self.bundle.environment_id
            or environment_digest != self.bundle.environment_digest
        ):
            raise ValueError("NeMo Gym environment contract does not match the bundle")

        sessions = tuple(rollout.session_id for rollout in self.rollouts)
        indexes = tuple(rollout.example_index for rollout in self.rollouts)
        if len(set(sessions)) != len(sessions):
            raise ValueError("NeMo Gym rollout session IDs must be unique")
        if tuple(sorted(set(indexes))) != indexes:
            raise ValueError("NeMo Gym rollout example indexes must be sorted and unique")
        refit = self.rollouts[0].refit
        for rollout in self.rollouts:
            if (
                rollout.environment_id != environment.id
                or rollout.environment_digest != environment_digest
            ):
                raise ValueError("NeMo Gym rollout environment binding mismatch")
            expected_reward = environment.verifier.combine_reward_components(
                rollout.reward_components
            )
            if not math.isclose(
                rollout.total_reward, expected_reward, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("NeMo Gym weighted reward total mismatch")
            if rollout.refit != refit:
                raise ValueError("NeMo Gym rollout batch spans multiple refit receipts")

        rollout_payload = [item.model_dump(mode="json") for item in self.rollouts]
        expected_batch_digest = canonical_hash(rollout_payload)
        token_payload = [
            {
                "example_index": rollout.example_index,
                "message_tokens": [
                    item.model_dump(mode="json") for item in rollout.message_tokens
                ],
            }
            for rollout in self.rollouts
        ]
        expected_token_digest = canonical_hash(token_payload)
        expected_refit_digest = canonical_hash(refit.model_dump(mode="json"))
        expected_mean = sum(item.total_reward for item in self.rollouts) / len(self.rollouts)
        expected = {
            "rollout_batch_digest": expected_batch_digest,
            "token_evidence_digest": expected_token_digest,
            "refit_receipt_digest": expected_refit_digest,
        }
        for field, digest in expected.items():
            current = getattr(self, field)
            if current and current != digest:
                raise ValueError(f"NeMo Gym {field.replace('_', ' ')} mismatch")
            if not current:
                object.__setattr__(self, field, digest)
        if not math.isfinite(self.mean_total_reward) or not math.isclose(
            self.mean_total_reward, expected_mean, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("NeMo Gym mean total reward mismatch")

        evidence_payload = self.model_dump(mode="json", exclude={"evidence_digest"})
        expected_evidence_digest = canonical_hash(evidence_payload)
        if self.evidence_digest and self.evidence_digest != expected_evidence_digest:
            raise ValueError("NeMo Gym campaign evidence digest mismatch")
        if not self.evidence_digest:
            object.__setattr__(self, "evidence_digest", expected_evidence_digest)
        return self

    def assert_attempt(self, attempt: ActionAttempt) -> None:
        expected = {
            "workspace_id": attempt.workspace_id,
            "campaign_id": attempt.campaign_id,
            "study_id": attempt.study_id,
            "action_id": attempt.action_id,
            "attempt_id": attempt.attempt_id,
            "manifest_revision": attempt.manifest_revision,
            "claim_generation": attempt.claim_generation,
            "candidate_digest": attempt.candidate_digest,
        }
        if any(getattr(self, field) != value for field, value in expected.items()):
            raise ValueError("NeMo Gym campaign evidence does not match the action attempt")

    def bounded_reference(self, *, artifact_id: str, artifact_sha256: str) -> dict[str, Any]:
        refit = self.rollouts[0].refit
        return {
            "artifact_id": artifact_id,
            "artifact_sha256": artifact_sha256,
            "bundle_digest": self.bundle.bundle_digest,
            "environment_id": self.bundle.environment_id,
            "environment_digest": self.bundle.environment_digest,
            "rollout_batch_digest": self.rollout_batch_digest,
            "token_evidence_digest": self.token_evidence_digest,
            "refit_receipt_digest": self.refit_receipt_digest,
            "rollout_count": len(self.rollouts),
            "mean_total_reward": self.mean_total_reward,
            "training_step": refit.training_step,
            "policy_revision": refit.policy_revision,
        }


def build_nemo_gym_campaign_evidence(
    attempt: ActionAttempt,
    *,
    bundle_manifest: Mapping[str, Any],
    environment: EnvironmentSpec,
    rollout_payloads: Sequence[Mapping[str, Any]],
) -> NemoGymCampaignEvidence:
    """Validate Gym output and bind it to one durable campaign attempt."""

    rollouts = validate_nemo_gym_rollout_batch(rollout_payloads, environment)
    receipts = tuple(
        NemoGymRolloutReceipt(
            session_id=rollout.session_id,
            example_index=rollout.example_index,
            environment_id=rollout.environment_id,
            environment_digest=rollout.environment_digest,
            message_tokens=tuple(
                NemoGymMessageTokenReceipt(
                    item_id=item.item_id,
                    prompt_token_ids=item.prompt_token_ids,
                    generation_token_ids=item.generation_token_ids,
                    generation_log_probs=item.generation_log_probs,
                )
                for item in rollout.message_tokens
            ),
            reward_components=rollout.reward_components,
            total_reward=rollout.total_reward,
            refit=NemoGymRefitEvidence(
                refit_id=rollout.refit.refit_id,
                training_step=rollout.refit.training_step,
                source_checkpoint_sha256=rollout.refit.source_checkpoint_sha256,
                policy_revision=rollout.refit.policy_revision,
                generation_revision=rollout.refit.generation_revision,
                synchronized=rollout.refit.synchronized,
            ),
        )
        for rollout in rollouts
    )
    return NemoGymCampaignEvidence(
        workspace_id=attempt.workspace_id,
        campaign_id=attempt.campaign_id,
        study_id=attempt.study_id,
        action_id=attempt.action_id,
        attempt_id=attempt.attempt_id,
        manifest_revision=attempt.manifest_revision,
        claim_generation=attempt.claim_generation,
        candidate_digest=attempt.candidate_digest,
        environment_contract=environment.to_dict(),
        bundle=NemoGymBundleIdentity.model_validate(dict(bundle_manifest)),
        rollouts=receipts,
        mean_total_reward=sum(item.total_reward for item in receipts) / len(receipts),
    )


def write_nemo_gym_campaign_evidence(
    path: Path, evidence: NemoGymCampaignEvidence
) -> Path:
    """Write canonical JSON without accepting an existing or symlinked destination."""

    destination = path.resolve()
    if path.is_symlink() or destination.exists():
        raise FileExistsError("NeMo Gym campaign evidence destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            evidence.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return destination


def load_nemo_gym_campaign_evidence(
    path: Path, *, expected_attempt: ActionAttempt | None = None
) -> NemoGymCampaignEvidence:
    """Load bounded evidence and optionally enforce the exact attempt binding."""

    candidate = path.resolve()
    if path.is_symlink() or not candidate.is_file():
        raise ValueError("NeMo Gym campaign evidence must be a regular file")
    if candidate.stat().st_size > _MAX_EVIDENCE_BYTES:
        raise ValueError("NeMo Gym campaign evidence exceeds the size limit")
    try:
        evidence = NemoGymCampaignEvidence.model_validate_json(
            candidate.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("NeMo Gym campaign evidence is invalid") from exc
    if expected_attempt is not None:
        evidence.assert_attempt(expected_attempt)
    return evidence


__all__ = [
    "NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME",
    "NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA",
    "NemoGymBundleIdentity",
    "NemoGymCampaignEvidence",
    "NemoGymMessageTokenReceipt",
    "NemoGymRefitEvidence",
    "NemoGymRolloutReceipt",
    "build_nemo_gym_campaign_evidence",
    "load_nemo_gym_campaign_evidence",
    "write_nemo_gym_campaign_evidence",
]
