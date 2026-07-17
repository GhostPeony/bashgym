"""Optional NeMo Gym adapters for BashGym environment and rollout evidence.

The module deliberately has no NeMo Gym import at module load time. Bundle
generation and evidence validation stay available in a normal BashGym install;
only the generated resources-server entrypoint requires NeMo Gym at runtime.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.star_count import (
    canonical_star_count_answer,
    score_star_count_prediction,
    star_count_environment_spec,
)

_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_LICENSE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .()+/_-]{0,127}$")
_TOKEN_FIELDS = (
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
)
_RESOURCE_ROOT = PurePosixPath("resources_servers/bashgym_star_count")
_MAX_BUNDLE_FILES = 10_000
_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
_APP_SOURCE = """\
\"\"\"NeMo Gym entrypoint for the pinned BashGym star-count environment.\"\"\"

from bashgym.environments.nemo_gym import run_star_count_resources_server


if __name__ == \"__main__\":
    run_star_count_resources_server()
"""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _immutable_revision(value: str, *, label: str) -> str:
    normalized = str(value).strip().casefold()
    if not _GIT_REVISION.fullmatch(normalized):
        raise ValueError(f"{label} must be an immutable 40-character git revision")
    return normalized


def _dataset_license(value: str) -> str:
    normalized = str(value).strip()
    if not _LICENSE_IDENTIFIER.fullmatch(normalized):
        raise ValueError("dataset license must be one explicit single-line identifier")
    return normalized


def _environment_digest(environment: EnvironmentSpec) -> str:
    errors = environment.validation_errors()
    if errors:
        raise ValueError("invalid BashGym environment: " + "; ".join(errors))
    return _canonical_hash(environment.to_dict())


def _load_verified_star_count_dataset(dataset_directory: Path) -> dict[str, Any]:
    candidate = dataset_directory.expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("star-count dataset must be a regular directory")
    root = candidate.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("star-count dataset must contain a regular manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "star_count_dataset.v1":
        raise ValueError("unsupported star-count dataset manifest")
    manifest_identity = {key: value for key, value in manifest.items() if key != "dataset_digest"}
    if _canonical_hash(manifest_identity) != manifest.get("dataset_digest"):
        raise ValueError("star-count dataset manifest digest mismatch")
    file_records = manifest.get("files")
    if not isinstance(file_records, list) or not file_records:
        raise ValueError("star-count manifest requires a non-empty file inventory")
    expected_paths: set[str] = set()
    for record in file_records:
        if not isinstance(record, Mapping):
            raise ValueError("star-count manifest file records must be objects")
        relative = PurePosixPath(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("star-count manifest contains an unsafe file path")
        normalized_path = relative.as_posix()
        if normalized_path in expected_paths:
            raise ValueError("star-count manifest contains duplicate file paths")
        expected_paths.add(normalized_path)
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("star-count manifest file is missing or not regular")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError("star-count manifest file size mismatch")
        if _sha256_file(path) != record.get("sha256"):
            raise ValueError("star-count manifest file digest mismatch")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != expected_paths:
        raise ValueError("star-count manifest inventory does not exactly bind dataset files")
    for split in ("train", "validation", "heldout"):
        if not (root / f"{split}.jsonl").is_file():
            raise ValueError(f"star-count dataset is missing {split}.jsonl")
    return {"root": root, "manifest": manifest}


def _image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _nemo_gym_star_count_record(
    record: Mapping[str, Any],
    *,
    dataset_root: Path,
    environment_digest: str,
    expected_split: str,
) -> dict[str, Any]:
    relative_image = PurePosixPath(str(record.get("image", "")))
    if relative_image.is_absolute() or ".." in relative_image.parts or not relative_image.parts:
        raise ValueError("star-count record contains an unsafe image path")
    image_path = dataset_root.joinpath(*relative_image.parts)
    if image_path.is_symlink() or not image_path.is_file():
        raise ValueError("star-count record image is missing or not regular")
    expected_counts = dict(record.get("counts") or {})
    expected_answer = canonical_star_count_answer(expected_counts)
    if record.get("answer") != expected_answer:
        raise ValueError("star-count record answer does not match its counts")
    prompt = str(record.get("prompt", ""))
    if not prompt:
        raise ValueError("star-count record prompt is required")
    example_id = str(record.get("example_id", ""))
    if not _IDENTIFIER.fullmatch(example_id):
        raise ValueError("star-count record requires a stable example ID")
    if record.get("split") != expected_split:
        raise ValueError("star-count record split does not match its dataset file")
    return {
        "environment_digest": environment_digest,
        "environment_id": "star-count-v1",
        "example_id": example_id,
        "expected_answer": expected_answer,
        "expected_counts": expected_counts,
        "responses_create_params": {
            "input": [
                {
                    "role": "developer",
                    "content": (
                        "Count the colored stars in the supplied image and return only "
                        "the exact requested red, blue, green, yellow format."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": _image_data_url(image_path),
                            "detail": "auto",
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                },
            ],
            "max_output_tokens": 64,
            "parallel_tool_calls": False,
        },
        "split": expected_split,
    }


def _config_text(*, dataset_license: str) -> str:
    return f"""\
bashgym_star_count:
  resources_servers:
    bashgym_star_count:
      entrypoint: app.py
      domain: other
      verified: false
      description: Deterministic BashGym four-color star counting
      value: Improve visual counting and exact structured output
bashgym_star_count_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      max_steps: 1
      resources_server:
        type: resources_servers
        name: bashgym_star_count
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/bashgym_star_count/data/train.jsonl
        license: {dataset_license}
      - name: validation
        type: validation
        jsonl_fpath: resources_servers/bashgym_star_count/data/validation.jsonl
        license: {dataset_license}
      - name: heldout
        type: validation
        jsonl_fpath: resources_servers/bashgym_star_count/data/heldout.jsonl
        license: {dataset_license}
"""


def export_star_count_nemo_gym_bundle(
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    nemo_gym_revision: str,
    bashgym_revision: str,
    dataset_license: str,
) -> dict[str, Any]:
    """Export one deterministic, path-independent NeMo Gym resources bundle."""

    nemo_revision = _immutable_revision(nemo_gym_revision, label="NeMo Gym revision")
    bashgym_source_revision = _immutable_revision(bashgym_revision, label="BashGym revision")
    license_id = _dataset_license(dataset_license)
    loaded = _load_verified_star_count_dataset(Path(dataset_directory))
    dataset_root: Path = loaded["root"]
    dataset_manifest: dict[str, Any] = loaded["manifest"]
    environment = star_count_environment_spec()
    environment_digest = _environment_digest(environment)

    split_payloads: dict[str, str] = {}
    example_ids: set[str] = set()
    for split in ("train", "validation", "heldout"):
        source = dataset_root / f"{split}.jsonl"
        records = [
            _nemo_gym_star_count_record(
                json.loads(line),
                dataset_root=dataset_root,
                environment_digest=environment_digest,
                expected_split=split,
            )
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"star-count {split} split cannot be empty")
        split_ids = {record["example_id"] for record in records}
        if len(split_ids) != len(records) or example_ids.intersection(split_ids):
            raise ValueError("star-count example IDs must be unique across splits")
        example_ids.update(split_ids)
        split_payloads[split] = "".join(_canonical_json(record) + "\n" for record in records)

    destination_candidate = Path(output_directory).expanduser()
    if destination_candidate.is_symlink():
        raise FileExistsError("NeMo Gym bundle destination cannot be a symlink")
    destination = destination_candidate.resolve()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError("NeMo Gym bundle destination is not a regular directory")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("NeMo Gym bundle destination is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    resources_root = destination.joinpath(*_RESOURCE_ROOT.parts)
    data_root = resources_root / "data"
    config_root = resources_root / "configs"
    data_root.mkdir(parents=True)
    config_root.mkdir(parents=True)

    for split, payload in split_payloads.items():
        (data_root / f"{split}.jsonl").write_text(
            payload,
            encoding="utf-8",
        )

    (resources_root / "app.py").write_text(_APP_SOURCE, encoding="utf-8")
    (config_root / "bashgym_star_count.yaml").write_text(
        _config_text(dataset_license=license_id), encoding="utf-8"
    )
    (destination / "environment_contract.json").write_text(
        _canonical_json(environment.to_dict()) + "\n",
        encoding="utf-8",
    )

    files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        relative = path.relative_to(destination).as_posix()
        files.append(
            {"path": relative, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
        )
    identity = {
        "schema_version": "bashgym_nemo_gym_bundle.v1",
        "bashgym_source_revision": bashgym_source_revision,
        "dataset_digest": dataset_manifest["dataset_digest"],
        "dataset_license": license_id,
        "environment_digest": environment_digest,
        "environment_id": environment.id,
        "files": files,
        "nemo_gym_source_revision": nemo_revision,
        "resources_server_id": "bashgym_star_count",
        "verified": False,
    }
    manifest = {**identity, "bundle_digest": _canonical_hash(identity)}
    (destination / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _validated_bundle_members(archive_path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    archive = archive_path.expanduser()
    if archive.is_symlink() or not archive.is_file() or archive.suffix.casefold() != ".zip":
        raise ValueError("NeMo Gym bundle archive must be a regular ZIP file")
    members: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(archive.resolve()) as bundle:
            entries = bundle.infolist()
            if len(entries) > _MAX_BUNDLE_FILES:
                raise ValueError("NeMo Gym bundle archive contains too many files")
            if sum(entry.file_size for entry in entries) > _MAX_BUNDLE_BYTES:
                raise ValueError("NeMo Gym bundle archive exceeds the extraction limit")
            for entry in entries:
                relative = PurePosixPath(entry.filename.replace("\\", "/"))
                mode = entry.external_attr >> 16
                if (
                    entry.is_dir()
                    or relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or stat.S_ISLNK(mode)
                    or relative.as_posix() in members
                ):
                    raise ValueError("NeMo Gym bundle archive contains an unsafe path")
                members[relative.as_posix()] = bundle.read(entry)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("NeMo Gym bundle archive is invalid") from exc

    try:
        manifest = json.loads(members["bundle_manifest.json"].decode("utf-8"))
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("NeMo Gym bundle archive has no valid manifest") from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != "bashgym_nemo_gym_bundle.v1"
    ):
        raise ValueError("unsupported NeMo Gym bundle manifest")
    identity = {key: value for key, value in manifest.items() if key != "bundle_digest"}
    if _canonical_hash(identity) != manifest.get("bundle_digest"):
        raise ValueError("NeMo Gym bundle manifest digest mismatch")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("NeMo Gym bundle manifest requires a file inventory")
    expected_paths: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("NeMo Gym bundle file records must be objects")
        relative = PurePosixPath(str(record.get("path", "")))
        path = relative.as_posix()
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or path == "bundle_manifest.json"
        ):
            raise ValueError("NeMo Gym bundle manifest contains an unsafe file path")
        payload = members.get(path)
        if payload is None:
            raise ValueError("NeMo Gym bundle archive is missing a manifest file")
        if len(payload) != record.get("size_bytes"):
            raise ValueError("NeMo Gym bundle file size mismatch")
        if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
            raise ValueError("NeMo Gym bundle file digest mismatch")
        expected_paths.append(path)
    if expected_paths != sorted(set(expected_paths)):
        raise ValueError("NeMo Gym bundle file inventory must be sorted and unique")
    if set(members) != {"bundle_manifest.json", *expected_paths}:
        raise ValueError("NeMo Gym bundle archive contains unbound files")
    required = {
        "environment_contract.json",
        (_RESOURCE_ROOT / "app.py").as_posix(),
        (_RESOURCE_ROOT / "configs/bashgym_star_count.yaml").as_posix(),
        (_RESOURCE_ROOT / "data/train.jsonl").as_posix(),
        (_RESOURCE_ROOT / "data/validation.jsonl").as_posix(),
    }
    if not required.issubset(members):
        raise ValueError("NeMo Gym bundle archive is missing required runtime files")
    try:
        environment_payload = json.loads(members["environment_contract.json"].decode("utf-8"))
        environment = EnvironmentSpec.from_dict(environment_payload)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("NeMo Gym bundle environment contract is invalid") from exc
    if (
        environment.id != manifest.get("environment_id")
        or environment.validation_errors()
        or _canonical_hash(environment_payload) != manifest.get("environment_digest")
    ):
        raise ValueError("NeMo Gym bundle environment contract identity mismatch")
    return members, dict(manifest)


def create_nemo_gym_bundle_archive(
    bundle_directory: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Create one deterministic transport archive from an exported Gym bundle."""

    root_candidate = Path(bundle_directory).expanduser()
    if root_candidate.is_symlink() or not root_candidate.is_dir():
        raise ValueError("NeMo Gym bundle must be a regular directory")
    root = root_candidate.resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("NeMo Gym bundle archive destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=6)
    _members, manifest = _validated_bundle_members(destination)
    return {
        "schema_version": "bashgym_nemo_gym_bundle_archive.v1",
        "archive_sha256": _sha256_file(destination),
        "bundle_digest": manifest["bundle_digest"],
        "size_bytes": destination.stat().st_size,
    }


def extract_nemo_gym_bundle_archive(
    archive_path: str | Path, destination: str | Path
) -> dict[str, Any]:
    """Validate and safely extract one exact Gym bundle archive."""

    members, manifest = _validated_bundle_members(Path(archive_path))
    root = Path(destination).expanduser().resolve()
    if root.exists() and (root.is_symlink() or not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("NeMo Gym bundle extraction destination is not empty")
    root.mkdir(parents=True, exist_ok=True)
    for relative, payload in members.items():
        target = root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return manifest


def inspect_nemo_gym_bundle_archive(archive_path: str | Path) -> dict[str, Any]:
    """Return the validated, path-independent identity of a Gym bundle archive."""

    _members, manifest = _validated_bundle_members(Path(archive_path))
    return manifest


def _response_output_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    fragments: list[str] = []
    for output in response.get("output") or []:
        if not isinstance(output, Mapping):
            continue
        for content in output.get("content") or []:
            if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                fragments.append(content["text"])
    return "\n".join(fragments)


def score_star_count_nemo_response(
    response: Mapping[str, Any], expected_counts: Mapping[str, Any]
) -> dict[str, Any]:
    """Score a NeMo Gym response with BashGym's authoritative component verifier."""

    score = score_star_count_prediction(_response_output_text(response), dict(expected_counts))
    components = score.reward_components()
    reward = star_count_environment_spec().verifier.combine_reward_components(components)
    return {
        "correct": score.exact,
        "predicted_counts": score.predicted_counts,
        "reward": reward,
        "reward_components": components,
    }


def _token_ids(value: Any, *, label: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a token ID sequence")
    result = tuple(value)
    if not allow_empty and not result:
        raise ValueError(f"{label} cannot be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result):
        raise ValueError(f"{label} must contain non-negative integer token IDs")
    return result


def _log_probs(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("generation_log_probs must be a numeric sequence")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError("generation_log_probs must be finite")
    return result


@dataclass(frozen=True)
class NemoGymMessageTokenEvidence:
    """Model-server token evidence copied from one generated message unchanged."""

    item_id: str
    prompt_token_ids: tuple[int, ...]
    generation_token_ids: tuple[int, ...]
    generation_log_probs: tuple[float, ...]

    @classmethod
    def from_message(cls, message: Mapping[str, Any]) -> NemoGymMessageTokenEvidence:
        present = {field for field in _TOKEN_FIELDS if field in message}
        if present != set(_TOKEN_FIELDS):
            raise ValueError("message-level token IDs and logprobs must all be present")
        item_id = str(message.get("id", ""))
        if not _IDENTIFIER.fullmatch(item_id):
            raise ValueError("token evidence requires a stable message item ID")
        prompt = _token_ids(message["prompt_token_ids"], label="prompt_token_ids")
        generation = _token_ids(message["generation_token_ids"], label="generation_token_ids")
        log_probs = _log_probs(message["generation_log_probs"])
        if len(generation) != len(log_probs):
            raise ValueError("generation token IDs and logprobs must have the same length")
        return cls(
            item_id=item_id,
            prompt_token_ids=prompt,
            generation_token_ids=generation,
            generation_log_probs=log_probs,
        )


def assert_message_token_evidence_preserved(
    generated_message: Mapping[str, Any], carried_message: Mapping[str, Any]
) -> None:
    """Reject any retokenization or mutation between generated and carried messages."""

    generated = NemoGymMessageTokenEvidence.from_message(generated_message)
    carried = NemoGymMessageTokenEvidence.from_message(carried_message)
    if generated != carried:
        raise ValueError("message-level token evidence changed across turns")


@dataclass(frozen=True)
class NemoGymRefitReceipt:
    refit_id: str
    training_step: int
    source_checkpoint_sha256: str
    policy_revision: int
    generation_revision: int
    synchronized: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NemoGymRefitReceipt:
        receipt = cls(
            refit_id=str(payload.get("refit_id", "")),
            training_step=int(payload.get("training_step", -1)),
            source_checkpoint_sha256=str(payload.get("source_checkpoint_sha256", "")),
            policy_revision=int(payload.get("policy_revision", -1)),
            generation_revision=int(payload.get("generation_revision", -1)),
            synchronized=payload.get("synchronized") is True,
        )
        if not _IDENTIFIER.fullmatch(receipt.refit_id):
            raise ValueError("refit receipt requires a stable refit ID")
        if receipt.training_step < 0 or receipt.policy_revision < 0:
            raise ValueError("refit training step and policy revision must be non-negative")
        if not _HEX_DIGEST.fullmatch(receipt.source_checkpoint_sha256):
            raise ValueError("refit source checkpoint must be an exact SHA-256 digest")
        if not receipt.synchronized:
            raise ValueError("refit receipt must confirm synchronization")
        if receipt.generation_revision != receipt.policy_revision:
            raise ValueError("refit generation revision does not match the policy revision")
        return receipt


@dataclass(frozen=True)
class NemoGymRolloutEvidence:
    session_id: str
    example_index: int
    environment_id: str
    environment_digest: str
    message_tokens: tuple[NemoGymMessageTokenEvidence, ...]
    reward_components: dict[str, float]
    total_reward: float
    refit: NemoGymRefitReceipt

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], environment: EnvironmentSpec
    ) -> NemoGymRolloutEvidence:
        session_id = str(payload.get("session_id", ""))
        if not _IDENTIFIER.fullmatch(session_id):
            raise ValueError("rollout evidence requires a stable session ID")
        example_index = int(payload.get("example_index", -1))
        if example_index < 0:
            raise ValueError("rollout example index must be non-negative")
        environment_id = str(payload.get("environment_id", ""))
        environment_digest = str(payload.get("environment_digest", ""))
        expected_environment_digest = _environment_digest(environment)
        if environment_id != environment.id or environment_digest != expected_environment_digest:
            raise ValueError("rollout environment binding does not match the expected environment")
        response = payload.get("response")
        if not isinstance(response, Mapping):
            raise ValueError("rollout evidence requires a response object")
        tokens = tuple(
            NemoGymMessageTokenEvidence.from_message(item)
            for item in response.get("output") or []
            if isinstance(item, Mapping) and any(field in item for field in _TOKEN_FIELDS)
        )
        if not tokens:
            raise ValueError("rollout evidence requires model-server message-level token IDs")
        raw_components = payload.get("reward_components")
        if not isinstance(raw_components, Mapping):
            raise ValueError("rollout evidence requires reward components")
        components = {str(key): float(value) for key, value in raw_components.items()}
        expected_total = environment.verifier.combine_reward_components(components)
        total_reward = float(payload.get("total_reward", math.nan))
        if not math.isfinite(total_reward) or not math.isclose(
            total_reward, expected_total, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("weighted reward total does not match reward components")
        raw_refit = payload.get("refit")
        if not isinstance(raw_refit, Mapping):
            raise ValueError("rollout evidence requires an exact refit receipt")
        return cls(
            session_id=session_id,
            example_index=example_index,
            environment_id=environment_id,
            environment_digest=environment_digest,
            message_tokens=tokens,
            reward_components=components,
            total_reward=total_reward,
            refit=NemoGymRefitReceipt.from_dict(raw_refit),
        )


def validate_nemo_gym_rollout_batch(
    payloads: Sequence[Mapping[str, Any]], environment: EnvironmentSpec
) -> tuple[NemoGymRolloutEvidence, ...]:
    """Validate and restore async rollout results to original example order."""

    rollouts = tuple(NemoGymRolloutEvidence.from_dict(payload, environment) for payload in payloads)
    if not rollouts:
        raise ValueError("rollout batch cannot be empty")
    sessions = [rollout.session_id for rollout in rollouts]
    if len(sessions) != len(set(sessions)):
        raise ValueError("rollout session IDs must be unique")
    indexes = [rollout.example_index for rollout in rollouts]
    if len(indexes) != len(set(indexes)):
        raise ValueError("rollout example indexes must be unique")
    return tuple(sorted(rollouts, key=lambda rollout: rollout.example_index))


def build_star_count_resources_server() -> type:
    """Build the NeMo Gym server class only inside an optional Gym runtime."""

    try:
        from nemo_gym.base_resources_server import (
            BaseResourcesServerConfig,
            BaseVerifyRequest,
            BaseVerifyResponse,
            SimpleResourcesServer,
        )
    except ImportError as exc:  # pragma: no cover - requires optional NeMo Gym runtime
        raise RuntimeError(
            "NeMo Gym is optional; run this entrypoint inside a pinned NeMo Gym environment"
        ) from exc

    class StarCountVerifyRequest(BaseVerifyRequest):
        expected_counts: dict[str, int]

    class StarCountVerifyResponse(BaseVerifyResponse):
        correct: bool
        predicted_counts: dict[str, int] | None
        reward_components: dict[str, float]

    class BashGymStarCountResourcesServer(SimpleResourcesServer):
        config: BaseResourcesServerConfig

        async def verify(self, body: StarCountVerifyRequest) -> StarCountVerifyResponse:
            result = score_star_count_nemo_response(
                body.response.model_dump(mode="json"), body.expected_counts
            )
            return StarCountVerifyResponse(
                **body.model_dump(),
                reward=result["reward"],
                correct=result["correct"],
                predicted_counts=result["predicted_counts"],
                reward_components=result["reward_components"],
            )

    BashGymStarCountResourcesServer.verify_request_model = StarCountVerifyRequest
    BashGymStarCountResourcesServer.verify_response_model = StarCountVerifyResponse
    return BashGymStarCountResourcesServer


def run_star_count_resources_server() -> None:
    build_star_count_resources_server().run_webserver()


__all__ = [
    "NemoGymMessageTokenEvidence",
    "NemoGymRefitReceipt",
    "NemoGymRolloutEvidence",
    "assert_message_token_evidence_preserved",
    "build_star_count_resources_server",
    "create_nemo_gym_bundle_archive",
    "extract_nemo_gym_bundle_archive",
    "export_star_count_nemo_gym_bundle",
    "inspect_nemo_gym_bundle_archive",
    "run_star_count_resources_server",
    "score_star_count_nemo_response",
    "validate_nemo_gym_rollout_batch",
]
