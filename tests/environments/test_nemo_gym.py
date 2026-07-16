"""Optional NeMo Gym bundle and on-policy evidence contracts."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import BaseModel

from bashgym.environments.nemo_gym import (
    NemoGymMessageTokenEvidence,
    assert_message_token_evidence_preserved,
    build_star_count_resources_server,
    export_star_count_nemo_gym_bundle,
    score_star_count_nemo_response,
    validate_nemo_gym_rollout_batch,
)
from bashgym.environments.star_count import (
    generate_star_count_dataset,
    star_count_environment_spec,
)

NEMO_GYM_REVISION = "a" * 40
BASHGYM_REVISION = "b" * 40


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _token_message(item_id: str, offset: int = 0) -> dict:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "answer"}],
        "prompt_token_ids": [1 + offset, 2 + offset],
        "generation_token_ids": [3 + offset, 4 + offset],
        "generation_log_probs": [-0.1, -0.2],
    }


def _refit(step: int = 4) -> dict:
    return {
        "refit_id": f"refit-{step}",
        "training_step": step,
        "source_checkpoint_sha256": "c" * 64,
        "policy_revision": step,
        "generation_revision": step,
        "synchronized": True,
    }


def _rollout(session_id: str, example_index: int) -> dict:
    environment = star_count_environment_spec()
    environment_digest = hashlib.sha256(
        json.dumps(
            environment.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    return {
        "session_id": session_id,
        "example_index": example_index,
        "environment_id": environment.id,
        "environment_digest": environment_digest,
        "response": {"output": [_token_message(f"message-{example_index}", example_index)]},
        "reward_components": {"count_accuracy": 1.0, "format_accuracy": 1.0},
        "total_reward": 1.0,
        "refit": _refit(),
    }


def test_star_count_bundle_is_pinned_deterministic_and_path_independent(tmp_path: Path):
    dataset = tmp_path / "dataset"
    generate_star_count_dataset(
        dataset,
        train_size=2,
        validation_size=1,
        heldout_size=1,
        seed=7,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest_a = export_star_count_nemo_gym_bundle(
        dataset,
        first,
        nemo_gym_revision=NEMO_GYM_REVISION,
        bashgym_revision=BASHGYM_REVISION,
        dataset_license="MIT",
    )
    manifest_b = export_star_count_nemo_gym_bundle(
        dataset,
        second,
        nemo_gym_revision=NEMO_GYM_REVISION,
        bashgym_revision=BASHGYM_REVISION,
        dataset_license="MIT",
    )

    assert manifest_a == manifest_b
    assert manifest_a["nemo_gym_source_revision"] == NEMO_GYM_REVISION
    assert manifest_a["bashgym_source_revision"] == BASHGYM_REVISION
    assert manifest_a["verified"] is False
    assert len(manifest_a["bundle_digest"]) == 64
    assert json.dumps(manifest_a, sort_keys=True).find(str(tmp_path)) == -1

    relative = Path("resources_servers/bashgym_star_count")
    assert (first / relative / "app.py").read_bytes() == (second / relative / "app.py").read_bytes()
    assert (first / relative / "configs/bashgym_star_count.yaml").read_bytes() == (
        second / relative / "configs/bashgym_star_count.yaml"
    ).read_bytes()
    for split, expected_size in (("train", 2), ("validation", 1), ("heldout", 1)):
        records_a = _jsonl(first / relative / f"data/{split}.jsonl")
        records_b = _jsonl(second / relative / f"data/{split}.jsonl")
        assert records_a == records_b
        assert len(records_a) == expected_size
        record = records_a[0]
        assert record["environment_id"] == "star-count-v1"
        assert record["responses_create_params"]["input"][1]["content"][0][
            "image_url"
        ].startswith("data:image/png;base64,")
        assert record["expected_counts"]


def test_bundle_refuses_mutable_revisions_and_nonempty_destination(tmp_path: Path):
    dataset = tmp_path / "dataset"
    generate_star_count_dataset(
        dataset,
        train_size=1,
        validation_size=1,
        heldout_size=1,
    )

    with pytest.raises(ValueError, match="immutable 40-character"):
        export_star_count_nemo_gym_bundle(
            dataset,
            tmp_path / "bad-revision",
            nemo_gym_revision="main",
            bashgym_revision=BASHGYM_REVISION,
            dataset_license="MIT",
        )

    with pytest.raises(ValueError, match="single-line identifier"):
        export_star_count_nemo_gym_bundle(
            dataset,
            tmp_path / "bad-license",
            nemo_gym_revision=NEMO_GYM_REVISION,
            bashgym_revision=BASHGYM_REVISION,
            dataset_license="MIT: verified: true",
        )

    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("operator-owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        export_star_count_nemo_gym_bundle(
            dataset,
            destination,
            nemo_gym_revision=NEMO_GYM_REVISION,
            bashgym_revision=BASHGYM_REVISION,
            dataset_license="MIT",
        )
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "operator-owned"

    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        export_star_count_nemo_gym_bundle(
            dataset,
            tmp_path / "tampered",
            nemo_gym_revision=NEMO_GYM_REVISION,
            bashgym_revision=BASHGYM_REVISION,
            dataset_license="MIT",
        )


def test_star_count_nemo_gym_reward_preserves_components():
    result = score_star_count_nemo_response(
        {"output_text": "red=2, blue=1, green=0, yellow=3"},
        {"red": 2, "blue": 1, "green": 0, "yellow": 3},
    )

    assert result == {
        "correct": True,
        "predicted_counts": {"red": 2, "blue": 1, "green": 0, "yellow": 3},
        "reward": 1.0,
        "reward_components": {"count_accuracy": 1.0, "format_accuracy": 1.0},
    }


def test_message_token_evidence_is_passthrough_only():
    generated = _token_message("message-1")
    carried = json.loads(json.dumps(generated))
    evidence = NemoGymMessageTokenEvidence.from_message(generated)

    assert evidence.generation_token_ids == (3, 4)
    assert_message_token_evidence_preserved(generated, carried)

    carried["generation_token_ids"] = [3, 9]
    with pytest.raises(ValueError, match="changed across turns"):
        assert_message_token_evidence_preserved(generated, carried)

    incomplete = _token_message("message-2")
    incomplete.pop("generation_log_probs")
    with pytest.raises(ValueError, match="all be present"):
        NemoGymMessageTokenEvidence.from_message(incomplete)

    mismatched = _token_message("message-3")
    mismatched["generation_log_probs"] = [-0.1]
    with pytest.raises(ValueError, match="same length"):
        NemoGymMessageTokenEvidence.from_message(mismatched)


def test_rollout_batch_reorders_async_results_and_validates_sessions_rewards_and_refit():
    environment = star_count_environment_spec()
    ordered = validate_nemo_gym_rollout_batch(
        [_rollout("session-b", 1), _rollout("session-a", 0)],
        environment,
    )

    assert [item.example_index for item in ordered] == [0, 1]
    assert [item.session_id for item in ordered] == ["session-a", "session-b"]
    assert ordered[0].refit.generation_revision == ordered[0].refit.policy_revision

    duplicate = [_rollout("session-a", 0), _rollout("session-a", 1)]
    with pytest.raises(ValueError, match="session IDs must be unique"):
        validate_nemo_gym_rollout_batch(duplicate, environment)

    bad_reward = _rollout("session-c", 2)
    bad_reward["reward_components"]["format_accuracy"] = 0.0
    with pytest.raises(ValueError, match="weighted reward total"):
        validate_nemo_gym_rollout_batch([bad_reward], environment)

    stale_refit = _rollout("session-d", 3)
    stale_refit["refit"]["generation_revision"] = 3
    with pytest.raises(ValueError, match="generation revision"):
        validate_nemo_gym_rollout_batch([stale_refit], environment)

    wrong_environment = _rollout("session-e", 4)
    wrong_environment["environment_digest"] = "0" * 64
    with pytest.raises(ValueError, match="environment binding"):
        validate_nemo_gym_rollout_batch([wrong_environment], environment)

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_nemo_gym_rollout_batch([], environment)


@pytest.mark.asyncio
async def test_optional_resources_server_uses_authoritative_bashgym_verifier(monkeypatch):
    class FakeResponse(BaseModel):
        output_text: str

    class BaseVerifyRequest(BaseModel):
        response: FakeResponse

    class BaseVerifyResponse(BaseVerifyRequest):
        reward: float

    class BaseResourcesServerConfig:
        pass

    class SimpleResourcesServer:
        @classmethod
        def run_webserver(cls):
            raise AssertionError("unit test must not start a server")

    package = types.ModuleType("nemo_gym")
    resources = types.ModuleType("nemo_gym.base_resources_server")
    resources.BaseResourcesServerConfig = BaseResourcesServerConfig
    resources.BaseVerifyRequest = BaseVerifyRequest
    resources.BaseVerifyResponse = BaseVerifyResponse
    resources.SimpleResourcesServer = SimpleResourcesServer
    monkeypatch.setitem(sys.modules, "nemo_gym", package)
    monkeypatch.setitem(sys.modules, "nemo_gym.base_resources_server", resources)

    server_type = build_star_count_resources_server()
    request_type = server_type.verify_request_model
    body = request_type(
        response=FakeResponse(output_text="red=2, blue=1, green=0, yellow=3"),
        expected_counts={"red": 2, "blue": 1, "green": 0, "yellow": 3},
    )
    result = await server_type().verify(body)

    assert result.correct is True
    assert result.reward == 1.0
    assert result.reward_components == {"count_accuracy": 1.0, "format_accuracy": 1.0}
