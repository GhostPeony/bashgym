"""One-submission NeMo Gym adapter for isolated coding/recovery evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from bashgym.environments.contracts import (
    BuildSpec,
    EnvironmentSpec,
    RewardComponentSpec,
    RolloutSpec,
    VerifierSpec,
)
from bashgym.environments.docker_coding import run_docker_environment_attempt, validate_pinned_image
from bashgym.environments.nemo_gym import (
    _canonical_hash,
    _immutable_revision,
    _response_output_text,
    _sha256_file,
)
from bashgym.environments.personal_coding import (
    PERSONAL_CODING_SPLITS,
    environment_content_digest,
    load_personal_coding_bundle,
    personal_coding_environment_specs,
)
from bashgym.environments.rollout import RolloutCommandPlan

SERVER_ID = "bashgym_personal_coding"


def export_personal_coding_nemo_gym_bundle(
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    nemo_gym_revision: str,
    bashgym_revision: str,
    sandbox_image: str,
) -> dict[str, Any]:
    """Export pinned data and a Docker verifier; confirmation remains opt-in."""
    nemo_revision = _immutable_revision(nemo_gym_revision, label="NeMo Gym revision")
    source_revision = _immutable_revision(bashgym_revision, label="BashGym revision")
    image = validate_pinned_image(sandbox_image)
    specs = load_personal_coding_bundle(dataset_directory)
    dataset = json.loads((Path(dataset_directory) / "manifest.json").read_text(encoding="utf-8"))
    destination = Path(output_directory)
    if destination.is_symlink() or (
        destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise FileExistsError("NeMo Gym destination must be an empty regular directory")
    environment = EnvironmentSpec(
        id="personal-coding-v1",
        instruction="Repair authored repositories and recover failed tool commands.",
        domain="coding",
        source="authored_synthetic",
        license="MIT",
        build=BuildSpec(dockerfile="", base_image=image),
        rollout=RolloutSpec(harness="bashgym-docker-coding-v1"),
        verifier=VerifierSpec(
            kind="coding_unittest",
            path="verify.py",
            reward_components=[RewardComponentSpec("verified_tests")],
        ),
        metadata={
            "dataset_digest": dataset["dataset_digest"],
            "environment_ids": [spec.id for spec in specs],
            "adapter": "one_submission_commands",
            "confirmation_opt_in": True,
        },
    )
    environment_digest = _canonical_hash(environment.to_dict())
    resource_root = destination / "resources_servers" / SERVER_ID
    (resource_root / "data").mkdir(parents=True)
    (resource_root / "configs").mkdir()
    for split in PERSONAL_CODING_SPLITS:
        records = []
        for spec in specs:
            if spec.metadata["split"] != split:
                continue
            visible = "\n\n".join(
                f"File: {path}\n{content}"
                for path, content in sorted(spec.files.items())
                if path != "verify.py"
            )
            records.append(
                {
                    "environment_id": environment.id,
                    "environment_digest": environment_digest,
                    "example_id": spec.id,
                    "split": split,
                    "environment_spec": spec.to_dict(),
                    "sandbox_image": image,
                    "responses_create_params": {
                        "input": [
                            {
                                "role": "developer",
                                "content": 'Return only JSON {"commands":["shell command", ...]} to repair the task. Commands run sequentially in a fresh offline container. Every command starts in /workspace and filesystem changes persist. Do not edit tests or the verifier.',
                            },
                            {"role": "user", "content": spec.instruction + "\n\n" + visible},
                        ],
                        "max_output_tokens": 2048,
                        "parallel_tool_calls": False,
                    },
                }
            )
        (resource_root / "data" / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records
            ),
            encoding="utf-8",
        )
    (resource_root / "app.py").write_text(
        'from bashgym.environments.personal_coding_nemo import run_personal_coding_resources_server\n\nif __name__ == "__main__":\n    run_personal_coding_resources_server()\n',
        encoding="utf-8",
    )
    config = f"""{SERVER_ID}:
  resources_servers:
    {SERVER_ID}:
      entrypoint: app.py
      domain: coding
      verified: false
      sandbox_image: {image}
      description: Authored repository repair and failed-tool recovery
      value: Evaluate code repair with protected tests in offline Docker episodes
{SERVER_ID}_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      max_steps: 1
      resources_server:
        type: resources_servers
        name: {SERVER_ID}
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/{SERVER_ID}/data/train.jsonl
        license: MIT
      - name: dev
        type: validation
        jsonl_fpath: resources_servers/{SERVER_ID}/data/dev.jsonl
        license: MIT
"""
    (resource_root / "configs" / f"{SERVER_ID}.yaml").write_text(config, encoding="utf-8")
    (destination / "environment_contract.json").write_text(
        json.dumps(environment.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
    )
    files = [
        {
            "path": path.relative_to(destination).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    identity = {
        "schema_version": "bashgym_nemo_gym_bundle.v1",
        "bashgym_source_revision": source_revision,
        "nemo_gym_source_revision": nemo_revision,
        "dataset_digest": dataset["dataset_digest"],
        "dataset_license": "MIT",
        "environment_digest": environment_digest,
        "environment_id": environment.id,
        "resources_server_id": SERVER_ID,
        "sandbox_image": image,
        "verified": False,
        "files": files,
    }
    manifest = {**identity, "bundle_digest": _canonical_hash(identity)}
    (destination / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def score_personal_coding_nemo_response(
    response, environment_spec, sandbox_image: str, *, client=None
):
    """Execute a candidate command list against the fixed authored task contract."""
    spec = EnvironmentSpec.from_dict(environment_spec)
    known = {item.id: item for item in personal_coding_environment_specs()}
    if (
        spec.id not in known
        or environment_content_digest(spec) != environment_content_digest(known[spec.id])
        or spec.metadata.get("content_sha256") != environment_content_digest(spec)
    ):
        raise ValueError("unknown or modified personal coding environment contract")
    payload = json.loads(_response_output_text(response))
    commands = payload.get("commands") if isinstance(payload, dict) else None
    if (
        not isinstance(commands, list)
        or not commands
        or len(commands) > spec.rollout.max_tool_calls
        or any(
            not isinstance(command, str) or not command.strip() or len(command) > 16384
            for command in commands
        )
    ):
        raise ValueError("coding response requires a bounded list of nonempty shell commands")
    result = run_docker_environment_attempt(
        RolloutCommandPlan(spec, commands), image=sandbox_image, client=client
    )
    return {
        "reward": result.attempt.reward,
        "correct": result.attempt.passed,
        "verifier_status": result.attempt.verifier_status,
        "reward_components": {"verified_tests": result.attempt.reward},
        "attempt": result.attempt.to_dict(),
    }


def build_personal_coding_resources_server() -> type:
    try:
        from nemo_gym.base_resources_server import (
            BaseResourcesServerConfig,
            BaseVerifyRequest,
            BaseVerifyResponse,
            SimpleResourcesServer,
        )
    except ImportError as exc:
        raise RuntimeError("NeMo Gym is optional; use a pinned NeMo Gym environment") from exc

    class CodingResourcesServerConfig(BaseResourcesServerConfig):
        sandbox_image: str

    class CodingVerifyRequest(BaseVerifyRequest):
        environment_spec: dict[str, Any]
        sandbox_image: str

    class CodingVerifyResponse(BaseVerifyResponse):
        correct: bool
        verifier_status: str
        reward_components: dict[str, float]

    class PersonalCodingResourcesServer(SimpleResourcesServer):
        config: CodingResourcesServerConfig

        async def verify(self, body: CodingVerifyRequest) -> CodingVerifyResponse:
            approved_image = validate_pinned_image(self.config.sandbox_image)
            if body.sandbox_image != approved_image:
                raise ValueError("request sandbox image differs from the configured image")
            result = await asyncio.to_thread(
                score_personal_coding_nemo_response,
                body.response.model_dump(mode="json"),
                body.environment_spec,
                approved_image,
            )
            return CodingVerifyResponse(
                **body.model_dump(),
                reward=result["reward"],
                correct=result["correct"],
                verifier_status=result["verifier_status"],
                reward_components=result["reward_components"],
            )

    PersonalCodingResourcesServer.verify_request_model = CodingVerifyRequest
    PersonalCodingResourcesServer.verify_response_model = CodingVerifyResponse
    PersonalCodingResourcesServer.config_model = CodingResourcesServerConfig
    return PersonalCodingResourcesServer


def run_personal_coding_resources_server() -> None:
    build_personal_coding_resources_server().run_webserver()
