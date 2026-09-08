"""Versioned authored coding tasks and executable verifier contracts."""

import hashlib
import json
import subprocess
import sys
import types

import pytest

import bashgym.environments as environments

REPAIRS = {
    "train-slug": {"solution.py": "def slug(text):\n    return '-'.join(text.lower().split())\n"},
    "train-config": {"settings.json": '{"enabled":true,"prefix":"ready"}'},
    "dev-merge": {
        "solution.py": "def merge_counts(left, right):\n    result = dict(left)\n    for key, count in right.items():\n        result[key] = result.get(key, 0) + count\n    return result\n"
    },
    "dev-csv": {
        "main.py": "import csv, sys\nwith open(sys.argv[1], encoding='utf-8', newline='') as stream:\n    print(sum(int(row['score']) for row in csv.DictReader(stream)))\n"
    },
    "confirmation-unique": {
        "solution.py": "def unique(items):\n    return list(dict.fromkeys(items))\n"
    },
    "confirmation-unicode": {
        "main.py": "from pathlib import Path\nimport sys\npath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'assets/text.txt'\nprint(path.read_text(encoding='utf-8').upper())\n"
    },
}


def run_verifier(workspace):
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-I", str(workspace / "verify.py")],
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.mark.parametrize("task_suffix", list(REPAIRS))
def test_authored_task_fails_before_repair_and_passes_independent_tests(tmp_path, task_suffix):
    specs = environments.personal_coding_environment_specs()
    spec = next(spec for spec in specs if spec.id.endswith(task_suffix))
    workspace = environments.materialize_environment(spec, tmp_path).path
    before = run_verifier(workspace)
    assert before.returncode != 0
    for name, source in REPAIRS[task_suffix].items():
        (workspace / name).write_text(source, encoding="utf-8")
    after = run_verifier(workspace)
    assert after.returncode == 0, after.stderr
    assert json.loads(after.stdout)["tests_run"] == spec.verifier.metadata["test_count"]


def test_candidate_cannot_forge_success_report_by_exiting_during_import(tmp_path):
    from bashgym.environments.docker_coding import _verified_result
    from bashgym.environments.rollout import CommandObservation

    spec = next(
        spec
        for spec in environments.personal_coding_environment_specs()
        if spec.id.endswith("train-slug")
    )
    workspace = environments.materialize_environment(spec, tmp_path).path
    forged = {
        "schema_version": "bashgym.coding_tests.v1",
        "tests_run": 4,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    (workspace / "solution.py").write_text(
        "import os\nprint(" + repr(json.dumps(forged)) + ", flush=True)\nos._exit(0)\n",
        encoding="utf-8",
    )
    result = run_verifier(workspace)
    observation = CommandObservation(
        "verify", "/workspace", result.returncode, result.stdout, result.stderr, 0
    )
    assert _verified_result(observation, 4)[0] is False


def api(name):
    function = getattr(environments, name, None)
    assert callable(function), f"Missing executable environment API: {name}"
    return function


def test_personal_bundle_has_independent_reproducible_splits(tmp_path):
    build = api("build_personal_coding_bundle")
    load = api("load_personal_coding_bundle")
    first = build(tmp_path / "first")
    second = build(tmp_path / "second")
    assert first == second
    assert first["schema_version"] == "bashgym.personal_coding.v1"
    assert first["provenance"]["origin"] == "authored_synthetic"
    assert len(first["dataset_digest"]) == 64
    all_ids = set()
    all_digests = set()
    for split in ("train", "dev", "confirmation"):
        specs = load(tmp_path / "first", split=split)
        assert len(specs) == 2
        assert {spec.metadata["task_family"] for spec in specs} == {
            "repository_repair",
            "tool_recovery",
        }
        for spec in specs:
            assert spec.validation_errors() == []
            assert spec.metadata["split"] == split
            assert spec.id not in all_ids
            assert spec.metadata["content_sha256"] not in all_digests
            all_ids.add(spec.id)
            all_digests.add(spec.metadata["content_sha256"])
            assert spec.rollout.harness == "bashgym-docker-coding-v1"
            assert spec.build.network_disabled
            assert spec.verifier.metadata["test_count"] >= 3


def test_personal_bundle_rejects_modified_inventory_and_preserves_existing_files(tmp_path):
    build = api("build_personal_coding_bundle")
    load = api("load_personal_coding_bundle")
    directory = tmp_path / "bundle"
    build(directory)
    with pytest.raises(FileExistsError):
        build(directory)
    path = directory / "train.jsonl"
    records = path.read_text(encoding="utf-8")
    path.write_text(records.replace("solution.py", "changed.py"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest|inventory"):
        load(directory)


def test_personal_bundle_loader_rejects_split_relabeling_even_with_new_file_hash(tmp_path):
    build = api("build_personal_coding_bundle")
    load = api("load_personal_coding_bundle")
    directory = tmp_path / "bundle"
    build(directory)
    path = directory / "confirmation.jsonl"
    data = path.read_text(encoding="utf-8").replace('"confirmation"', '"train"')
    path.write_text(data, encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if item["path"] == path.name:
            item.update(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size
            )
    identity = {key: value for key, value in manifest.items() if key != "dataset_digest"}
    manifest["dataset_digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split|content"):
        load(directory)


def test_personal_nemo_bundle_uses_existing_archive_path_without_certification(tmp_path):
    build = api("build_personal_coding_bundle")
    export = api("export_personal_coding_nemo_gym_bundle")
    source = tmp_path / "source"
    build(source)
    manifest = export(
        source,
        tmp_path / "gym",
        nemo_gym_revision="a" * 40,
        bashgym_revision="b" * 40,
        sandbox_image="python@sha256:" + "c" * 64,
    )
    assert manifest["verified"] is False
    assert manifest["resources_server_id"] == "bashgym_personal_coding"
    archive = tmp_path / "gym.zip"
    environments.create_nemo_gym_bundle_archive(tmp_path / "gym", archive)
    inspected = environments.inspect_nemo_gym_bundle_archive(archive)
    assert inspected["bundle_digest"] == manifest["bundle_digest"]
    config = (
        tmp_path
        / "gym/resources_servers/bashgym_personal_coding/configs/bashgym_personal_coding.yaml"
    ).read_text(encoding="utf-8")
    assert "data/confirmation.jsonl" not in config


@pytest.mark.asyncio
async def test_personal_resources_server_rejects_image_override_before_scoring(monkeypatch):
    from pydantic import BaseModel

    import bashgym.environments.personal_coding_nemo as adapter

    class Response(BaseModel):
        output_text: str

    class Request(BaseModel):
        response: Response

    class VerifyResponse(Request):
        reward: float

    module = types.ModuleType("nemo_gym.base_resources_server")
    module.BaseResourcesServerConfig = BaseModel
    module.BaseVerifyRequest = Request
    module.BaseVerifyResponse = VerifyResponse
    module.SimpleResourcesServer = type("SimpleResourcesServer", (), {})
    monkeypatch.setitem(sys.modules, "nemo_gym", types.ModuleType("nemo_gym"))
    monkeypatch.setitem(sys.modules, "nemo_gym.base_resources_server", module)

    def refuse_execution(*args, **kwargs):
        pytest.fail("Unapproved request image reached sandbox scoring")

    monkeypatch.setattr(adapter, "score_personal_coding_nemo_response", refuse_execution)
    server_type = adapter.build_personal_coding_resources_server()
    server = server_type()
    server.config = types.SimpleNamespace(sandbox_image="python@sha256:" + "a" * 64)
    body = server_type.verify_request_model(
        response=Response(output_text='{"commands":["echo ok"]}'),
        environment_spec=environments.personal_coding_environment_specs()[0].to_dict(),
        sandbox_image="python@sha256:" + "b" * 64,
    )
    with pytest.raises(ValueError, match="image"):
        await server.verify(body)
