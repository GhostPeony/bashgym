"""Activation CLI preserves reviewed physical model and evaluator output contracts."""

import json
from types import SimpleNamespace

import pytest

from bashgym.campaigns.activation import _validate_definition_bindings
from bashgym.campaigns.contracts import StageKind
from bashgym.cli import build_parser
from tests.campaigns.test_autoresearch_activation import _activation_fixture


@pytest.fixture
def activation_cli(tmp_path, monkeypatch):
    definition, request = _activation_fixture(tmp_path)
    executor = request.executor_profile
    training = executor.stage_profile(StageKind.FULL_TRAINING)
    evaluation = executor.stage_profile(StageKind.DEVELOPMENT_EVALUATION)
    receipt_path = tmp_path / "model-receipt.json"
    receipt_path.write_text(executor.registered_base_model.artifact_receipt.model_dump_json())
    monkeypatch.setattr(
        "bashgym.campaigns.autoresearch.load_autoresearch_template_definitions",
        lambda directory: (definition,),
    )

    async def device(*args):
        return SimpleNamespace(
            id="device",
            host=executor.host,
            port=executor.port,
            username=executor.username,
            key_path=executor.key_path,
            work_dir=executor.remote_work_dir,
        )

    monkeypatch.setattr("bashgym.device_registry.DeviceRegistry.get_device", device)

    async def no_preflight(*args, **kwargs):
        pytest.fail("CLI validation/plan must not launch remote preflight")

    monkeypatch.setattr("bashgym.gym.remote_trainer.RemoteTrainer.preflight_check", no_preflight)
    captured = []

    def validate_and_capture(definition, request, **kwargs):
        _validate_definition_bindings(definition, request)
        captured.append(request)
        return SimpleNamespace(doctor=None, model_dump=lambda **kw: {"applied": False})

    monkeypatch.setattr("bashgym.campaigns.activation.activate_autoresearch", validate_and_capture)
    argv = [
        "campaign",
        "activate-autoresearch",
        "--template",
        definition.template_id,
        "--workspace-id",
        request.workspace_id,
        "--device-id",
        "device",
        "--project-name",
        "Coding",
        "--owner-actor-id",
        "owner",
        "--dataset-id",
        request.dataset.dataset_id,
        "--dataset-name",
        "Coding dev",
        "--remote-dataset-path",
        executor.registered_evaluation_dataset.remote_dataset_path,
        "--dataset-content-digest",
        request.dataset_version.content_digest,
        "--dataset-source-id",
        executor.registered_evaluation_dataset.source_id,
        "--evaluator-file",
        str(evaluation.script_path),
        "--evaluation-budget-reservation",
        "0.1",
        "--evaluation-name",
        "Coding dev",
        "--remote-model-path",
        executor.registered_base_model.remote_model_path,
        "--source-repository",
        str(request.source_profile.repository_path),
        "--source-entrypoint",
        training.code_lineage_binding.entrypoint_path,
        "--mutation-path",
        "trainer=bashgym/gym/trainer.py",
        "--training-script",
        str(training.script_path),
        "--training-input",
        str(training.input_files[0]),
        "--executor-profile",
        "new-coding-executor",
        "--full-budget-reservation",
        "0.2",
        "--data-dir",
        str(tmp_path / "state"),
        "--json",
    ]
    return argv, receipt_path, captured


def test_activation_cli_builds_real_v2_request_and_additional_evidence_outputs(
    activation_cli, capsys
):
    argv, receipt, captured = activation_cli
    args = build_parser().parse_args(
        argv
        + [
            "--model-artifact-receipt",
            str(receipt),
            "--evaluation-output",
            "coding_task_results.json",
            "--evaluation-output",
            "coding_task_results.json",
        ]
    )
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    request = captured[0]
    assert request.executor_profile.registered_base_model.schema_version.endswith("v2")
    assert request.executor_profile.registered_base_model.artifact_receipt.model_dump(
        mode="json"
    ) == json.loads(receipt.read_text())
    assert request.executor_profile.stage_profile(
        StageKind.DEVELOPMENT_EVALUATION
    ).output_paths == ("autoresearch_evaluation.json", "coding_task_results.json")


def test_activation_cli_default_evidence_output_is_preserved(activation_cli, capsys):
    argv, receipt, captured = activation_cli
    args = build_parser().parse_args(argv + ["--model-artifact-receipt", str(receipt)])
    assert args.func(args) == 0
    assert captured[0].executor_profile.stage_profile(
        StageKind.DEVELOPMENT_EVALUATION
    ).output_paths == ("autoresearch_evaluation.json",)


@pytest.mark.parametrize("case", ["missing", "wrong_revision", "malformed", "unsafe_output"])
def test_invalid_physical_inputs_fail_before_remote_preflight(activation_cli, case):
    argv, receipt, captured = activation_cli
    extra = ["--apply"]
    if case != "missing":
        extra += ["--model-artifact-receipt", str(receipt)]
    if case == "wrong_revision":
        value = json.loads(receipt.read_text())
        value["revision"] = "b" * 40
        receipt.write_text(json.dumps(value))
    if case == "malformed":
        receipt.write_text('{"model_id":"not a receipt"}')
    if case == "unsafe_output":
        extra += ["--evaluation-output", "../outside.json"]
    args = build_parser().parse_args(argv + extra)
    with pytest.raises(ValueError):
        args.func(args)
    assert not captured
