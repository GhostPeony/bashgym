"""Code benchmark ABI and evidence tests; no model or container execution."""

import hashlib
import json
import signal
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from bashgym.campaigns import first_party_coding_runner as runner
from bashgym.campaigns.autoresearch_evidence import (
    AutoResearchEvaluationContext,
    AutoResearchEvaluationEvidence,
    evaluation_context_bytes,
)


@pytest.fixture
def inputs(tmp_path):
    config = dict(
        schema_version="first_party_coding_config.v1",
        scope="smoke",
        split="test",
        source="benchmark",
        revision="a" * 40,
        expected_task_count=3,
        primary_metric="pass_fraction",
        sandbox_image="python@sha256:" + "a" * 64,
        max_input_tokens=1024,
        max_new_tokens=64,
        generation_timeout_seconds=30,
        test_timeout_seconds=2,
        dtype="float32",
        device="cpu",
        seed=7,
        completion_protocol="raw",
        max_seconds=60,
    )
    rows = [
        dict(
            task_id=f"task/{index}",
            prompt="def add(a, b):\n",
            test="def check(candidate): assert candidate(1, 2) == 3",
            entry_point="add",
            provenance={k: config[k] for k in ("split", "source", "revision")},
        )
        for index in range(3)
    ]
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    context = AutoResearchEvaluationContext(
        workspace_id="workspace",
        campaign_id="campaign",
        study_id="study",
        action_id="action",
        attempt_id="attempt",
        candidate_digest="a" * 64,
        evaluation_suite_id="suite",
        evaluation_code_digest="b" * 64,
        dataset_version_id="data",
        dataset_content_digest=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        evaluated_model_manifest_digest="c" * 64,
    )
    context_path = tmp_path / "autoresearch_evaluation_context.json"
    context_path.write_bytes(evaluation_context_bytes(context))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture only")
    return dict(
        context_path=context_path,
        model_directory=model,
        dataset_path=dataset,
        output_path=tmp_path / "autoresearch_evaluation.json",
        config_path=config_path,
    )


def test_fixed_denominator_and_real_evidence_schema(inputs):
    prompts = []
    statuses = iter(("passed", "failed", "test_timeout"))
    result = runner.run(
        **inputs,
        complete=lambda prompt: prompts.append(prompt) or "    return a+b\n",
        evaluate=lambda *args: next(statuses),
    )
    parsed = AutoResearchEvaluationEvidence.model_validate_json(inputs["output_path"].read_bytes())
    assert parsed == result
    assert parsed.metrics == {"pass_fraction": 1 / 3}
    assert parsed.evaluated_model_manifest_digest == "c" * 64
    assert prompts == ["def add(a, b):\n"] * 3
    counts = parsed.slice_metrics["coding_benchmark"]
    assert counts["scope"] == "smoke" and counts["task_count"] == 3
    assert counts["passed"] == counts["failed"] == counts["test_timeout"] == 1
    assert counts["complete"] and not counts["adversarial_reward_integrity_verified"]
    artifact = inputs["output_path"].with_name("coding_task_results.json").read_bytes()
    assert hashlib.sha256(artifact).hexdigest() == counts["task_results_sha256"]
    results = json.loads(artifact)
    assert results["tasks"] == [
        {"task_id": "task/0", "status": "passed"},
        {"task_id": "task/1", "status": "failed"},
        {"task_id": "task/2", "status": "test_timeout"},
    ]
    assert b"prompt" not in artifact and b"return a+b" not in artifact


@pytest.mark.parametrize(
    "boundary",
    [
        "def other():",
        "class Other:",
        "if True:",
        "import math",
        "from math import pi",
        "print('other')",
        "# unrelated example",
        "```python",
    ],
)
def test_humaneval_body_stops_before_dedented_continuation(boundary):
    body = "    def nested(value):\n        return value\n    return nested(a + b)\n\n"
    raw = body + boundary + '\n    """unfinished unrelated text'
    assert runner.process_completion(raw, "humaneval_body_v1") == body
    assert runner.process_completion(raw, "raw") == raw
    namespace = {}
    exec("def add(a, b):\n" + runner.process_completion(raw, "humaneval_body_v1"), namespace)
    assert namespace["add"](1, 2) == 3


def test_completion_protocol_is_required_and_does_not_repair_invalid_body(inputs):
    config = json.loads(inputs["config_path"].read_text())
    config.pop("completion_protocol")
    with pytest.raises(ValueError, match="completion_protocol"):
        runner.CodingRunnerConfig.model_validate(config)
    invalid = "    return (a +\n"
    assert runner.process_completion(invalid, "humaneval_body_v1") == invalid


def test_run_applies_declared_completion_protocol_and_records_it(inputs):
    config = json.loads(inputs["config_path"].read_text())
    config["completion_protocol"] = "humaneval_body_v1"
    inputs["config_path"].write_text(json.dumps(config))
    raw = '    return a + b\n\ndef unrelated():\n    """unfinished docstring'

    def evaluate(task, completion, selected):
        assert selected.completion_protocol == "humaneval_body_v1"
        namespace = {}
        exec(task.prompt + completion, namespace)
        return "passed" if namespace[task.entry_point](1, 2) == 3 else "failed"

    result = runner.run(**inputs, complete=lambda prompt: raw, evaluate=evaluate)
    summary = result.slice_metrics["coding_benchmark"]
    assert summary["completion_protocol"] == "humaneval_body_v1"
    assert result.metrics["pass_fraction"] == 1
    artifact = json.loads(inputs["output_path"].with_name("coding_task_results.json").read_text())
    assert artifact["completion_protocol"] == "humaneval_body_v1"


@pytest.mark.parametrize("phase", ["generation", "evaluation"])
def test_infrastructure_failure_aborts_without_dropping_tasks(inputs, phase):
    calls = []

    def generate(prompt):
        calls.append("generate")
        if phase == "generation":
            raise RuntimeError("private backend detail")
        return "    return a+b\n"

    def evaluate(*args):
        calls.append("evaluate")
        raise RuntimeError("private Docker detail")

    result = runner.run(**inputs, complete=generate, evaluate=evaluate)
    counts = result.slice_metrics["coding_benchmark"]
    assert counts["infrastructure_error"] == 1 and counts["not_run"] == 2
    assert not counts["complete"] and result.metrics["pass_fraction"] == 0
    assert calls == (["generate"] if phase == "generation" else ["generate", "evaluate"])
    assert "private" not in inputs["output_path"].read_text()
    outcomes = json.loads(inputs["output_path"].with_name("coding_task_results.json").read_bytes())
    assert outcomes["tasks"][0]["error_category"] == phase + "_runtime_error"
    assert [row["status"] for row in outcomes["tasks"][1:]] == ["not_run", "not_run"]


def test_generation_timeout_stops_before_verifier_or_next_task(inputs):
    def timeout(prompt):
        raise TimeoutError()

    result = runner.run(
        **inputs, complete=timeout, evaluate=lambda *args: pytest.fail("must not evaluate")
    )
    assert result.slice_metrics["coding_benchmark"]["generation_timeout"] == 1
    assert result.slice_metrics["coding_benchmark"]["not_run"] == 2


@pytest.mark.parametrize("change", ["digest", "answer", "split", "duplicate", "count", "adapter"])
def test_invalid_inputs_rejected_before_completion(inputs, change):
    if change == "adapter":
        (inputs["model_directory"] / "adapter_config.json").write_text("{}")
    elif change == "count":
        config = json.loads(inputs["config_path"].read_text())
        config["expected_task_count"] = 4
        inputs["config_path"].write_text(json.dumps(config))
    else:
        rows = [json.loads(row) for row in inputs["dataset_path"].read_text().splitlines()]
        if change == "answer":
            rows[0]["canonical_solution"] = "must not be accepted"
        elif change == "split":
            rows[0]["provenance"]["split"] = "train"
        elif change == "duplicate":
            rows[1]["task_id"] = rows[0]["task_id"]
        else:
            rows[0]["prompt"] += "changed"
        inputs["dataset_path"].write_text("".join(json.dumps(row) + "\n" for row in rows))
        if change != "digest":
            context = json.loads(inputs["context_path"].read_text())
            context["dataset_content_digest"] = hashlib.sha256(
                inputs["dataset_path"].read_bytes()
            ).hexdigest()
            inputs["context_path"].write_text(json.dumps(context))
    with pytest.raises(ValueError):
        runner.run(**inputs, complete=lambda prompt: pytest.fail("must not generate"))
    assert not inputs["output_path"].exists()


@pytest.mark.parametrize("scope,exit_code", [("smoke", 3), ("development", 0)])
def test_existing_worker_cli_abi_uses_mandatory_pinned_config(
    inputs, monkeypatch, scope, exit_code
):
    config = json.loads(inputs["config_path"].read_text())
    config["scope"] = scope
    inputs["config_path"].write_text(json.dumps(config))
    seen = []

    def execute_inline(argv, *, timeout):
        assert timeout == config["max_seconds"]
        runner._run_child(*argv[3:])

    monkeypatch.setattr(runner, "run_evaluation_process", execute_inline)
    monkeypatch.setattr(runner, "LocalCompletion", lambda *args: lambda prompt: "    return a+b\n")
    monkeypatch.setattr(
        runner, "evaluate_task", lambda *args: seen.append(args[0].task_id) or "passed"
    )
    argv = [
        "--config",
        str(inputs["config_path"]),
        "--context",
        str(inputs["context_path"]),
        "--model-dir",
        str(inputs["model_directory"]),
        "--dataset",
        str(inputs["dataset_path"]),
        "--output",
        str(inputs["output_path"]),
    ]
    assert runner.main(argv) == exit_code and len(seen) == 3
    with pytest.raises(SystemExit):
        runner.main(argv[2:])


def test_incomplete_cli_run_is_nonzero_and_not_adoptable(inputs, monkeypatch):
    monkeypatch.setattr(
        runner, "run_evaluation_process", lambda argv, **kwargs: runner._run_child(*argv[3:])
    )
    monkeypatch.setattr(runner, "LocalCompletion", lambda *args: lambda prompt: "completion")
    monkeypatch.setattr(runner, "evaluate_task", lambda *args: "infrastructure_error")
    args = []
    for flag, key in (
        ("context", "context_path"),
        ("model-dir", "model_directory"),
        ("dataset", "dataset_path"),
        ("output", "output_path"),
        ("config", "config_path"),
    ):
        args += ["--" + flag, str(inputs[key])]
    assert runner.main(args) == 2


def test_deadline_is_mandatory_and_timeout_cannot_publish_success(inputs, monkeypatch):
    config = json.loads(inputs["config_path"].read_text())
    config.pop("max_seconds")
    with pytest.raises(ValueError, match="max_seconds"):
        runner.CodingRunnerConfig.model_validate(config)

    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(runner, "run_evaluation_process", timeout)
    args = [
        value
        for name, key in (
            ("context", "context_path"),
            ("model-dir", "model_directory"),
            ("dataset", "dataset_path"),
            ("output", "output_path"),
            ("config", "config_path"),
        )
        for value in ("--" + name, str(inputs[key]))
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        runner.main(args)
    assert not inputs["output_path"].exists()


def test_child_termination_runs_context_cleanup_and_restores_handlers(monkeypatch):
    events = []
    prior = signal.getsignal(signal.SIGTERM)

    def interrupted_run(*args):
        try:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        finally:
            events.append("episode_cleanup")

    monkeypatch.setattr(runner, "run", interrupted_run)
    with pytest.raises(KeyboardInterrupt, match="coding_evaluation_interrupted"):
        runner._run_child("context", "model", "dataset", "output", "config")
    assert events == ["episode_cleanup"]
    assert signal.getsignal(signal.SIGTERM) == prior


def test_evaluation_deadline_terminates_hanging_owned_subprocess(tmp_path, monkeypatch):
    import psutil

    from bashgym.campaigns import sft_runner

    original = subprocess.Popen
    owned = []

    def capture(*args, **kwargs):
        assert not kwargs.get("start_new_session", False)
        process = original(*args, **kwargs)
        owned.append(process)
        return process

    monkeypatch.setattr(sft_runner.subprocess, "Popen", capture)
    script = tmp_path / "hanging_evaluation.py"
    descendant_pid = tmp_path / "descendant.pid"
    script.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid))\ntime.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_evaluation_process([sys.executable, str(script), str(descendant_pid)], timeout=2)
    assert len(owned) == 1 and owned[0].poll() is not None
    assert descendant_pid.exists()
    assert not psutil.pid_exists(int(descendant_pid.read_text()))


@pytest.mark.parametrize("status", ["passed", "failed", "test_timeout", "infrastructure_error"])
def test_docker_boundary_uses_protected_humaneval_verifier(inputs, monkeypatch, status):
    captured = {}
    config = runner.CodingRunnerConfig.model_validate_json(inputs["config_path"].read_bytes())
    task = runner.CodingTask.model_validate_json(
        inputs["dataset_path"].read_bytes().splitlines()[0]
    )

    class Episode:
        def __init__(self, spec, *, image):
            from bashgym.environments.docker_coding import _validate_environment

            _validate_environment(spec)
            captured.update(spec=spec, image=image)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            captured["closed"] = True

        def tampered(self):
            return False

        def run(self, command, *, verifier):
            assert verifier and command == "python -I /workspace/verify.py"
            report = dict(
                schema_version="bashgym.coding_tests.v1",
                tests_run=1,
                failures=int(status != "passed"),
                errors=0,
                skipped=0,
                status=status,
            )
            return SimpleNamespace(
                timeout=False, blocked=False, exit_code=0, stdout=json.dumps(report)
            )

    monkeypatch.setattr(runner, "DockerCodingEpisode", Episode)
    assert runner.evaluate_task(task, "    return a+b\n", config) == status
    spec = captured["spec"]
    assert "from human_eval.execution import check_correctness" in spec.files["verify.py"]
    assert spec.metadata["protected_paths"] == ["task.json"]
    assert captured["image"] == config.sandbox_image and captured["closed"]
    assert "canonical_solution" not in spec.files["task.json"]


@pytest.mark.parametrize("mode", ["normal", "timeout", "input_limit"])
def test_generation_is_greedy_local_only_and_explicit_dict_outputs(inputs, monkeypatch, mode):
    calls = {}

    class Tensor:
        shape = (1, 4)

        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            calls["prompt"] = prompt
            return {"input_ids": Tensor(), "attention_mask": Tensor()}

        def decode(self, tokens, **kwargs):
            return "    return a+b\n"

    class Sequence:
        def __getitem__(self, key):
            return [1]

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(sequences=Sequence())

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = kwargs
        return Tokenizer()

    def load_model(path, **kwargs):
        calls["model"] = kwargs
        return Model()

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(float32="float32", inference_mode=nullcontext)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
            set_seed=lambda seed: None,
        ),
    )
    config = runner.CodingRunnerConfig.model_validate_json(inputs["config_path"].read_bytes())
    if mode == "input_limit":
        config = config.model_copy(update={"max_input_tokens": 2})
        with pytest.raises(ValueError, match="coding_prompt_token_limit"):
            runner.LocalCompletion(inputs["model_directory"], config)("prompt")
        assert "generate" not in calls
        return
    if mode == "timeout":
        ticks = iter((0, config.generation_timeout_seconds + 1))
        monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
        with pytest.raises(TimeoutError, match="coding_generation_timeout"):
            runner.LocalCompletion(inputs["model_directory"], config)("prompt")
    else:
        assert (
            runner.LocalCompletion(inputs["model_directory"], config)("prompt")
            == "    return a+b\n"
        )
    for key in ("tokenizer", "model"):
        assert calls[key]["local_files_only"] and not calls[key]["trust_remote_code"]
    assert calls["model"]["return_dict"]
    assert calls["generate"]["return_dict_in_generate"]
    assert calls["generate"]["max_time"] == config.generation_timeout_seconds
    assert calls["generate"]["do_sample"] is False
