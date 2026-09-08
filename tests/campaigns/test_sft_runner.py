"""Campaign SFT orchestration checks without loading training dependencies."""

import ast
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bashgym.campaigns.sft_runner import (
    CampaignSFTConfig,
    _campaign_metrics_callback,
    _execute_generated,
    build_training_script,
    run_sft,
    run_training_process,
    validate_token_lengths,
)


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2"}')
    (model / "model.safetensors").write_bytes(b"fixture-base")
    (model / "tokenizer_config.json").write_text("{}")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Implement identity."},
                    {"role": "assistant", "content": "def identity(x): return x"},
                ]
            }
        )
        + "\n"
    )
    output = tmp_path / "run"
    output.mkdir()
    launch = output / "launch_manifest.json"
    launch.write_text(
        json.dumps(
            {
                "schema_version": "campaign_remote_launch_manifest.v2",
                "run_id": "attempt-test",
                "recipe_digest": "a" * 64,
                "registered_base_model": {
                    "remote_model_path": str(model),
                    "model_digest": "b" * 64,
                },
            }
        )
    )
    config = CampaignSFTConfig(dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
    return config, model, dataset, output, launch


def _execute_fixture(script, result, timeout):
    work = script.parent
    merged = work / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type":"qwen2"}')
    (merged / "model.safetensors").write_bytes(b"fixture-merged")
    (merged / "tokenizer_config.json").write_text("{}")
    (work.parent / "training_metrics.jsonl").write_text('{"step":16,"loss":1.2}\n')
    result.write_text(json.dumps({"global_step": 16, "train_loss": 1.2}))


def test_publishes_merged_candidate_and_training_evidence(inputs):
    config, model, dataset, output, launch = inputs
    result = run_sft(config, model, dataset, output, launch, execute=_execute_fixture)
    assert result["status"] == "completed"
    assert result["optimizer_steps"] == 16
    assert result["dataset_sha256"] == config.dataset_sha256
    assert (output / "final" / "model.safetensors").read_bytes() == b"fixture-merged"
    assert not (output / "final" / "adapter_config.json").exists()
    assert json.loads((output / "training_manifest.json").read_text()) == result


@pytest.mark.parametrize("defect", ["wrong_model", "wrong_data", "empty_data", "quantized"])
def test_rejects_unbound_or_incompatible_inputs_before_execution(inputs, defect):
    config, model, dataset, output, launch = inputs
    if defect == "wrong_model":
        doc = json.loads(launch.read_text())
        doc["registered_base_model"]["remote_model_path"] = str(model.parent / "other")
        launch.write_text(json.dumps(doc))
    elif defect == "wrong_data":
        dataset.write_text(dataset.read_text() + "\n")
    elif defect == "empty_data":
        dataset.write_text("")
        config = config.model_copy(update={"dataset_sha256": hashlib.sha256(b"").hexdigest()})
    else:
        (model / "config.json").write_text('{"quantization_config":{"quant_method":"fp4"}}')
    called = []
    with pytest.raises(ValueError):
        run_sft(config, model, dataset, output, launch, execute=lambda *a: called.append(a))
    assert not called
    assert not (output / "final").exists()


@pytest.mark.parametrize("defect", ["no_merge", "partial_steps", "nan_loss", "timeout"])
def test_execution_failure_never_publishes_candidate(inputs, defect):
    config, model, dataset, output, launch = inputs

    def execute(script, result, timeout):
        if defect == "timeout":
            raise subprocess.TimeoutExpired("fixture", timeout)
        _execute_fixture(script, result, timeout)
        if defect == "no_merge":
            (script.parent / "merged" / "model.safetensors").unlink()
        elif defect == "partial_steps":
            result.write_text('{"global_step":2,"train_loss":1.2}')
        else:
            result.write_text('{"global_step":16,"train_loss":NaN}')

    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        run_sft(config, model, dataset, output, launch, execute=execute)
    assert not (output / "final").exists()
    assert json.loads((output / "training_manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("backend", ["plain", "unsloth"])
def test_generated_trainer_receives_structured_metrics_callback(inputs, backend):
    config, model, dataset, output, _ = inputs
    config = config.model_copy(update={"backend": backend})
    script = build_training_script(
        config, model, dataset, output / ".sft-work", output / "training_metrics.jsonl"
    )
    tree = ast.parse(script)
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "SFTTrainer"
    )
    callback_arg = next(k.value for k in call.keywords if k.arg == "callbacks")
    marker = object()
    actual = eval(
        compile(ast.Expression(callback_arg), "<callbacks>", "eval"),
        {
            "callbacks": ["existing"],
            "_campaign_metrics_callback": lambda path: marker,
        },
    )
    assert actual == ["existing", marker]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        ):
            kwargs = {k.arg: k.value for k in node.keywords}
            assert ast.literal_eval(kwargs["local_files_only"]) is True
            assert ast.literal_eval(kwargs["trust_remote_code"]) is False


def test_token_validation_rejects_truncation_and_handles_batch_encoding(inputs):
    config, _, dataset, _, _ = inputs

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is True
            assert kwargs["add_generation_prompt"] is False
            return {"input_ids": list(range(513))}

    with pytest.raises(ValueError, match="sequence_limit"):
        validate_token_lengths(dataset, Tokenizer(), config.max_seq_length)


def test_token_validation_accounts_for_eos_after_template_newline(inputs):
    config, _, dataset, _, _ = inputs

    class Tokenizer:
        eos_token = "<eos>"
        bos_token = None

        def apply_chat_template(self, messages, **kwargs):
            if kwargs["tokenize"]:
                return {"input_ids": list(range(512))}
            return "full conversation<eos>\n"

        def __call__(self, *, text):
            assert text == "full conversation<eos>\n<eos>"
            return {"input_ids": list(range(513))}

    with pytest.raises(ValueError, match="sequence_limit"):
        validate_token_lengths(dataset, Tokenizer(), config.max_seq_length)


def test_actual_subprocess_timeout_terminates_owned_process(tmp_path, monkeypatch):
    from bashgym.campaigns import sft_runner

    original = subprocess.Popen
    processes = []

    def track(*args, **kwargs):
        assert not kwargs.get("start_new_session", False)
        assert "process_group" not in kwargs
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(sft_runner.subprocess, "Popen", track)
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(20)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        run_training_process([sys.executable, str(script)], timeout=0.1)
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_callback_writes_real_log_values_in_campaign_stream_format(tmp_path, monkeypatch):
    from bashgym.campaigns.metrics import parse_metric_lines

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    path = tmp_path / "metrics.jsonl"
    callback = _campaign_metrics_callback(str(path))
    state = SimpleNamespace(global_step=3)
    callback.on_log(
        None, state, None, logs={"loss": 1.25, "learning_rate": 0.0001, "untrusted_text": "ignored"}
    )
    points = parse_metric_lines(tuple(path.read_text().splitlines()))
    assert points[0].step == 3
    assert points[0].values == {"loss": 1.25, "learning_rate": 0.0001}
    with pytest.raises(ValueError, match="nonfinite"):
        callback.on_log(None, state, None, logs={"loss": float("nan")})
    assert len(path.read_text().splitlines()) == 1


def test_child_validates_tokens_then_reads_actual_trainer_state(tmp_path, monkeypatch, inputs):
    _, model, dataset, _, _ = inputs
    calls = []

    class Tokenizer:
        eos_token = "<eos>"
        bos_token = None

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1, 2, 3]} if kwargs["tokenize"] else "conversation<eos>"

        def __call__(self, *, text):
            return {"input_ids": [1, 2, 3]}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    script = tmp_path / "generated-fixture.py"
    script.write_text(
        "from types import SimpleNamespace\n"
        "trainer = SimpleNamespace(state=SimpleNamespace(global_step=16, log_history=[{'loss': 1.4}, {'train_loss': 1.2}]))\n"
    )
    result = tmp_path / "result.json"
    _execute_generated(str(script), str(result), str(dataset), str(model), "512", "plain")
    assert json.loads(result.read_text()) == {"global_step": 16, "train_loss": 1.2}
    assert calls == [(str(model), {"local_files_only": True, "trust_remote_code": False})]
