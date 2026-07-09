import ast
from pathlib import Path

from bashgym.gym.trainer import Trainer, TrainerConfig, TrainingRun, TrainingStrategy


def test_session_distillation_script_uses_masked_hinted_kl_loss():
    config = TrainerConfig(
        base_model="tiny-local-model",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
        session_distillation_alpha=0.75,
        session_distillation_temperature=1.25,
        session_distillation_min_confidence=0.65,
    )
    run = TrainingRun(
        run_id="session-distill-test",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
        base_model=config.base_model,
        dataset_path=Path("data/session_distillation_records.jsonl"),
        output_path=Path("/tmp/bashgym-session-distill"),
    )

    script = Trainer(config)._generate_session_distillation_script(run)

    ast.parse(script)
    assert "class SessionDistillationTrainer(Trainer):" in script
    assert "original_context" in script
    assert "hinted_context" in script
    assert "teacher_logits = teacher_outputs.logits.detach()" in script
    assert "F.kl_div" in script
    assert "target_span_only" in script
    assert "session_distillation_loss" in script
    assert "MIN_CONFIDENCE = 0.65" in script
    assert "remove_unused_columns=False" in script


def test_session_distillation_script_escapes_windows_local_model_path():
    config = TrainerConfig(
        base_model=r"C:\Users\Cade\AppData\Local\Temp\tiny-model",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
    )
    run = TrainingRun(
        run_id="session-distill-windows-path",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
        base_model=config.base_model,
        dataset_path=Path(r"C:\Users\Cade\AppData\Local\Temp\records.jsonl"),
        output_path=Path(r"C:\Users\Cade\AppData\Local\Temp\out"),
    )

    script = Trainer(config)._generate_session_distillation_script(run)

    ast.parse(script)
    assert 'MODEL_NAME = "C:/Users/Cade/AppData/Local/Temp/tiny-model"' in script


def _sd_run(config, *, dataset="data/session_distillation_records.jsonl", out="/tmp/bashgym-sd"):
    return TrainingRun(
        run_id="sd-test",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
        base_model=config.base_model,
        dataset_path=Path(dataset),
        output_path=Path(out),
    )


def test_session_distillation_script_trains_in_bf16_without_fp16_amp():
    # fp16 weights + fp16 AMP crashes with "Attempting to unscale FP16 gradients".
    config = TrainerConfig(
        base_model="tiny-local-model",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
    )
    script = Trainer(config)._generate_session_distillation_script(_sd_run(config))
    ast.parse(script)
    assert "torch.bfloat16" in script
    assert "fp16=torch.cuda.is_available()" not in script
    assert "bf16=" in script


def test_session_distillation_script_adds_lora_when_enabled():
    config = TrainerConfig(
        base_model="tiny-local-model",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
        use_lora=True,
    )
    script = Trainer(config)._generate_session_distillation_script(_sd_run(config))
    ast.parse(script)
    assert "LoraConfig" in script
    assert "get_peft_model" in script


def test_session_distillation_script_does_not_trust_remote_code():
    config = TrainerConfig(
        base_model="tiny-local-model",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
    )
    script = Trainer(config)._generate_session_distillation_script(_sd_run(config))
    assert "trust_remote_code=True" not in script


def test_session_distillation_script_rejects_injection_in_base_model():
    config = TrainerConfig(
        base_model='x"; import os; os.system("echo pwned")\n#',
        strategy=TrainingStrategy.SESSION_DISTILLATION,
    )
    try:
        Trainer(config)._generate_session_distillation_script(_sd_run(config))
    except ValueError as exc:
        assert "base_model" in str(exc)
    else:
        raise AssertionError("unsafe base_model should be rejected")


def test_session_distillation_script_uses_relative_paths_when_remote():
    config = TrainerConfig(
        base_model="tiny-local-model",
        strategy=TrainingStrategy.SESSION_DISTILLATION,
    )
    run = _sd_run(config, dataset=r"C:\Users\Cade\records.jsonl", out=r"C:\Users\Cade\out")
    script = Trainer(config)._generate_session_distillation_script(run, remote=True)
    ast.parse(script)
    assert 'DATASET_PATH = "records.jsonl"' in script
    assert 'OUTPUT_DIR = "."' in script


def test_remote_launch_spec_runs_session_distillation_script_without_unsloth():
    # Remote launch must run the uploaded session-distillation script (not the
    # default train_sft.py), not gate on Unsloth (broken on some aarch64 GPUs),
    # and bake no local absolute paths.
    config = TrainerConfig(
        base_model="tiny-local-model", strategy=TrainingStrategy.SESSION_DISTILLATION
    )
    run = _sd_run(config, dataset=r"C:\Users\Cade\records.jsonl", out=r"C:\Users\Cade\out")
    content, script_name, require_unsloth = Trainer(config)._remote_launch_spec(run)
    assert script_name == "train_session_distillation.py"
    assert require_unsloth is False
    ast.parse(content)
    assert 'DATASET_PATH = "records.jsonl"' in content
    assert 'OUTPUT_DIR = "."' in content


def test_session_distillation_config_rejects_unsupported_mask_policy():
    try:
        TrainerConfig(
            base_model="tiny-local-model",
            session_distillation_mask_policy="full_context",
        )
    except ValueError as exc:
        assert "session_distillation_mask_policy" in str(exc)
    else:
        raise AssertionError("unsupported mask policy should fail validation")


def test_training_python_can_be_overridden_for_runtime_smokes(tmp_path, monkeypatch):
    runtime = tmp_path / "python.exe"
    runtime.write_text("", encoding="utf-8")
    monkeypatch.setenv("BASHGYM_TRAINING_PYTHON", str(runtime))

    assert Trainer(TrainerConfig(base_model="tiny-local-model"))._get_training_python() == str(
        runtime
    )


def test_training_subprocess_env_strips_quoted_hf_cache_paths(monkeypatch):
    monkeypatch.setenv("HF_HOME", r'"F:\huggingface"')
    monkeypatch.setenv("TRANSFORMERS_CACHE", r'"F:\huggingface"')

    env = Trainer(TrainerConfig(base_model="tiny-local-model"))._training_subprocess_env()

    assert env["HF_HOME"] == r"F:\huggingface"
    assert env["TRANSFORMERS_CACHE"] == r"F:\huggingface"


def test_parse_session_distillation_metrics_extracts_all_keys():
    from bashgym.gym.trainer import _parse_session_distillation_metrics

    line = (
        "{'session_distillation_loss': 0.42, 'session_distillation_kl': 0.10, "
        "'session_distillation_ce': 0.31, 'session_distillation_masked_tokens': 12}"
    )
    metrics = _parse_session_distillation_metrics(line)
    assert metrics["session_distillation_loss"] == 0.42
    assert metrics["session_distillation_kl"] == 0.10
    assert metrics["session_distillation_ce"] == 0.31
    assert metrics["session_distillation_masked_tokens"] == 12


def test_parse_session_distillation_metrics_empty_on_plain_loss_line():
    from bashgym.gym.trainer import _parse_session_distillation_metrics

    assert _parse_session_distillation_metrics("{'loss': 0.5, 'epoch': 1.0, 'step': 3}") == {}


def test_session_distillation_script_renders_context_with_chat_template():
    # Re-score in the chat/tool-call serving format when the tokenizer has a
    # chat template; target stays tokenized separately so original/hinted align.
    config = TrainerConfig(
        base_model="tiny-local-model", strategy=TrainingStrategy.SESSION_DISTILLATION
    )
    script = Trainer(config)._generate_session_distillation_script(_sd_run(config))
    ast.parse(script)
    assert "apply_chat_template" in script
    assert "tokenizer(target, add_special_tokens=False)" in script


def test_parse_bashgym_metrics_line_extracts_throughput_and_gpu():
    from bashgym.gym.trainer import _parse_bashgym_metrics_line

    line = 'noise [bashgym-metrics] {"tokens_per_second": 1234.5, "gpu_memory_gb": 6.2, "gpu_utilization": 87} tail'
    metrics = _parse_bashgym_metrics_line(line)
    assert metrics["tokens_per_second"] == 1234.5
    assert metrics["gpu_memory_gb"] == 6.2
    assert metrics["gpu_utilization"] == 87


def test_parse_bashgym_metrics_line_ignores_non_marker_lines():
    from bashgym.gym.trainer import _parse_bashgym_metrics_line

    assert _parse_bashgym_metrics_line("{'loss': 0.5, 'epoch': 1.0}") == {}
    assert _parse_bashgym_metrics_line("[bashgym-metrics] not-json") == {}


def test_session_distillation_script_emits_throughput_telemetry():
    config = TrainerConfig(
        base_model="tiny-local-model", strategy=TrainingStrategy.SESSION_DISTILLATION
    )
    script = Trainer(config)._generate_session_distillation_script(_sd_run(config))
    ast.parse(script)
    assert "BashGymThroughputCallback" in script
    assert "include_num_input_tokens_seen" in script
    assert "[bashgym-metrics]" in script
    # VRAM + GPU-util reads must be guarded so CPU smokes / GB10 degrade gracefully.
    assert "reset_peak_memory_stats" in script
    assert "pynvml" in script


def test_plain_sft_script_emits_throughput_telemetry():
    from bashgym.families import resolve_family_profile

    config = TrainerConfig(base_model="tiny-local-model")
    run = TrainingRun(
        run_id="sft-throughput",
        strategy=TrainingStrategy.SFT,
        base_model=config.base_model,
        dataset_path=Path("data/train.jsonl"),
        output_path=Path("/tmp/bashgym-sft"),
    )
    profile = resolve_family_profile(config.base_model)
    script = Trainer(config)._generate_sft_script_plain(run, profile)
    ast.parse(script)
    assert "BashGymThroughputCallback" in script
    assert "include_num_input_tokens_seen" in script
    assert "[bashgym-metrics]" in script


def test_progress_line_parser_parses_loss_step_grad_and_eta():
    from bashgym.gym.trainer import _ProgressLineParser

    p = _ProgressLineParser(batch_size=2, num_epochs=3, learning_rate=2e-4)
    assert p.feed("just a log line", now=0.0) is None  # no metric → no payload
    payload = p.feed("{'loss': 0.5, 'epoch': 1.0, 'step': 4, 'grad_norm': 1.1}", now=1.0)
    assert payload["loss"] == 0.5
    assert payload["step"] == 4
    assert payload["grad_norm"] == 1.1
    assert payload["total_epochs"] == 3
    assert payload["learning_rate"] == 2e-4
    assert payload["samples_processed"] == 8  # step * batch_size
    assert p.had_loss is True


def test_progress_line_parser_includes_resource_metrics_without_loss():
    from bashgym.gym.trainer import _ProgressLineParser

    p = _ProgressLineParser(batch_size=1, num_epochs=1, learning_rate=1e-4)
    payload = p.feed('[bashgym-metrics] {"tokens_per_second": 500, "gpu_memory_gb": 6.0}', now=1.0)
    assert payload["tokens_per_second"] == 500
    assert payload["gpu_memory_gb"] == 6.0
    assert p.had_loss is False


def test_progress_line_parser_accepts_quoted_remote_metric_values():
    """The remote SSH log stream carries TRL dicts with quoted string values."""
    from bashgym.gym.trainer import _ProgressLineParser

    p = _ProgressLineParser(batch_size=1, num_epochs=1, learning_rate=1e-4)
    payload = p.feed(
        "{'loss': '11.35', 'grad_norm': '21.62', 'learning_rate': '1.111e-05', 'epoch': '1.5'}",
        now=1.0,
    )
    assert payload["loss"] == 11.35
    assert payload["grad_norm"] == 21.62
    assert p.had_loss is True
    assert p.loss_lines == 1


def test_progress_line_parser_uses_last_tqdm_snapshot_in_glued_line():
    """tqdm \\r updates can arrive glued into one streamed line — last one wins."""
    from bashgym.gym.trainer import _ProgressLineParser

    p = _ProgressLineParser(batch_size=1, num_epochs=1, learning_rate=1e-4)
    glued = " 0%|          | 0/10 [00:00<?, ?it/s] 30%|███       | 3/10 [00:45<01:45, 15.0s/it]"
    payload = p.feed(glued, now=1.0)
    assert payload["step"] == 3
    assert payload["total_steps"] == 10
