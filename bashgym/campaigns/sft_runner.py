"""Bounded campaign entrypoint around BashGym's existing generated SFT recipe.

Training dependencies remain in the execution environment. The worker supplies
the launch manifest and pins this script, its config, and its input files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import runpy
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from bashgym.campaigns.contracts import FrozenContractModel


class CampaignSFTConfig(FrozenContractModel):
    schema_version: Literal["campaign_sft_config.v1"] = "campaign_sft_config.v1"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: Literal["plain", "unsloth"] = "plain"
    max_steps: int = Field(default=16, ge=1, le=1000)
    max_seq_length: int = Field(default=512, ge=32, le=4096)
    batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=64)
    lora_r: int = Field(default=8, ge=1, le=64)
    lora_alpha: int = Field(default=16, ge=1, le=128)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.01)
    seed: Literal[42] = 42
    max_seconds: int = Field(default=1200, ge=1, le=1200)
    max_examples: int = Field(default=128, ge=1, le=10000)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _regular_path(path: Path, *, directory: bool = False) -> Path:
    original = path.expanduser()
    if not original.is_absolute() or original.is_symlink():
        raise ValueError("campaign_sft_requires_absolute_regular_path")
    resolved = original.resolve(strict=True)
    # Generated recipes embed paths in Python strings. Reject unrepresentable
    # inputs before script generation rather than accepting executable text.
    if any(character in resolved.as_posix() for character in "\"'\n\r\x00"):
        raise ValueError("campaign_sft_unsupported_path_characters")
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise ValueError("campaign_sft_requires_regular_path")
    return resolved


def _check_model(directory: Path) -> None:
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("campaign_sft_model_links_unsupported")
    config = json.loads((directory / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("campaign_sft_quantized_base_unsupported")
    if (directory / "adapter_config.json").exists():
        raise ValueError("campaign_sft_requires_full_checkpoint")
    weights = list(directory.glob("*.safetensors"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise ValueError("campaign_sft_model_weights_missing")
    if not (directory / "tokenizer_config.json").is_file():
        raise ValueError("campaign_sft_tokenizer_missing")


def _load_rows(path: Path, limit: int) -> list[dict]:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("campaign_sft_dataset_too_large")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and set(row).intersection(
            {"input_ids", "prompt", "completion", "tools", "chat_template_kwargs", "conversations"}
        ):
            raise ValueError("campaign_sft_preprocessed_or_alternate_format_unsupported")
        messages = row.get("messages") if isinstance(row, dict) else None
        if (
            not isinstance(messages, list)
            or len(messages) < 2
            or not all(isinstance(message, dict) for message in messages)
            or messages[-1].get("role") != "assistant"
            or not messages[-1].get("content")
            or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            )
        ):
            raise ValueError("campaign_sft_invalid_messages")
        rows.append(row)
        if len(rows) > limit:
            raise ValueError("campaign_sft_example_limit_exceeded")
    if not rows:
        raise ValueError("campaign_sft_empty_dataset")
    return rows


def validate_token_lengths(dataset: Path, tokenizer, max_length: int) -> None:
    """Check both chat and rendered-text representations used by TRL/Unsloth.

    Plain TRL preserves messages; the Unsloth formatter can strip BOS and append
    EOS to rendered text. Require every supported representation to fit, so an
    EOS added after a trailing newline cannot cause hidden truncation.
    """
    for row in _load_rows(dataset, 10000):
        encoded = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False, return_dict=True
        )
        tokens = encoded["input_ids"]
        if not tokens or len(tokens) > max_length:
            raise ValueError("campaign_sft_example_exceeds_sequence_limit")
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        if not isinstance(text, str) or not tokenizer.eos_token:
            raise ValueError("campaign_sft_template_eos_unavailable")
        representations = [text]
        bos = getattr(tokenizer, "bos_token", None)
        if bos and text.startswith(bos):
            representations.append(text[len(bos) :])
        for rendered in representations:
            if not rendered.endswith(tokenizer.eos_token):
                rendered += tokenizer.eos_token
            rendered_ids = tokenizer(text=rendered)["input_ids"]
            if not rendered_ids or len(rendered_ids) > max_length:
                raise ValueError("campaign_sft_example_exceeds_sequence_limit")


def _campaign_metrics_callback(path: str):
    from transformers import TrainerCallback

    class CampaignMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            allowed = {"loss", "train_loss", "learning_rate", "grad_norm", "epoch", "train_runtime"}
            values = {}
            for key, value in (logs or {}).items():
                if key not in allowed:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if not math.isfinite(value):
                    raise ValueError("campaign_sft_nonfinite_training_metric")
                values[key] = float(value)
            if values:
                with Path(path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": int(state.global_step), **values}) + "\n")

    return CampaignMetricsCallback()


def build_training_script(
    config: CampaignSFTConfig, model: Path, dataset: Path, work: Path, metrics: Path
) -> str:
    """Reuse the trainer; add local-only loads and its campaign metric callback."""
    from bashgym.gym.trainer import Trainer, TrainerConfig, TrainingRun, TrainingStrategy

    recipe = TrainerConfig(
        base_model=model.as_posix(),
        sft_backend=config.backend,
        load_in_4bit=False,
        use_lora=True,
        use_liger=False,
        max_steps=config.max_steps,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        learning_rate=config.learning_rate,
        num_epochs=1,
        logging_steps=1,
        save_steps=config.max_steps,
        checkpoint_limit=1,
        artifact_retention="deployable",
        auto_export_gguf=False,
        auto_deploy_ollama=False,
        auto_push_hf=False,
        output_dir=str(work),
        early_stopping_patience=0,
    )
    recipe.validate_backend_recipe(config.backend)
    run = TrainingRun(
        run_id="campaign-sft",
        strategy=TrainingStrategy.SFT,
        base_model=model.as_posix(),
        dataset_path=dataset,
        output_path=work,
    )
    tree = ast.parse(Trainer(recipe)._generate_sft_script(run))
    trainer_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            node.keywords = [
                k for k in node.keywords if k.arg not in {"local_files_only", "trust_remote_code"}
            ]
            node.keywords.extend(
                [
                    ast.keyword(arg="local_files_only", value=ast.Constant(True)),
                    ast.keyword(arg="trust_remote_code", value=ast.Constant(False)),
                ]
            )
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "FastLanguageModel":
                for keyword in node.keywords:
                    if keyword.arg == "dtype":
                        keyword.value = ast.Attribute(
                            value=ast.Name(id="torch", ctx=ast.Load()),
                            attr="bfloat16",
                            ctx=ast.Load(),
                        )
        if isinstance(node.func, ast.Name) and node.func.id == "SFTTrainer":
            trainer_calls += 1
            callbacks = next(k for k in node.keywords if k.arg == "callbacks")
            callbacks.value = ast.BinOp(
                left=ast.BoolOp(
                    op=ast.Or(), values=[callbacks.value, ast.List(elts=[], ctx=ast.Load())]
                ),
                op=ast.Add(),
                right=ast.List(
                    elts=[
                        ast.Call(
                            func=ast.Name(id="_campaign_metrics_callback", ctx=ast.Load()),
                            args=[ast.Constant(str(metrics))],
                            keywords=[],
                        )
                    ],
                    ctx=ast.Load(),
                ),
            )
        if isinstance(node.func, ast.Name) and node.func.id == "SFTConfig":
            for keyword in node.keywords:
                if keyword.arg in {"bf16", "fp16"}:
                    keyword.value = ast.Constant(keyword.arg == "bf16")
    if trainer_calls != 1:
        raise ValueError("campaign_sft_generated_trainer_contract_changed")
    # This import is dependency-light, including before Unsloth's first import.
    tree.body.insert(
        1,
        ast.ImportFrom(
            module="bashgym.campaigns.sft_runner",
            names=[ast.alias(name="_campaign_metrics_callback")],
            level=0,
        ),
    )
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def _execute_generated(
    script: str, result: str, dataset: str, model: str, max_length: str, backend: str
) -> None:
    if backend == "unsloth":
        # Unsloth must patch the training stack before transformers is imported.
        import importlib

        importlib.import_module("unsloth")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    validate_token_lengths(Path(dataset), tokenizer, int(max_length))
    del tokenizer
    namespace = runpy.run_path(script, run_name="__main__")
    state = namespace["trainer"].state
    losses = [row.get("train_loss", row.get("loss")) for row in state.log_history]
    losses = [value for value in losses if isinstance(value, (int, float))]
    if not losses or not math.isfinite(losses[-1]):
        raise RuntimeError("campaign_sft_training_loss_unavailable")
    _write_json(
        Path(result), {"global_step": int(state.global_step), "train_loss": float(losses[-1])}
    )


def run_training_process(argv: list[str], *, timeout: float) -> None:
    """Bound the whole owned subprocess, including model loading and final merge."""
    environment = dict(os.environ)
    environment.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_DISABLED="true"
    )
    # Inherit the campaign supervisor's group: its pause/resume/force-stop must
    # reach the actual trainer as well as this wrapper.
    process = subprocess.Popen(argv, env=environment)

    def terminate() -> None:
        if process.poll() is not None:
            return
        import psutil

        try:
            owned = psutil.Process(process.pid)
            descendants = owned.children(recursive=True)
            targets = [*reversed(descendants), owned]
        except psutil.NoSuchProcess:
            process.wait(timeout=5)
            return
        for target in targets:
            try:
                target.terminate()
            except psutil.NoSuchProcess:
                pass
        _, survivors = psutil.wait_procs(targets, timeout=5)
        for target in survivors:
            try:
                target.kill()
            except psutil.NoSuchProcess:
                pass
        process.wait(timeout=5)

    def interrupted(signum, frame):
        terminate()
        raise InterruptedError("campaign_sft_interrupted")

    previous = {}
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        code = process.wait(timeout=timeout)
        if code:
            raise RuntimeError("campaign_sft_training_process_failed")
    finally:
        terminate()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_sft(
    config: CampaignSFTConfig,
    model_dir: Path,
    dataset: Path,
    output: Path,
    launch_manifest: Path,
    *,
    execute: Callable[[Path, Path, int], None] | None = None,
) -> dict:
    model = _regular_path(model_dir, directory=True)
    dataset = _regular_path(dataset)
    output = _regular_path(output, directory=True)
    _check_model(model)
    if output == model or output.is_relative_to(model) or model.is_relative_to(output):
        raise ValueError("campaign_sft_output_overlaps_model")
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != config.dataset_sha256:
        raise ValueError("campaign_sft_dataset_digest_mismatch")
    rows = _load_rows(dataset, config.max_examples)
    launch = json.loads(_regular_path(launch_manifest).read_text())
    sources = [
        launch[key] for key in ("registered_base_model", "remote_resident_model") if launch.get(key)
    ]
    if (
        launch.get("schema_version") != "campaign_remote_launch_manifest.v2"
        or len(sources) != 1
        or Path(sources[0]["remote_model_path"]).resolve() != model
        or not launch.get("run_id")
        or not launch.get("recipe_digest")
    ):
        raise ValueError("campaign_sft_launch_model_binding_mismatch")
    owned = [
        output / name
        for name in ("final", ".sft-work", "training_metrics.jsonl", "training_manifest.json")
    ]
    if any(path.exists() or path.is_symlink() for path in owned):
        raise FileExistsError("campaign_sft_output_already_exists")
    work = output / ".sft-work"
    work.mkdir()
    script = work / "train.py"
    metrics = output / "training_metrics.jsonl"
    result_path = work / "result.json"
    script.write_text(
        build_training_script(config, model, dataset, work, metrics), encoding="utf-8"
    )
    manifest = {
        "schema_version": "training_manifest.v1",
        "status": "running",
        "run_id": launch["run_id"],
        "recipe_digest": launch["recipe_digest"],
        "model_source": sources[0],
        "dataset_sha256": config.dataset_sha256,
        "train_examples": len(rows),
        "effective_config": config.model_dump(mode="json"),
        "generated_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "candidate_artifact": "final",
        "training_method": "lora_sft",
        "weight_dtype": "bfloat16",
        "artifact_retention": "deployable",
    }
    _write_json(output / "training_manifest.json", manifest)
    started = time.monotonic()
    try:
        if execute is not None:
            execute(script, result_path, config.max_seconds)
        else:
            bootstrap = "import sys; from bashgym.campaigns.sft_runner import _execute_generated; _execute_generated(*sys.argv[1:])"
            run_training_process(
                [
                    sys.executable,
                    "-c",
                    bootstrap,
                    str(script),
                    str(result_path),
                    str(dataset),
                    str(model),
                    str(config.max_seq_length),
                    config.backend,
                ],
                timeout=config.max_seconds,
            )
        result = json.loads(result_path.read_text())
        if result.get("global_step") != config.max_steps or not math.isfinite(
            result.get("train_loss", float("nan"))
        ):
            raise RuntimeError("campaign_sft_incomplete_training_result")
        merged = work / "merged"
        _check_model(merged)
        from bashgym.campaigns.metrics import parse_metric_lines

        points = parse_metric_lines(tuple(metrics.read_text().splitlines()))
        if not points or not any("loss" in p.values or "train_loss" in p.values for p in points):
            raise RuntimeError("campaign_sft_training_metrics_missing")
        merged.rename(output / "final")
        manifest.update(
            status="completed",
            optimizer_steps=result["global_step"],
            train_loss=result["train_loss"],
        )
    except BaseException as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        manifest.update(
            completed_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started,
        )
        _write_json(output / "training_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--launch-manifest", type=Path, default=Path("launch_manifest.json"))
    args = parser.parse_args(argv)
    config = CampaignSFTConfig.model_validate_json(args.config.read_text())
    run_sft(
        config,
        args.model_dir,
        args.dataset,
        args.output.absolute(),
        args.launch_manifest.absolute(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
