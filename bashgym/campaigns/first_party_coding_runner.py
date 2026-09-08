"""Pinned, single-completion code benchmarks through the campaign evaluator ABI.

Generated Python runs only in Docker through the installed HumanEval checker.
This implements ordinary benchmark correctness, not adversarial reward integrity.
Profiles must collect both autoresearch_evaluation.json and coding_task_results.json.
Transformers max_time is cooperative; the execution adapter still needs a hard
worker/process deadline. Smoke scope exits 3 and cannot produce a completed seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from bashgym.campaigns.autoresearch_evidence import (
    MAX_AUTORESEARCH_EVALUATION_BYTES,
    AutoResearchEvaluationContext,
    AutoResearchEvaluationEvidence,
)
from bashgym.campaigns.contracts import FrozenContractModel, Identifier, utc_now
from bashgym.environments.contracts import BuildSpec, EnvironmentSpec, RolloutSpec, VerifierSpec
from bashgym.environments.docker_coding import DockerCodingEpisode, validate_pinned_image


class CodingRunnerConfig(FrozenContractModel):
    schema_version: Literal["first_party_coding_config.v1"]
    scope: Literal["smoke", "development"]
    split: str = Field(min_length=1, max_length=160)
    source: str = Field(min_length=1, max_length=160)
    revision: str = Field(min_length=1, max_length=160)
    expected_task_count: int = Field(strict=True, ge=1, le=10000)
    primary_metric: Identifier
    sandbox_image: str
    max_input_tokens: int = Field(strict=True, ge=1, le=131072)
    max_new_tokens: int = Field(strict=True, ge=1, le=32768)
    generation_timeout_seconds: float = Field(gt=0, le=3600, allow_inf_nan=False)
    test_timeout_seconds: float = Field(gt=0, le=300, allow_inf_nan=False)
    dtype: Literal["float32", "float16", "bfloat16"]
    device: str
    seed: int = Field(strict=True, ge=0, le=2**32 - 1)
    completion_protocol: Literal["raw", "humaneval_body_v1"]

    @field_validator("sandbox_image")
    @classmethod
    def pinned_image(cls, value):
        return validate_pinned_image(value)

    @field_validator("device")
    @classmethod
    def exact_device(cls, value):
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", value):
            raise ValueError("coding_device_invalid")
        return value


class CodingTaskProvenance(FrozenContractModel):
    split: str = Field(min_length=1, max_length=160)
    source: str = Field(min_length=1, max_length=160)
    revision: str = Field(min_length=1, max_length=160)


class CodingTask(FrozenContractModel):
    task_id: str = Field(min_length=1, max_length=240)
    prompt: str = Field(min_length=1, max_length=131072)
    test: str = Field(min_length=1, max_length=131072)
    entry_point: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    provenance: CodingTaskProvenance


def _read_file(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("coding_input_file_invalid")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("coding_input_file_invalid")
    return data


def load_tasks(path: Path, context: AutoResearchEvaluationContext, config: CodingRunnerConfig):
    data = _read_file(path, 64 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != context.dataset_content_digest:
        raise ValueError("coding_dataset_digest_mismatch")
    tasks = [CodingTask.model_validate_json(row) for row in data.splitlines() if row.strip()]
    if len(tasks) != config.expected_task_count or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("coding_dataset_task_count_invalid")
    expected = {key: getattr(config, key) for key in ("split", "source", "revision")}
    if any(task.provenance.model_dump() != expected for task in tasks):
        raise ValueError("coding_dataset_provenance_mismatch")
    return tasks


def validate_model_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir() or (path / "adapter_config.json").exists():
        raise ValueError("coding_requires_full_local_model")
    _read_file(path / "config.json", 1024 * 1024)
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise ValueError("coding_requires_full_local_model")
    if any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError("coding_requires_full_local_model")
    return path.resolve()


class LocalCompletion:
    """Local continuation with cooperative max_time; requires an outer hard deadline."""

    def __init__(self, model_directory: Path, config: CodingRunnerConfig):
        self.directory, self.config = model_directory, config
        self.model = self.tokenizer = None

    def __call__(self, prompt: str) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        config = self.config
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.directory, local_files_only=True, trust_remote_code=False
            )
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.directory,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=getattr(torch, config.dtype),
                    return_dict=True,
                )
                .to(config.device)
                .eval()
            )
        set_seed(config.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[-1] > config.max_input_tokens:
            raise ValueError("coding_prompt_token_limit")
        inputs = {key: value.to(config.device) for key, value in inputs.items()}
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=config.max_new_tokens,
                max_time=config.generation_timeout_seconds,
                return_dict_in_generate=True,
            )
        if time.monotonic() - started >= config.generation_timeout_seconds:
            raise TimeoutError("coding_generation_timeout")
        return self.tokenizer.decode(
            output.sequences[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )


def process_completion(completion: str, protocol: Literal["raw", "humaneval_body_v1"]) -> str:
    """Apply the declared textual stop convention without repairing generated code.

    Body completion ends at the first column-zero declaration, import, print,
    comment, conditional, or Markdown fence. Indented nested code is preserved.
    This is a fixed delimiter protocol, not Python parsing or semantic repair;
    it deliberately applies the same boundaries even inside multiline strings.
    """
    if protocol == "raw":
        return completion
    if protocol != "humaneval_body_v1":
        raise ValueError("coding_completion_protocol_invalid")
    boundary = re.search(
        r"^(?:def[ \t]|class[ \t]|if[ \t]|import[ \t]|from[ \t]|print(?:[ \t]|\()|#|```)",
        completion,
        flags=re.MULTILINE,
    )
    return completion[: boundary.start()] if boundary else completion


# check_correctness performs prompt + completion + test + check(entry_point).
# The checker and its dependencies must already exist in the pinned runtime image.
VERIFIER_SOURCE = """import json
from pathlib import Path
from human_eval.execution import check_correctness

payload = json.loads(Path("/workspace/task.json").read_text(encoding="utf-8"))
result = check_correctness(payload["problem"], payload["completion"], payload["timeout"])
status = "passed" if result["passed"] else ("test_timeout" if result["result"] == "timed out" else "failed")
print(json.dumps({"schema_version": "bashgym.coding_tests.v1", "tests_run": 1,
    "failures": int(not result["passed"]), "errors": 0, "skipped": 0, "status": status}))
"""


def evaluate_task(task: CodingTask, completion: str, config: CodingRunnerConfig) -> str:
    payload = {
        "problem": task.model_dump(exclude={"provenance"}),
        "completion": completion,
        "timeout": config.test_timeout_seconds,
    }
    spec = EnvironmentSpec(
        id="benchmark-" + hashlib.sha256(task.task_id.encode()).hexdigest()[:24],
        instruction="Evaluate the pinned code benchmark completion.",
        source="registered_code_benchmark",
        domain="coding",
        build=BuildSpec(dockerfile="", network_disabled=True),
        rollout=RolloutSpec(timeout_sec=config.test_timeout_seconds + 10),
        verifier=VerifierSpec(
            kind="coding_unittest",
            path="verify.py",
            timeout_sec=config.test_timeout_seconds + 5,
            metadata={"test_count": 1},
        ),
        files={"verify.py": VERIFIER_SOURCE, "task.json": json.dumps(payload)},
        metadata={"protected_paths": ["task.json"]},
    )
    with DockerCodingEpisode(spec, image=config.sandbox_image) as episode:
        observation = episode.run("python -I /workspace/verify.py", verifier=True)
        if (
            observation.timeout
            or observation.blocked
            or observation.exit_code != 0
            or episode.tampered()
        ):
            return "infrastructure_error"
        try:
            report = json.loads(observation.stdout.splitlines()[-1])
            status = report["status"]
            if (
                report["schema_version"] != "bashgym.coding_tests.v1"
                or type(report["tests_run"]) is not int
                or report["tests_run"] != 1
                or report["errors"] != 0
                or report["skipped"] != 0
                or status not in {"passed", "failed", "test_timeout"}
                or type(report["failures"]) is not int
                or report["failures"] != int(status != "passed")
            ):
                raise ValueError("coding_report_invalid")
            return status
        except (KeyError, TypeError, ValueError, IndexError):
            return "infrastructure_error"


def _error_category(exc: Exception, phase: str) -> str:
    """Keep actionable categories without copying paths, prompts, or backend secrets."""
    message = str(exc)
    if re.fullmatch(r"coding_[a-z_]{1,100}", message):
        return message
    normalized = message.lower()
    if isinstance(exc, ImportError):
        category = "missing_dependency"
    elif "out of memory" in normalized:
        category = "out_of_memory"
    elif "not recognize this architecture" in normalized or "unsupported" in normalized:
        category = "unsupported_model_or_runtime"
    elif isinstance(exc, OSError):
        category = "io_error"
    elif isinstance(exc, ValueError):
        category = "invalid_runtime_input"
    else:
        category = "runtime_error"
    return phase + "_" + category


def run(
    context_path: Path,
    model_directory: Path,
    dataset_path: Path,
    output_path: Path,
    config_path: Path,
    *,
    complete=None,
    evaluate=None,
) -> AutoResearchEvaluationEvidence:
    context = AutoResearchEvaluationContext.model_validate_json(_read_file(context_path, 65536))
    config = CodingRunnerConfig.model_validate_json(_read_file(config_path, 65536))
    tasks = load_tasks(dataset_path, context, config)
    model_directory = validate_model_directory(model_directory)
    task_results_path = output_path.with_name("coding_task_results.json")
    if output_path == task_results_path or any(
        path.is_symlink() or path.exists() for path in (output_path, task_results_path)
    ):
        raise ValueError("coding_output_already_exists")
    complete = complete or LocalCompletion(model_directory, config)
    evaluate = evaluate or evaluate_task
    started = utc_now()
    counts = dict.fromkeys(
        (
            "passed",
            "failed",
            "test_timeout",
            "generation_timeout",
            "infrastructure_error",
            "not_run",
        ),
        0,
    )
    outcomes = []
    for index, task in enumerate(tasks):
        try:
            completion = complete(task.prompt)
            if not isinstance(completion, str) or len(completion.encode()) > 1024 * 1024:
                raise ValueError("coding_completion_invalid")
            completion = process_completion(completion, config.completion_protocol)
        except TimeoutError:
            counts["generation_timeout"] += 1
            counts["not_run"] = len(tasks) - index - 1
            outcomes.append(
                {
                    "task_id": task.task_id,
                    "status": "generation_timeout",
                    "error_category": "coding_generation_timeout",
                }
            )
            break
        except Exception as exc:
            counts["infrastructure_error"] += 1
            counts["not_run"] = len(tasks) - index - 1
            outcomes.append(
                {
                    "task_id": task.task_id,
                    "status": "infrastructure_error",
                    "error_category": _error_category(exc, "generation"),
                }
            )
            break
        error_category = None
        try:
            status = evaluate(task, completion, config)
        except Exception as exc:
            status = "infrastructure_error"
            error_category = _error_category(exc, "evaluation")
        if status not in {"passed", "failed", "test_timeout", "infrastructure_error"}:
            status = "infrastructure_error"
            error_category = "evaluation_invalid_status"
        counts[status] += 1
        outcome = {"task_id": task.task_id, "status": status}
        if status == "infrastructure_error":
            outcome["error_category"] = error_category or "evaluation_infrastructure_error"
        outcomes.append(outcome)
        if status == "infrastructure_error":
            counts["not_run"] = len(tasks) - index - 1
            break
    outcomes.extend(
        {"task_id": task.task_id, "status": "not_run"} for task in tasks[len(outcomes) :]
    )
    task_results = json.dumps(
        {
            "schema_version": "coding_task_results.v1",
            "scope": config.scope,
            "completion_protocol": config.completion_protocol,
            "attempt_id": context.attempt_id,
            "dataset_content_digest": context.dataset_content_digest,
            "evaluated_model_manifest_digest": context.evaluated_model_manifest_digest,
            "tasks": outcomes,
        },
        sort_keys=True,
    ).encode()
    if len(task_results) > 4 * 1024 * 1024:
        raise ValueError("coding_task_results_too_large")
    evidence = AutoResearchEvaluationEvidence(
        **context.model_dump(exclude={"schema_version", "workspace_id", "dataset_content_digest"}),
        metrics={config.primary_metric: counts["passed"] / len(tasks)},
        slice_metrics={
            "coding_benchmark": {
                "scope": config.scope,
                "completion_protocol": config.completion_protocol,
                "split": config.split,
                "task_count": len(tasks),
                "task_results_file": task_results_path.name,
                "task_results_sha256": hashlib.sha256(task_results).hexdigest(),
                **counts,
                "complete": not (counts["infrastructure_error"] or counts["generation_timeout"]),
                "adversarial_reward_integrity_verified": False,
            }
        },
        started_at=started,
        completed_at=utc_now(),
    )
    encoded = evidence.model_dump_json(indent=2).encode()
    if len(encoded) > MAX_AUTORESEARCH_EVALUATION_BYTES:
        raise ValueError("coding_evidence_too_large")
    with task_results_path.open("xb") as handle:
        handle.write(task_results)
    with output_path.open("xb") as handle:
        handle.write(encoded)
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("context", "model-dir", "dataset", "output", "config"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.context, args.model_dir, args.dataset, args.output, args.config)
    summary = result.slice_metrics["coding_benchmark"]
    if not summary["complete"]:
        return 2
    return 0 if summary["scope"] == "development" else 3


if __name__ == "__main__":
    raise SystemExit(main())
