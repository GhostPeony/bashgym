"""Offline-first LoRA SFT and evaluation for the star-count VLM environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from bashgym.environments.star_count import score_star_count_prediction

_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_LORA_EXCLUDES = ("vision_tower", "multi_modal_projector", "audio_tower")
_MAX_ARCHIVE_FILES = 10_000
_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_extracted_dataset(root: Path) -> None:
    """Verify the generator identity and every content-addressed dataset member."""

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("star-count archive is missing its manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        dataset_digest = manifest["dataset_digest"]
        identity = {key: value for key, value in manifest.items() if key != "dataset_digest"}
        expected_digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if dataset_digest != expected_digest or not isinstance(files, list):
            raise ValueError
        declared_paths: set[str] = set()
        for item in files:
            relative = str(item["path"])
            path = (root / relative).resolve()
            path.relative_to(root)
            if relative in declared_paths or not path.is_file():
                raise ValueError
            declared_paths.add(relative)
            if path.stat().st_size != int(item["size_bytes"]) or _sha256(path) != item["sha256"]:
                raise ValueError
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual_paths != declared_paths:
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("star-count archive manifest verification failed") from exc


@dataclass(frozen=True)
class StarCountVLMRecipe:
    model_id: str
    model_revision: str
    max_steps: int = 160
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    max_length: int = 1024
    seed: int = 42
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id is required")
        if not _IMMUTABLE_REVISION.fullmatch(self.model_revision):
            raise ValueError("model_revision must be an immutable revision")
        if self.max_steps <= 0 or self.batch_size <= 0:
            raise ValueError("training step and batch limits must be positive")
        if self.gradient_accumulation_steps <= 0 or self.max_length <= 0:
            raise ValueError("accumulation and sequence limits must be positive")
        if self.learning_rate <= 0 or self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA hyperparameters must be positive")


def load_star_count_records(jsonl_path: str | Path) -> list[dict[str, Any]]:
    path = Path(jsonl_path).expanduser().resolve()
    root = path.parent
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        image_path = (root / str(record.get("image", ""))).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"star-count image escapes dataset root at line {line_number}"
            ) from exc
        if not image_path.is_file():
            raise ValueError(f"star-count image is missing at line {line_number}")
        if not isinstance(record.get("messages"), list) or len(record["messages"]) < 2:
            raise ValueError(f"star-count messages are invalid at line {line_number}")
        if not isinstance(record.get("counts"), dict):
            raise ValueError(f"star-count labels are invalid at line {line_number}")
        records.append({**record, "image_path": image_path})
    if not records:
        raise ValueError("star-count dataset is empty")
    return records


def extract_star_count_archive(
    archive_path: str | Path, destination: str | Path
) -> Path:
    """Extract one bounded dataset archive without links or path traversal."""

    archive = Path(archive_path).expanduser().resolve()
    root = Path(destination).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"star-count extraction directory is not empty: {root.name}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > _MAX_ARCHIVE_FILES:
            raise ValueError("star-count archive contains too many files")
        if sum(member.file_size for member in members) > _MAX_ARCHIVE_BYTES:
            raise ValueError("star-count archive exceeds the extraction limit")
        for member in members:
            parts = Path(member.filename.replace("\\", "/")).parts
            mode = member.external_attr >> 16
            if (
                not parts
                or Path(member.filename).is_absolute()
                or ".." in parts
                or stat.S_ISLNK(mode)
            ):
                raise ValueError("star-count archive contains an unsafe path")
            target = (root / Path(*parts)).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError("star-count archive contains an unsafe path") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    _verify_extracted_dataset(root)
    return root


class StarCountVLMDataCollator:
    """Batch image/text examples while supervising only the assistant answer."""

    def __init__(self, processor: Any, *, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = []
        full_texts = []
        prompt_texts = []
        for example in examples:
            with Image.open(example["image_path"]) as image:
                images.append(image.convert("RGB").copy())
            messages = example["messages"]
            full_texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
            )

        batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = batch["input_ids"].clone()
        for index, (prompt_text, image) in enumerate(zip(prompt_texts, images, strict=True)):
            prompt = self.processor(
                text=prompt_text,
                images=[image],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prompt_length = int(prompt["attention_mask"][0].sum().item())
            labels[index, : min(prompt_length, labels.shape[1])] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return dict(batch)


class _RecordDataset:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _model_classes():
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    return AutoProcessor, AutoVLM


def _load_base(recipe: StarCountVLMRecipe):
    import torch

    from bashgym.families.patches import apply_patches

    apply_patches(["gemma4_clippable_linear"])
    auto_processor, auto_vlm = _model_classes()
    common = {
        "revision": recipe.model_revision,
        "local_files_only": recipe.local_files_only,
        "trust_remote_code": False,
    }
    processor = auto_processor.from_pretrained(recipe.model_id, **common)
    model = auto_vlm.from_pretrained(
        recipe.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **common,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, processor


def train_star_count_lora(
    recipe: StarCountVLMRecipe,
    *,
    train_jsonl: str | Path,
    validation_jsonl: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train a language-side LoRA while retaining image processor inputs."""

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(recipe.seed)
    model, processor = _load_base(recipe)
    model = get_peft_model(
        model,
        LoraConfig(
            r=recipe.lora_rank,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=0.0,
            target_modules=list(_LORA_TARGETS),
            exclude_modules=list(_LORA_EXCLUDES),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    train_records = load_star_count_records(train_jsonl)
    validation_records = load_star_count_records(validation_jsonl)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            per_device_train_batch_size=recipe.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=recipe.gradient_accumulation_steps,
            max_steps=recipe.max_steps,
            learning_rate=recipe.learning_rate,
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=max(10, recipe.max_steps // 4),
            save_strategy="steps",
            save_steps=max(10, recipe.max_steps // 4),
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            report_to="none",
            seed=recipe.seed,
        ),
        train_dataset=_RecordDataset(train_records),
        eval_dataset=_RecordDataset(validation_records),
        data_collator=StarCountVLMDataCollator(processor, max_length=recipe.max_length),
    )
    result = trainer.train()
    adapter_dir = output / "final_adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    summary = {
        "schema_version": "star_count_training_result.v1",
        "model_id": recipe.model_id,
        "model_revision": recipe.model_revision,
        "training_method": "lora_sft",
        "max_steps": recipe.max_steps,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "train_loss": float(result.training_loss),
        "adapter_artifact": "final_adapter",
    }
    (output / "training_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def summarize_star_count_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("star-count evaluation requires examples")
    scores = [
        score_star_count_prediction(str(row["prediction"]), row["counts"])
        for row in rows
    ]
    count_accuracy = sum(score.count_accuracy for score in scores) / len(scores)
    format_accuracy = sum(score.format_accuracy for score in scores) / len(scores)
    return {
        "primary_metric": "exact_count_accuracy",
        "exact_count_accuracy": sum(score.exact for score in scores) / len(scores),
        "count_accuracy": count_accuracy,
        "format_accuracy": format_accuracy,
        "mean_reward": 0.95 * count_accuracy + 0.05 * format_accuracy,
        "example_count": len(scores),
    }


def evaluate_star_count_model(
    recipe: StarCountVLMRecipe,
    *,
    heldout_jsonl: str | Path,
    output_path: str | Path,
    adapter_path: str | Path | None = None,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    import torch

    model, processor = _load_base(recipe)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, Path(adapter_path).expanduser().resolve())
    model.eval()
    model.to("cuda")
    rows = []
    for record in load_star_count_records(heldout_jsonl):
        with Image.open(record["image_path"]) as image:
            rgb = image.convert("RGB")
        prompt = processor.apply_chat_template(
            record["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=prompt, images=[rgb], return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        decoder = getattr(processor, "decode", None) or processor.tokenizer.decode
        prediction = decoder(new_tokens, skip_special_tokens=True).strip()
        rows.append(
            {
                "example_id": record["example_id"],
                "prediction": prediction,
                "counts": record["counts"],
            }
        )
    metrics = summarize_star_count_predictions(rows)
    payload = {
        "schema_version": "star_count_evaluation_result.v1",
        "model_id": recipe.model_id,
        "model_revision": recipe.model_revision,
        "adapter_evaluated": adapter_path is not None,
        "metrics": metrics,
        "predictions": rows,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BashGym star-count VLM runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--model-id", required=True)
        command.add_argument("--model-revision", required=True)
        command.add_argument("--allow-download", action="store_true")
    train = subparsers.choices["train"]
    train.add_argument("--train-jsonl")
    train.add_argument("--validation-jsonl")
    train.add_argument("--dataset-archive")
    train.add_argument("--output", required=True)
    train.add_argument("--max-steps", type=int, default=160)
    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument("--heldout-jsonl")
    evaluate.add_argument("--dataset-archive")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--adapter")
    args = parser.parse_args(argv)
    recipe = StarCountVLMRecipe(
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_steps=getattr(args, "max_steps", 160),
        local_files_only=not args.allow_download,
    )
    if args.command == "train":
        if args.dataset_archive:
            dataset_root = extract_star_count_archive(
                args.dataset_archive, Path(args.output).expanduser().resolve().parent / "dataset"
            )
            train_jsonl = dataset_root / "train.jsonl"
            validation_jsonl = dataset_root / "validation.jsonl"
        else:
            if not args.train_jsonl or not args.validation_jsonl:
                parser.error("train requires --dataset-archive or both split JSONL paths")
            train_jsonl = args.train_jsonl
            validation_jsonl = args.validation_jsonl
        result = train_star_count_lora(
            recipe,
            train_jsonl=train_jsonl,
            validation_jsonl=validation_jsonl,
            output_dir=args.output,
        )
    else:
        heldout_jsonl = args.heldout_jsonl
        if args.dataset_archive:
            dataset_root = extract_star_count_archive(
                args.dataset_archive, Path(args.output).expanduser().resolve().parent / "dataset"
            )
            heldout_jsonl = dataset_root / "heldout.jsonl"
        elif not heldout_jsonl:
            parser.error("evaluate requires --dataset-archive or --heldout-jsonl")
        result = evaluate_star_count_model(
            recipe,
            heldout_jsonl=heldout_jsonl,
            output_path=args.output,
            adapter_path=args.adapter,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "StarCountVLMDataCollator",
    "StarCountVLMRecipe",
    "evaluate_star_count_model",
    "extract_star_count_archive",
    "load_star_count_records",
    "main",
    "summarize_star_count_predictions",
    "train_star_count_lora",
]
