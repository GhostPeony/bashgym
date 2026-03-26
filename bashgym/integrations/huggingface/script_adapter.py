"""
Cloud Script Adapter for HuggingFace Unsloth Jobs

Transforms locally-generated Unsloth training scripts for cloud execution
via `hf jobs uv run`. Adds PEP 723 inline dependencies, HuggingFace Hub
dataset loading, trackio monitoring, and model push to Hub.

Usage:
    from bashgym.integrations.huggingface.script_adapter import (
        CloudScriptConfig, generate_cloud_script
    )

    config = CloudScriptConfig(
        strategy="sft",
        dataset_repo="username/my-dataset",
        output_repo="username/my-model",
        base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        hardware="a10g-small",
    )
    script = generate_cloud_script(config)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


PEP_723_HEADER = """# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth[cu124]",
#     "trl>=0.15",
#     "datasets",
#     "torch",
#     "trackio",
#     "huggingface_hub",
# ]
# ///
"""


@dataclass
class CloudScriptConfig:
    """Configuration for generating a cloud-ready training script."""

    strategy: str = "sft"
    """Training strategy: 'sft', 'dpo', or 'distillation'."""

    dataset_repo: str = ""
    """HuggingFace dataset repository (e.g., 'username/my-dataset')."""

    output_repo: str = ""
    """HuggingFace model repository to push results (e.g., 'username/my-model')."""

    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    """Base model to fine-tune."""

    hardware: str = "a10g-small"
    """Hardware tier for the job."""

    # Hyperparameters
    num_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1

    # DPO-specific
    dpo_beta: float = 0.1

    # Distillation-specific
    teacher_model: str = "claude-sonnet-4-20250514"
    distillation_alpha: float = 0.5
    teacher_temperature: float = 0.7

    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    def validate(self) -> list[str]:
        """Validate the configuration. Returns list of error messages."""
        errors = []
        if self.strategy not in ("sft", "dpo", "distillation"):
            errors.append(
                f"Invalid strategy '{self.strategy}'. Must be 'sft', 'dpo', or 'distillation'."
            )
        if not self.dataset_repo:
            errors.append("dataset_repo is required.")
        if not self.output_repo:
            errors.append("output_repo is required.")
        if not self.base_model:
            errors.append("base_model is required.")
        if self.num_epochs < 1:
            errors.append("num_epochs must be at least 1.")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive.")
        return errors


def generate_cloud_script(config: CloudScriptConfig) -> str:
    """
    Generate a cloud-ready Unsloth training script.

    Creates a temporary TrainerConfig + TrainingRun from the cloud config,
    calls the appropriate Trainer script generation method, then adapts
    the result for cloud execution.

    Args:
        config: Cloud script configuration.

    Returns:
        Complete Python script ready for `hf jobs uv run`.

    Raises:
        ValueError: If configuration is invalid.
    """
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid cloud script config: {'; '.join(errors)}")

    # Import trainer classes
    from bashgym.gym.trainer import Trainer, TrainerConfig, TrainingRun, TrainingStrategy

    # Map strategy string to enum
    strategy_map = {
        "sft": TrainingStrategy.SFT,
        "dpo": TrainingStrategy.DPO,
        "distillation": TrainingStrategy.DISTILLATION,
    }
    strategy_enum = strategy_map[config.strategy]

    # Build a TrainerConfig from the cloud config
    trainer_config = TrainerConfig(
        base_model=config.base_model,
        strategy=strategy_enum,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_epochs=config.num_epochs,
        max_seq_length=config.max_seq_length,
        warmup_ratio=config.warmup_ratio,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        dpo_beta=config.dpo_beta,
        teacher_model=config.teacher_model,
        distillation_alpha=config.distillation_alpha,
        teacher_temperature=config.teacher_temperature,
        output_dir="./output",
    )

    # Build a TrainingRun with placeholder paths (will be replaced by adapter)
    run = TrainingRun(
        run_id=f"cloud_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        strategy=strategy_enum,
        base_model=config.base_model,
        dataset_path=Path("__PLACEHOLDER_DATASET__"),
        output_path=Path("./output"),
    )

    # Create trainer and generate the raw script
    trainer = Trainer(trainer_config)

    if config.strategy == "sft":
        raw_script = trainer._generate_unsloth_sft_script(run)
    elif config.strategy == "dpo":
        raw_script = trainer._generate_unsloth_dpo_script(run)
    elif config.strategy == "distillation":
        raw_script = trainer._generate_distillation_script(run)
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    # Adapt for cloud execution
    return _adapt_for_cloud(raw_script, config)


def _adapt_for_cloud(script: str, config: CloudScriptConfig) -> str:
    """
    Transform a local Unsloth training script for cloud execution.

    Modifications:
    - Prepends PEP 723 inline dependency header
    - Replaces local dataset loading with HF Hub loading
    - Adds trackio monitoring
    - Adds model/tokenizer push to Hub
    - Replaces local output paths with ./output
    """
    adapted = script

    # 1. Replace local dataset loading with HF Hub dataset loading
    # Match patterns like: load_dataset("json", data_files="...")
    adapted = re.sub(
        r'load_dataset\(\s*"json"\s*,\s*data_files\s*=\s*"[^"]*"\s*,\s*split\s*=\s*"train"\s*\)',
        f'load_dataset("{config.dataset_repo}", split="train")',
        adapted,
    )
    # Also handle the placeholder
    adapted = re.sub(
        r'load_dataset\(\s*"json"\s*,\s*data_files\s*=\s*"__PLACEHOLDER_DATASET__"\s*,\s*split\s*=\s*"train"\s*\)',
        f'load_dataset("{config.dataset_repo}", split="train")',
        adapted,
    )

    # 1b. Replace local validation dataset loading block with Hub-based loading
    # The local scripts load val data from a file path; cloud scripts should use
    # the "test" split from the same Hub dataset (if it exists).
    val_block_pattern = (
        r'# Load validation dataset if available\n'
        r'[ \t]*val_dataset = None\n'
        r'[ \t]*val_dataset_path = "[^"]*"\n'
        r'[ \t]*if val_dataset_path and os\.path\.exists\(val_dataset_path\):\n'
        r'[ \t]*print\("Loading validation dataset\.\.\."\)\n'
        r'[ \t]*val_dataset = load_dataset\("json", data_files=val_dataset_path, split="train"\)\n'
        r'[ \t]*print\(f"Validation set: \{[^}]*\} examples"\)'
    )
    val_block_replacement = (
        f'# Load validation dataset from Hub (if "test" split exists)\n'
        f'    val_dataset = None\n'
        f'    try:\n'
        f'        val_dataset = load_dataset("{config.dataset_repo}", split="test")\n'
        f'        print(f"Validation set: {{len(val_dataset)}} examples")\n'
        f'    except ValueError:\n'
        f'        print("No test split found, skipping validation")'
    )
    adapted = re.sub(val_block_pattern, val_block_replacement, adapted)

    # Also handle top-level (unindented) validation blocks (e.g., in DPO scripts)
    val_block_pattern_toplevel = (
        r'# Load validation dataset if available\n'
        r'val_dataset = None\n'
        r'val_dataset_path = "[^"]*"\n'
        r'if val_dataset_path and os\.path\.exists\(val_dataset_path\):\n'
        r'[ \t]+print\("Loading validation dataset\.\.\."\)\n'
        r'[ \t]+val_dataset = load_dataset\("json", data_files=val_dataset_path, split="train"\)\n'
        r'[ \t]+print\(f"Validation set: \{[^}]*\} examples"\)'
    )
    val_block_replacement_toplevel = (
        f'# Load validation dataset from Hub (if "test" split exists)\n'
        f'val_dataset = None\n'
        f'try:\n'
        f'    val_dataset = load_dataset("{config.dataset_repo}", split="test")\n'
        f'    print(f"Validation set: {{len(val_dataset)}} examples")\n'
        f'except ValueError:\n'
        f'    print("No test split found, skipping validation")'
    )
    adapted = re.sub(val_block_pattern_toplevel, val_block_replacement_toplevel, adapted)

    # 2. Replace hardcoded output_dir paths with ./output
    adapted = re.sub(
        r'output_dir\s*=\s*"[^"]*"',
        'output_dir="./output"',
        adapted,
    )

    # 3. Replace save_pretrained paths
    adapted = re.sub(
        r'\.save_pretrained\(\s*"[^"]*?/final"\s*\)',
        '.save_pretrained("./output/final")',
        adapted,
    )
    adapted = re.sub(
        r'\.save_pretrained_merged\(\s*\n?\s*"[^"]*?/merged"',
        '.save_pretrained_merged(\n        "./output/merged"',
        adapted,
    )

    # 4. Add trackio import after existing imports
    trackio_import = "import trackio\ntrackio.init()\n"
    # Insert after the last import block
    import_end = _find_last_import_line(adapted)
    if import_end is not None:
        lines = adapted.split("\n")
        lines.insert(import_end + 1, "")
        lines.insert(import_end + 2, trackio_import.rstrip())
        adapted = "\n".join(lines)

    # 5. Add hub push after model saving
    hub_push = f"""
# Push to HuggingFace Hub
print("Pushing model to Hub...")
model.push_to_hub("{config.output_repo}")
tokenizer.push_to_hub("{config.output_repo}")
"""
    # Insert before the final "print" (Training/Distillation complete)
    adapted = re.sub(
        r'(print\("(?:Training|DPO training|Knowledge distillation) complete!"\))',
        hub_push.rstrip() + "\n\n\\1",
        adapted,
    )

    # 6. Prepend PEP 723 header (before the shebang/docstring)
    adapted = PEP_723_HEADER + "\n" + adapted

    return adapted


def _find_last_import_line(script: str) -> int | None:
    """Find the line index of the last top-level import statement."""
    lines = script.split("\n")
    last_import = None
    in_docstring = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track docstrings
        if '"""' in stripped:
            if in_docstring:
                in_docstring = False
                continue
            # Check for single-line docstring
            if stripped.count('"""') >= 2:
                continue
            in_docstring = True
            continue

        if in_docstring:
            continue

        # Match import lines (top-level only, not indented)
        if line and not line[0].isspace():
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = i

    return last_import
