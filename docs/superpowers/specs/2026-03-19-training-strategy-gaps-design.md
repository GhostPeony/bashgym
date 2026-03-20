# Training Strategy Gaps: Distillation, GRPO, RLVR

**Date**: 2026-03-19
**Status**: Approved
**Context**: BashGym was selected for AI Tinkerer's newsletter. The description claims GRPO, RLVR, and distillation — three training strategies that generate scripts but return fake metrics instead of executing them. This spec fixes all three.

---

## Scope

Changes across `bashgym/gym/trainer.py`, `bashgym/api/routes.py`, `bashgym/api/schemas.py`, and `requirements-training.txt`:

1. **Distillation** — wire `_train_with_distillation()` to execute its generated script
2. **GRPO** — rewrite generated script to use `trl.GRPOTrainer` with tiered reward functions, wire `_run_grpo_loop()` to execute it
3. **RLVR** — thin wrapper that runs GRPO with `reward_mode="verification"`
4. **API schemas** — add `RLVR` to schemas.py `TrainingStrategy` enum, add `grpo_reward_mode` to `TrainingRequest`
5. **API routes** — update dispatch for RLVR strategy, wire `log_callback`/`pid_callback` for all strategies
6. **TrainerConfig** — add missing `weight_decay` and `use_gradient_checkpointing` fields (pre-existing bug fix)
7. **Dependencies** — bump `trl>=0.15.0` in `requirements-training.txt`

---

## 1. Distillation Fix

### Problem

`_train_with_distillation()` (line 899) generates a valid training script via `_generate_distillation_script()` but never executes it. Instead it returns hardcoded fake metrics.

### Solution

Replace the simulation block (lines 917-936) with subprocess execution matching the SFT pattern in `_train_with_unsloth_sft()` (line 465):

1. Call `self._get_training_python()` for CUDA-capable Python
2. `subprocess.Popen([python_exe, str(script_path)], ...)` with stdout streaming
3. Parse HuggingFace training output: loss, epoch, step, grad_norm, tqdm progress
4. Call `callback()` with parsed metrics on each update
5. Call `log_callback()` with raw log lines for WebSocket streaming
6. Call `pid_callback()` for process control (suspend/resume)
7. Track loss curve via `run.add_loss_point()`
8. Set `run.metrics` from final parsed values
9. Raise `RuntimeError` on non-zero exit code

### Method signature update

```python
def _train_with_distillation(
    self,
    run: TrainingRun,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    pid_callback: Optional[Callable[[int, "TrainingRun"], None]] = None,
) -> None:
```

### Script template changes

None. The generated script (`_generate_distillation_script()`) is already correct — loads student with Unsloth, trains with SFTTrainer on teacher-generated data, saves merged model. The `distillation_loss` function defined in the script is available for future logit-level distillation but the current offline approach (training on teacher traces) is the correct implementation.

### `train_distillation()` update

Pass through `log_callback` and `pid_callback` from caller to `_train_with_distillation()`.

---

## 2. GRPO Fix

### Problem

`_run_grpo_loop()` (line 1504) generates a script via `_generate_grpo_script()` but never executes it. The generated script uses the old `PPOTrainer` API and has placeholder rewards (`rewards = [0.5 for _ in responses]`).

### 2a. New `TrainerConfig` fields

```python
grpo_reward_mode: str = "syntax"  # "syntax", "execution", "verification"

# Pre-existing bug fix: these fields are referenced by _save_model_profile() but never defined
weight_decay: float = 0.01
use_gradient_checkpointing: bool = True
```

### 2b. New generated script

Replace `_generate_grpo_script()` entirely. New script structure:

```python
#!/usr/bin/env python3
"""GRPO Training Script — trl.GRPOTrainer with tiered rewards"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import torch, ast, subprocess, tempfile, os, sys, re

# === Configuration (templated from TrainerConfig) ===
REWARD_MODE = "{self.config.grpo_reward_mode}"
NUM_GENERATIONS = {self.config.grpo_num_generations}

# === Reward Functions ===

def syntax_reward(completions, **kwargs):
    """Tier 1: AST parse check. 1.0 if valid Python, 0.0 if not."""
    rewards = []
    for completion in completions:
        # Extract code from completion
        code = extract_code(completion[0]["content"])
        try:
            ast.parse(code)
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(0.0)
    return rewards

def execution_reward(completions, **kwargs):
    """Tier 2: Subprocess execution. 1.0 if exit 0, 0.0 otherwise."""
    rewards = []
    for completion in completions:
        code = extract_code(completion[0]["content"])
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, timeout=10
            )
            rewards.append(1.0 if result.returncode == 0 else 0.0)
        except (subprocess.TimeoutExpired, Exception):
            rewards.append(0.0)
    return rewards

def verification_reward(completions, prompts, tests, **kwargs):
    """Tier 3: Run test cases. Returns passed/total."""
    rewards = []
    for completion, test_code in zip(completions, tests):
        code = extract_code(completion[0]["content"])
        reward = run_verification(code, test_code)
        rewards.append(reward)
    return rewards

def extract_code(text):
    """Extract code block from model output."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

def run_verification(code, test_code):
    """Execute code + tests in temp dir, return passed/total."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write solution
        sol_path = os.path.join(tmpdir, "solution.py")
        with open(sol_path, "w") as f:
            f.write(code)
        # Write tests
        test_path = os.path.join(tmpdir, "test_solution.py")
        with open(test_path, "w") as f:
            f.write(test_code)
        # Run pytest
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
                capture_output=True, text=True, timeout=30, cwd=tmpdir
            )
            # Parse "X passed, Y failed"
            output = result.stdout
            passed = len(re.findall(r"PASSED", output))
            failed = len(re.findall(r"FAILED", output))
            total = passed + failed
            return passed / total if total > 0 else 0.0
        except (subprocess.TimeoutExpired, Exception):
            return 0.0

# Select reward function
REWARD_FN = {
    "syntax": syntax_reward,
    "execution": execution_reward,
    "verification": verification_reward,
}[REWARD_MODE]

# === Model Setup (Unsloth + LoRA) ===
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, ...)

# === Dataset ===
dataset = load_dataset("json", data_files="...", split="train")

# === GRPO Training ===
grpo_config = GRPOConfig(
    output_dir="...",
    num_generations=NUM_GENERATIONS,
    per_device_train_batch_size=...,
    gradient_accumulation_steps=...,
    learning_rate=...,
    num_train_epochs=...,
    logging_steps=...,
    save_steps=...,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,  # trl uses processing_class, not tokenizer
    reward_funcs=[REWARD_FN],
    train_dataset=dataset,
    args=grpo_config,
)

trainer.train()

# Save model
model.save_pretrained(".../final")
tokenizer.save_pretrained(".../final")
model.save_pretrained_merged(".../merged", tokenizer, save_method="merged_16bit")
```

### 2c. Wire execution in `_run_grpo_loop()`

Replace simulation block (lines 1528-1544) with subprocess execution matching the SFT pattern:

- `subprocess.Popen` with `stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(Path.cwd())`
- Parse GRPO-specific metrics from trl output: `reward`, `kl`, `policy_loss`, plus standard loss/epoch/step
- Support `log_callback` and `pid_callback`
- Track loss curve

### Method signature update

```python
def _run_grpo_loop(
    self,
    run: TrainingRun,
    verifier_fn: Callable[[str, str], float],  # kept for API compat, not used in script
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    pid_callback: Optional[Callable[[int, "TrainingRun"], None]] = None,
) -> None:
```

### `train_grpo()` signature update

Add `log_callback` and `pid_callback` parameters, pass through to `_run_grpo_loop()`.

---

## 3. RLVR Implementation

### Problem

`RLVR = "rlvr"` exists in `TrainingStrategy` enum but has zero implementation.

### Solution

RLVR = GRPO with `reward_mode="verification"`. Add to `Trainer` class (not `GRPOTrainer`):

```python
import copy

def train_rlvr(
    self,
    dataset_path: Path,
    run_id: Optional[str] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    pid_callback: Optional[Callable[[int, "TrainingRun"], None]] = None,
) -> TrainingRun:
    """
    Run RL with Verifiable Rewards.

    This is GRPO with verification-based reward signals.
    Dataset must include 'tests' field with pytest-compatible test code.
    """
    # Use a deep copy to avoid thread-safety issues with shared config (mutable list fields)
    grpo_config = copy.deepcopy(self.config)
    grpo_config.grpo_reward_mode = "verification"

    grpo_trainer = GRPOTrainer(grpo_config)
    run = grpo_trainer.train_grpo(
        dataset_path=dataset_path,
        verifier_fn=lambda p, r: 0.0,  # Not used — reward is in-script
        run_id=run_id,
        callback=callback,
        log_callback=log_callback,
        pid_callback=pid_callback,
    )
    # Override strategy label
    run.strategy = TrainingStrategy.RLVR
    return run
```

---

## 4. API Schema Updates

### `schemas.py`

Add `RLVR` to the `TrainingStrategy` enum (currently missing — API would reject RLVR requests):

```python
class TrainingStrategy(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    RLVR = "rlvr"            # NEW
    DISTILLATION = "distillation"
```

Add `grpo_reward_mode` to `TrainingRequest`:

```python
grpo_reward_mode: Optional[str] = "syntax"  # "syntax", "execution", "verification"
```

---

## 5. API Route Updates

### `routes.py` dispatch

**Fix GRPO routing bug**: Currently `app.state.trainer` is `Trainer` (not `GRPOTrainer`), so `hasattr(app.state.trainer, 'train_grpo')` is always `False` and GRPO always falls through to simulation. Fix by instantiating `GRPOTrainer` in the dispatch block (same pattern as the RLVR wrapper):

```python
elif request.strategy == TrainingStrategy.GRPO:
    grpo_trainer = GRPOTrainer(app.state.trainer.config)
    run = grpo_trainer.train_grpo(
        dataset_path=dataset_path,
        verifier_fn=lambda p, r: 0.0,
        run_id=run_id,
        callback=callback.on_progress_sync,
        log_callback=log_callback,
        pid_callback=pid_callback,
    )
```

Add RLVR dispatch:

```python
elif request.strategy == TrainingStrategy.RLVR:
    run = app.state.trainer.train_rlvr(
        dataset_path=dataset_path,
        run_id=run_id,
        callback=callback.on_progress_sync,
        log_callback=log_callback,
        pid_callback=pid_callback,
    )
```

Remove the GRPO fallback simulation block.

Wire `grpo_reward_mode` from request to config before dispatch.

Update distillation dispatch to pass `log_callback` and `pid_callback`.

---

## 6. Dependency Update

In `requirements-training.txt`:

```
trl>=0.15.0  # was >=0.7.0, need GRPOTrainer
```

---

## Dataset Format

### SFT / DPO / Distillation

No change. Existing NeMo JSONL format with `messages` array.

### GRPO (syntax/execution mode)

```json
{"prompt": [{"role": "user", "content": "Write a function that reverses a string"}]}
```

### GRPO/RLVR (verification mode)

```json
{
  "prompt": [{"role": "user", "content": "Write a function that reverses a string"}],
  "tests": "from solution import *\n\ndef test_reverse():\n    assert reverse_string('hello') == 'olleh'\n    assert reverse_string('') == ''"
}
```

---

## Files Modified

| File | Change |
|------|--------|
| `bashgym/gym/trainer.py` | Fix `_train_with_distillation()`, rewrite `_generate_grpo_script()`, fix `_run_grpo_loop()`, add `train_rlvr()`, add `grpo_reward_mode`/`weight_decay`/`use_gradient_checkpointing` to config |
| `bashgym/api/routes.py` | Fix GRPO routing bug, add RLVR dispatch, remove simulation fallback, wire `log_callback`/`pid_callback` for distillation |
| `bashgym/api/schemas.py` | Add `RLVR` to `TrainingStrategy` enum, add `grpo_reward_mode` to `TrainingRequest` |
| `requirements-training.txt` | Bump `trl>=0.15.0` |

---

## What This Does NOT Change

- SFT and DPO training (already working)
- Trace capture pipeline
- Frontend UI (strategies already listed in dropdowns; reward mode is API-configurable only for now)
- Remote SSH training (scripts are executed the same way locally or remotely — adding remote SSH for distillation/GRPO is a follow-up)
- AutoResearch system
- Provider abstraction layer

---

## Verification Plan

1. Generate a small test dataset with prompts + tests
2. Run distillation with a real dataset, confirm subprocess executes and metrics stream
3. Run GRPO with `reward_mode="syntax"`, confirm trl.GRPOTrainer executes
4. Run RLVR (verification mode) with test dataset, confirm rewards compute from test results
5. Check WebSocket log streaming works for all three strategies
