# Training Strategy Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make distillation, GRPO, and RLVR training strategies execute real training instead of returning fake metrics.

**Architecture:** Each strategy generates a Python training script and executes it via subprocess (same pattern as the working SFT implementation). GRPO uses `trl.GRPOTrainer` with three tiered reward functions. RLVR is a thin wrapper that runs GRPO with verification-based rewards.

**Tech Stack:** Python, trl (GRPOTrainer/GRPOConfig), Unsloth, subprocess, FastAPI

**Spec:** `docs/superpowers/specs/2026-03-19-training-strategy-gaps-design.md`

---

### Task 1: Add missing TrainerConfig fields

**Files:**
- Modify: `bashgym/gym/trainer.py:52-116` (TrainerConfig dataclass)

- [ ] **Step 1: Add three missing fields to TrainerConfig**

In `bashgym/gym/trainer.py`, after the `grpo_temperature` field (line 89), add:

```python
    grpo_reward_mode: str = "syntax"  # "syntax", "execution", "verification"
```

After the `use_flash_attention` field (line 108), add:

```python
    # Missing fields referenced by _save_model_profile()
    weight_decay: float = 0.01
    use_gradient_checkpointing: bool = True
```

- [ ] **Step 2: Verify no import errors**

Run: `python -c "from bashgym.gym.trainer import TrainerConfig; c = TrainerConfig(); print(c.grpo_reward_mode, c.weight_decay, c.use_gradient_checkpointing)"`
Expected: `syntax 0.01 True`

- [ ] **Step 3: Commit**

```bash
git add bashgym/gym/trainer.py
git commit -m "fix: add missing TrainerConfig fields (grpo_reward_mode, weight_decay, use_gradient_checkpointing)"
```

---

### Task 2: Wire distillation subprocess execution

**Files:**
- Modify: `bashgym/gym/trainer.py:845-936` (train_distillation + _train_with_distillation)

- [ ] **Step 1: Update `train_distillation()` signature**

In `bashgym/gym/trainer.py`, replace the `train_distillation` method signature (lines 845-850) with:

```python
    def train_distillation(
        self,
        dataset_path: Path,
        run_id: Optional[str] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        pid_callback: Optional[Callable[[int, "TrainingRun"], None]] = None,
    ) -> TrainingRun:
```

Update the call at line 880 to pass through the new params:

```python
            self._train_with_distillation(run, callback, log_callback, pid_callback)
```

- [ ] **Step 2: Fix path normalization in `_generate_distillation_script()`**

**CRITICAL (Windows bug):** The distillation script template at line 938 uses `{run.dataset_path}` and `{run.output_path}` directly without forward-slash normalization. On Windows, backslashes become escape sequences in the generated Python strings (`\U`, `\t`, etc.), crashing the script.

At the top of `_generate_distillation_script()`, add path normalization (same pattern as SFT/DPO):

```python
    def _generate_distillation_script(self, run: TrainingRun) -> str:
        """Generate Knowledge Distillation training script."""
        # Use forward slashes for cross-platform compatibility
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")
```

Then replace all `{run.dataset_path}` with `{dataset_path}` and `{run.output_path}` with `{output_path}` in the template string.

- [ ] **Step 3: Replace `_train_with_distillation()` simulation with real execution**

Replace the entire `_train_with_distillation` method (lines 899-936) with subprocess execution. The new method should match `_train_with_unsloth_sft()` (lines 500-635) exactly, with these name changes:

- Log prefix: `[Distillation]` instead of `[Training]`
- Script filename: already `train_distillation.py` (set at line 907)

Here is the complete replacement method:

```python
    def _train_with_distillation(
        self,
        run: TrainingRun,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        pid_callback: Optional[Callable[[int, "TrainingRun"], None]] = None,
    ) -> None:
        """Train using knowledge distillation (offline — teacher traces as training data)."""
        import re

        script_content = self._generate_distillation_script(run)
        script_path = run.output_path / "train_distillation.py"
        run.output_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)

        python_exe = self._get_training_python()
        logger.info(f"Starting Distillation run: {run.run_id}")
        logger.info(f"Teacher: {self.config.teacher_model}")
        logger.info(f"Student: {self.config.base_model}")
        logger.info(f"Dataset: {run.dataset_path}")
        logger.info(f"Output: {run.output_path}")
        logger.info(f"Script: {script_path}")
        logger.info(f"Python: {python_exe}")

        try:
            process = subprocess.Popen(
                [python_exe, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd())
            )

            run.pid = process.pid
            logger.info(f"Distillation subprocess started with PID {process.pid}")

            if pid_callback:
                try:
                    pid_callback(process.pid, run)
                except Exception as e:
                    logger.warning(f"pid_callback error: {e}")

            last_loss = None
            last_epoch = 0
            last_step = 0
            last_grad_norm = None
            samples_processed = 0
            start_time = datetime.now(timezone.utc)
            estimated_total_steps = 1000

            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"[Distillation] {line}")

                    if log_callback:
                        try:
                            log_callback(line)
                        except Exception as e:
                            logger.warning(f"Log callback error: {e}")

                    loss_match = re.search(r"'loss':\s*([\d.]+)", line)
                    epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
                    step_match = re.search(r"'step':\s*(\d+)", line)
                    grad_norm_match = re.search(r"'grad_norm':\s*([\d.]+)", line)
                    progress_match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)
                    unsloth_steps_match = re.search(r"Total steps\s*=\s*(\d+)", line)

                    if loss_match:
                        last_loss = float(loss_match.group(1))
                    if epoch_match:
                        last_epoch = int(float(epoch_match.group(1)))
                    if step_match:
                        last_step = int(step_match.group(1))
                        samples_processed = last_step * self.config.batch_size
                    if grad_norm_match:
                        last_grad_norm = float(grad_norm_match.group(1))
                    if unsloth_steps_match:
                        estimated_total_steps = int(unsloth_steps_match.group(1))
                    if progress_match:
                        last_step = int(progress_match.group(2))
                        estimated_total_steps = int(progress_match.group(3))
                        samples_processed = last_step * self.config.batch_size

                    eta = None
                    if last_step > 0 and estimated_total_steps > 0:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        steps_remaining = estimated_total_steps - last_step
                        if steps_remaining > 0:
                            time_per_step = elapsed / last_step
                            eta_seconds = steps_remaining * time_per_step
                            if eta_seconds < 60:
                                eta = f"{int(eta_seconds)}s"
                            elif eta_seconds < 3600:
                                eta = f"{int(eta_seconds / 60)}m"
                            else:
                                eta = f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"

                    if callback and (progress_match or loss_match):
                        callback({
                            "epoch": last_epoch,
                            "total_epochs": self.config.num_epochs,
                            "step": last_step,
                            "total_steps": estimated_total_steps,
                            "loss": last_loss,
                            "learning_rate": self.config.learning_rate,
                            "grad_norm": last_grad_norm,
                            "eta": eta,
                            "samples_processed": samples_processed,
                        })

                    if loss_match and last_loss is not None and last_step > 0:
                        run.add_loss_point(
                            step=last_step,
                            loss=last_loss,
                            epoch=last_epoch,
                            learning_rate=self.config.learning_rate,
                        )

            return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(f"Distillation script exited with code {return_code}")

            run.metrics = {
                "final_loss": last_loss or 0.0,
                "epochs_completed": self.config.num_epochs,
                "samples_processed": samples_processed,
                "teacher_model": self.config.teacher_model,
            }

            logger.info(f"Distillation completed. Model saved to: {run.output_path}")

        except FileNotFoundError as e:
            raise RuntimeError(f"Python interpreter not found: {e}")
        except Exception as e:
            logger.error(f"Distillation training failed: {e}")
            raise
```

- [ ] **Step 4: Verify it compiles**

Run: `python -c "from bashgym.gym.trainer import Trainer; t = Trainer(); print('distillation OK')"`
Expected: `distillation OK`

- [ ] **Step 5: Commit**

```bash
git add bashgym/gym/trainer.py
git commit -m "feat: wire distillation training to execute generated script via subprocess"
```

---

### Task 3: Rewrite GRPO generated script with trl.GRPOTrainer

**Files:**
- Modify: `bashgym/gym/trainer.py:1546-1644` (GRPOTrainer._generate_grpo_script)

**IMPORTANT — trl reward function API:** The reward function signature for `trl.GRPOTrainer` (v0.15+) passes `completions` as a list of lists of chat-format dicts (e.g., `[[{"role": "assistant", "content": "..."}]]`). Each inner list is one completion's messages. Access the content via `completion[0]["content"]`. The `**kwargs` receives all extra dataset columns (including `prompts`, `tests`, etc.). If the trl API changes in a future version, the reward functions may need adjustment — verify against the installed version's docs.

- [ ] **Step 1: Replace `_generate_grpo_script()` entirely**

Replace the entire method body of `_generate_grpo_script()` in the `GRPOTrainer` class (lines 1546-1644). The new generated script must:

Use forward-slash paths (same as SFT/DPO scripts):
```python
dataset_path = str(run.dataset_path).replace("\\", "/")
output_path = str(run.output_path).replace("\\", "/")
```

The generated script template (f-string) must contain:

**Imports:**
```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig
import torch, ast, subprocess, tempfile, os, sys, re
```
(Note: rename to `TRLGRPOTrainer` to avoid name clash with the BashGym class)

**Configuration section** with templated values:
```python
REWARD_MODE = "{self.config.grpo_reward_mode}"
NUM_GENERATIONS = {self.config.grpo_num_generations}
```

**Three reward functions** exactly as specified in the design spec (section 2b):
- `syntax_reward(completions, **kwargs)` — `ast.parse()` check, returns list of 1.0/0.0
- `execution_reward(completions, **kwargs)` — subprocess run with 10s timeout, returns list of 1.0/0.0
- `verification_reward(completions, prompts, tests, **kwargs)` — runs pytest in tempdir, returns list of passed/total

**Helper functions:**
- `extract_code(text)` — extract code from markdown fences or raw text
- `run_verification(code, test_code)` — write solution + tests to tempdir, run pytest, parse PASSED/FAILED counts

**Reward selection:**
```python
REWARD_FN = {{"syntax": syntax_reward, "execution": execution_reward, "verification": verification_reward}}[REWARD_MODE]
```
(Note: double braces `{{` `}}` to escape f-string)

**Model setup** (standard Unsloth pattern, same as SFT):
```python
if __name__ == "__main__":
    import gc
    print("Clearing GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {{torch.cuda.get_device_name(0)}}")
        print(f"Available VRAM: {{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f}} GB")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="{self.config.base_model}",
        max_seq_length={self.config.max_seq_length},
        dtype=torch.float16,
        load_in_4bit={self.config.load_in_4bit},
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r={self.config.lora_r},
        target_modules={self.config.lora_target_modules},
        lora_alpha={self.config.lora_alpha},
        lora_dropout={self.config.lora_dropout},
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
```

**Dataset loading:**
```python
    dataset = load_dataset("json", data_files="{dataset_path}", split="train")
```

**GRPOConfig:**
```python
    grpo_config = GRPOConfig(
        output_dir="{output_path}",
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size={self.config.batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        num_train_epochs={self.config.num_epochs},
        learning_rate={self.config.learning_rate},
        logging_steps={self.config.logging_steps},
        save_steps={self.config.save_steps},
        save_total_limit=3,
        fp16=True,
    )
```

**GRPOTrainer init** (use `processing_class`, not `tokenizer`):
```python
    trainer = TRLGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[REWARD_FN],
        train_dataset=dataset,
        args=grpo_config,
    )
```

**Training + save:**
```python
    print("Starting GRPO training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained("{output_path}/final")
    tokenizer.save_pretrained("{output_path}/final")
    model.save_pretrained_merged("{output_path}/merged", tokenizer, save_method="merged_16bit")
    print("GRPO training complete!")
```

- [ ] **Step 2: Verify the method generates valid Python**

Run: `python -c "
from bashgym.gym.trainer import GRPOTrainer, TrainerConfig, TrainingRun
from pathlib import Path
t = GRPOTrainer(TrainerConfig(grpo_reward_mode='syntax'))
run = TrainingRun(run_id='test', strategy=t.config.strategy, base_model=t.config.base_model, dataset_path=Path('data/test.jsonl'), output_path=Path('/tmp/test'))
script = t._generate_grpo_script(run)
compile(script, '<grpo_script>', 'exec')
print('Script compiles OK')
print('Has GRPOTrainer:', 'TRLGRPOTrainer' in script)
print('Has syntax_reward:', 'syntax_reward' in script)
print('Has processing_class:', 'processing_class' in script)
"`
Expected: All three checks print True.

- [ ] **Step 3: Commit**

```bash
git add bashgym/gym/trainer.py
git commit -m "feat: rewrite GRPO script to use trl.GRPOTrainer with tiered reward functions"
```

---

### Task 4: Wire GRPO subprocess execution

**Files:**
- Modify: `bashgym/gym/trainer.py:1456-1544` (GRPOTrainer.train_grpo + _run_grpo_loop)

- [ ] **Step 1: Update `train_grpo()` signature**

In the `GRPOTrainer` class, update `train_grpo()` (line 1456) to add `log_callback` and `pid_callback`:

```python
    def train_grpo(
        self,
        dataset_path: Path,
        verifier_fn: Callable[[str, str], float],
        run_id: Optional[str] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        pid_callback: Optional[Callable[[int, "TrainingRun"], None]] = None,
    ) -> TrainingRun:
```

Update the call at line 1490 to pass through:

```python
            self._run_grpo_loop(run, verifier_fn, callback, log_callback, pid_callback)
```

- [ ] **Step 2: Replace `_run_grpo_loop()` simulation with real execution**

Update the `_run_grpo_loop` method signature (line 1504) to:

```python
    def _run_grpo_loop(
        self,
        run: TrainingRun,
        verifier_fn: Callable[[str, str], float],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        pid_callback: Optional[Callable[[int, "TrainingRun"], None]] = None,
    ) -> None:
```

Replace the simulation block (lines 1528-1544) with subprocess execution. Same pattern as the distillation fix from Task 2, with these differences:

- Script path is `run.output_path / "train_grpo.py"`
- Log prefix is `[GRPO Training]`
- Additional metric parsing for GRPO-specific trl output:
  - `'reward':\s*([\d.-]+)` or `'mean_reward':\s*([\d.-]+)` → avg_reward
  - `'kl':\s*([\d.-]+)` → kl_divergence
  - `'policy_loss':\s*([\d.-]+)` → policy_loss
- Callback dict includes `avg_reward`, `kl_divergence`, `policy_loss` in addition to the standard fields
- Final `run.metrics` includes `final_avg_reward`, `final_kl_divergence`, `final_policy_loss`

Keep the existing prints at lines 1519-1520 (start message) and the script generation at lines 1523-1526. Replace everything after line 1527 (the simulation block) through line 1544.

- [ ] **Step 3: Verify it compiles**

Run: `python -c "from bashgym.gym.trainer import GRPOTrainer; t = GRPOTrainer(); print('GRPO execution OK')"`
Expected: `GRPO execution OK`

- [ ] **Step 4: Commit**

```bash
git add bashgym/gym/trainer.py
git commit -m "feat: wire GRPO training to execute generated script via subprocess"
```

---

### Task 5: Implement RLVR as GRPO wrapper

**Files:**
- Modify: `bashgym/gym/trainer.py:165-179` (Trainer class — add train_rlvr method)

- [ ] **Step 1: Add `import copy` to the top of the file**

Add `import copy` to the imports section at the top of `bashgym/gym/trainer.py` (after line 21, near the other stdlib imports).

- [ ] **Step 2: Add `train_rlvr()` method to the `Trainer` class**

Add this method to the `Trainer` class, after the `train_distillation()` method (after line 897):

```python
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
        grpo_config = copy.deepcopy(self.config)
        grpo_config.grpo_reward_mode = "verification"

        grpo_trainer = GRPOTrainer(grpo_config)
        run = grpo_trainer.train_grpo(
            dataset_path=dataset_path,
            verifier_fn=lambda p, r: 0.0,
            run_id=run_id,
            callback=callback,
            log_callback=log_callback,
            pid_callback=pid_callback,
        )
        run.strategy = TrainingStrategy.RLVR
        return run
```

- [ ] **Step 3: Verify it compiles**

Run: `python -c "from bashgym.gym.trainer import Trainer; t = Trainer(); print(hasattr(t, 'train_rlvr'))"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add bashgym/gym/trainer.py
git commit -m "feat: implement RLVR as GRPO wrapper with verification-based rewards"
```

---

### Task 6: Update API schemas

**Files:**
- Modify: `bashgym/api/schemas.py:33-38` (TrainingStrategy enum)
- Modify: `bashgym/api/schemas.py:125-166` (TrainingRequest model)

- [ ] **Step 1: Add RLVR to TrainingStrategy enum**

In `bashgym/api/schemas.py`, replace lines 33-38:

```python
class TrainingStrategy(str, Enum):
    """Available training strategies."""
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    RLVR = "rlvr"
    DISTILLATION = "distillation"
```

- [ ] **Step 2: Add grpo_reward_mode to TrainingRequest**

In `bashgym/api/schemas.py`, after the `grpo_temperature` field (line 147), add:

```python
    grpo_reward_mode: str = Field("syntax", description="GRPO reward mode: syntax, execution, or verification")
```

- [ ] **Step 3: Verify schema compiles**

Run: `python -c "from bashgym.api.schemas import TrainingStrategy, TrainingRequest; print(TrainingStrategy.RLVR.value); r = TrainingRequest(grpo_reward_mode='verification'); print(r.grpo_reward_mode)"`
Expected: `rlvr` then `verification`

- [ ] **Step 4: Commit**

```bash
git add bashgym/api/schemas.py
git commit -m "feat: add RLVR strategy and grpo_reward_mode to API schemas"
```

---

### Task 7: Update API route dispatch

**Files:**
- Modify: `bashgym/api/routes.py:871-978` (training dispatch block)

- [ ] **Step 1: Wire grpo_reward_mode into TrainerConfig construction**

In `bashgym/api/routes.py`, in the config construction block (lines 872-902), add `grpo_reward_mode` after `grpo_temperature` (around line 892):

```python
                    grpo_reward_mode=getattr(request, 'grpo_reward_mode', 'syntax'),
```

- [ ] **Step 2: Add GRPOTrainer import**

At the top of `routes.py` where trainer is imported, ensure `GRPOTrainer` is imported:

```python
from bashgym.gym.trainer import Trainer, TrainerConfig, TrainingStrategy as TS, TrainingRun, GRPOTrainer
```

Check the existing import line first — it may already import some of these. Just add `GRPOTrainer` to the existing import.

- [ ] **Step 3: Update distillation dispatch to pass log_callback and pid_callback**

Replace lines 939-944:

```python
                elif request.strategy == TrainingStrategy.DISTILLATION:
                    run = app.state.trainer.train_distillation(
                        dataset_path=dataset_path,
                        run_id=run_id,
                        callback=callback.on_progress_sync,
                        log_callback=callback.on_log_sync,
                        pid_callback=_on_pid,
                    )
```

- [ ] **Step 4: Replace GRPO dispatch block and add RLVR**

Replace the entire `else:` block (lines 945-969) with two explicit elif branches:

```python
                elif request.strategy.value == "grpo":
                    grpo_trainer = GRPOTrainer(app.state.trainer.config)
                    run = grpo_trainer.train_grpo(
                        dataset_path=dataset_path,
                        verifier_fn=lambda p, r: 0.0,
                        run_id=run_id,
                        callback=callback.on_progress_sync,
                        log_callback=callback.on_log_sync,
                        pid_callback=_on_pid,
                    )
                elif request.strategy.value == "rlvr":
                    run = app.state.trainer.train_rlvr(
                        dataset_path=dataset_path,
                        run_id=run_id,
                        callback=callback.on_progress_sync,
                        log_callback=callback.on_log_sync,
                        pid_callback=_on_pid,
                    )
                else:
                    raise ValueError(f"Unknown training strategy: {request.strategy}")
```

Note: Use `request.strategy.value == "grpo"` instead of `request.strategy == TrainingStrategy.GRPO` because `routes.py` imports its own `TrainingStrategy` from `schemas.py` which is a different class than the one in `trainer.py`. The `.value` comparison avoids cross-enum issues.

- [ ] **Step 5: Verify routes compile**

Run: `python -c "from bashgym.api.routes import create_app; app = create_app(); print('Routes OK')"`
Expected: `Routes OK` (may have warnings about missing env vars, that's fine)

- [ ] **Step 6: Commit**

```bash
git add bashgym/api/routes.py
git commit -m "feat: fix GRPO routing, add RLVR dispatch, wire log/pid callbacks for all strategies"
```

---

### Task 8: Bump trl dependency

**Files:**
- Modify: `requirements-training.txt`

- [ ] **Step 1: Update trl version requirement**

In `requirements-training.txt`, change line 15:

```
trl>=0.15.0                # Reinforcement learning for LLMs (SFT, DPO, GRPO)
```

- [ ] **Step 2: Commit**

```bash
git add requirements-training.txt
git commit -m "deps: bump trl>=0.15.0 for GRPOTrainer support"
```

---

### Task 9: Verify all strategies compile end-to-end

**Files:** None (verification only)

- [ ] **Step 1: Verify trainer imports and method availability**

Run:
```bash
python -c "
from bashgym.gym.trainer import Trainer, GRPOTrainer, TrainerConfig, TrainingStrategy
t = Trainer()
g = GRPOTrainer()

# Check all strategies have methods
assert hasattr(t, 'train_sft'), 'Missing train_sft'
assert hasattr(t, 'train_dpo'), 'Missing train_dpo'
assert hasattr(t, 'train_distillation'), 'Missing train_distillation'
assert hasattr(t, 'train_rlvr'), 'Missing train_rlvr'
assert hasattr(g, 'train_grpo'), 'Missing train_grpo'

# Check config fields
c = TrainerConfig()
assert c.grpo_reward_mode == 'syntax'
assert c.weight_decay == 0.01
assert c.use_gradient_checkpointing == True

print('All trainer checks passed')
"
```

- [ ] **Step 2: Verify API schemas**

Run:
```bash
python -c "
from bashgym.api.schemas import TrainingStrategy, TrainingRequest

# RLVR exists in schema
assert TrainingStrategy.RLVR.value == 'rlvr'

# grpo_reward_mode in request
r = TrainingRequest(strategy='grpo', grpo_reward_mode='verification')
assert r.grpo_reward_mode == 'verification'

print('All schema checks passed')
"
```

- [ ] **Step 3: Verify app creates successfully**

Run:
```bash
python -c "from bashgym.api.routes import create_app; app = create_app(); print(f'App created with {len(app.routes)} routes')"
```

- [ ] **Step 4: Verify GRPO script generation with all reward modes**

Run:
```bash
python -c "
from bashgym.gym.trainer import GRPOTrainer, TrainerConfig, TrainingRun
from pathlib import Path

for mode in ['syntax', 'execution', 'verification']:
    config = TrainerConfig(grpo_reward_mode=mode)
    t = GRPOTrainer(config)
    run = TrainingRun(
        run_id='test', strategy=config.strategy,
        base_model=config.base_model,
        dataset_path=Path('test.jsonl'),
        output_path=Path('/tmp/test')
    )
    script = t._generate_grpo_script(run)
    compile(script, f'<grpo_{mode}>', 'exec')
    assert f'REWARD_MODE = \"{mode}\"' in script
    print(f'{mode} script: OK')

print('All GRPO scripts compile')
"
```
