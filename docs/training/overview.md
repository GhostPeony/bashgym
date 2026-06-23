# BashGym Training Overview

This guide explains the training gym from first principles: what data enters the
system, what each training strategy is trying to teach, and which evidence proves
that a trained model is actually better.

For exact knobs and recipes, read [strategy-guide.md](strategy-guide.md). For
world-model objectives, read [world-models.md](world-models.md). For diagnosis
during and after a run, read [metrics-runbook.md](metrics-runbook.md).

---

## The mental model

BashGym trains coding agents from verified work. A useful training example is
not just a prompt and an answer. It is a trajectory:

```text
task + repo/context + commands/tool calls -> outputs/diffs/tests/verifier result
```

The gym keeps the whole loop visible:

```text
capture/import -> classify -> generate examples -> train -> evaluate -> deploy
       ^                                                        |
       |                                                        v
       +---------------------- collect new traces <-------------+
```

The key rule is simple: loss curves are not release evidence. Verifiers, tests,
pass@k, holdout gates, tamper checks, and external benchmarks decide whether the
student is good enough to route traffic to.

---

## What BashGym trains from

| Source | What it contains | Best first use |
|---|---|---|
| Gold traces | Successful AI coding sessions captured from Claude Code, Codex, Gemini, Copilot, OpenCode, and similar tools. | SFT baseline and DPO chosen examples. |
| Failed or lower-tier traces | Attempts with bad choices, failed commands, weak verification, or low quality scores. | DPO rejected examples and failure analysis. |
| Custom JSONL | Prebuilt NeMo/TRL-style `messages` datasets. | Importing curated external or hand-authored data. |
| Security datasets | Malware, phishing, or other labeled security corpora converted into examples. | Security-specialist behavior or classification tasks. |
| Terminal environments | Executable tasks with a prompt, workspace, verifier, rollout attempts, and pass/fail reward. | GRPO/RLVR, DPPO replay, pass@k, holdout gates, and world-model replay data. |

Gold traces teach the model what good behavior looks like. Terminal
environments prove whether the behavior survives interaction with a real shell,
repo, and verifier.

---

## What the training strategies teach

| Strategy | Teaches | Needs | Proves itself with |
|---|---|---|---|
| SFT | Imitation: reproduce successful trace format, tool use, and problem-solving style. | Gold examples. | Eval loss plus heldout pass@k. |
| DPO | Preference: choose the better response for the same prompt. | Chosen/rejected pairs. | Preference accuracy, reward margin, heldout behavior. |
| GRPO/RLVR | Outcome optimization: improve completions using verifier rewards. | Reward variation across sampled attempts. | Reward, reward_std, pass@1/pass@k, verifier status. |
| Distillation | Compression: move teacher behavior into a smaller student. | Teacher outputs or teacher-on-policy budget. | Student pass@k and quality against teacher baseline. |
| Cascade RL | Curriculum: train domain specialists in stages, then merge or distill. | Enough examples per domain. | Per-domain gates and final generalist holdout. |
| DPPO replay | Terminal rollout optimization with behavior/train logprob replay and trust-region masks. | Served-model rollouts with logprobs and a backend such as verl/SkyRL/open-instruct. | Mask telemetry, reward, pass@k, backend smoke artifacts. |

Start with SFT unless you already have a verifier-rich terminal environment and
reward variation. SFT teaches the student the language and shape of the task.
RL improves outcomes only after the model can produce attempts worth comparing.

---

## The basic training path

1. Import or capture traces.

   Use the existing agent history first. Hooks are for future sessions; the
   backlog gives the gym data on day one.

2. Classify and curate.

   Promote verified, complete sessions to gold. Keep failed or weak sessions for
   DPO negatives, but do not mix them into SFT as if they were success examples.

3. Generate examples.

   SFT examples use structured tool-call chat messages. DPO examples must pair a
   chosen and rejected answer for the same prompt.

4. Train the first student with SFT.

   Use QLoRA on small/local hardware. Use the remote device or cloud path for
   larger models, longer sequences, or full fine-tunes.

5. Evaluate before routing.

   Run heldout trace evals, executable environment pass@k, spurious-reward
   controls, tamper canaries, and any relevant public benchmark ingest before
   treating a model as shippable.

6. Route conservatively.

   The student does not need to replace the teacher everywhere. Route narrow
   tasks it passes, fall back to the teacher when confidence or gates are weak,
   then collect the new traces for the next cycle.

---

## Where world models fit

BashGym's JEPA-style world-model work is about predicting useful latent terminal
dynamics, not every raw terminal byte. The transition of interest is:

```text
task + command/history -> next observation, diff, test state, verifier state
```

Two objectives are wired today:

| Objective | Adds | Current status |
|---|---|---|
| ECHO | Auxiliary observation-token prediction loss for terminal outputs. | Config and replay contract are wired; trainer backend integration is the next step. |
| RWML | Embedding-space reward for predicting next terminal state. | Pure reward/transition layer and replay contract are wired; real backend loop is pending. |

Treat ECHO/RWML metrics as diagnostics and auxiliary learning signals until they
correlate with heldout pass@k and safety metrics. Do not use them as release
gates by themselves.

---

## Read next

- [strategy-guide.md](strategy-guide.md) - concrete starting settings and when to use each strategy.
- [world-models.md](world-models.md) - ECHO/RWML contracts, defaults, replay telemetry, and boundaries.
- [metrics-runbook.md](metrics-runbook.md) - how to diagnose flat pass@k, zero reward variance, timeouts, verifier errors, and tamper attempts.
- [glossary.md](glossary.md) - compact definitions for the training vocabulary.
- [agent-cli.md](agent-cli.md) - machine-readable CLI commands agents can call for setup and replay analysis.
- [../TRAINING_DATA_GUIDE.md](../TRAINING_DATA_GUIDE.md) - trace format and data pipeline reference.
- [../training-config-guide.md](../training-config-guide.md) - existing Training Config panel reference.

## Source references

- [../../tasks/jepa-bashgym-action-plan-2026-06-23.md](../../tasks/jepa-bashgym-action-plan-2026-06-23.md)
- [../../tasks/jepa-worldmodel-hardware-handoff-2026-06-23.md](../../tasks/jepa-worldmodel-hardware-handoff-2026-06-23.md)
- [../../bashgym/gym/trainer.py](../../bashgym/gym/trainer.py)
- [../../bashgym/gym/terminal_rl.py](../../bashgym/gym/terminal_rl.py)
- [../../bashgym/eval/dppo_replay.py](../../bashgym/eval/dppo_replay.py)
