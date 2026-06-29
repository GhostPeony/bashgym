# RLHF Handbook Comparison And BashGym Gap Plan

Date: 2026-06-28

This document compares BashGym against Nathan Lambert's RLHF Book and the
current open RLHF/post-training ecosystem. It is intended to guide the next
BashGym training-platform implementation pass.

Primary sources reviewed:

- RLHF Book repository: https://github.com/natolambert/rlhf-book
- RLHF Book code README: https://github.com/natolambert/rlhf-book/blob/main/code/README.md
- RLHF Book chapters: https://github.com/natolambert/rlhf-book/tree/main/book/chapters
- RLHF Book teaching materials: https://github.com/natolambert/rlhf-book/tree/main/teach
- Hugging Face TRL: https://huggingface.co/docs/trl/en/index
- OpenRLHF: https://openrlhf.readthedocs.io/
- SkyRL: https://docs.skyrl.ai/docs
- verl: https://verl.readthedocs.io/
- Harbor Terminal-Bench: https://www.harborframework.com/docs/tutorials/running-terminal-bench
- RewardBench 2: https://openreview.net/forum?id=fb0G86Dewb
- CUARewardBench: https://arxiv.org/abs/2510.18596
- SkyRL-Agent: https://arxiv.org/abs/2511.16108

Local BashGym sources reviewed:

- [capability-map.md](capability-map.md)
- [training-methods-reference.md](training-methods-reference.md)
- [strategy-guide.md](strategy-guide.md)
- [metrics-runbook.md](metrics-runbook.md)
- [world-models.md](world-models.md)
- [gx10-eval-checklist.md](gx10-eval-checklist.md)
- [external-review-packet.md](external-review-packet.md)
- `bashgym/cli.py`
- `bashgym/gym/trainer.py`
- `bashgym/eval/*`
- `bashgym/environments/*`
- `bashgym/gym/world_model_backend.py`

---

## Executive Read

BashGym is already directionally strong as a terminal-agent post-training
control plane. It is strongest where generic RLHF stacks are thin: trace import,
terminal environment contracts, verifier-backed rollouts, pass@k, holdout
gates, reward-hacking canaries, spurious-reward controls, external benchmark
evidence, DPPO replay contracts, and conservative release logic.

BashGym is not yet a full replacement for TRL, verl, SkyRL, or OpenRLHF as a
distributed trainer. It should not try to become all of those at once. The
highest-leverage role is:

```text
BashGym owns data, environment, replay, evidence, and promotion contracts.
External trainers own scalable optimization until BashGym proves a narrower
native backend path is worth owning.
```

The main gaps are:

1. Reproducibility packaging: no canonical run card or single required
   release-evidence bundle yet.
2. Preference quality: DPO exists, but metadata, pair difficulty, tie/noise, and
   length-ratio contracts should be stricter before scaling preference tuning.
3. Reward modeling: BashGym lacks first-class Preference RM, ORM, and PRM
   workflows comparable to the RLHF Book and TRL/OpenRLHF.
4. Backend proof: DPPO/ECHO/RWML contracts exist, but installed-backend GX10
   proof is still pending.
5. Education: BashGym has good docs, but not enough guided "reader experiment
   path" material with expected metrics, failure modes, and matched controls.

---

## What RLHF Book Adds

### 1. RLHF is a sequence, not an algorithm

The handbook treats RLHF as a post-training loop:

```text
SFT/instruction tuning
  -> preference or verifier data
  -> reward model, direct alignment, rejection sampling, or policy gradients
  -> evaluation, safety checks, and iteration
```

BashGym already reflects this in its strategy docs. The next improvement is to
make the sequence operational in the UI and CLI: every run should know where it
sits in the ladder, which prior artifact it depends on, and which next evidence
gate it must satisfy.

### 2. SFT teaches the interface

RLHF Book's SFT example focuses on the base-to-assistant transition: the base
model continues text, while the SFT model learns the chat template and stops.

BashGym equivalent:

- Tool-call format.
- Command style.
- Repo conventions.
- Final verification behavior.
- Recovery after errors.

Current BashGym docs already tell operators to start with SFT or distillation
when pass@k is zero. The gap is a small first-run tutorial that shows the same
observable transition for a BashGym trace: invalid action format at step 0,
valid command/tool use after a small SFT smoke.

### 3. Preference data quality dominates method choice

RLHF Book repeatedly points back to preference data: UI, ranking/rating format,
ties, label noise, annotator/source provenance, multi-turn context, and bias.

BashGym has DPO and decision-DPO surfaces, but the platform should require
stronger DPO pair metadata before serious runs:

- `pair_id`
- `prompt_hash`
- chosen and rejected trace ids
- pair-generation method
- annotator, judge, or verifier provenance
- label strength, tie, or noise flags
- quality scores
- chosen/rejected length ratio
- task family and domain
- split and decontamination metadata

This is a P0 because weak pairs can make DPO metrics look good while heldout
behavior gets worse.

### 4. Reward models are a real missing lane

RLHF Book includes Preference RM, ORM, and PRM examples. TRL exposes
`RewardTrainer` and `PRMTrainer`; OpenRLHF exposes reward-model training as part
of its SFT/RM/DPO path. RewardBench 2 and CUARewardBench make this gap sharper:
reward models are not only for RLHF, but also for best-of-N selection, trajectory
scoring, process feedback, and reward audits.

BashGym has verifiers and DPO, but no first-class learned reward-model workflow.
This should become a P1 lane:

- Preference RM for pairwise trajectory or response preferences.
- ORM for full terminal rollout success likelihood.
- PRM for step-level command/reasoning quality.
- Heldout RM eval and RewardBench-style pair accuracy.
- Agent/task-specific reward-model evals before using learned rewards for
  training or selection.

### 5. Policy gradients are a family

RLHF Book's policy-gradient chapter covers REINFORCE, RLOO, PPO, GRPO, GSPO,
CISPO, DAPO, and related implementation details. BashGym correctly keeps GRPO
and verifier-backed RL as the first-class terminal path because multiple
attempts per prompt/environment are natural for shell tasks.

Important comparison:

| Method | RLHF Book role | BashGym implication |
|---|---|---|
| REINFORCE | Simple policy gradient baseline. | Good future backend baseline for terminal rollouts. |
| RLOO | Multi-sample per-prompt baseline. | Natural for terminal tasks; useful before full PPO complexity. |
| PPO | Mature RLHF method with value function and KL. | Backend candidate through TRL/verl/OpenRLHF, not BashGym's first native path. |
| GRPO | Group-relative advantage without value model. | Correct BashGym first-class RLVR path when reward groups have contrast. |
| DAPO/Dr. GRPO | Stability/length/difficulty-bias variants. | Keep as backend/profile options for terminal RL. |

Current BashGym strength: `reward_std`, `frac_reward_zero_std`, active sampling,
zero-std filtering, pass@k, holdout, and tamper controls already match the core
operational lessons.

### 6. Rejection sampling needs matched controls

RLHF Book's rejection-sampling implementation compares reward-selected data
against random-selection baselines with the same sample budget. This is a useful
pattern BashGym should adopt more broadly:

```text
sample N terminal rollouts
score with verifier/RM/judge
select top completions or trajectories
SFT on selected traces
compare against random-selected traces with identical budget
```

BashGym should add this as a P1/P2 guided workflow because it is simpler than
online RL and can answer whether the reward signal is useful before expensive
backend work.

### 7. Evaluation variance and over-optimization must be first-class

RLHF Book's over-optimization and evaluation chapters reinforce BashGym's
conservative stance: reward, loss, KL, entropy, and preference accuracy are not
promotion evidence by themselves.

BashGym is strong here already. It has:

- heldout trace eval
- executable environment pass@k
- grouped holdouts
- paired base-vs-candidate comparison
- spurious-reward controls
- reward-hacking canaries
- tamper/protected-file checks
- external benchmark ingest
- diagnostic-only world-model quality lane

The missing step is enforcement: release tools should fail closed when required
sections are absent, rather than treating evidence as optional context.

---

## Where BashGym Is Strong

### Environment-first training and eval

BashGym's central idea is stronger than generic prompt/completion RLHF for coding
agents: the unit is a verified terminal trajectory, not a chat response.

This lines up with TRL's newer Harbor integration and SkyRL-Agent's long-horizon
software-engineering direction. BashGym is already well-positioned to generate,
materialize, split, replay, and evaluate terminal environments.

### Release gates and safety controls

The eval stack is unusually strong:

- `environment_passk.py`
- `environment_holdout.py`
- `environment_holdout_comparison.py`
- `environment_spurious_reward.py`
- `environments/canaries.py`
- `release_gate.py`
- external benchmark ingest and release-evidence attachment

This is a real differentiator. Many post-training stacks train well but leave
promotion and safety evidence to downstream users.

### DPPO/ECHO/RWML contracts are scoped honestly

BashGym has DPPO replay, train-logprob enrichment, Binary-TV/KL mask telemetry,
world-model replay payloads, ECHO/RWML adapter hooks, and smoke-bundle planning.

The docs correctly say this is backend-dependent and diagnostic until proven.
That honest boundary is important. It should stay.

### Agent-readable CLI and docs

The CLI can expose capabilities, docs, plans, replay summaries, smoke bundles,
and run analysis in JSON. This is exactly the right shape for AI/ML reviewers and
future agents to inspect platform state.

### Data and trace factory

BashGym already has trace capture, data factory, masking, decision-DPO,
quality scoring, deduplication, decontamination, environment generation, and
public dataset scoring. That makes it more domain-specific than a generic RLHF
trainer.

---

## Where BashGym Lacks

### P0 gaps

1. **Canonical run card / release bundle**

   The external review packet already calls this out. It should become a schema,
   CLI command, and promotion requirement.

   Minimum fields:

   ```yaml
   run_id:
   git_commit:
   branch:
   base_model:
   model_family_profile:
   training_method:
   training_command:
   training_config:
   data_artifacts:
   split_manifest:
   decontamination_manifest:
   backend:
   backend_version:
   hardware:
   seed:
   thresholds:
   metrics_path:
   release_evidence_path:
   smoke_bundle_path:
   outputs:
   known_limitations:
   decision:
   ```

2. **Strict preference pair contract**

   DPO pairs should not be just `prompt/chosen/rejected`. They need provenance,
   comparability, quality, and contamination metadata.

3. **GX10 installed-backend proof**

   DPPO/ECHO/RWML should not be claimed beyond contracts until one real backend
   smoke consumes replay, logs mask/world-model metrics, writes outputs, and is
   followed by before/after pass@k.

4. **Metric authority as enforcement**

   BashGym already categorizes metrics, but the operator path should make the
   distinction unavoidable:

   - setup checks block training
   - signal-quality checks block scaling
   - safety checks block promotion
   - behavior evidence controls routing
   - world-model metrics remain diagnostic

5. **Dependency/API drift checks**

   TRL v1 has moved quickly. Generated SFT/DPO/GRPO scripts should be smoke-run
   in a fresh pinned training environment before broad claims.

### P1 gaps

1. **First-class reward-model training**

   Add Preference RM, ORM, and PRM lanes. This is the largest method gap versus
   RLHF Book, TRL, and OpenRLHF.

2. **Rejection sampling / best-of-N with matched controls**

   Add a workflow that samples multiple terminal rollouts, scores them, selects
   winners, SFTs on winners, and compares against random-selection controls.

3. **Direct-alignment variants as recipes**

   Keep DPO first-class. Add IPO, KTO, ORPO, SimPO, DPO-Norm, and cDPO as
   backend/export recipes after pair quality gates exist.

4. **Reliability evals for tool-use agents**

   Add repeated-trial reliability, stricter tool schema validity, BFCL-style tool
   argument scoring, and "passes once vs reliably passes" distinction.

5. **Canonical backend decision**

   Recommendation:

   - TRL: reference semantics and local SFT/DPO/GRPO scripts.
   - SkyRL: first terminal-agent backend candidate because of custom
     environments, tools, and long-horizon agent orientation.
   - verl: scale-out GRPO/PPO backend candidate.
   - OpenRLHF: later scale-out and algorithm-family candidate.
   - Axolotl: YAML recipe import/export, not runtime core.

### P2 gaps

1. Reward authoring and reward audit UX.
2. World-model correlation dashboard.
3. Educational failure labs.
4. Dataset/environment/benchmark cards.
5. Public "first experiment path" docs modeled after RLHF Book's reader path.

---

## Reviewer Questions Resolved

### Answered at recommendation level

| Question | Answer |
|---|---|
| Is the method sequence correct? | Yes. Use SFT or distillation before RL unless executable attempts already have reward contrast. |
| Are GRPO/RLVR the right first RL methods for terminal-agent tasks? | Yes, when verifiers exist and sampled attempts sometimes pass and sometimes fail. |
| What operational metrics are mandatory before scaling? | Tokens/sec, peak GPU memory, OOMs, backend import status, timeout rate, verifier error rate, tamper rate, reward variance, behavior/train logprob readiness. |
| How should terminal environments be split? | Use grouped non-overlap by repo, task family, source, fixture/seed, and solution-pattern proxies; preserve hash contamination manifests. |
| What reward-hacking canaries should be mandatory? | Verifier tamper, test tamper, private fixture tamper, environment manifest tamper, plus heldout/private tests and shortcut probes. |
| How should external benchmarks be attached? | As release evidence with harness manifest, thresholds, failures, leakage/decontamination notes, and claim tier. |
| What statistical comparison should be standard? | Unbiased pass@k plus paired clustered bootstrap for base-vs-candidate comparisons; require intervals to clear zero for improvement claims. |
| What is "good enough to route narrowly"? | The candidate passes behavior and safety gates for a defined task region, with fallback outside that region. |
| Strongest part of BashGym? | Executable terminal trajectories plus conservative evidence gates. |
| Weakest assumption? | That DPPO/ECHO/RWML contracts will translate into backend training gains. |
| What should be de-scoped? | Broad SOTA claims, global teacher replacement, and world-model release gating before correlation evidence. |
| Most important missing artifact? | The run card / canonical release bundle. |

### Partially answered, empirical proof required

| Question | Current recommendation | Remaining proof |
|---|---|---|
| Is DPPO replay the right abstraction? | Yes as a handoff for multi-step rollouts with behavior/train logprobs and trust-region telemetry. | One installed backend must consume it and improve or preserve pass@k. |
| Are ECHO/RWML useful? | Coherent as JEPA-style auxiliary diagnostics and curriculum signals. | Prove correlation with heldout pass@k, command count, timeout, and safety. |
| Are current gates enough for first release? | Enough for narrow routing if mandatory. | Broad claims need external benchmarks, decontamination, and calibrated thresholds. |
| Which backend should be canonical? | SkyRL first for terminal-agent integration; verl for scale-out smoke; TRL for reference semantics. | Finalize after one successful installed-backend smoke. |
| What metrics are missing? | Metric families exist. | Need claim-tier thresholds, mandatory safety thresholds, run cards, and dataset cards. |

### Still open

1. Does ECHO/RWML quality predict better terminal-agent behavior?
2. Which backend is easiest to keep green for BashGym DPPO/ECHO/RWML smoke?
3. What thresholds separate local smoke, narrow routing, and broad claims?
4. Which reward model benchmark is best for terminal/coding trajectories?

---

## Action Plan

### P0: Button Up Evidence And Contracts

1. **Implement canonical run cards.**

   Add a schema, writer, validator, CLI topic, and release-gate ingestion path.
   Every serious run should produce a run card before promotion.

2. **Tighten DPO/preference pair validation.**

   Add required metadata for serious DPO runs and warnings for lightweight local
   experiments. Include length ratio, prompt identity, source ids, quality, label
   strength, and split metadata.

3. **Make release evidence fail closed.**

   Promotion should explicitly report missing required sections instead of
   silently treating absent evidence as unknown.

4. **Run one GX10 backend smoke.**

   Use a tiny DPPO replay with `include_world_model=true`, pick SkyRL or verl,
   preserve logs/config/output listings, and follow with a small before/after
   pass@k run.

5. **Add generated-script API drift smoke.**

   In a fresh pinned training env, execute the smallest SFT, DPO, and GRPO script
   generation/import path against the current TRL/Unsloth stack.

### P1: Fill Method Gaps

1. **Add Reward Model training lane.**

   Support Preference RM, ORM, and PRM contracts, metrics, heldout pair accuracy,
   and optional use for best-of-N/rejection sampling.

2. **Add rejection sampling workflow.**

   Use BashGym terminal rollouts as the completion pool. Always compare
   reward-selected traces against random-selection baselines with the same
   sample budget.

3. **Add direct-alignment recipe variants.**

   Keep DPO canonical. Add DPO-Norm, cDPO, IPO, KTO, ORPO, and SimPO as
   documented backend/export recipes with pair-quality and heldout gates.

4. **Add reliability evals.**

   Track pass@1, pass@k, repeated-trial stability, tool-call validity,
   argument-level scoring, command count, and timeout/tamper deltas.

5. **Turn docs into guided operator workflows.**

   Add "first experiment path" docs:

   - SFT first student
   - DPO preference refinement
   - GRPO/RLVR terminal RL
   - rejection sampling
   - DPPO/GX10 backend proof
   - ECHO/RWML diagnostic-only world-model pass

### P2: Make Learning Safer And Easier

1. Build reward authoring/audit UX with unit tests and adversarial examples.
2. Add world-model quality correlation tracking.
3. Add failure labs for common RLHF mistakes.
4. Add dataset, environment, benchmark, and run cards.
5. Publish a compact external-review bundle once P0 artifacts exist.

---

## Near-Term Implementation Order

Recommended next engineering sequence:

1. Run card schema + CLI generator.
2. Release-evidence fail-closed validation.
3. Strict preference-pair validator.
4. GX10 backend smoke using existing smoke-bundle path.
5. Reward-model lane design spec.
6. Rejection-sampling workflow.
7. Direct-alignment recipe variants.
8. Reliability evals and failure labs.

The first three are local and should not require GX10. The fourth turns the
current DPPO/ECHO/RWML work from a contract into evidence. The later items close
method and education gaps surfaced by RLHF Book.
