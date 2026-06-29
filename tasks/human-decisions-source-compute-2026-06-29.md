# Human Decisions: Source Library And Cloud Compute

Date: 2026-06-29

These decisions should be made by Cade before the next product-facing slice.
The current implementation is safe locally: source cards, CLI/API discovery,
recommendations, source manifests, and eval-only guardrails are already in place.

---

## D1: Eval-Only Override Policy

Question: should BashGym ever allow eval-only sources, such as Terminal-Bench,
SWE-bench, BFCL, tau-bench, RewardBench, or CUARewardBench, to be exported for
training?

### Option A: Hard block by default, admin override only

Pros:

- Strongest open-source credibility. Benchmark leakage is one of the easiest
  ways to undermine training claims.
- Matches the RLHF research lesson that eval evidence must stay independent
  from optimization data.
- Keeps release-gate claims clean and easier for external reviewers to trust.

Cons:

- Slower experimentation when a user intentionally wants benchmark-derived
  demonstrations or synthetic variants.
- Requires an override mechanism and audit trail for legitimate research cases.

### Option B: Warn by default, allow user override in UI/CLI

Pros:

- More flexible for power users.
- Useful for quick demos, ablations, and local-only experiments.

Cons:

- Easier to accidentally contaminate a training run.
- Harder to make strong public claims unless every override is carefully logged.

Recommendation: Option A. Keep eval-only sources blocked for training by
default, allow an explicit override only with a required reason saved to the
source manifest and run card.

---

## D2: First Cloud Launcher Priority

Question: which cloud/GPU launcher should become the first first-class target
after local and SSH/GX10?

### Option A: SkyPilot first

Pros:

- Strong fit for multi-cloud GPU jobs and reproducible launch YAML.
- Good bridge between local scripts, SSH-style execution, and cloud GPUs.
- Avoids locking BashGym into one rented-GPU provider too early.

Cons:

- Users still need cloud credentials and quota configured outside BashGym.
- More abstraction means more provider-specific errors to explain clearly.

### Option B: dstack first

Pros:

- Also open and portable across cloud/GPU providers.
- Good developer experience for reproducible GPU jobs.
- Fits the "bring your own provider" direction.

Cons:

- Smaller ecosystem surface than SkyPilot in many ML workflows.
- May require extra docs for provider setup and artifact sync.

### Option C: RunPod or Modal first

Pros:

- Easier for many users to understand: paste API key, pick GPU, run job.
- Faster path to a visible hosted-training flow.

Cons:

- More vendor-specific.
- Harder to preserve portability across GX10, cloud, and future backends.

Recommendation: Option A. Add SkyPilot dry-run config first, keep dstack as the
second open launcher, then add RunPod/Modal after the compute target schema is
stable.

---

## D3: First Source Expansion Set

Question: after the P0 source cards, which adapter family should we implement
first?

### Option A: Preference and reward-model sources

Examples: UltraFeedback, HelpSteer2, RewardBench/CUARewardBench eval adapters.

Pros:

- Directly closes the RLHF Book gap around reward models, ORM, and PRM.
- Supports DPO, reward-model training, and rejection sampling sooner.
- Mostly data-format work, less remote execution risk.

Cons:

- Less visually exciting than terminal-agent benchmarks.
- Generic preference data will not by itself improve terminal-agent behavior.

### Option B: Terminal/coding agent eval and environment sources

Examples: Harbor/Terminal-Bench, SWE-bench, SWE-agent/OpenHands trajectories.

Pros:

- Best aligned with BashGym's core terminal-agent differentiator.
- Feeds environment specs, pass@k, holdouts, DPPO replay, and release evidence.
- Easier to explain publicly as "BashGym is for executable agent training."

Cons:

- More adapter complexity and more sandbox/runtime edge cases.
- Higher contamination risk because benchmarks are public and often eval-only.

### Option C: Tool-use benchmarks first

Examples: BFCL and tau-bench.

Pros:

- Adds practical reliability signals for tool-call validity and argument quality.
- Smaller integration surface than full terminal environments.

Cons:

- Less central to terminal RL and DPPO/ECHO/RWML.
- Does not solve reward-model or terminal-environment gaps first.

Recommendation: Option B if the product story is terminal-agent training first.
Option A if we want the fastest path to RLHF-method completeness.

---

## D4: Human Approval Boundary For Remote Work

Question: when should BashGym require explicit human approval before launching
remote compute?

### Option A: Approval before any remote or billable action

Pros:

- Safest default for cost control and secret handling.
- Matches the existing GX10 checklist posture.

Cons:

- Adds friction for users running repeated experiments.

### Option B: Approval once per compute target

Pros:

- Lower friction after a target is trusted.
- Still creates an initial audit boundary.

Cons:

- A stale target or config change can still spend money unexpectedly.

Recommendation: Option A for now. Later, add a trusted-target mode with budget
limits and expiration.

---

## D5: First Installed Backend For GX10 Proof

Question: which backend should BashGym prove first for DPPO plus ECHO/RWML
diagnostic smoke runs on the GX10?

### Option A: SkyRL first

Pros:

- Best aligned with terminal-agent environments, tool use, and long-horizon
  agent workflows.
- Stronger product fit for BashGym's differentiator: executable terminal
  trajectories rather than generic prompt/completion RL.
- Likely easiest story to share with ML/agent-eval reviewers.

Cons:

- Integration surface may be narrower or less familiar than broad RL stacks.
- If SkyRL APIs shift, BashGym may need a maintained adapter shim.

### Option B: verl first

Pros:

- Stronger broad scale-out RL infrastructure story.
- Good fit for PPO/GRPO-style experimentation with distributed execution.
- More likely to map to common open RL training discussions.

Cons:

- Terminal-agent custom environment integration may require more glue.
- Less directly expressive of the BashGym terminal-trajectory thesis.

### Option C: OpenRLHF first

Pros:

- Familiar RLHF stack with SFT, RM, DPO, PPO-style paths.
- Good alignment with the RLHF Handbook method family.

Cons:

- Less directly centered on executable agent environments.
- May be better as the second proof after the terminal-agent backend path is
  established.

Recommendation: Option A if the immediate goal is to validate BashGym's unique
terminal-agent platform story. Option B if the immediate goal is to prove
scale-out RL compatibility to ML infra reviewers.

---

## D6: Claim-Tier Thresholds

Question: what evidence should BashGym require before saying a model is ready
for local smoke, narrow routing, or broad public claims?

Implementation status: RunCard validation now has `local_smoke`,
`narrow_routing`, and `broad_public_claim` scaffolding. The remaining human
decision is whether these recommended defaults should become the product policy
and what exact numeric thresholds should be attached to each tier.

### Option A: Conservative three-tier release ladder

Pros:

- Cleanest public credibility story.
- Makes it hard to accidentally overclaim from loss curves, reward curves, or
  diagnostic-only world-model metrics.
- Gives users a practical middle ground: narrow routing can ship before broad
  replacement claims.

Cons:

- Requires more UI and docs work to explain why a run can pass one tier but fail
  another.
- Slows demos that want one simple "ship" button.

Suggested thresholds:

- Local smoke: run completes, metrics exist, no OOM, no malformed artifacts.
- Narrow routing: heldout trace behavior is non-regressive, environment pass@k
  meets task-family threshold, tamper/spurious controls pass, run card complete.
- Broad public claim: narrow-routing evidence plus external benchmark manifests,
  decontamination/split evidence, paired comparison, and reproducible backend
  logs.

### Option B: Single promotion threshold

Pros:

- Simpler product surface.
- Easier for beginners to understand.

Cons:

- Forces very different claims into one bucket.
- Increases risk that a local smoke is mistaken for a public performance claim.

### Option C: User-defined custom tiers

Pros:

- Flexible for labs and advanced users.
- Can adapt to security, coding, tool-use, or company-specific requirements.

Cons:

- Harder to compare runs across users.
- Easier to configure weak evidence accidentally.

Recommendation: Option A. Implement three fixed claim tiers first, then allow
custom thresholds after the defaults are trusted.
