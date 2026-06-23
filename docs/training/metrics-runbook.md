# Training Metrics Runbook

Use this runbook while a training run is live or when a model looks better in
loss but not better in behavior. The north star is executable evidence:
pass@1/pass@k, heldout gates, verifier status, tamper controls, and external
benchmarks.

For a machine-readable first pass over local artifacts, run:

```bash
bashgym training analyze --run-id <run-id> --json
```

You can also pass explicit artifacts:

```bash
bashgym training analyze \
  --metrics data/models/<run-id>/metrics.jsonl \
  --replay data/dppo_replay/<run-id>.jsonl \
  --release-evidence artifacts/<run-id>-release.json \
  --json
```

The analyzer highlights missing heldout evidence, zero reward variance, high
timeout/tamper rates, release-gate blockers, and world-model replay coverage
without ECHO/RWML quality metrics.

When a backend smoke produces world-model quality metrics, attach them to the
heldout release run as diagnostic evidence:

```json
{
  "world_model_quality": {
    "metrics": {
      "echo_loss": { "first": 1.2, "last": 0.8 },
      "rwml_pass_rate": 0.72,
      "embedding_distance_mean": 0.12,
      "exit_code_accuracy": 0.9,
      "test_result_accuracy": 0.84
    },
    "coverage": {
      "world_model_records": 16,
      "rwml_transitions": 42
    }
  }
}
```

This evidence appears in the combined release verdict and Training/Evaluator
dashboards, but it remains diagnostic. Ship/no-ship is still decided by trace,
environment, safety, and external benchmark gates.

---

## What to monitor by strategy

| Strategy | Primary metrics | Release evidence |
|---|---|---|
| SFT | Train loss, eval loss, grad norm, token count, truncation warnings. | Heldout trace eval, executable environment pass@k. |
| DPO | Chosen/rejected rewards, reward margin, preference accuracy, chosen/rejected logprobs. | Heldout trace eval and task behavior against the SFT baseline. |
| GRPO/RLVR | Reward, `reward_std`, `frac_reward_zero_std`, KL, entropy, verifier status, timeouts. | pass@1/pass@k, holdout gate, spurious-reward control, tamper canaries. |
| DPPO | Behavior logprobs ready, train logprobs ready, replay-required records, trust-region mask telemetry. | Backend smoke artifacts plus pass@k before/after. |
| ECHO/RWML | Replay coverage, ECHO loss, RWML pass rate, embedding-distance distribution, exit-code/test-result prediction accuracy. | Diagnostic release evidence only until correlated with heldout pass@k and safety. |
| Cascade | Per-stage loss/reward, per-domain pass@k, stage-to-stage forgetting. | Domain holdouts and final generalist holdout. |

---

## Symptom: loss improves but pass@k is flat

Likely causes:

- The model is learning formatting or local style, not outcome-producing action.
- Max sequence length truncates the verifier, final fix, or recovery step.
- Training examples include weak or unverified sessions.
- The heldout environments require skills missing from the training data.

Actions:

1. Inspect a sample of generated examples for truncation and missing final
   verification.
2. Compare train repos/domains against heldout repos/domains.
3. Add verifier-backed examples or terminal environments for the missing skill.
4. Use SFT or teacher distillation before RL if pass@k is still zero.
5. Move to GRPO/RLVR only once sampled attempts sometimes pass.

Done when:

- pass@k improves on heldout environments without increased tamper, timeout, or
  verifier-error rates.

---

## Symptom: `reward_std` is zero or `frac_reward_zero_std` is near 1.0

Likely causes:

- Every sampled attempt gets the same reward.
- The reward function is too binary for the current model.
- Tasks are too hard, so all attempts fail.
- Tasks are too easy, so all attempts pass.

Actions:

1. Enable zero-std filtering and active sampling.
2. Raise group size toward 16-32 if compute allows.
3. Add graded rewards or intermediate verifier signals.
4. If all attempts fail, train with SFT/distillation or easier curriculum first.
5. If all attempts pass, keep those environments for regression and train on
   harder ones.

Done when:

- Non-zero-std prompt groups are selected consistently and pass@k changes in the
  intended direction.

---

## Symptom: pass@k is all zero

Likely causes:

- The model cannot produce a valid first action.
- The environment is too hard for the current checkpoint.
- The verifier is broken or unreachable.
- The prompt budget hides important environment observations.

Actions:

1. Run one local/manual rollout to confirm the verifier can pass.
2. Inspect rollout observations and verifier output.
3. Increase observation prompt budget if important shell output is missing.
4. Create easier curriculum environments from the same task family.
5. Use teacher distillation or SFT warm start before RL.

Done when:

- pass@k rises above zero and reward groups have contrast.

---

## Symptom: pass@k is high but heldout gate blocks

Likely causes:

- The training/eval split leaks task family, fixture, or solution pattern.
- The model learned a narrow shortcut.
- The pass@k set is too easy compared with holdout.

Actions:

1. Check contamination manifests and split keys.
2. Run holdout comparison against the base model.
3. Run spurious-reward negative controls.
4. Add harder environments from unseen task families.
5. Require external benchmark evidence for broad claims.

Done when:

- Holdout gate ships, comparison beats base by the required delta, and controls
  stay below the allowed false-positive rate.

---

## Symptom: timeouts are high

Likely causes:

- The model loops or over-explores.
- Max tool calls per episode is too high for the environment.
- Commands hang or wait for input.
- Observations are too long and bury the useful state.

Actions:

1. Inspect command traces for repeated commands or blocking commands.
2. Lower max tool calls for smoke runs.
3. Add command timeout or safe-shell guardrails.
4. Shorten observations while preserving verifier-relevant output.
5. Add SFT examples that show concise terminal recovery.

Done when:

- Timeout rate drops without lowering pass@k.

---

## Symptom: verifier errors are high

Likely causes:

- The environment setup is incomplete.
- The verifier command depends on missing files, packages, or paths.
- The model tampers with verifier inputs.
- The verifier is nondeterministic.

Actions:

1. Run verifier-only checks on materialized environments.
2. Inspect protected-file manifests.
3. Run reward-hacking canaries.
4. Separate verifier failures from model failures in reporting.
5. Fix the environment before training on its rewards.

Done when:

- Verifier errors are rare and failures mostly reflect model behavior, not broken
  infrastructure.

---

## Symptom: tamper attempts appear

Likely causes:

- The model found a reward shortcut.
- Protected files are writable or not checked.
- Training data contains bad examples of modifying tests or verifiers.

Actions:

1. Treat tamper as a release blocker.
2. Preserve and inspect the workspace.
3. Confirm manifest checksums cover verifier scripts, tests, fixtures, and
   `env.json`.
4. Add or strengthen reward-hacking canaries.
5. Remove or relabel training examples that reward test/verifier edits.

Done when:

- Tamper canaries are guarded and heldout attempts show no verifier or fixture
  tampering.

---

## Symptom: DPO reward margin grows but behavior worsens

Likely causes:

- Chosen/rejected pairs are not actually for the same prompt.
- Rejected examples are too obviously bad.
- Beta or learning rate is too aggressive.
- The run is over-optimizing preference labels and forgetting task behavior.

Actions:

1. Audit pairs for prompt identity and comparable length.
2. Lower LR or beta.
3. Use fewer epochs.
4. Re-run heldout trace eval against the SFT checkpoint.
5. Rebuild pairs from decision-level mistakes rather than whole unrelated traces.

Done when:

- Preference accuracy improves without heldout regression.

---

## Symptom: world-model coverage exists but quality is unclear

Likely causes:

- Replay telemetry is only coverage telemetry.
- The backend has not produced predicted next states.
- ECHO/RWML metrics are not correlated against pass@k yet.

Actions:

1. Confirm replay coverage: records, transitions, history depth, ECHO segments.
2. Run a tiny backend smoke that logs ECHO loss or RWML prediction quality.
3. Compare prediction-error outliers to failed rollouts.
4. Use high-error transitions for curriculum mining.
5. Attach `world_model_quality` to release evidence as diagnostic context.
6. Do not use world-model metrics as release blockers until pass@k correlation is
   demonstrated.

Done when:

- World-model quality improves on heldout transitions and predicts useful
  changes in pass@k, command count, timeout rate, or tamper rate.

---

## Minimum release checklist

Before routing real traffic to a trained student:

- Heldout trace eval is not worse than the baseline.
- Environment pass@k improves or meets the required threshold.
- Holdout gate passes on unseen groups.
- Base-vs-candidate comparison passes if a baseline exists.
- Spurious-reward controls do not pass by chance.
- Reward-hacking canaries are guarded.
- External benchmark evidence is attached for broad capability claims.
- World-model quality evidence is attached when ECHO/RWML was enabled, and it is
  interpreted as diagnostic context.
- No unresolved verifier-error or tamper pattern remains.

---

## Source references

- [../../bashgym/eval/environment_passk.py](../../bashgym/eval/environment_passk.py)
- [../../bashgym/eval/environment_holdout.py](../../bashgym/eval/environment_holdout.py)
- [../../bashgym/eval/environment_holdout_comparison.py](../../bashgym/eval/environment_holdout_comparison.py)
- [../../bashgym/eval/environment_spurious_reward.py](../../bashgym/eval/environment_spurious_reward.py)
- [../../bashgym/eval/release_gate.py](../../bashgym/eval/release_gate.py)
- [../../bashgym/gym/trainer.py](../../bashgym/gym/trainer.py)
- [../../bashgym/eval/dppo_replay.py](../../bashgym/eval/dppo_replay.py)
