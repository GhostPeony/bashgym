# ECHO and RWML World Models

BashGym's world-model layer helps a terminal agent learn the dynamics around its
actions. It is JEPA-style in spirit: learn useful predictive structure about the
next state, not a perfect byte-for-byte simulator of the terminal.

The BashGym transition is:

```text
instruction + prior command/observation history + next command
    -> next terminal observation / diff / test state / verifier state
```

Verifiers, pass@k, holdout gates, tamper checks, and external benchmarks remain
the source of truth. ECHO and RWML are auxiliary objectives, curriculum signals,
and diagnostics until they are shown to correlate with heldout outcomes.

---

## What is wired today

| Layer                   | Status                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| Training API fields     | `TrainingRequest` accepts ECHO/RWML knobs.                                                    |
| Trainer config          | `TrainerConfig.world_model_settings()` returns a dedicated settings contract.                 |
| Training metadata       | Enabled ECHO/RWML settings are recorded in run metadata.                                      |
| DPPO replay export      | Replay records can include optional `world_model` payloads.                                   |
| Replay summary          | Replay metadata reports ECHO/RWML coverage.                                                   |
| DPPO smoke launch       | ECHO/RWML settings are exported as `BASHGYM_DPPO_*` env vars.                                 |
| Backend trainer adapter | `WorldModelTrainerAdapter` builds ECHO loss hooks and TRL/verl RWML reward hooks from replay. |
| Real backend loop       | Pending: an installed external DPPO/GRPO backend must call the adapter in its trainer.        |

This means the contract is ready, but a successful GPU/backend training run is
still separate work.

---

## ECHO

ECHO adds an auxiliary loss that asks the policy to predict terminal observation
tokens caused by its own actions.

```text
total loss = policy loss + echo_aux_lambda * environment_prediction_loss
```

In BashGym:

- `action` spans are the commands or assistant actions the policy owns.
- `observation` spans are terminal outputs, stderr, test output, and similar
  environment feedback.
- Replay JSONL stores role-tagged text spans, not token ids.
- The trainer backend tokenizes those spans with the model tokenizer, builds
  observation masks, and calls `echo_augmented_loss()`.

Starter:

```text
echo_enabled=true
echo_aux_lambda=0.05
```

Watch:

- ECHO environment-prediction loss.
- Observation-token or observation-character coverage.
- Exit-code and test-result prediction accuracy when available.
- Whether ECHO loss improves without hurting pass@k.

Do not:

- Fabricate token masks in replay export. The replay layer has text only.
- Treat low ECHO loss as release evidence without pass@k or holdout proof.

---

## RWML

RWML turns rollout steps into next-state prediction triplets and rewards a model
when its predicted next state is close to the real next state in embedding space.

```text
reward = 1 if 1 - cosine(embedding(predicted), embedding(actual)) < threshold
       = 0 otherwise
```

Each transition contains:

```json
{
  "instruction": "Create answer.txt containing ok.",
  "prior": [["ls", "README.md\n"]],
  "action": "echo ok > answer.txt",
  "next_state": ""
}
```

Starter:

```text
rwml_enabled=true
rwml_distance_threshold=0.2
rwml_easy_pass_rate_threshold=0.8
rwml_easy_keep_probability=0.1
rwml_history_window=4
rwml_embedding_model=<embedding model id>
rwml_kl_beta=0.0
```

Watch:

- RWML binary pass rate.
- Embedding-distance distribution.
- Mean and p95 embedding distance against the starter threshold.
- Easy vs hard transition retention.
- Prediction-error outliers for curriculum mining.
- Whether RWML improvements correlate with heldout pass@k.

Use RWML for:

- Pre-RL world-model learning.
- Ranking transitions by prediction difficulty.
- Candidate command reranking experiments.
- Keeping useful data from zero-std RL groups that policy-gradient updates drop.

Do not:

- Hardcode an embedding model. Pass it from the live provider/catalog.
- Use imagined multi-step rollouts as trusted evidence early. Start with
  one-step prediction and reranking.

---

## Replay payloads

DPPO replay records omit world-model data by default. When
`include_world_model=True`, each record gets an additive `world_model` key:

```json
{
  "world_model": {
    "rwml_transitions": [
      {
        "instruction": "...",
        "prior": [["act", "state"]],
        "action": "...",
        "next_state": "stdout\nstderr"
      }
    ],
    "echo": {
      "segments": [
        { "role": "action", "text": "ls" },
        { "role": "observation", "text": "README.md\n" }
      ],
      "n_action_chars": 2,
      "n_observation_chars": 10,
      "note": "Text-only ECHO summary..."
    }
  }
}
```

The base schema stays `bashgym.dppo_replay.v1`. Consumers that do not understand
`world_model` can ignore it.

---

## Replay coverage metrics

`summarize_world_model_payloads()` reports coverage, not prediction quality:

| Metric                             | Meaning                                              |
| ---------------------------------- | ---------------------------------------------------- |
| `records`                          | Replay records that include a `world_model` payload. |
| `records_missing_world_model`      | Replay records without the payload.                  |
| `rwml_transitions`                 | Number of action -> next-state triplets.             |
| `rwml_mean_transitions_per_record` | Average RWML triplets per enriched replay record.    |
| `rwml_mean_prior_pairs`            | Average prior history pairs per RWML transition.     |
| `rwml_max_prior_pairs`             | Largest history depth seen in the replay.            |
| `echo_segments`                    | Count of role-tagged action/observation text spans.  |
| `echo_action_chars`                | Action-text character coverage.                      |
| `echo_observation_chars`           | Observation-text character coverage.                 |
| `echo_observation_char_fraction`   | Observation chars divided by total ECHO chars.       |

These metrics answer "did the replay carry enough world-model material?" They
do not answer "did the world model become accurate?"

---

## Backend integration contract

A DPPO/GRPO backend should:

1. Read replay records with optional `world_model`.
2. Tokenize ECHO `segments` with the same model tokenizer used for training.
3. Build masks with `build_echo_masks()`.
4. Add `WorldModelTrainerAdapter.apply_echo_loss()` inside the backend's `compute_loss`.
5. Build RWML reward scoring with `build_trl_rwml_reward_func()` or
   `build_verl_rwml_reward_fn()`.
6. Use the adapter's cached/batched embedding path before real training runs.
7. Log quality metrics separately from replay coverage metrics.

Import targets:

```python
from bashgym.gym.world_model_backend import (
    WorldModelTrainerAdapter,
    WorldModelTrainerSettings,
    build_trl_rwml_reward_func,
    build_verl_rwml_reward_fn,
)
```

For TRL `GRPOTrainer`, use BashGym's verifier reward plus the RWML reward
function in `reward_funcs`. TRL custom rewards receive completions and dataset
columns; use an `actual_next_state` column or replay order for targets. ECHO is
not a reward function: subclass the trainer loss path and call
`adapter.apply_echo_loss(base_loss, logits, input_ids=...)` after the base GRPO
loss is computed.

For verl, point a custom reward shim at `build_verl_rwml_reward_fn()` for the
documented `data_source, solution_str, ground_truth, extra_info` signature. verl
reward managers are sufficient for RWML scoring, but ECHO still requires a
trainer/loss-path hook.

For SkyRL, the environment/reward loop is the best long-term fit for terminal
trajectories. Register BashGym as a text environment, keep RWML in reward
postprocessing, and apply ECHO through a policy-loss hook or trainer subclass.
The ECHO implementation should preserve separate action, terminal-observation,
and warning/synthetic masks.

Recommended metric names:

```text
echo_loss
rwml_pass_rate
embedding_distance_mean
embedding_distance_p95
exit_code_accuracy
test_result_accuracy
```

The Training Monitor parses those names from backend stat dictionaries and shows
the World-Model Quality panel automatically when they appear.

To carry the same diagnostics into a heldout release verdict, attach:

```json
{
  "world_model_quality": {
    "metrics": {
      "echo_loss": { "first": 1.2, "last": 0.8 },
      "rwml_pass_rate": 0.72,
      "embedding_distance_mean": 0.12,
      "exit_code_accuracy": 0.9
    },
    "coverage": {
      "world_model_records": 16,
      "rwml_transitions": 42
    }
  }
}
```

The combined release gate records this as diagnostic evidence. It does not block
or allow shipping by itself.

The launch planner exports these settings for the backend:

```text
BASHGYM_DPPO_ECHO_ENABLED
BASHGYM_DPPO_ECHO_LAMBDA
BASHGYM_DPPO_RWML_ENABLED
BASHGYM_DPPO_RWML_DISTANCE_THRESHOLD
BASHGYM_DPPO_RWML_EASY_PASS_RATE_THRESHOLD
BASHGYM_DPPO_RWML_EASY_KEEP_PROBABILITY
BASHGYM_DPPO_RWML_HISTORY_WINDOW
BASHGYM_DPPO_RWML_EMBEDDING_MODEL
BASHGYM_DPPO_RWML_KL_BETA
BASHGYM_DPPO_WORLD_MODEL_ADAPTER
BASHGYM_DPPO_ECHO_LOSS_HOOK
BASHGYM_DPPO_TRL_RWML_REWARD_FACTORY
BASHGYM_DPPO_VERL_RWML_REWARD_FACTORY
```

Before moving this to a private or cloud GPU target, produce the local smoke bundle:

```bash
bashgym training smoke-bundle \
  --replay data/dppo_replay/latest.jsonl \
  --output-dir data/backend-smokes/latest \
  --backend auto \
  --json
```

The bundle writes a replay summary, an ECHO/RWML backend probe, the exact launch
environment contract, and a readiness report. Treat it as the preflight gate:

- `contract_ready=false` means replay/logprob/world-model data is missing.
- `optimizer_ready=false` means train-policy logprob enrichment is still needed.
- `backend_launch_ready=false` with `contract_ready=true` means the replay handoff
  is shaped correctly and the next step is installing/configuring the backend.

---

## Promotion path

Use this progression before treating world-model metrics as product evidence:

1. Replay coverage looks healthy.
2. Tiny backend smoke run completes.
3. ECHO/RWML quality metrics improve on heldout transitions.
4. Candidate command reranking improves pass@k or reduces command count.
5. Improvements survive holdout gates, spurious-reward controls, tamper checks,
   and external benchmarks.
6. Only then consider adding world-model metrics to release evidence.

---

## Source references

- [../../bashgym/gym/echo.py](../../bashgym/gym/echo.py)
- [../../bashgym/gym/echo_trainer.py](../../bashgym/gym/echo_trainer.py)
- [../../bashgym/gym/rwml.py](../../bashgym/gym/rwml.py)
- [../../bashgym/gym/world_model_backend.py](../../bashgym/gym/world_model_backend.py)
- [../../bashgym/eval/dppo_replay.py](../../bashgym/eval/dppo_replay.py)
- [../../bashgym/gym/dppo_launcher.py](../../bashgym/gym/dppo_launcher.py)
- [../../tests/eval/test_dppo_replay.py](../../tests/eval/test_dppo_replay.py)
- [../../tests/gym/test_world_model_config.py](../../tests/gym/test_world_model_config.py)
- [../../tests/gym/test_world_model_trainer_adapter.py](../../tests/gym/test_world_model_trainer_adapter.py)
