# BashGym Platform Flywheels

BashGym should be explained as a set of connected flywheels, not one giant
training menu. The original product loop stays at the center:

```text
coding work
  -> trace extraction
  -> curation and dataset generation
  -> training method
  -> eval and routing
  -> more coding work
```

The other flywheels support, strengthen, or govern that core loop. They should
be shown as separate loops in product docs and UI so users do not confuse source
discovery, reward modeling, terminal RL, world-model diagnostics, and cloud
compute with the same job.

## Flywheel Map

| Flywheel | Role | Starts from | Produces | Primary user question |
|---|---|---|---|---|
| Core trace-to-training | Main BashGym product loop. | Real coding sessions. | Gold traces, datasets, trained adapters, routing evidence. | How do my coding sessions become a better local/open model? |
| Data quality and Data Designer | Turns raw traces or sources into better artifacts. | Traces, source cards, external files, synthetic gaps. | SFT examples, DPO pairs, reward examples, dataset cards. | Is this data good enough to train on? |
| Preference and reward | Learns or validates what "better" means. | Chosen/rejected traces, reward examples, verifier outcomes. | DPO pairs, reward models, reward evals, selection scores. | Can we rank or score behavior safely? |
| Terminal RL and environments | Optimizes behavior inside executable tasks. | Environment specs and rollout attempts. | Pass@k, verifier rewards, DPPO replay, backend evidence. | Can the model improve by acting in a real shell? |
| JEPA/ECHO/RWML diagnostics | Learns predictive terminal state signals. | Action/observation spans and transitions. | World-model metrics, curriculum signals, diagnostic evidence. | Does the model understand terminal dynamics better? |
| Source and domain library | Finds public or internal data/eval sources safely. | Curated source cards. | Source manifests, converted artifacts, eval manifests. | What data or benchmark should I use, and is it training-safe? |
| AutoResearch recipes | Searches for better data mixes and curricula. | Candidate sources, domains, quality thresholds, eval targets. | Ranked recipes, cost/improvement evidence, Data Designer handoff. | Which data mix improves the model most? |
| Compute and RunCards | Makes runs reproducible and auditable. | Training plan, compute target, artifacts, metrics. | Launch plan, logs, synced artifacts, promotion verdict. | Can we prove what ran and what it changed? |
| Education and recipes | Keeps operators oriented. | Method docs, defaults, failure examples. | Guided paths, setting help, failure labs. | What should I run next, and what do these knobs mean? |

## 1. Core Trace-To-Training Flywheel

This is the original BashGym concept and should remain the first explanation.

```text
agent/coder works in a repo
  -> BashGym captures traces
  -> traces are scrubbed, scored, classified, and segmented
  -> Data Designer creates trainable examples
  -> SFT, DPO, distillation, GRPO/RLVR, or DPPO trains a model
  -> heldout traces, environments, and release gates evaluate it
  -> routing sends narrow safe work to the trained model
  -> new sessions become new traces
```

What belongs here:

- Live and historical capture from Claude Code, Codex, Gemini, OpenCode,
  Copilot CLI, and similar tools.
- Trace normalization, quality scoring, PII scrubbing, and failure labeling.
- SFT examples from gold traces.
- DPO pairs from same-prompt better/worse behavior.
- Heldout trace eval and conservative routing.

What does not belong here:

- Public benchmark ingestion as default training data.
- World-model metrics as release proof.
- Cloud launch details before the data and evidence are valid.

Core action items:

- Keep the README and Training Overview centered on this loop.
- Add UI labels that show where a user is in the loop: Capture, Curate,
  Generate, Train, Evaluate, Route.
- Attach dataset cards and RunCards to every serious handoff between stages.

## 2. Data Quality And Data Designer Flywheel

This flywheel improves the core loop's fuel.

```text
raw traces or sources
  -> filtering, splitting, dedup, quality scoring
  -> Data Designer transformations
  -> dataset cards and artifact validators
  -> training plan handoff
  -> eval feedback updates the next data recipe
```

Use this when:

- Gold traces are too sparse.
- Failed traces need to become DPO negatives.
- A source needs conversion into SFT, DPO, reward, or environment artifacts.
- The model is learning format but not solving tasks.

Key artifacts:

- `training_examples.jsonl`
- `dpo_pairs.jsonl`
- `reward_examples.jsonl`
- `environment_specs.jsonl`
- `dataset_card.json`
- `source_manifest.json`

Implementation actions:

- Wire source adapters into Data Designer prepare flows.
- Preserve split, contamination, quality, source, and prompt-hash metadata.
- Add UI explanations for why a row is accepted, warned, or blocked.

## 3. Preference And Reward Flywheel

This flywheel teaches BashGym how to compare or score behavior.

```text
chosen/rejected behavior or reward labels
  -> strict validators
  -> DPO, reward model, ORM, or PRM
  -> heldout reward eval
  -> rejection sampling or trajectory scoring
  -> behavior eval proves whether the signal helps
```

Use this when:

- The model can produce plausible attempts but makes lower-quality choices.
- There are meaningful failed or weaker alternatives for the same prompt.
- A verifier, judge, or reward model can score rollouts with non-zero contrast.

Do not confuse this with:

- Release evidence by itself. Reward accuracy, reward margin, and calibration
  are signal quality metrics. Behavior gates still decide promotion.

Implementation actions:

- Add real TRL/OpenRLHF-style reward backend integration.
- Keep RewardBench and CUARewardBench eval-only by default.
- Add rejection sampling with selected-vs-random matched controls.

## 4. Terminal RL And Environment Flywheel

This is the TMax-style executable agent loop.

```text
environment specs
  -> rollout attempts
  -> verifier rewards and pass@k
  -> active sampling and non-zero reward groups
  -> GRPO/RLVR or DPPO backend training
  -> holdout, canary, and spurious-reward gates
  -> harder or broader environments
```

Use this when:

- SFT has taught the model the action format.
- The model can produce attempts worth scoring.
- Verifier rewards have variation across sampled attempts.

Key risks:

- Zero reward variance burns compute.
- Public benchmark leakage weakens claims.
- Verifier hacking can look like improvement.

Implementation actions:

- Pick the first installed backend for GX10 proof.
- Run one tiny DPPO smoke with saved replay, logs, metrics, output listing, and
  RunCard evidence.
- Keep environment pass@k and safety gates as the behavior proof.

## 5. JEPA/ECHO/RWML Diagnostic Flywheel

This is a supporting world-model loop, not a promotion loop yet.

```text
terminal actions and observations
  -> ECHO action/observation spans
  -> RWML state transitions
  -> auxiliary losses or diagnostic probes
  -> world-model quality metrics
  -> curriculum/routing hypotheses
  -> correlation against behavior gates
```

Use this when:

- You want to understand whether the model predicts terminal consequences.
- You want curriculum signals for harder environments.
- You are running a backend that can emit ECHO/RWML quality metrics.

Boundary:

- ECHO/RWML metrics stay diagnostic until they correlate with heldout pass@k,
  timeout rate, command count, and safety metrics.

Implementation actions:

- Wire/test hooks inside the chosen backend.
- Save ECHO/RWML metrics in RunCards.
- Build correlation reports before using these metrics for release or routing.

## 6. Source And Domain Library Flywheel

This flywheel expands the platform beyond private traces while preserving
credibility.

```text
source card
  -> eligibility and risk review
  -> adapter conversion
  -> source manifest and dataset/eval card
  -> training, eval, or Data Designer handoff
  -> results update source recommendations
```

Use this when:

- A user wants public data, external evals, or domain-specific corpora.
- AutoResearch needs candidate sources.
- A reviewer asks where training/eval data came from.

Boundary:

- Eval-only sources are not training data unless there is an explicit override
  reason saved to the source manifest and RunCard.

Implementation actions:

- Implement fixture-backed source adapters first.
- Add one SFT adapter, one preference/reward adapter, and one terminal
  environment adapter.
- Keep eval-only benchmark adapters separate from training adapters.

## 7. AutoResearch Data Recipe Flywheel

This flywheel searches for better recipes instead of relying on guesswork.

```text
source candidates and constraints
  -> proposed data recipe
  -> small run or simulated objective
  -> heldout behavior and cost metrics
  -> winning recipe export
  -> Data Designer and training plan handoff
```

Use this when:

- There are multiple safe training sources.
- You need to tune source weights, domains, quality thresholds, or sample sizes.
- Cost per improvement matters.

Implementation actions:

- Add comparison charts.
- Run one small real two-source search.
- Track cost per pass@k point.
- Export the best recipe into Data Designer and RunCards.

## 8. Compute And Evidence Flywheel

This flywheel makes real training reproducible.

```text
training plan
  -> compute target preflight
  -> dry-run launch config
  -> local, GX10, or cloud run
  -> metrics, logs, checkpoints, and artifacts
  -> RunCard validation
  -> release or iterate
```

Use this when:

- A run moves beyond local fixture smoke.
- GX10 or cloud GPUs are involved.
- A model claim needs to be reviewed or reproduced.

Boundary:

- Remote or billable actions should require explicit human approval until
  trusted-target budgets and expirations exist.

Implementation actions:

- Add Training Config compute target selection.
- Add artifact sync plans.
- Add status/log commands after launch paths are trusted.
- Keep secret values redacted in every payload and log.

## 9. Education And Onboarding Flywheel

This flywheel reduces confusion as the platform gains power.

```text
operator intent
  -> method recommendation
  -> guided recipe and starter settings
  -> smoke run
  -> metric interpretation
  -> failure lab or next recipe
```

Use this when:

- A user asks what method to run.
- A setting is technical enough to need context.
- A run fails or plateaus.

Implementation actions:

- Add beginner recipes per method.
- Add hardware-specific defaults.
- Add failure labs for contamination, length bias, zero reward variance, reward
  hacking, over-optimization, and single-pass unreliability.
- Add UI field help that ties settings to metrics and failure modes.

## Product Segmentation Rule

Every new feature should answer one of these questions:

1. Does it improve the core trace-to-training loop?
2. Does it improve data quality, reward quality, environment quality, or compute
   reproducibility for that loop?
3. Does it make the loop easier to understand without weakening evidence?

If a feature does not answer one of those questions, it belongs in backlog or
research notes, not the primary product path.
