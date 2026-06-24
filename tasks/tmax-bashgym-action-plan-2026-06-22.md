# TMax to BashGym Action Plan

Compiled: 2026-06-22

## Executive Summary

TMax is important for BashGym because it validates the exact direction BashGym is already aiming at: train small open models into stronger terminal agents using executable tasks, verifiable rewards, long-horizon rollouts, and rigorous evals. The difference is that TMax makes the executable environment the first-class training unit, while BashGym currently treats traces, generated tool-use conversations, rewards, and benchmark harnesses as mostly separate pieces.

The plan: make BashGym produce and train on TMax-style terminal environments, keep BashGym's personal-trace advantage, and add the stability/reproducibility rails TMax/Olmo/DPPO show are needed before expensive RL runs.

## Research Takeaways

1. Recipe work matters more than novelty right now.
   Nathan Lambert's post frames TMax as an open RL recipe: data, algorithm, infrastructure, pitfalls, and artifacts. BashGym should follow that pattern by publishing a repeatable "BashGym terminal RL recipe", not just adding isolated features.

2. The training unit should be an executable environment, not only a transcript.
   TMax-15K contains 14,600 RL environments with task instructions, source files, container build context, and programmatic verifiers. The WAI blog describes a compositional pipeline over structured axes, plus Docker-style build validation and graded verifiers. BashGym has trace capture and real-tool generation, but needs a first-class environment package.

3. Difficulty control is the data moat.
   TMax samples across domain, skill, persona, fixture, language, task complexity, command complexity, and verifier style, then tracks balance and pass@k. BashGym should stop treating "more traces" as the main lever and start tracking whether tasks are too easy, impossible, or genuinely learnable.

4. Outcome-only RL works, but only with stability engineering.
   TMax uses DPPO, token-level loss, zero-standard-deviation filtering, active sampling, FP32 LM head, large group size, long context, and asynchronous vLLM rollouts. Olmo 3 independently supports active sampling, zero-gradient filtering, token-level loss, and careful decontamination. DPPO's core lesson is that PPO/GRPO ratio clipping is unstable for LLM token distributions because it over-constrains low-probability tokens and under-constrains large distribution shifts.

5. Strong post-trained models may not want SFT warm-starts.
   TMax finds SFT hurts Qwen 3.5 9B but helps older Qwen 3 8B. BashGym should choose SFT warm-start conditionally by base model and should prove it by ablation, not bake in "SFT then RL" as default.

6. External eval has to be multi-harness and contamination-aware.
   TMax reports Terminal-Bench, Terminal-Bench Lite/2.1, SWE-Bench, AIME, and harness-transfer results. Olmo 3 adds spurious-reward negative controls for contamination. BashGym already has held-out trace eval, Terminal-Bench command builders, pass@k math, and decontamination utilities; the action is to wire them into a release gate.

## Current BashGym Fit

- Training config already exposes SFT/DPO/GRPO/RLVR, vLLM toggles, backend dispatch, and loss variants in `bashgym/gym/trainer.py`.
- Cascade RL already splits domains into file ops, bash, search, and multi-step reasoning in `bashgym/gym/cascade_scheduler.py`.
- Data Designer already supports real sandbox MCP tool-use in `bashgym/factory/designer_pipelines/mcp_tool_use.py`.
- The sandbox/verifier layer exists in `bashgym/arena/sandbox.py`, `bashgym/mcp/sandbox_server.py`, and `bashgym/judge/verifier.py`.
- Eval scaffolding exists for held-out trace gates, episode pass@k, Terminal-Bench, BFCL, SWE-bench, and forgetting evals in `bashgym/eval/`.
- Decontamination exists via 13-gram and 3-gram Jaccard gates in `bashgym/datasets/decontaminate.py`.

The missing centerpiece is an executable environment dataset layer that connects generation -> build/smoke -> rollout -> verifier reward -> eval record.

## Action Plan

### Phase 0 - Ground Truth Baseline, No Training

Goal: make TMax measurable inside BashGym before we change training code.

- Add a TMax artifact importer that can read:
  - `allenai/TMax-15K`
  - `allenai/tmax-15k-open-instruct`
  - `tmax/TMax-15K-Harbor@latest`
- Create `bashgym/environments/contracts.py` with:
  - `EnvironmentSpec`
  - `EnvironmentAxis`
  - `VerifierSpec`
  - `FixtureSpec`
  - `BuildSpec`
  - `RolloutSpec`
- Convert 25 TMax tasks into `EnvironmentSpec` and preserve original metadata.
- Run a 10-task smoke with one current BashGym student/base model through the existing Harbor or Terminal-Bench command seam.
- Record:
  - build success rate
  - pass@1/pass@4/pass@8
  - mean turns and tokens/run
  - verifier failures
  - timeout rate

Exit criteria:

- One checked-in fixture file under `tests/fixtures/tmax_envs/`.
- Unit tests for importer and `EnvironmentSpec` serialization.
- A local baseline report in `artifacts/tmax_baseline_2026-06-22.json`.

### Phase 1 - Make Executable Environments First-Class

Goal: turn BashGym from "trace trainer" into "trace + executable environment trainer".

- Add `bashgym/environments/`:
  - `contracts.py`: typed schemas.
  - `builder.py`: writes task directory, source files, verifier, Dockerfile/compose.
  - `loader.py`: reads local/TMax/Harbor-like environment bundles.
  - `metrics.py`: domain balance, skill balance, pass@k difficulty summaries.
  - `decontaminate.py`: wraps existing decontamination against benchmark task text.
- Add a Data Designer pipeline `terminal_env_generation`:
  - Sample axes: domain, skill, persona, fixture, language, task complexity, command complexity, verifier kind.
  - Generate task instructions, files, verifier, and optional fixture.
  - Build/smoke the environment once.
  - Mark rows with `build_passed`, `verifier_kind`, `difficulty_bucket`, and `passes_quality`.
- Upgrade the existing `mcp_tool_use` pipeline so it can attach a verifier artifact, not only a judged transcript.
- Add graded verifier types:
  - exact success
  - metric threshold
  - adversarial corpus
  - fuzz equivalence
  - multi-protocol service check

Exit criteria:

- Generate 50 BashGym-native environments locally.
- >= 90% build success.
- Each environment has a runnable verifier.
- Domain/skill/task-complexity balance report is emitted.

### Phase 2 - Adopt TMax/Olmo Stability Settings Before Expensive RL

Goal: stop wasting RL runs on known bad regimes.

- Add a named training profile: `terminal_rl_tmax_like`.
- Extend `TrainerConfig` with explicit fields instead of overloading existing ones:
  - `grpo_group_size` default 32 for terminal RL.
  - `prompts_per_rollout_batch` default 8.
  - `max_tool_calls_per_episode` default 64.
  - `token_level_loss` toggle.
  - `filter_zero_std_groups` toggle.
  - `active_sampling` toggle.
  - `lm_head_fp32` toggle.
  - `interleaved_thinking` toggle.
  - `sft_warm_start_policy`: `never|weak_models_only|always|ablation`.
- Implement active sampling in the GRPO/RLVR path:
  - Continue sampling prompts until a batch has enough non-zero-advantage groups.
  - Track dropped all-zero/all-one groups separately.
  - Surface `frac_reward_zero_std` and active-sampling refill counts in the UI.
- Keep SFT warm-start off for strong modern post-trained models by default.
- Add a warning/gate when terminal RL starts with group size < 16 or without zero-std filtering.

Exit criteria:

- Existing GRPO tests pass.
- New tests prove zero-std filtering and active sampling maintain a full effective batch.
- Simulated cascade run reports group size, zero-std fraction, and refill count.

### Phase 3 - DPPO as a Deliberate Backend, Not a Flag Name

Goal: add DPPO only where we can verify the math and logprob plumbing.

- First path: add an external backend wrapper for a known DPPO-capable stack if available in the training environment, likely `verl`, `SkyRL`, or TMax's open-instruct fork.
- Second path: if keeping TRL, add `dppo_binary_tv` only behind a backend capability check.
- Validate against TMax released rollouts/logprobs:
  - Binary-TV divergence mask matches expected behavior on sampled tokens.
  - Trust region is anchored to the rollout/behavior policy, not recomputed policy.
  - Threshold defaults to the TMax-style range, with config override.
- Add telemetry:
  - train/inference logprob max diff
  - mean absolute policy mismatch
  - percentage of masked updates
  - collapse detector after 200-300 steps

Exit criteria:

- DPPO unit tests over synthetic logits.
- One smoke run on a tiny environment set proves the backend starts, logs mask stats, and saves artifacts.
- GRPO remains the stable fallback.

### Phase 4 - Persistent Terminal Harness and Rollout Infrastructure

Goal: train the actual behavior we care about: long-running terminal work.

- Add a persistent-shell harness module:
  - `bashgym/arena/harness.py`
  - command execution with persistent cwd/env
  - max tool calls
  - per-command timeout
  - observation truncation
  - submit marker
  - format-error recovery
- Use the same harness for:
  - generated environment rollouts
  - SFT trajectory generation
  - RL rollouts
  - evaluation
- Build a sandbox pool:
  - prebuilt per-domain base images
  - workspace reset/reuse
  - resource contention telemetry
  - network-off by default
- Add reward-hacking guardrails:
  - verifier file tamper detection
  - task file manifest checksums
  - forbidden edits to `/tests`, verifier scripts, and private fixtures unless task allows it
  - audit logs for suspicious verifier/pass shortcuts

Exit criteria:

- One local 20-environment rollout batch completes without manual intervention.
- Harness emits per-episode steps, tokens, tool calls, timeout, reward, and verifier status.
- Reward-hacking canaries fail when the agent tampers with the checker.

### Phase 5 - Evaluation Gate: TMax-Style, BashGym-Owned

Goal: make "better" mean statistically and operationally better.

- Promote S6 external benchmarks from roadmap to release gate:
  - Terminal-Bench 2.0/2.1 via Harbor or official `tb run`.
  - Terminal-Bench Lite for cheaper frequent checks.
  - SWE-bench Verified Lite/50.
  - BFCL-V4 local categories.
  - Forgetting suite through lm-eval/vLLM.
- Add multi-harness evaluation:
  - BashGym persistent shell harness
  - mini-swe-agent style harness
  - OpenHands if practical
  - Terminus/Harbor where official settings require it
- Add an environment holdout gate:
  - split by source, repo, domain, and generator seed
  - report pass@1/pass@4/pass@8
  - run paired bootstrap clustered by task family/session
- Add Olmo-style spurious-reward control:
  - random binary reward training must not improve the benchmark.
  - if it does, block release and inspect contamination.

Exit criteria:

- `POST /api/eval/heldout` can include environment pass@k plus precomputed
  environment holdout, holdout-comparison, and spurious-reward gate evidence.
- Model registry records Terminal-Bench, BashGym environment holdout, forgetting, and contamination verdict.
- Deploy is blocked on regression.

### Phase 6 - AutoResearch Over Data Recipe, Not Just Hyperparameters

Goal: make BashGym's flywheel optimize the environment recipe itself.

- Extend `SchemaSearchSpace` to mutate:
  - axis weights
  - verifier-kind mix
  - fixture-kind mix
  - task/command complexity buckets
  - persona selection
  - domain balancing target
- Optimize for a target learnability band:
  - too easy: high pass@1 and high pass@8
  - too hard: near-zero pass@8
  - target: moderate pass@1 with meaningful pass@8 lift
- Add reward from downstream micro-RL/eval, not only SFT loss.
- Feed high-loss/high-failure signatures from TraceResearcher back into environment generation.

Exit criteria:

- AutoResearch can produce a new environment mix proposal with measured pass@k/difficulty/balance changes.
- The proposal is exportable as a reproducible recipe file.

## Recommended Implementation Order

1. Build `EnvironmentSpec` and TMax importer.
2. Generate/report 50 BashGym-native executable environments.
3. Wire environment pass@k into existing eval service.
4. Add active sampling + terminal RL profile to GRPO/RLVR.
5. Run a small RL smoke on generated environments.
6. Add DPPO backend only after rollout logprob plumbing is observable.
7. Promote Terminal-Bench/SWE-bench/BFCL to a release gate.
8. Let AutoResearch mutate environment axes after the fixed recipe works.

## Do Not Do

- Do not train on Terminal-Bench tasks or anything that can leak into reported evals.
- Do not assume SFT warm-start helps strong post-trained bases.
- Do not rely on LLM judges for RL rewards where a verifier can exist.
- Do not start long RL runs with small group size, no active sampling, or no zero-std telemetry.
- Do not call generated transcripts "terminal RL data" unless they include executable state and a verifier.

## Verification Checklist

- Unit tests:
  - `tests/environments/test_contracts.py`
  - `tests/environments/test_tmax_importer.py`
  - `tests/factory/test_terminal_env_generation.py`
  - `tests/gym/test_terminal_rl_profile.py`
  - `tests/gym/test_active_sampling.py`
  - `tests/eval/test_environment_passk.py`
- Local smoke:
  - generate 10 environments
  - build all 10
  - run verifier against empty baseline
  - run one model rollout attempt
  - compute pass@1
- Release candidate:
  - held-out trace gate passes
  - environment pass@k improves
  - forgetting suite does not regress
  - Terminal-Bench Lite improves or holds
  - no contamination or spurious-reward improvement

## Sources Read

- Nathan Lambert, "TMax: An open RL recipe for terminal agents" (2026-06-22): https://natolambert.substack.com/p/tmax-an-open-rl-recipe-for-terminal
- Ivison et al., "TMax: A Simple Recipe for Terminal Agents" PDF and WAI blog: https://github.com/hamishivi/tmax/blob/master/assets/paper.pdf and https://wai-org.com/blog/tmax/
- TMax code and artifacts: https://github.com/hamishivi/tmax and https://huggingface.co/collections/allenai/tmax
- Terminal-Bench: https://www.tbench.ai/
- Qi et al., "Rethinking the Trust Region in LLM Reinforcement Learning" / DPPO: https://arxiv.org/abs/2602.04879
- Team Olmo, "Olmo 3": https://arxiv.org/abs/2512.13961

## Implementation Research Update - 2026-06-22

Additional current sources sharpen the Phase 0/1 implementation:

- Terminal-World (May 2026): uses agent skills as the central synthesis primitive and co-derives task instructions, environments, and teacher trajectories. BashGym should treat skills as first-class environment axes, not only trace labels. Source: https://arxiv.org/abs/2605.20876
- LiteCoder-Terminal (May 2026): argues for zero-dependency executable and verifiable environment generation from domain specifications, including LiteCoder-Terminal-SFT and LiteCoder-Terminal-RL. BashGym's environment builder should support local, self-contained bundles before remote/Harbor execution. Source: https://arxiv.org/abs/2605.29559
- Prime Intellect Verifiers/Environments Hub: frames RL environments as reusable artifacts for training, evaluation, synthetic data, and harness experimentation. BashGym's `EnvironmentSpec` should therefore support all four uses, not only RL rollouts. Sources: https://github.com/PrimeIntellect-ai/verifiers and https://docs.primeintellect.ai/tutorials-environments/environments
- TerminalWorld benchmark (May 2026): reverse-engineers tasks from real terminal recordings, producing 1,530 validated tasks and a 200-task verified subset. BashGym should keep an ingestion path for real terminal recordings alongside synthetic generation. Source: https://arxiv.org/html/2605.22535v1
- TermiGen and Endless Terminals reinforce the same direction: scalable environment synthesis and verifier-backed terminal tasks are now the bottleneck, not prompt-only data. Sources: https://arxiv.org/html/2602.07274 and https://arxiv.org/html/2601.16443v1
- ECHO (May 2026): adds an environment-observation prediction loss to terminal-agent RL, turning stdout/errors/files/logs from rollouts into dense supervision without extra rollouts and roughly doubling GRPO pass@1 on TerminalBench-2.0 in their report. BashGym rollout/eval records should therefore keep action-token and observation-token telemetry rather than only final verifier rewards. Source: https://arxiv.org/abs/2605.24517
- Endless Terminals v3 (January 2026): emphasizes persistent Apptainer-backed shell sessions that preserve filesystem, environment, and process state across turns. BashGym's first rollout harness should therefore keep cwd/env/process accounting stable before adding Docker/Harbor pools. Source: https://arxiv.org/html/2601.16443v3
- Terminal-Bench 2.1 (2026): fixes benchmark task issues and moves toward continuous validation. BashGym release gates should pin exact benchmark versions and keep benchmark-ingest paths separate from training environment generation. Source: https://snorkel.ai/leaderboard/terminal-bench-2-1/
- ProRL Agent (March 2026): frames multi-turn agent RL infrastructure as "rollout-as-a-service", decoupled from the trainer and backed by standardized sandbox environments. BashGym's rollout path should therefore be callable as an API/service first, then reused by GRPO/DPPO trainers instead of hiding rollout collection inside training scripts. Source: https://arxiv.org/html/2603.18815v1
- TACO (April/May 2026): compresses terminal observations by learning workflow-adaptive filtering rules, preserving task-relevant signals while reducing low-value output. BashGym should keep raw observations for audit/ECHO-style supervision while adding prompt-side observation truncation/compression knobs for served-model rollouts. Source: https://arxiv.org/html/2604.19572v2
- Recent reward-hacking work, adversarial benchmark-hardening work, and Terminal-Bench integrity updates show terminal agents will exploit verifier surfaces when those surfaces are visible or mutable. BashGym should treat verifier scripts, hidden tests, private fixtures, and task manifests as protected artifacts with explicit tamper telemetry before any rollout is scored. Sources: https://arxiv.org/abs/2605.02964, https://arxiv.org/html/2604.17596v1, https://arxiv.org/html/2606.08960, https://metr.org/blog/2025-06-05-recent-reward-hacking/, https://www.tbench.ai/news/leaderboard-integrity-update

## Implementation Research Update - 2026-06-23

Additional current benchmark work sharpens Phase 5 release-gate design:

- Terminal-Bench Challenges (June 2026): benchmark owners are adding challenge-style task releases and official submission workflows, which makes train/eval separation and exact manifest tracking more important than ad hoc local scores. BashGym should freeze grouped environment holdouts and record their manifests before any external benchmark claim. Source: https://www.tbench.ai/news/terminal-bench-challenges
- What Makes a Good Terminal-Agent Benchmark Task (April 2026): task quality depends on realistic terminal work, reliable executable scoring, and resistance to shortcut solutions. BashGym's environment gate should therefore score only held-out task groups and carry verifier/tamper status beside pass@k, not just aggregate success. Source: https://arxiv.org/pdf/2604.28093
- Terminal-Bench paper (January 2026): the benchmark treats terminal tasks as containerized, executable agent evaluations with official harness semantics. BashGym should keep its local persistent-shell eval as a development harness, then map release candidates onto official Terminal-Bench harnesses without mixing those benchmark tasks into training data. Source: https://arxiv.org/html/2601.11868v1
- Olmo 3 technical report (December 2025): reinforces explicit decontamination, held-out evaluation manifests, and negative controls for post-training claims. BashGym should keep content hashes for environment train/holdout splits and block releases when identical executable task payloads appear on both sides. Source: https://www.datocms-assets.com/64837/1765558567-olmo_3_technical_report-4.pdf
- Spurious Rewards v2 (February 2026): random, format-only, and incorrect rewards can produce benchmark gains on some model families when the benchmark/model pipeline is contaminated or high-prior behaviors are amplified. BashGym should require a negative-control audit beside the environment holdout gate: a random/spurious reward control must not clear the same release threshold or erase the observed lift. Source: https://arxiv.org/abs/2506.10947
- Reward Hacking Benchmark (ICML 2026): tool-using RL agents show measurable exploit rates across shortcut, metadata, and evaluator-tamper opportunities, while simple environment hardening cuts exploit rates sharply. BashGym should keep tamper canaries and spurious-reward controls as separate pre-release checks because "passes the verifier" and "learned the task" can diverge. Source: https://arxiv.org/abs/2605.02964
- Efficient Benchmarking of AI Agents (March 2026): shows random benchmark subsets can have high variance across seeds and that reliable agent rankings need task-selection and uncertainty discipline, not only raw aggregate scores. BashGym's environment holdout gate should therefore compare base and candidate attempts with paired deltas and clustered bootstrap intervals before making a release claim. Source: https://arxiv.org/abs/2603.23749
- Berkeley RDI trustworthy-benchmark work: public agent benchmarks can be brittle or gamed when tasks, validators, or scoring assumptions leak. BashGym should make "holdout gate passed with no content-hash overlap" a local precondition before running public leaderboard harnesses. Sources: https://rdi.berkeley.edu/blog/trustworthy-benchmarks/, https://moogician.github.io/blog/2026/trustworthy-benchmarks-cont/
- Harbor (latest GitHub release v0.15.0 on June 19, 2026): the Terminal-Bench creators now describe Harbor as the official harness for `terminal-bench@2.0`, with `harbor run --dataset terminal-bench@2.0 --agent ... --model ...` as the current command path. BashGym should surface Harbor commands next to legacy `tb run` commands and treat Harbor rollout token/reward metadata as a future DPPO/ECHO bridge. Sources: https://github.com/harbor-framework/harbor, https://www.harborframework.com/docs/training-workflows/rl, https://www.tbench.ai/

## Implementation Status - 2026-06-22

Completed Phase 0/1 foundation:

- Added `bashgym/environments/` with `EnvironmentSpec`, axis/fixture/verifier/build/rollout contracts, TMax/Harbor-style normalization, diversity metrics, benchmark decontamination, and local materialization helpers.
- Registered a `terminal_env_generation` Data Designer pipeline skeleton so BashGym can generate environment recipes through the same Factory surface as trace/data synthesis.
- Added `/api/environments/*` routes for pipeline metadata, normalize/import, decontamination, and materialization.
- Added Factory -> Environment Lab UI for importing JSON/JSONL, inspecting environment mix balance, filtering benchmark overlap, and writing selected bundles.
- Added environment pass@k eval plumbing: `bashgym/eval/environment_passk.py`, `POST /api/eval/environments/passk`, optional model-registry recording, and an Environment Lab pass@k panel that accepts rollout-attempt JSON/JSONL. Reports include pass@k, success rate, timeout rate, verifier statuses, and action/observation token telemetry for ECHO-style follow-up work.
- Added local persistent-shell rollouts: `bashgym/environments/rollout.py`, service plumbing, `POST /api/eval/environments/local-rollout-passk`, and Environment Lab controls for running command-script attempts against materialized environments. This proves the environment -> command execution -> verifier -> pass@k loop locally before wiring served-model policies.
- Added served-model policy rollouts: prompt/parse helpers for one-command-at-a-time terminal agents, OpenAI-compatible endpoint resolution, `POST /api/eval/environments/model-rollout-passk`, service fan-out over attempts per environment, and Environment Lab controls for endpoint/model/attempt settings. This moves BashGym toward a ProRL-style rollout service that trainers can call later.
- Advanced Phase 3 DPPO plumbing: sampled served-model rollouts can be exported as DPPO replay JSONL artifacts, external train-policy scorers can enrich those artifacts with train logprobs, and BashGym now computes DPPO Binary-TV/Binary-KL mask telemetry over the enriched batch through both API and Environment Lab UI.
- Added backend-specific DPPO smoke-launch planning: BashGym probes/selects `verl`, SkyRL, or TMax/open-instruct, builds a tiny smoke command from the scored replay artifact, writes `launch_dppo_smoke.sh`, and exposes the plan in Environment Lab. A real GPU smoke run is still pending on an installed backend stack.
- Added Phase 4 reward-hacking guardrails: materialized environments now write a protected-file manifest for `env.json`, verifier scripts, test directories, and private/protected fixtures; local and served-model rollouts audit the workspace before verification; tampered attempts get `verifier_status="tampered"` plus structured audit metadata; Environment Lab surfaces a guard tile with changed paths.
- Exposed TACO/ECHO-aligned observation budgeting for served rollouts: `max_observation_chars` now flows through API/service/UI, is recorded on attempt metadata, caps only the prompt-side observation context, and keeps full raw stdout/stderr in the rollout artifact for audit and future dense-supervision work.
- Added built-in reward-hacking canaries inspired by RHB/Terminal Wrench/hacker-fixer research: verifier tamper, hidden-test tamper, private-fixture tamper, and task-manifest tamper attempts now run through the real local rollout harness; `/api/eval/environments/reward-hacking-canaries` and the Environment Lab Guardrail Canaries panel summarize whether each exploit is caught before scoring.
- Added the Phase 5 environment holdout gate: `bashgym/eval/environment_holdout.py` builds deterministic grouped train/holdout splits, hashes executable task content to detect copied payloads across the split, evaluates holdout-only pass@k, and blocks release on low pass@1, excessive timeout/tamper rate, or contamination. `/api/eval/environments/holdout-gate` and the Environment Lab Holdout Gate panel expose the same split manifest, leakage, and verdict.
- Closed the first registry gap in Phase 5: model profiles now persist `environment_holdout_evals`, `/api/eval/verdict/{model_id}` exposes the latest environment holdout verdict, and `/api/eval/environments/holdout-gate` can record the split manifest/verdict plus holdout pass@k benchmark scores against a registry model id. Environment Lab now has registry model/record controls for the Holdout Gate panel.
- Added Harbor-native external benchmark command generation: `benchmark-commands` now includes `harbor_terminal_bench` by default alongside legacy `terminal_bench`, BFCL, SWE-bench, and forgetting commands.
- Added the Phase 5 spurious-reward negative-control audit: `bashgym/eval/environment_spurious_reward.py` reuses the deterministic environment holdout split, evaluates observed holdout pass@k, then compares it against either provided spurious-control attempts or deterministic random-label trials. `/api/eval/environments/spurious-reward-control` and the Environment Lab Spurious Reward Control panel expose the audit, including control pass@k summaries, observed-control lift, and ship/hold reasons.
- Added the Phase 5 paired comparison gate: `bashgym/eval/environment_holdout_comparison.py` compares base and candidate rollout attempts on the same deterministic environment holdout, computes per-environment pass@k deltas, runs clustered paired bootstrap by task family/domain/source/repo/seed, and blocks release when the CI does not clear zero or candidate operational rates exceed thresholds. `/api/eval/environments/holdout-comparison` and the Environment Lab Holdout Comparison Gate expose the result.
- Added the unified Phase 5 release verdict combiner: `bashgym/eval/release_gate.py` folds precomputed environment evidence into `/api/eval/heldout` results, preserving trace/forgetting behavior when no evidence is supplied while blocking registry-recorded release verdicts on failed environment holdout, paired comparison, or spurious-reward gates. This maps the Terminal-Bench/PostTrainBench contamination concern and Spurious Rewards negative-control lesson into the same ship/hold record used by the main held-out trace gate.
- Added the Evaluator release-evidence UI: `frontend/src/components/evaluator/HeldoutGatePanel.tsx` now lets the held-out trace gate attach environment pass@k, holdout, holdout-comparison, and spurious-reward JSON evidence to `/api/eval/heldout`; combined reports render trace/environment pass-or-hold status in the verdict panel. `frontend/src/services/api.ts` now types the request evidence, combined release-gate report, and environment holdout verdict fields.
- Added standalone external benchmark result ingestion: `bashgym/eval/benchmarks_ext.py` now normalizes aggregate summaries, named result maps/lists, and Harbor-style trial reward JSON into `BenchmarkReport`; `POST /api/eval/benchmarks/external-ingest` records Harbor/Terminal-Bench, BFCL, SWE-bench, or other public harness scores into the model registry. The Evaluator Held-out Gate panel now lets users paste official harness output after running the generated commands and record normalized scores against the selected model.
- Added external benchmark release evidence: `bashgym/eval/release_gate.py` now treats normalized external benchmark reports as a distinct release-gate lane with optional minimum-score thresholds, harness failure blocking, and manifest preservation. `/api/eval/heldout` accepts this evidence under `environment_evidence.external_benchmarks`, and the Evaluator release-evidence UI includes an `External benchmarks` JSON field plus Trace/Environment/External pass-or-hold status.
- Added first-class BFCL/SWE-bench result adapters: `parse_bfcl_results` understands official BFCL score JSON/CSV-style rows, V4 weighted category scores, and per-category drilldown metrics; `parse_swebench_results` understands SWE-bench `results.json`, `sb-cli get-report`-style summaries, and `instance_results` rows with per-repository resolution rates. The Evaluator external-ingest UI now previews those drilldown metrics after recording a public harness result.
- Advanced S7 flywheel automation: MOPD is confirmed unstubbed in `/api/cascade/distill`; `bashgym/api/pipeline_routes.py` now wires pipeline gold-threshold triggers into a CascadeStartRequest-compatible payload and queues the existing `CascadeScheduler` on the app event loop; `PipelineConfig` and the Electron Pipeline panel now expose cascade base model, mode, stage steps, min examples, remote SSH, and repo-domain trigger settings. This follows the Nemotron-Cascade/MOPD pattern of routing domain-specialized teachers by data source and the Hermes/NemoClaw pattern of persistent trace/runtime state feeding the next loop.
- Added Phase 6/S7 environment recipe AutoResearch: `EnvironmentRecipeSearchSpace` now mutates sample size, axis weights, verifier-kind mix, fixture-kind mix, target pass@1, and deterministic seeds over imported `EnvironmentSpec` pools; `/api/autoresearch/environment-recipe/propose` runs a bounded AutoResearch loop and optionally writes a reproducible proposal JSON; the AutoResearch dashboard now includes an Environment Recipe Proposals panel for source path, export path, search budget, selected IDs, balance metrics, and mean pass@1.

Verification:

- `python -m pytest tests/api/test_environment_routes.py tests/environments tests/factory/test_data_designer.py -q -o addopts=` -> 106 passed.
- `python -m ruff check bashgym/environments bashgym/api/environment_routes.py bashgym/factory/designer_pipelines/terminal_env_generation.py tests/environments tests/api/test_environment_routes.py` -> clean.
- `python -m pytest tests/eval/test_environment_passk.py tests/eval/test_passk.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/environments -q -o addopts=` -> 50 passed.
- `python -m pytest tests/environments/test_rollout.py tests/environments/test_builder.py tests/eval/test_environment_passk.py tests/api/test_eval_routes.py -q -o addopts=` -> 28 passed.
- `python -m pytest tests/environments/test_rollout.py tests/eval/test_service.py tests/api/test_eval_routes.py -q -o addopts=` -> 44 passed.
- `python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_passk.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/factory/test_data_designer.py -q -o addopts=` -> 142 passed.
- `python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_passk.py tests/eval/test_service.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/factory/test_data_designer.py -q -o addopts=` -> 166 passed.
- `python -m pytest tests/eval/test_dppo_replay.py tests/api/test_eval_routes.py -q -o addopts=` -> 24 passed.
- `python -m pytest tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_eval_routes.py -q -o addopts=` -> 31 passed.
- `python -m pytest tests/environments/test_builder.py tests/environments/test_rollout.py -q -o addopts=` -> 17 passed.
- `python -m pytest tests/eval/test_service.py tests/api/test_eval_routes.py -q -o addopts=` -> 41 passed.
- `python -m pytest tests/environments/test_canaries.py tests/environments/test_builder.py tests/environments/test_rollout.py tests/api/test_eval_routes.py -q -o addopts=` -> 43 passed.
- `python -m pytest tests/eval/test_environment_holdout.py tests/eval/test_environment_passk.py tests/api/test_eval_routes.py -q -o addopts=` -> 34 passed.
- `python -m ruff check bashgym/eval/environment_holdout.py bashgym/eval/service.py bashgym/api/eval_routes.py tests/eval/test_environment_holdout.py tests/api/test_eval_routes.py` -> clean.
- `python -m ruff check bashgym/eval/service.py tests/eval/test_service.py tests/api/test_eval_routes.py` -> clean.
- `python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` -> 157 passed.
- `python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/eval/test_heldout_registry.py tests/eval/test_benchmarks_ext.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` -> 177 passed.
- `python -m pytest tests/eval/test_environment_spurious_reward.py tests/api/test_eval_routes.py -q -o addopts=` -> 32 passed.
- `python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_environment_spurious_reward.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/eval/test_heldout_registry.py tests/eval/test_benchmarks_ext.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` -> 184 passed.
- `python -m pytest tests/eval/test_environment_holdout_comparison.py tests/api/test_eval_routes.py -q -o addopts=` -> 33 passed.
- `python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_environment_holdout_comparison.py tests/eval/test_environment_spurious_reward.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/eval/test_heldout_registry.py tests/eval/test_benchmarks_ext.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` -> 190 passed.
- `python -m pytest tests/eval/test_release_gate.py tests/api/test_eval_routes.py -q -o addopts=` -> 35 passed.
- `python -m pytest tests/environments tests/eval/test_release_gate.py tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_environment_holdout_comparison.py tests/eval/test_environment_spurious_reward.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/eval/test_heldout_registry.py tests/eval/test_benchmarks_ext.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` -> 196 passed.
- `python -m ruff check bashgym/eval bashgym/api/eval_routes.py tests/eval/test_release_gate.py tests/api/test_eval_routes.py` -> clean.
- `npm run typecheck` and `npm run lint` in `frontend/` -> clean after the release-evidence UI.
- `python -m pytest tests/eval/test_benchmarks_ext.py tests/eval/test_service.py tests/api/test_eval_routes.py -q -o addopts=` -> 69 passed.
- `python -m pytest tests/environments tests/eval/test_release_gate.py tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_environment_holdout_comparison.py tests/eval/test_environment_spurious_reward.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/eval/test_heldout_registry.py tests/eval/test_benchmarks_ext.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` -> 201 passed.
- `python -m pytest tests/eval/test_release_gate.py tests/api/test_eval_routes.py -q -o addopts=` -> 40 passed after adding external benchmark release evidence.
- `python -m pytest tests/environments tests/eval/test_release_gate.py tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_environment_holdout_comparison.py tests/eval/test_environment_spurious_reward.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/eval/test_heldout_registry.py tests/eval/test_benchmarks_ext.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` -> 205 passed after both S6 slices.
- `python -m pytest tests/eval/test_benchmarks_ext.py tests/eval/test_service.py tests/api/test_eval_routes.py -q -o addopts=` -> 74 passed after adding first-class BFCL/SWE-bench adapters.
- `python -m pytest tests/environments tests/eval/test_environment_passk.py tests/eval/test_environment_holdout.py tests/eval/test_environment_spurious_reward.py tests/eval/test_environment_holdout_comparison.py tests/eval/test_passk.py tests/eval/test_service.py tests/eval/test_dppo_replay.py tests/eval/test_heldout_registry.py tests/eval/test_benchmarks_ext.py tests/eval/test_release_gate.py tests/api/test_eval_routes.py tests/api/test_environment_routes.py tests/gym/test_terminal_rl_profile.py tests/gym/test_grpo_script.py tests/gym/test_dppo.py tests/gym/test_dppo_backend.py tests/gym/test_dppo_launcher.py tests/api/test_training_schema.py -q -o addopts=` -> 209 passed after the adapter drilldown slice.
- `python -m pytest tests/pipeline/test_cascade_trigger.py tests/pipeline/test_config.py tests/pipeline/test_threshold_monitor.py tests/pipeline/test_orchestrator.py tests/api/test_cascade_routes.py tests/trace_capture/test_hermes_importer.py -q -o addopts=` -> 39 passed after wiring the pipeline-to-cascade trigger.
- `python -m pytest tests/pipeline tests/api/test_cascade_routes.py tests/api/test_autoresearch_routes.py tests/trace_capture/test_hermes_importer.py tests/gym/test_cascade_scheduler.py tests/gym/test_cascade_distillation.py tests/gym/test_autoresearch.py -q -o addopts=` -> 160 passed after the S7 trigger bridge.
- `python -m pytest tests/gym/test_environment_recipe_search_space.py tests/api/test_autoresearch_routes.py -q -o addopts=` -> 20 passed after adding environment recipe AutoResearch.
- `python -m pytest tests/gym/test_environment_recipe_search_space.py tests/gym/test_autoresearch.py tests/api/test_autoresearch_routes.py tests/api/test_environment_routes.py tests/environments tests/pipeline/test_cascade_trigger.py tests/pipeline/test_config.py tests/pipeline/test_threshold_monitor.py tests/pipeline/test_orchestrator.py tests/trace_capture/test_hermes_importer.py -q -o addopts=` -> 126 passed after the recipe proposal slice.
- Live backend smoke: `POST /api/autoresearch/environment-recipe/propose` against `tests/fixtures/tmax_envs/tmax_demo_001.jsonl` returned `bashgym.environment_recipe_proposal.v1` and wrote a proposal JSON under `artifacts/`.
- `python -m ruff check bashgym/eval/benchmarks_ext.py bashgym/eval/service.py bashgym/api/eval_routes.py tests/eval/test_benchmarks_ext.py tests/eval/test_service.py tests/api/test_eval_routes.py` -> clean.
- `python -m ruff check bashgym/eval/release_gate.py bashgym/api/eval_routes.py tests/eval/test_release_gate.py tests/api/test_eval_routes.py` -> clean.
- `python -m ruff check bashgym/eval/benchmarks_ext.py bashgym/eval/__init__.py tests/eval/test_benchmarks_ext.py` -> clean after the adapter drilldown slice.
- `python -m ruff check bashgym/pipeline/config.py bashgym/pipeline/orchestrator.py bashgym/api/pipeline_routes.py bashgym/api/routes.py tests/pipeline/test_cascade_trigger.py` -> clean after the S7 trigger bridge.
- `python -m ruff check bashgym/gym/environment_recipe_search_space.py bashgym/gym/__init__.py bashgym/api/autoresearch_routes.py bashgym/api/schemas.py tests/gym/test_environment_recipe_search_space.py tests/api/test_autoresearch_routes.py` -> clean after the recipe proposal slice.
- Live backend smoke: `POST /api/eval/benchmarks/external-ingest` returned a normalized `harbor_terminal_bench` report from Harbor-style trial reward JSON.
- Live backend smoke: `POST /api/eval/benchmarks/external-ingest` returned a normalized `bfcl_v4` report with 63.0% weighted score and category metrics from BFCL V4-style category JSON.
- In-process API smoke: isolated `PUT /api/pipeline/config` accepted cascade trigger settings and `POST /api/pipeline/trigger/cascade` returned `status=queued` with the temp gold-trace directory in the request payload.
- `git diff --check` -> clean.
- `npm run typecheck` and `npm run lint` in `frontend/` -> clean.
- Chrome/Playwright smoke against `http://127.0.0.1:5175` with backend on `http://127.0.0.1:8003`: Environment Lab rendered and the Holdout Gate panel appeared with split/threshold controls. Screenshot saved to ignored `artifacts/environment-holdout-gate-smoke.png`.
- Chrome/Playwright smoke after registry controls: Environment Lab rendered the Holdout Gate, `REGISTRY MODEL`, and `Record gate` controls with no console errors. Screenshot saved to ignored `artifacts/environment-holdout-registry-controls-smoke.png`.
- Chrome/Playwright smoke after spurious-control UI: Environment Lab rendered the Spurious Reward Control panel with no console errors. Screenshot saved to ignored `artifacts/environment-spurious-reward-control-smoke.png`; a stubbed action smoke imported one environment, seeded baseline attempts, clicked Audit, and rendered "Spurious control clear" with no console errors (`artifacts/environment-spurious-reward-control-action-smoke.png`). A live backend call to `/api/eval/environments/spurious-reward-control` returned `bashgym.environment_spurious_reward_control.v1`.
- Chrome/Playwright smoke after holdout-comparison UI: stubbed Environment Lab action flow imported one environment, filled candidate/base attempt JSON, clicked Compare, and rendered "Holdout comparison clear" with no console errors (`artifacts/environment-holdout-comparison-action-smoke.png`). A live backend call to `/api/eval/environments/holdout-comparison` returned `bashgym.environment_holdout_comparison.v1`.
- Chrome/Playwright smoke after Evaluator release-evidence UI: opened Evaluator -> Held-out Gate, enabled `Release evidence`, pasted holdout-gate JSON, rendered "1 JSON object attached", and saw no console errors. Screenshot saved to ignored `artifacts/evaluator-release-evidence-smoke.png`.
- Chrome/Playwright smoke after external benchmark ingest UI: opened Evaluator -> Held-out Gate, rendered `External benchmark suite`, posted sample Harbor-style trial JSON through `Record result`, and saw `harbor_terminal_bench: 66.7% (2/3)` with no console errors. Screenshot saved to ignored `artifacts/evaluator-external-benchmark-ingest-action-smoke.png`.
- Chrome/Playwright smoke after external benchmark release evidence UI: opened Evaluator -> Held-out Gate, enabled `Release evidence`, and rendered the `External benchmarks` JSON field with no console errors. Screenshot saved to ignored `artifacts/evaluator-external-benchmark-release-evidence-smoke.png`.
- Chrome/Playwright smoke after BFCL/SWE adapter UI: opened Evaluator -> Held-out Gate, posted BFCL V4-style category JSON through `Record result`, and saw `bfcl_v4: 63.0%` plus `category.agentic`, `category.multi_turn`, and `bfcl_v4_weighted_score` drilldown with no console errors. Screenshot saved to ignored `artifacts/evaluator-external-adapter-smoke.png`.
- Chrome/Playwright smoke after environment recipe UI: opened AutoResearch, filled the Environment Recipe Proposals form with the TMax fixture path, submitted a mocked successful proposal response, and rendered metric, selected ID, export path, domain balance, and mean pass@1 with no console errors. Screenshot saved to ignored `artifacts/autoresearch-environment-recipe-proposal-smoke.png`.
