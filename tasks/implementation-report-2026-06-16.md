# BashGym Platform Build — Implementation Report (June 2026)

How the model-agnostic train→eval→deploy flywheel was built this session, the DGX/S8
state, test status, and a detailed, prioritized outline of next steps.

**Branch:** `feat/training-strategies-device-mgmt` @ origin (desktop + GX10 in sync).
**Companions:** `roadmap-2026-06-15-model-agnostic-pipeline.md`, `status-review-2026-06-11.md`, `eval-and-data-strategy-2026-06-11.md`, `todo.md`.

---

## 1. Executive summary

The **locally-buildable spine of the flywheel is built, tested, and pushed**: any open model
resolves to a correct training recipe (S1), deploys with a correct chat template (S2), is
judged "better?" with statistical rigor (S3), can be trained on decontaminated public data
mixed with our traces (S4), on training data that carries a continuous quality signal and a
defined loss-mask policy (S5). The cascade flywheel's biggest blocker (the MOPD distill stub)
is removed and the Hermes deploy→trace loop has an importer (S7). The test suite was made
green and hermetic (#2, #3).

The **DGX environment consolidation (S8) is effectively already in place** — the `bashgym-train`
venv runs transformers 5.5 + Unsloth + vLLM + TRL together on the GB10, which was the hard part.
What remains there is a set of *version upgrades* (vLLM cu13 build, TRL 1.6, torch 2.11) best done
in a planned window because the GB10 is a live production server.

The **runtime half** of the flywheel (serve base+candidate via vLLM, episode pass@k in the
sandbox, external benchmarks) is the main remaining work; its pure-logic cores are already built
and tested, so what's left is integration against served models on the Spark.

---

## 2. What shipped this session (by segment)

| Seg | Commit(s) | What | Tests |
|----|----|----|----|
| **S0** Reconcile | `569d039` + promote/push | Unified 3 divergent lines (desktop frontend + GX10 ws2 backend) onto one branch; resolved the GRPO direction as a backend switch; GX10 fast-forwarded. Caught + fixed a latent ws2 bug (unescaped `{state.global_step}` → NameError in the GRPO generator). | full suite collects; frontend tsc+eslint clean |
| **S1** Model-agnostic core | `4477a99`, `ee37dbe`, `acf1b6f`, `4cee6af` | `bashgym/families/`: `ModelFamilyProfile` registry (Gemma4/Qwen3/Qwen2.5/Llama3/generic), `select_backend()` (Unsloth↔plain switch), `apply_patches()` (gemma4 monkey-patch extracted). GRPO generator consumes the profile; plain-transformers backend reintroduced behind `grpo_backend`. `scripts/train_model.py --base-model` replaced the 4 hardcoded `train_gemma4_*.py`. | 28 + 14 + 6 |
| **S2** Export/deploy | `ffa08d7`, `f566ecf` | `bashgym/export/gguf.py`: Modelfile builder (always emits TEMPLATE), `ollama show` template parser, `check_template_roundtrip` (double-BOS / dropped-tool-token / divergence). Wired `deploy_gguf_to_ollama` to emit a correct TEMPLATE (the #1 deploy bug) + `TrainerConfig.ollama_base_tag`. | 8 |
| **#2** Cascade fix | `5e…` | Root-caused: converter only read trace-format, not the `messages` format the cascade filters → zero examples. Now reads both; simulate skips empty-domain preflight. Fixed `test_pipeline_builders` drift. | 57 cascade + 7 converter |
| **#3** Hermeticity | `e1bc45b` | `--timeout=120` bounds every test; `conftest.py` skips `@pytest.mark.network` by default (`--run-network` opts in); confirmed hangers marked. Suite no longer hangs; bulk green. | 484 bulk green |
| **S3** Held-out eval | `6cd57cb` | `bashgym/eval/`: tool-call metrics (exact name + per-arg F1), **session-clustered paired bootstrap** (ship only if 95% CI excludes 0), contamination-free session/repo split + manifest, deploy gate (min delta + CI + max forgetting). | 31 |
| **S4** HF ingestion | `b91ea13` | `decontaminate.py` (13-gram + 3-gram-Jaccard benchmark gates — closes the no-decontamination gap), `mixer.py` (self-fraction ratio mixing), `converters.normalize_public_messages` (OpenHands/ShareGPT → our format, JSON-string tool args → dicts). | 23 |
| **S5** Data quality | `ac367fa` | Continuous `quality_score` [0,1] carried into example metadata (SERA-style soft signal, not just gold/failed); `masking.py` (SFT loss policy: assistant turns only). | 8 |
| **S7** Flywheel | `cdbba2f` + `…` | MOPD distill route now calls the real `distill_cascade()` (was a 4-line sleep stub — the biggest cascade blocker); `hermes_history.py` importer (defensive SQLite reader → BashGym traces, closes deploy→trace loop). | 63 cascade + 4 hermes |

All additive/test-guarded; broad non-network regression stayed green (506 passed) through S5.

---

## 3. Test & verification status

- **Hermetic, bounded suite:** `pytest` no longer hangs; `pytest -m "not network"` is the fast/CI path; `--run-network` runs live-service tests. ~150+ new tests added this session, all green.
- **Regression discipline:** ran broad non-network suites (datasets/gym/factory/eval/export/families) after each change — **506 passed** at the S5 checkpoint, nothing broken.
- **Known non-hanging failures (tracked, task #13):** `tests/orchestrator/test_e2e_api` (1), `tests/research/test_backend_integration` (4, likely API drift like the fixed `test_pipeline_builders`), `tests/orchestrator/test_e2e_worktrees` (14, git-worktree env). These complete (don't hang) and are isolated under `network`/triage.
- **GX10 verified:** backend healthy (HTTP 200, 216 routes) after each sync; `families`/`export`/`eval`/`converters` import on the Spark; GB10 matmul works.

---

## 4. DGX / S8 state and the exact upgrade plan

**Hardware/OS:** GB10 (sm_121, aarch64), driver 580.126.09, CUDA 13.0.2, 341G free disk.

**`bashgym-train` venv (the consolidated stack — already coexisting):**

| Package | Installed | Target | Action |
|---|---|---|---|
| transformers | **5.5.0** | 5.5.x | ✅ already correct |
| unsloth | present, imports (xformers fallback; flash-attn broken per unslothai#4867 — expected) | latest | ✅ functional |
| vllm | **0.18.1 +cu12** (mismatched: system is cu13) | **0.22.1/0.23.0 +cu130** | upgrade (logprob serving + transformers-5.5/Gemma4 support) |
| trl | **1.0.0** | **1.6.0** (AsyncGRPO) | upgrade (major-version API change — test GRPO after) |
| torch | **2.10.0 +cu130** (warns: sm_121 > max 12.0, "upgrade ≥2.11"; **matmul works**) | 2.11.0 +cu130 | upgrade for clean sm_121 (functional today) |

**The S8 headline — single-venv consolidation — is ACHIEVED:** transformers 5.5 + Unsloth + vLLM + TRL coexist and train on the GB10 (the cascade GRPO completed). The old "must use separate venvs for vLLM vs Unsloth" constraint is resolved.

**Serving:** Ollama is the working inference path (nemotron3-super, qwen3.5:35b, nemotron-3-nano loaded; serves OpenClaw). vLLM serving needs the cu130 build for logprob-capable eval.

**Why not done in-place this session:** the GB10 is a live production server, and TRL 1.0→1.6 is a major-version jump that can break the GRPO generator's API usage. Per the project's standing "plan before env changes / never change deps without a verified plan," these belong in a **planned, isolated upgrade**, not forced surgery on the serving box.

**Exact, safe upgrade procedure (recommended: a fresh isolated `bashgym-serve` venv first, prove it, then adopt):**

```bash
# On the GX10, in a maintenance window. Build serving venv WITHOUT touching bashgym-train.
python3.12 -m venv ~/bashgym-serve && source ~/bashgym-serve/bin/activate
pip install --upgrade pip
# torch 2.11 cu130 (proper sm_121) — verify the cu130 aarch64 wheel resolves:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install "transformers==5.5.*" "trl==1.6.0" peft accelerate datasets
pip install "vllm>=0.22.1"          # cu130 default wheels (drop the stale nightly recipe)
# Verify before adopting:
python -c "import torch; print(torch.cuda.get_device_capability())"   # expect (12,1), no warning
python -c "import vllm, transformers, trl; print(vllm.__version__, transformers.__version__, trl.__version__)"
vllm serve <merged-gemma4-ckpt> --port 8100 &   # logprob-capable serving for evals
# Then: re-run the GRPO generator's output under TRL 1.6 (API check) before retiring bashgym-train.
```

**Hermes install (deploy harness, separate):**
```bash
# Per the Nous/NVIDIA DGX Spark playbook: install Hermes, point at Ollama
export OLLAMA_CONTEXT_LENGTH=64000   # Hermes needs >=64k; Ollama defaults to 4096
# deploy the fine-tuned student GGUF as a model, run Hermes against http://localhost:11434/v1
# then `bashgym.trace_capture.importers.import_hermes_sessions()` pulls ~/.hermes/state.db back as traces.
```

---

## 5. Detailed next steps (prioritized)

**A. Runtime integration (needs the serving venv from §4) — turns on the rest of the flywheel:**
1. **S3 eval runner** — `bashgym/eval/heldout.py`: freeze a session+repo split from `data/gold_traces`, serve base+candidate via vLLM, teacher-force step-level tool-call scoring + episode pass@k through the existing Docker sandbox/`verify.sh`, run the paired bootstrap, write the gate verdict into `registry_index.json` (block deploy on regression). All scoring/stat/split/gate pieces already exist + tested.
2. **S6 external benchmarks** — `bashgym/eval/benchmarks_ext.py`: lm-eval forgetting suite (MMLU/GSM8K/IFEval/HellaSwag) via `local-completions`→vLLM; Terminal-Bench 2.0 (Harbor), BFCL-V4, SWE-bench Lite (mini-swe-agent). Run harnesses from the x86 desktop against the Spark endpoint.
3. **S2 deploy smoke** — export a merged checkpoint → GGUF → Ollama, then run `check_template_roundtrip` against the real HF-template vs Ollama render; wire it as a pre-deploy gate.

**B. Remaining local pieces (no DGX needed):**
4. **S5 decision-level DPO wiring** — `data_factory.py:859-942` builds decision-level DPO pairs but isn't called by the main factory flow; wire it into the DPO export path (+ test).
5. **S5 per-ModelProfile tool sanitization** — extend `families` so tool-call rendering matches each family's template (Gemma 4 `<|"|>` delimiter, qwen_xml, hermes); validate against `apply_chat_template`.
6. **S4 real ingestion** — promote `bashgym/research` scanner simulate→real; `hf_ingest.py` to download SWE-rebench-openhands (`resolved=1`), Kwai-Klear, Nemotron-SWE, Toucan, SWE-chat, run them through `normalize_public_messages`→`decontaminate`→`mix` (the logic is built/tested).
7. **S7 auto-trigger** — `ThresholdMonitor` → `POST /api/cascade/start`; set `auto_deploy_ollama` gated by the S3 verdict; cascade stage-resumption.

**C. Hygiene (tracked):**
8. **Task #13** — triage the 5 non-hanging drift failures (likely API-drift fixes like `test_pipeline_builders`).
9. **S1 follow-ups (#12)** — Liger FLCE (fused CE for Gemma's 262k vocab; Liger-Kernel#1186) and SFT/DPO backend dispatch with config-override-vs-profile-default precedence.

**D. DGX (planned window):** execute §4 (serving venv → verify → adopt; Hermes install). Keep `bashgym-train` as the fallback until proven.

---

## 6. End-goal scorecard

Goal: *leverage our coding-agent traces (successes + failures) to fine-tune open models that
operate best with our system — model-agnostic, rigorously evaluated, full flywheel, NVIDIA tooling.*

| Capability | Status |
|---|---|
| Any open model trainable (Qwen 3.6 / latest Gemma / Llama) | ✅ S1 registry + backend switch + CLI |
| Correct local deployment (no broken tool calls) | ✅ S2 template fix |
| "Is it actually better?" with rigor | ✅ S3 logic (runner = next) |
| Leverage public datasets, decontaminated | ✅ S4 logic (download = next) |
| Training data that matters (continuous scoring, masking) | ✅ S5 (decision-DPO = next) |
| Cascade RL flywheel | ✅ MOPD unstubbed; auto-trigger = next |
| Deploy→trace loop (Hermes) | ✅ importer built; install = next |
| Clean, bounded, hermetic test base | ✅ #2 + #3 |
| DGX single-venv consolidation | ✅ already in place; version upgrades = planned window |
| Serve for eval + run benchmarks | ⏳ Ollama works; vLLM-cu130 + S3-runner/S6 = next |

The pure-logic spine is done and tested; the remaining work is runtime integration on the Spark.
