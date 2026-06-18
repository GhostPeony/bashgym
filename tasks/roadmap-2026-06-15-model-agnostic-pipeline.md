# BashGym → Model-Agnostic Training/Eval/Deploy Platform — Implementation Roadmap

> **For agentic workers:** Each segment below is sized to ~one session and produces working, testable software on its own. Execute with `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Segment 0 has bite-sized steps; Segment 1 has a detailed design; Segments 2–8 are specification cards that each become their own detailed plan at execution time (per `superpowers:writing-plans` scope-check — this spec spans multiple independent subsystems).

**Goal:** Turn BashGym into a platform where dropping in *any* new open model id (Qwen 3.6, latest Gemma, Llama, …) runs the full loop — train/improve on our traces + mixed-in HF public datasets, evaluate with rigor, export with a correct chat template, deploy to Ollama, and gate deployment on a statistically-sound "is it better?" — with the train→export→serve ergonomics of **Unsloth Studio** plus the data/curation/RL/eval **flywheel Unsloth Studio doesn't have.**

**Architecture:** A declarative `ModelProfile` registry is the spine: every model-specific fact (chat template, tool-call format, LoRA targets/exclusions, attention impl, correctness patches, GGUF template, default training backend) lives in data, not scattered `if "gemma" in name` code. Trainer, exporter, and evaluator all consume the profile. Training backends (Unsloth / plain-transformers / TRL+vLLM) sit behind one interface and are auto-selected by profile + platform probe — which permanently resolves the GRPO Unsloth-vs-plain fork as a *switch*, not an either/or. HF public datasets flow through a scanner → converter → decontamination → mix pipeline into the same unified `messages` format as our traces. A held-out, contamination-free eval harness with paired-bootstrap confidence intervals is the deploy gate.

**Tech stack:** Python 3.12, transformers 5.5.x, PEFT, TRL 1.6, Unsloth (where supported), Liger-Kernel (fused CE), vLLM (serving/eval + TRL-GRPO generation), llama.cpp/GGUF, Ollama, FastAPI + React/Electron frontend, DGX Spark (GB10/sm_121/aarch64/CUDA 13).

**Companion docs:** `tasks/status-review-2026-06-11.md` (full repo/ecosystem/gap review), `tasks/eval-and-data-strategy-2026-06-11.md` (eval audit, dataset shortlist, Unsloth gap analysis). All file:line refs and version numbers there.

---

## Target architecture (where every segment lands)

```
bashgym/
  families/            # NEW (S1 ✅ core landed) — renamed from models/ to avoid collision with
    profiles.py        #   bashgym.models.ModelProfile (= metadata for *trained* artifacts)
    backends.py        # ModelFamilyProfile + REGISTRY + resolve_family_profile(); select_backend()
    patches.py         # named correctness patches (gemma4_clippable_linear) via apply_patches()
  gym/
    trainer.py         # MODIFY (S1 PENDING) — script-gen consumes ModelFamilyProfile + selected backend; remove hardcoded gemma/qwen branches
  datasets/
    contracts.py       # EXISTS (ws2) — FormatContract (SFT/DPO/GRPO/DISTILLATION)
    converters.py      # EXISTS (ws2) + EXTEND (S4) — public-schema → unified messages
    validator.py       # EXISTS (ws2)
    hf_ingest.py       # NEW (S4) — download → convert → decontaminate → MinHash dedup → mix
    decontaminate.py   # NEW (S4) — 13-gram/3-gram/cosine gates vs benchmark repos
  research/            # EXISTS (ws2) — HF dataset scanner + DatasetSearchSpace (promote simulate→real in S4)
  export/
    gguf.py            # NEW (S2) — merge → convert_hf_to_gguf → quantize → Modelfile(TEMPLATE from profile) → round-trip test
  eval/
    heldout.py         # NEW (S3) — frozen split, tool-call AST metrics, episode pass@k, paired bootstrap
    gate.py            # NEW (S3) — pre-registered thresholds; registry gating; regression block
    benchmarks_ext.py  # NEW (S6) — lm-eval / Terminal-Bench / BFCL-V4 / SWE-bench adapters (vLLM endpoint)
  judge/               # EXISTS — verifier/semantic_judge/benchmarks/evaluator (wire into eval/ in S3,S6)
  trace_capture/importers/
    hermes_history.py  # NEW (S7) — ~/.hermes/state.db SQLite importer
  pipeline/            # EXISTS — ThresholdMonitor → cascade auto-trigger (wire in S7)
  gym/cascade_scheduler.py  # EXISTS — unstub MOPD route, resumption, auto-deploy gate (S7)
```

---

## Segment dependency graph & recommended order

```
S0 (reconcile) ──┬─▶ S1 (ModelProfile core) ──┬─▶ S2 (export/deploy) ──┐
                 │                             ├─▶ S3 (held-out eval) ──┼─▶ S7 (flywheel automation)
                 ├─▶ S4 (HF dataset ingest) ───┘                        │
                 └─▶ S5 (data quality) ─────────────────────────────────┘
                                                  S8 (DGX env, APPROVAL) ─▶ S6 (external benchmarks)
```

Recommended sequence: **S0 → S1 → (S2 ∥ S3 ∥ S4) → S5 → [S8 approval gate] → S6 → S7.** S2/S3/S4 are independent after S1 and can be parallel subagent sessions. S6 needs vLLM serving from S8 (the only approval-gated segment).

| Seg | Title | Approval? | Est. sessions | Unblocked by |
|----|-------|-----------|---------------|--------------|
| S0 | Reconcile divergent branches & land finished work | No | 1 | — |
| S1 | Model-agnostic training core (ModelProfile + backends + Liger) | No | 1–2 | S0 |
| S2 | Model-agnostic export & deploy (GGUF + Modelfile + round-trip) | No | 1 | S1 |
| S3 | Held-out trace eval + statistical gate + registry | No | 1–2 | S0 (S1 for serving the candidate) |
| S4 | HF public dataset ingestion (scan→convert→decontaminate→mix) | No | 1–2 | S0 |
| S5 | Data pipeline quality upgrades | No | 1 | S0 (S1 for per-profile tool sanitization) |
| S6 | External benchmark evals (lm-eval/TB2/BFCL/SWE-bench) | No (uses S8 serving) | 1 | S3, S8 |
| S7 | Flywheel automation (MOPD, auto-trigger, Hermes importer) | No | 1 | S1, S2, S3 |
| S8 | DGX environment consolidation | **YES** | 1 | S1 |

---

## Segment 0 — Reconcile divergent branches & land finished work

**Goal:** Collapse three divergent lines into one clean branch with a green test + typecheck suite, losing no work and resolving the GRPO direction as a backend switch.

> **Progress (2026-06-15): S0 COMPLETE on `integration/unify-2026-06-15` (`569d039`), not yet pushed.** Steps 1–5: backups (tag `backup/pre-reconcile-2026-06-15`, origin `backup/desktop-frontend-2026-06-15`); commits `69963e6`/`698d78e`/`554f4f9`; test_grpo_script 9/9. Steps 6–7: merged ws2 ↔ desktop — only **5 conflicts** (trainer.py, routes.py, 3 frontend), all resolved (trainer = ws2 Unsloth GRPO + my config params, plain-transformers preserved in `554f4f9` for S1; routes = both endpoint sets; frontend = unions). Caught + fixed a **latent ws2 bug** (unescaped `{state.global_step}` → NameError in the GRPO generator; ws2 had no generator test). Verified: GRPO 9/9, frontend tsc+eslint clean, 1530 collect, 190/194 pure-logic pass. **Known pre-existing (identical to ws2, not merge-caused): 4 cascade preflight tests fail; suite non-hermetic (api/research hang on network/SSH).** Step 8 (promote→`feat` + push + sync GX10) pending user go — affects shared remote.

**Current divergence (verified 2026-06-15):**
- Merge-base: `a2ea717` (Jun 9).
- **Local** `feat/training-strategies-device-mgmt` @ `7d2831c`: 6 commits (Jun 11), all frontend/observability — `8b5019a` repair typecheck/lint toolchains, `4f0ce1f` resolve 129 type errors, `3d4aeff` PTY lifecycle hardening, `59d9211` activity feed + router latency, `87dc773` streamed import progress + run metrics, `7d2831c` RunComparison + DatasetInspector. Plus uncommitted: `trainer.py` (plain-transformers GRPO for sm_121), `designer_pipelines/*` (ChatCompletionInferenceParams migration); untracked `datasets/`, `scripts/`, `train_gemma4_*.py`, `data-fixed/`, `data-pipeline-fixed/`.
- **ws2** `origin/ws2-hf-research`: 34 commits — `bashgym/research/` HF scanner + DatasetSearchSpace, multi-strategy cascade (SFT/DPO/GRPO per stage), SFT/DPO/GRPO subprocess hardening, **Unsloth** GRPO direction (`750e9d2`,`d4b8442`), HFDashboard Buckets/Research/Traces tabs, 11 eval/benchmark scripts.

**Reconciliation strategy:** Backend/training/research = **ws2 canonical** (real new capability). Frontend/typecheck = **local canonical** (it carries the toolchain repair + 129 type-error fixes ws2 lacks). The two frontend feature sets are additive (local: RunComparison/DatasetInspector/ActivityFeed; ws2: HFDashboard tabs) — combine, don't discard. Resolve `trainer.py` by keeping BOTH GRPO paths behind a backend switch (the seed of S1).

**Files:**
- Modify: `.gitignore`, `bashgym/gym/trainer.py`, `tests/gym/test_grpo_script.py`
- Merge surface (resolve, keep both features): `bashgym/api/routes.py`, `bashgym/api/websocket.py`, `bashgym/factory/dataset_inspector.py`, `bashgym/gym/run_metrics.py`, ~40 `frontend/src/**` files, `frontend/package.json`, `tests/{api,factory,gym}/*`
- Commit (finished, identical-to-ws2 or local-only): `bashgym/datasets/`, `scripts/`, `bashgym/factory/designer_pipelines/*`

- [ ] **Step 1: Safety anchors (DONE this session)** — local tag `backup/pre-reconcile-2026-06-15`, origin backup branch `backup/desktop-frontend-2026-06-15`; ws2/ws1 already on origin.

- [ ] **Step 2: gitignore generated data dirs**
  Add to `.gitignore`: `data-fixed/`, `data-pipeline-fixed/`, `data-rebuilt/`, `data-pipeline/`.
  Run: `git status --short` → Expected: those dirs no longer listed as `??`.

- [ ] **Step 3: Ensure lint tools, then lint the finished Python** (per standing rule)
  Run (project Python 3.12, NOT browser-use-env): `py -3.12 -m pip install ruff black` then `py -3.12 -m ruff check bashgym/gym/trainer.py bashgym/datasets/ scripts/ bashgym/factory/designer_pipelines/ --fix` and `py -3.12 -m black <same paths>`.
  Expected: clean exit.

- [ ] **Step 4: Fix the broken GRPO test** (`tests/gym/test_grpo_script.py` references nonexistent `TrainerConfig(grpo_use_vllm=...)`)
  Add to `TrainerConfig` (trainer.py:97-100 GRPO block): `grpo_use_vllm: bool = False` and `grpo_backend: str = "auto"  # auto|unsloth|plain|trl_vllm`. In `_generate_grpo_script`, emit a literal `use_vllm={self.config.grpo_use_vllm}` line so the test's `assert "use_vllm=False" in script` passes.
  Run: `py -3.12 -m pytest tests/gym/test_grpo_script.py -v` → Expected: PASS.

- [ ] **Step 5: Commit local finished work in logical chunks** (local only, no push yet)
  `git commit` each: (a) gitignore; (b) `feat(datasets): format contracts/converters/validator`; (c) `feat(factory): designer_pipelines → ChatCompletionInferenceParams`; (d) `feat(trainer): plain-transformers GRPO backend for GB10/sm_121 + grpo_backend/grpo_use_vllm config`; (e) `feat(scripts): trace→training-data pipeline + cascade/dpo utilities`; (f) `test(gym): align test_grpo_script with TrainerConfig`.
  Leave `.claude/settings.local.json` and `train_gemma4_*.py` uncommitted (latter consolidated in S1).

- [ ] **Step 6: Merge ws2 with documented conflict rules**
  Run: `git merge origin/ws2-hf-research`. Resolve: **backend/`bashgym/**` non-frontend → take ws2**; **`frontend/**` + typecheck config → take local, then re-add ws2's HFDashboard tab registration** (Sidebar/MainLayout/api.ts/App routing); **`trainer.py` GRPO → keep both backends** (ws2 Unsloth path under `grpo_backend in {auto,unsloth}`, local plain path under `grpo_backend=="plain"`); **`requirements*.txt` → union**. If any single file is too tangled to resolve confidently, `git checkout --theirs`/`--ours` the safer side and open a follow-up note — do NOT guess.

- [ ] **Step 7: Verify both stacks green**
  Run: `py -3.12 -m pytest tests/ -q` (foreground — never background) → Expected: PASS (note any pre-existing skips).
  Run: `cd frontend && npm install && npm run typecheck && npm run lint` → Expected: 0 errors (local's toolchain repair must survive the merge).
  Spot-check both feature sets present: grep for `RunComparison`, `DatasetInspector`, `HFDashboard`, `bashgym/research/` scanner.

- [ ] **Step 8: Commit merge + push**
  `git commit` the merge; `git push origin feat/training-strategies-device-mgmt`. Update PR #16 / open a fresh PR. (Push only after 7 is fully green — never push then fix.)

**Acceptance:** one branch; `pytest tests/` and `npm run typecheck` both green; ws2 research package + multi-strategy cascade AND local frontend dashboards all present; GRPO has selectable backends; finished work committed; branch pushed.

---

## Segment 1 — Model-agnostic training core (ModelProfile + backends + Liger FLCE)

**Goal:** Any HF model id resolves to a correct training script with zero code changes; the Unsloth-vs-plain-transformers choice becomes automatic; large-vocab memory is tamed with fused cross-entropy.

> **Progress (2026-06-15): core landed (`4477a99`).** `bashgym/families/{profiles,patches,backends}.py` + 28 tests (ruff+black clean). `ModelFamilyProfile` registry (Gemma4/Qwen3/Qwen2.5/Llama3/generic) + `resolve_family_profile()`; `select_backend()` makes Unsloth-vs-plain a switch; `apply_patches()` extracts the gemma4_clippable_linear monkey-patch into a reusable form. **Remaining: (a) wire trainer.py `_generate_grpo_script` (+ SFT/DPO) to consume the profile — replace hardcoded target_modules/excludes/patches and reintroduce the plain-transformers backend from `554f4f9` under `grpo_backend=="plain"`; (b) Liger FLCE opt-in (`use_liger_kernel`); (c) replace the 4 `train_gemma4_*.py` with one `scripts/train_model.py --base-model`.**

**Design — `bashgym/models/profiles.py`:**
```python
@dataclass(frozen=True)
class ModelProfile:
    family: str                       # "gemma4" | "qwen3" | "qwen2.5" | "llama3" | "generic"
    match: tuple[str, ...]            # lowercased substrings matched against base_model id
    tool_call_format: str             # "openai_json" | "gemma4_delimited" | "qwen_xml" | "hermes"
    lora_target_modules: tuple[str, ...]
    lora_exclude_modules: tuple[str, ...] = ()      # vision/audio towers for multimodal
    attn_implementation: str = "sdpa"               # "flash_attention_2" where supported
    dtype: str = "bfloat16"
    patches: tuple[str, ...] = ()                    # names resolved in patches.py
    thinking: bool = False                           # needs *-thinking chat template
    chat_template_override: str | None = None
    default_backend: str = "auto"                    # auto|unsloth|plain|trl_vllm
    gguf_template_source: str = "base"               # "base" = pull from base tokenizer; else explicit
    stop_tokens: tuple[str, ...] = ()

REGISTRY: tuple[ModelProfile, ...] = (
    ModelProfile("gemma4", ("gemma-4", "gemma4"),
        tool_call_format="gemma4_delimited",
        lora_target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
        lora_exclude_modules=("vision_tower","multi_modal_projector","audio_tower"),
        patches=("gemma4_clippable_linear",), thinking=True, default_backend="plain"),  # plain until unsloth#4867
    ModelProfile("qwen3", ("qwen3", "qwen-3", "qwen3.5", "qwen3.6"),
        tool_call_format="qwen_xml",
        lora_target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
        thinking=True, default_backend="auto"),
    # qwen2.5, llama3, generic fallback ...
)

def resolve_model_profile(base_model: str) -> ModelProfile: ...  # first match; generic fallback
```

**Backends — `bashgym/models/backends.py`:** `TrainingBackend` ABC with `load_model(profile, cfg)`, `wrap_lora(profile, cfg)`, `save_merged(path)`. `select_backend(profile, platform_probe)` → on GB10/sm_121 force `plain` for gemma4 (unsloth#4867); else honor `profile.default_backend`/`cfg.grpo_backend`. `platform_probe()` detects sm_121/aarch64 and Unsloth import health.

**Liger FLCE:** add `use_liger_kernel: bool` to TrainerConfig; in plain backend, set `TrainingArguments(use_liger_kernel=True)` or apply `LigerFusedLinearCrossEntropy` (Gemma 4 via Liger #1186 community patch — RMSNorm/GeGLU/FLCE, not RoPE).

**Key tests (TDD):**
- `resolve_model_profile("google/gemma-4-31B")` → family gemma4, has `gemma4_clippable_linear` patch, excludes vision_tower.
- `resolve_model_profile("Qwen/Qwen3.6-35B-A3B")` → family qwen3, qwen_xml tool format, no gemma patch.
- `resolve_model_profile("meta-llama/Llama-3.2-3B")` → llama3.
- Script generated for each contains the right `target_modules`, `attn_implementation`, and applies the patch only for gemma4.
- `select_backend(gemma4_profile, sm121_probe)` → "plain"; `select_backend(qwen_profile, x86_probe)` → "unsloth"/"auto".

**Acceptance:** setting `base_model` to a Gemma 4, Qwen 3.6, or Llama model produces a correct, backend-appropriate training script with no edits to trainer.py; the 4 `train_gemma4_*.py` root scripts are deleted/replaced by one `scripts/train_model.py` that takes `--base-model`.

---

## Segment 2 — Model-agnostic export & deploy (GGUF + Ollama Modelfile + template round-trip)

**Goal:** Replicate the single most valuable Unsloth feature we lost — `save_pretrained_gguf` + a correct Ollama Modelfile — for any model, killing the #1 deploy bug (wrong chat template → double-BOS, leaked thinking, broken tool calls).

**Files:** Create `bashgym/export/gguf.py`, `tests/export/test_gguf_template.py`; modify `trainer.py` auto-export hooks (`_auto_deploy_to_ollama` at trainer.py:518, `export_to_gguf` template at ~1950).

**Flow:** merge LoRA (compute-dtype, upcast adapters to fp32 for the merge math) → `convert_hf_to_gguf.py` → `llama-quantize` (profile/cfg quant) → generate Modelfile whose `TEMPLATE` + `PARAMETER stop` come from the ModelProfile / base tokenizer (`ollama show <base> --modelfile` reuse where a known-good base exists) → **train-vs-serve round-trip test**: render a fixed conversation with the training chat template and with the GGUF/Ollama template; assert token-equality (catches the Gemma 4 `<|"|>` delimiter + thinking-channel drift).

**Acceptance:** export a merged checkpoint → Ollama loads it → round-trip template test passes → a 5-prompt tool-call smoke suite returns parseable tool calls.

---

## Segment 3 — Held-out trace eval + statistical gate + registry integration

**Goal:** Replace today's contaminated, single-run win-count with a frozen, contamination-free, CI-backed "is it better?" that gates deployment. (Today's eval judges on the same `gold_traces/` pool training exports from — `eval_finetuned.py:37-39,241`.)

**Files:** Create `bashgym/eval/heldout.py`, `bashgym/eval/gate.py`, `tests/eval/test_heldout.py`, `tests/eval/test_gate.py`; modify the export path to exclude the frozen eval split; modify `~/.bashgym/models/registry_index.json` writer.

**Design:** Freeze a held-out split **by session AND repo** (whole sessions = in-distribution; whole repos = hard generalization) before training; persist split manifest (hashes) so the export pipeline can assert disjointness. Metrics: (1) step-level tool-name exact-match + per-argument AST F1 (parse tool_calls — never substring); (2) episode-level pass@1 + pass@8/pass^8 via the existing Docker sandbox + `verify.sh`. Stats: per-example paired deltas vs base → **paired bootstrap, 1,000 resamples, clustered by session** → ship only if 95% CI of the difference excludes zero (Anthropic arXiv:2411.00640). Pre-register thresholds in `gate.py` (e.g. trace pass@1 +5pts CI>0; forgetting ≤2pts). `gate.py` writes verdict to the registry and refuses `auto_deploy` on regression.

**Acceptance:** `python -m bashgym.eval.heldout --base X --candidate Y` emits a CI-backed verdict + per-metric table; a contamination test asserts train/eval split disjointness; a regression verdict blocks deploy in a dry-run.

---

## Segment 4 — HF public dataset ingestion (scan → convert → decontaminate → mix)

**Goal:** "Leverage HF public datasets for domain-specific training" — turn the ws2 research scanner from simulate-only into a real pipeline that pulls verified public trajectories, converts them to our format, decontaminates, dedups, and mixes with our traces at a tunable ratio.

**Files:** Promote `bashgym/research/` DatasetSearchSpace simulate→real (real eval already scaffolded); create `bashgym/datasets/hf_ingest.py`, `bashgym/datasets/decontaminate.py`, `tests/datasets/test_hf_ingest.py`, `tests/datasets/test_decontaminate.py`; extend `bashgym/datasets/converters.py`.

**Shortlist (verified, license-OK for Apache-2.0 project):** `nebius/SWE-rebench-openhands-trajectories` (67k, 32k `resolved=1`, CC-BY-4.0) #1; `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k`; `nvidia/Nemotron-SWE-v1` (CC-BY-4.0); `SWE-bench/SWE-smith-trajectories` (MIT); `Agent-Ark/Toucan-1.5M` (Apache-2.0, sample); `SALT-NLP/SWE-chat` (6k real Claude Code sessions — closest analog to ours).

**Flow:** scanner ranks → download `resolved=1` subset → converter maps OpenHands/SWE-agent schema → our `messages` (tool_call args as **dicts**, fixing the serialized-JSON gotcha) → `decontaminate.py` drops anything with 13-gram overlap / >0.7 3-gram / >0.85 cosine vs SWE-bench repos → MinHash dedup across the merged pool (self+public) → write mixed `train.jsonl` with self-trace slice at a configurable 10–30% (upsampled). Expose the mix ratio as an AutoResearch hyperparameter.

**Acceptance:** ingest a SWE-rebench `resolved=1` subset → unified format passes `validator.py` → decontamination report shows drops → mixed dataset builds; ratio is a config knob.

---

## Segment 5 — Data pipeline quality upgrades

**Goal:** Make the training sets *matter* — fix the catalogued extraction/scoring weaknesses that cap effectiveness.

**Files:** modify `bashgym/factory/data_factory.py` (continuous scores; decision-level DPO at 859-942; observation masking in SFT export), `bashgym/factory/trace_processor.py` (stop hard gold/failed binarization — carry `quality_calculator` continuous score as metadata), tool-call sanitization keyed by `ModelProfile.tool_call_format` (Gemma4 `<|"|>`, qwen_xml, hermes). Tests under `tests/factory/`.

**Acceptance:** SFT exports mask tool-output/observation tokens (train on assistant action tokens only); traces carry continuous verification scores instead of 3-bucket binarization; decision-level DPO produces pairs; tool_calls render correctly per profile.

---

## Segment 6 — External benchmark evals (lm-eval / Terminal-Bench 2.0 / BFCL-V4 / SWE-bench Lite)

**Goal:** Report the 2026-standard agentic numbers + a catastrophic-forgetting guard, all against a vLLM endpoint.

**Files:** create `bashgym/eval/benchmarks_ext.py` + adapters; results → registry. (Needs S8 vLLM serving.)

**Stack:** lm-evaluation-harness `local-completions` → vLLM (MMLU/GSM8K/IFEval/HellaSwag forgetting suite, >5pt drop = hard fail); Terminal-Bench 2.0 via Harbor (`--agent terminus-2`); BFCL-V4 local categories (non-live/multi-turn/hallucination); SWE-bench Verified Lite via mini-swe-agent v2 + official harness. Run harnesses from the x86 desktop against `http://192.168.50.173:<port>/v1`; Ollama/GGUF gets a separate smoke pass (no logprobs).

**Acceptance:** base-vs-candidate benchmark table with deltas; forgetting gate wired into `eval/gate.py`.

---

## Segment 7 — Flywheel automation (MOPD unstub + auto-trigger + Hermes importer)

**Goal:** Close the loop so the platform self-improves without manual kicks.

**Files:** modify `bashgym/api/cascade_routes.py:244-248` (replace the `asyncio.sleep` stub with the real `distill_cascade()` call — verified still a stub on 2026-06-11), `bashgym/pipeline/orchestrator.py` (ThresholdMonitor → `POST /api/cascade/start`), `bashgym/gym/cascade_scheduler.py` (stage-completion checkpointing for resumption; set `auto_deploy_ollama` gated by S3 `eval/gate.py`); create `bashgym/trace_capture/importers/hermes_history.py` (`~/.hermes/state.db` SQLite).

**Acceptance:** a threshold breach triggers a cascade → eval gate → conditional auto-deploy, demonstrated end-to-end on a tiny run; Hermes sessions import as traces.

---

## Segment 8 — DGX environment consolidation  *(APPROVAL-GATED — present this plan to the user before touching the Spark)*

**Goal:** One modern venv replacing the separate-venvs workaround, with vLLM serving for evals and Hermes as the student deployment harness.

**Plan to approve:** new venv with transformers 5.5.x + vLLM ≥0.22.1 (cu130 stable wheels) + TRL ≥1.6 + Unsloth latest (resolved upstream — vLLM ≥0.20 runs transformers 5.5.x); decide Unsloth path (official DGX Spark Docker `TORCH_CUDA_ARCH_LIST=12.1` vs 3 manual patches for #4867); stand up vLLM serving for S6; install Hermes Agent pointed at Ollama (`OLLAMA_CONTEXT_LENGTH=64000`); evaluate NeMo Gym v0.3 / NeMo RL v0.6 (LoRA-GRPO/GDPO) as backend candidates. Keep old venvs until the new one is proven on a Gemma 4 + Qwen 3.6 fine-tune + serve.

**Acceptance:** one venv trains Gemma 4 *and* Qwen 3.6 and serves both via vLLM; bf16 LoRA retained (GB10 bitsandbytes still blocked); documented; rollback path intact.

---

## Self-review notes
- **Spec coverage:** evals up-to-date → S3 (held-out) + S6 (benchmarks); better datasets → S4; local-model integration + "how do we know it's better" → S1/S2/S3 loop + gate; Unsloth parity → S1 (kernels/backends) + S2 (export); model-agnostic across Qwen 3.6/Gemma/etc → S1 ModelProfile (north star). All covered.
- **Risk hotspots:** S0 merge (mitigated by backups + abort-if-tangled rule); S8 is the only environment change and is approval-gated; lint env must be the project Python 3.12, not browser-use-env.
