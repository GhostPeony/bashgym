# Evals, Datasets, Local-Model Loop, and the Unsloth Gap — June 11, 2026

Companion to `status-review-2026-06-11.md`. Sources: 1 repo audit + 3 web research agents; key claims spot-verified (contamination in eval_finetuned.py, SWE-rebench dataset card, Liger-Kernel #1186).

---

## 1. Eval & test-suite audit (current state)

### What exists
- **ws2 eval scripts** (`scripts/` on ws2-hf-research): eval_finetuned.py (3 tiers: val-loss + tool prediction / Claude-Haiku judge on 25 traces / HumanEval pass@1), benchmark_finetuned_vs_base.py, benchmark_three_models.py (base vs SFT vs DPO), smoke + thorough functional tests.
- **Judge layer** (`bashgym/judge/`): verifier.py (task verification), evaluator.py (NeMo Evaluator client, localhost-only), semantic_judge.py (Claude trace-quality verdicts), benchmarks.py (1053-LOC BenchmarkRunner: HumanEval/MBPP/BigCodeBench/BFCL/...).
- **Router** (`bashgym/gym/router.py`): CONFIDENCE_BASED + PROGRESSIVE student handoff on rolling success_rate.
- 61 test files; **zero tests for judge/, eval scripts, or router** (only providers/test_router_integration.py).

### Critical weaknesses (verified)
1. **Train/test contamination**: eval_finetuned.py Tier 2 samples from `~/bashgym/data/gold_traces` — the same pool training data is exported from (lines 37–39, 241). Tier 1 val.jsonl split is not documented as session/repo-level.
2. **No statistical rigor anywhere**: single runs, n=15–30 for judge tiers, no seeds, no CIs, no paired tests. Verdict = naive win-count.
3. **Brittle metrics**: tool prediction via substring match on response text, not parsed tool_calls.
4. **Judge never calibrated** (does Haiku score correlate with execution success?), not blinded, single ordering (position bias).
5. **HumanEval is dead** as a 2026 benchmark (saturated); no agentic benchmark (Terminal-Bench, SWE-bench, BFCL) wired in despite benchmarks.py scaffolding.
6. **No catastrophic-forgetting suite**, no result tracking/regression detection, results land in untracked local JSON.
7. tests/gym/test_grpo_script.py broken (`grpo_use_vllm` field doesn't exist — pending GRPO reconciliation decision).

## 2. Target eval stack (2026 standards)

Reporting convention for agentic fine-tunes our size: **SWE-bench Verified (scaffold + context stated) + Terminal-Bench 2.0 + BFCL-V4 + tau2**, plus forgetting suite. HumanEval/MBPP are legacy.

**Architecture**: serve candidate + base via vLLM on the Spark (bf16 merged); run harnesses from desktop against `http://192.168.50.173:<port>/v1`. Ollama/GGUF gets a separate smoke suite (quantization + template shift scores; Ollama lacks logprobs so it can't run MCQ benchmarks).

- **Tier 1 — every checkpoint (~1–2h)**
  1. *Held-out trace eval (build in-repo — primary ship signal)*: ≥300 paired examples; split by **session AND repo**, frozen before training; step-level tool-name exact match + per-arg AST F1 (BFCL-style, parse args — not substrings); episode-level pass@1 + pass@8/pass^8 through our own sandbox+verifier; **paired bootstrap 1,000 resamples clustered by session**, ship only if 95% CI excludes zero (Anthropic arXiv:2411.00640 protocol).
  2. *Forgetting suite*: lm-evaluation-harness (`local-completions` → vLLM): MMLU 5-shot, GSM8K, IFEval, HellaSwag. >5pt drop on any = hard fail; IFEval is the tool-format-overfit canary.
- **Tier 2 — per release candidate (~½ day)**
  3. Terminal-Bench 2.0 via Harbor (`harbor run --dataset terminal-bench@2.0 --agent terminus-2 --model openai/<id>`), base vs FT, 2–3 runs.
  4. BFCL-V4 local categories (non-live, multi-turn, hallucination) from gorilla repo or inspect_evals port.
  5. SWE-bench Verified Lite/50-task subset via mini-swe-agent v2 + official harness.
- **Tier 3 — quarterly/optional**: tau2-bench, SWE-bench Multilingual, MCP-Atlas. Skip: OSWorld, aider polyglot, LiveCodeBench (wrong modality).
- **Judge hygiene** when LLM-judge is used: non-Gemma judge family, both orderings, length normalization, calibrate vs ~50 human labels.
- Useful harnesses: inspect-ai v0.3.225 (best general framework, Ollama/vLLM native), lm-eval-harness, Harbor, mini-swe-agent v2, NeMo Evaluator SDK 0.1.0 (bring-your-own-endpoint; reproduces Nemotron numbers; check aarch64 images).

**Test-suite fixes**: add tests/judge/ + tests for router and dataset validator; fix or quarantine test_grpo_script.py; unskip-audit test_remote_integration.py.

## 3. External datasets worth mixing in (verified shortlist)

| Priority | Dataset | Size / License | Why |
|---|---|---|---|
| 1 | `nebius/SWE-rebench-openhands-trajectories` | 67,074 traj (32k resolved=1), CC-BY-4.0 | Real GitHub issues, execution-verified, proper tool-call format (args serialized as strings — needs our dict sanitization) |
| 2 | `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` | 66k traj | Log-linear scaling evidence (Qwen3-8B → ~40% SBV) |
| 3 | `nvidia/Nemotron-SWE-v1` (+ Nemotron-Agentic-v1) | 59k / 335k, CC-BY-4.0 | NVIDIA-curated SFT-ready, permissive-repo-only |
| 4 | `SWE-bench/SWE-smith-trajectories` | 5k–76k, MIT | Trained SWE-agent-LM-32B (40.2% SBV) |
| 5 | `Agent-Ark/Toucan-1.5M` (sample 50–100k) | 1.65M, Apache-2.0 | MCP/tool-calling generalization (495 real MCP servers) |
| 6 | `SALT-NLP/SWE-chat` | 6,000 real Claude Code/Codex sessions | Closest analog to our own traces — ~3x our real-user slice |

YC landscape: OpenPipe (acquired by CoreWeave; **ART RL library is open-source** and relevant to GRPO), Datacurve (closed, lab-only), Osmosis/LLM Data Company/RunRL (services). Most usable non-dataset asset: **Prime Intellect Environments Hub** (2,500+ open RL envs + `verifiers` lib).

Mixing practice: success-only filtering (resolved=1); keep self-collected slice ~10–30% of mix (upsample 2–4x) — treat ratio as an AutoResearch hyperparameter; **mandatory decontamination vs SWE-bench repos before ever reporting SBV**; MinHash dedup across the merged pool (public sets overlap each other); format unification into our messages schema is the real work (`bashgym/datasets/converters.py` is the home for it).

## 4. Unsloth gap analysis (what we lose on the plain-transformers path)

Neutralized by GB10 (lose nothing): async-offload gradient checkpointing (unified memory), dynamic 4-bit quants (bitsandbytes broken anyway; bf16 LoRA is right), GRPO Standby weight-sharing (128GB blunts duplication; venv topology favors TRL server-mode/AsyncGRPO anyway). GA loss-norm fix: upstreamed in transformers ≥4.46.

**Real losses + cheapest replacements:**
1. **Fused/chunked cross-entropy** — Gemma 4's 262k vocab makes the logits tensor the dominant memory term. Fix: Liger-Kernel FLCE (Gemma 4 via open issue #1186 — working community impl: RMSNorm ✅ GeGLU ✅ FLCE ✅ RoPE ❌) or Apple cut-cross-entropy. Liger is pure Triton → works on aarch64/sm_121a.
2. **GGUF + Ollama Modelfile export with template verification** — the #1 Gemma 4 deploy bug (double-BOS, thought-channel leakage, broken tool calls). Fix: ~30-line path: merge → convert_hf_to_gguf.py → quantize → reuse `ollama show gemma4 --modelfile` TEMPLATE/stop block → **train-vs-serve template round-trip test**. ws2's scripts/export_unsloth.py is the starting point.
3. **Sequence packing** — TRL `packing="bfd"` needs FA2/FA3 (not SDPA); on SDPA, pack at dataset-mean length manually. See GB10 forum recipe (7.67x LoRA speedup, NGC PyTorch 26.03 + transformers 5.5.0 + Triton 3.6 + xformers TORCH_CUDA_ARCH_LIST=12.1 + packing + torch.compile): https://forums.developer.nvidia.com/t/370517
4. **Gemma 4 correctness fixes to audit in our path**: use_cache=False garbage logits on E2B/E4B with gradient checkpointing (we use it!), `layer_types[:-0]` cache crash on 31B/26B-A4B, re-pull base before export (template fixed upstream ~May 4).
5. **torch.compile** recovers much of the fused-kernel speed generically.

Note: Unsloth README now claims DGX Spark support but #4867 is still open — claim ahead of reality.

## 5. "How do we know it's better?" — the loop

Train → merge → **vLLM serve (base + candidate)** → Tier 1 gate (held-out trace CI + forgetting) → Tier 2 on release candidates → GGUF export w/ template test → Ollama smoke suite → deploy as Student → router PROGRESSIVE handoff → traces flow back. Wire eval results into the model registry (`registry_index.json`) and surface in the dashboard; regression = block deploy. Today's answer ("FT wins > base wins on contaminated n=25") is not evidence; after Tier 1 lands it becomes a pre-registered, CI-backed claim.
