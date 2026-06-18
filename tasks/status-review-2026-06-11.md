# BashGym Status Review — June 11, 2026

Synthesized from 4 repo-review reports (cascade-rl, training-stack, data-pipeline, repo-state) and 4 web-research reports (data-designer, nvidia-ecosystem, hermes-agent, best-practices).

---

## 1. Executive Summary

BashGym is in a strong but stalled position. The v0.2.0 AutoCurriculum Compiler (PR #13, merged 2026-03-26, +6405/−569, 236 new tests) shipped Cascade RL, schema evolution, and Data Designer multi-provider support. Since April 9 (last local commit a2ea717), local work has focused on DGX Spark GB10/sm_121 compatibility: GRPO moved off Unsloth to plain transformers+peft+trl, a Gemma 4 PEFT monkey-patch, Triton ptxas env fixes, a new `bashgym/datasets/` contracts layer, and 5 pipeline scripts — all uncommitted. The local branch is **4 commits behind origin/feat/training-strategies-device-mgmt and 3 merge commits behind origin/master** (verified via `git rev-list --count`). Separately, the DGX clone's `ws2-hf-research` branch holds 34 commits (~14k lines) that existed only on that machine until backed up to origin during this review.

Meanwhile, the upstream ecosystem moved decisively in BashGym's favor between April and June 2026:

- **The vLLM/Unsloth transformers conflict is RESOLVED upstream**: vLLM v0.20.0 (Apr 27, 2026) runs on transformers 5.5.x (PR #30566, merged Apr 15), CUDA 13.0.2 default wheels. One venv can now hold Unsloth + TRL ≥1.5/1.6 + vLLM on the Spark.
- **GRPO-needs-vLLM is no longer a conflict**: TRL 1.x still requires vLLM for GRPO generation, but since vLLM ≥0.20 coexists with Unsloth, this blocker dissolves. TRL v1.6.0 (Jun 11) moved AsyncGRPO to a separate CPU-only process.
- **GB10 bitsandbytes detection is still broken** — bf16 LoRA remains the right call, and best-practices research independently confirms QLoRA is NOT recommended for Gemma 4 26B-A4B MoE anyway.
- **Data Designer v0.6.1** (Jun 1) adds `AgentRolloutSeedSource` with native `CLAUDE_CODE` and `HERMES_AGENT` formats — directly overlapping BashGym's hand-rolled trace importers — plus a real CLI, async-by-default engine, run resumption, and workflow chaining. The repo's `ChatCompletionInferenceParams` migration is correct but targets v0.5.x; v0.6.x is current.
- **Hermes Agent** (Nous Research, MIT, ~180k stars, v0.16.0 Jun 6) is the NVIDIA-blessed DGX Spark agent harness — the natural deployment leg for BashGym students (Ollama endpoint) and a new trace source (`~/.hermes/state.db` SQLite).

Cascade RL is ~80% complete: the critical stub is the MOPD distillation route (`cascade_routes.py:244-248` sleeps instead of calling `distill_cascade()`), plus no auto-trigger from threshold monitors. The data pipeline has 20 identified weaknesses; the biggest are sparse verification (~70% of traces `verification_passed=None`), no code-execution validation in synthetic generation, and decision-level DPO existing but not wired in.

---

## 2. Repo State

**Branch**: feat/training-strategies-device-mgmt | **Last local commit**: a2ea717 (Apr 9, 2026, "feat: incremental dataset improvement pipeline for Gemma 4 training")

### Branch divergence (CRITICAL)

Local is **4 commits behind origin/feat/training-strategies-device-mgmt**:
- d4b8442 — fix: use gradient_checkpointing=True in GRPO script (Unsloth/TRL compat)
- 750e9d2 — fix: use Unsloth for GRPO script generation (fixes vLLM crash on DGX Spark)
- 4cee2e4 — fix: prevent 2.5GB memory spike in test_prompt_evolver_training
- 83e505f — fix: check both top-level and metadata.primary_repo in repo domain scanner

Local is **3 merge commits behind origin/master** (origin/master at a1bb8a1; PRs #14–16, all from this feature branch, merged Mar 26–Apr 10 — their content is already in local history).

**DGX-only work (now backed up)**: the DGX clone was on branch `ws2-hf-research` — 34 commits, ~13,988 insertions, never pushed until this review (now on origin as `ws2-hf-research`; `ws1-cascade-multi-strategy` is fully contained in it). Contents: `bashgym/research/` HF dataset scanner + scoring + DatasetSearchSpace + empirical ranking, cascade multi-strategy dispatch (SFT/DPO/GRPO per stage), GRPO/SFT/DPO subprocess hardening (LossPlateauStop, sentinel-race fix), `WORKSTREAMS.md` (ws1/ws2 marked complete, ws3 frontend wiring landed), 11 additional scripts (eval/benchmark/export/DPO-extraction). The local untracked `scripts/`, `bashgym/datasets/`, and modified `designer_pipelines/` files are byte-identical subsets of ws2 commits; local `trainer.py` working-tree changes (plain-transformers GRPO + Gemma4 PEFT patch) are NOT in ws2 and conflict with ws2's Unsloth-GRPO direction.

**Conflict warning**: remote commits d4b8442/750e9d2 moved GRPO script generation *back to Unsloth*, while local uncommitted trainer.py changes move GRPO *off Unsloth* to plain transformers+peft for sm_121. These must be reconciled before pull/push.

Other origin branches: origin/feat/dual-loop-evolution (eaf3702, Mar 17), origin/feat/integration-nodes (9191112, Feb 25).

### Git timeline arc
- Mar 20–26: v0.2.0 release (PR #13) — AutoCurriculum Compiler, Cascade RL scheduler, Data Designer multi-provider, 236 tests, Nemotron base models (Cascade-2-30B-A3B, Nano-4B, Mini-4B)
- Mar 27–Apr 2: Discord webhook (e00fa37), Gemma 4 models in config (f50bdf9), cascade filter fixes (9a0844a, cb2975d), Qwen3.5 tool_calls sanitization (90051a6), auto-classify background task (afd4682)
- Apr 1–9: subagent trace import (4c0057e), incremental dataset improvement (a2ea717)

### Uncommitted work

**Modified:**
- `bashgym/gym/trainer.py` (106+/34−) — GRPO sm_121 refactor (see §5). **FINISHED, ready to commit** (after reconciling with remote)
- `bashgym/factory/designer_pipelines/{__init__.py,coding_agent_dpo.py,tool_use_sft.py}` — `InferenceParameters` → `ChatCompletionInferenceParams` (Data Designer v0.5.x API rename). Affected lines: `__init__.py:57, 66, 75`; `coding_agent_dpo.py:57, 65`; `tool_use_sft.py:95`. **FINISHED, ready to commit**
- `.claude/settings.local.json` — local allowlist entries (ESLint, find, review skills). Likely keep local.

**Untracked:**
- `bashgym/datasets/` — `contracts.py` (format schemas: SFT/DPO/GRPO/Distillation), `converters.py` (format conversion), `validator.py`. **FINISHED, commit**
- `scripts/` (~1155 LOC): `fix_training_data.py` (415 LOC), `generate_dpo_dataset.py` (320 LOC, DPO pairs via Data Designer), `full_pipeline_rebuild.py` (164 LOC), `run_cascade_real.py` (105 LOC), `validate_dataset.py` (71 LOC). **FINISHED, commit**
- `tests/gym/test_grpo_script.py` (71 LOC) — **HALF-DONE**: references `TrainerConfig(grpo_use_vllm=...)` at lines 27, 32 but that field does not exist in `TrainerConfig` (trainer.py:56-147; actual GRPO fields at trainer.py:97-100 are only `grpo_num_generations`, `grpo_temperature`, `grpo_reward_mode`). Add the field or delete the tests.
- `train_gemma4_31b.py`, `train_gemma4_26b_a4b.py`, `train_gemma4_e4b.py`, `train_gemma4_e2b.py` (~125 LOC each) — near-identical Unsloth SFT scripts; consolidate into one parametrized script or delete.
- `data-fixed/` (353 train.jsonl examples + unsloth/ chatml/sharegpt variants), `data-pipeline-fixed/` (5,476 train.jsonl examples, same layout — output of the a2ea717 incremental improvement pipeline). **Gitignore both.**

### ML stack pins (requirements)
- requirements.txt: anthropic>=0.20.0, fastapi>=0.109.0, uvicorn>=0.27.0, pydantic>=2.5.0, datasets>=2.14.0, asyncssh>=2.14.0, pytest>=7.4.0
- requirements-training.txt: torch>=2.1.0, transformers>=4.36.0, datasets>=2.16.0, peft>=0.7.0, trl>=0.15.0, accelerate>=0.25.0, nemo-microservices>=0.1.0, bitsandbytes (commented, Linux only), unsloth (commented)
- These floors are far behind current (TRL 1.6.0, transformers 5.5.x, torch 2.11) — see Gap Analysis.

---

## 3. Cascade RL & AutoResearch Status

**Overall: ~80% complete for sequential domain training; AutoResearch fully implemented.**

### Implemented
- **CascadeScheduler.run_cascade()** (cascade_scheduler.py:325-543) — fully automated domain-by-domain sequential GRPO: filter dataset → GRPO → chain checkpoint to next stage; WebSocket progress callbacks; graceful stop.
- **Domain taxonomy** (cascade_scheduler.py:32-110): file_operations, bash_commands, search_and_navigate, multi_step_reasoning; **repo-based domains** via `RepoCascadeDomain`/`build_repo_domains()` (cascade_scheduler.py:118-193); weakest-first ordering via `sort_domains_by_loss()`.
- **Dataset filtering** (cascade_scheduler.py:602-694): .json & .jsonl, OpenAI messages & BashGym trace formats, trace→GRPO conversion via `trace_to_grpo_example()`, post-filter `validate_dataset(filtered_path, format="grpo")`.
- **GRPO integration**: `_run_stage()` (cascade_scheduler.py:545-578) builds `TrainerConfig` (strategy=GRPO, per-domain reward_mode, `use_remote_ssh` passthrough) → `GRPOTrainer.train_grpo()` (trainer.py:2054-2200+).
- **MOPD infrastructure**: `MOPDConfig` (cascade_scheduler.py:290-318), `create_mopd_config()` (cascade_scheduler.py:721-751), `distill_cascade()` (cascade_scheduler.py:759-917) → `Trainer.train_distillation()` with `_mopd_domain`/`_mopd_teacher` tagging.
- **AutoResearch** (autoresearch.py:37-676): `SearchSpace` ABC (37-60), SEARCH_SPACE (66-115: learning_rate 1e-6–1e-3 log-scale, lora_r 4–128, lora_alpha 8–256, lora_dropout 0–0.3, warmup_ratio 0–0.3, grad_accum 1–64, batch_size 1–32, max_seq_length 512–8192, load_in_4bit), `_simulate_loss()` heuristic (184-256), `HyperparamSearchSpace` (319-435), `AutoResearcher.run_loop()` evolutionary search with pause/resume/stop (442-676). Real mode: 10% dataset subset, 100 steps, eval_loss.
- **Schema Search Space** (schema_search_space.py): SCHEMA_SEARCH_SPACE (35-62), TEMPLATE_LIBRARY & FAILURE_TEMPLATE_MAP (69-150), `GemmaJudge` local-perplexity judge (157-227), two-stage eval — fast judge filter then micro-train (234-574).
- **API**: POST/GET /api/cascade/start|stop|status (cascade_routes.py:82-181), /api/autoresearch/start (autoresearch_routes.py:90-225), /api/autoresearch/schema-research/start (autoresearch_routes.py:597-709).
- **Tests**: tests/gym/test_cascade_scheduler.py:20-445, test_cascade_distillation.py:1-83, test_autoresearch.py:1-292, tests/api/test_cascade_routes.py:23-82.

### Stubbed / missing (blockers to full automation)
1. **MOPD route stub** — `POST /api/cascade/distill` (cascade_routes.py:184-272) sleeps `random.uniform(2.0, 5.0)` at lines 244-248 instead of calling `distill_cascade()`. **Key blocker**; fix is replacing ~4 lines.
2. **No auto-trigger**: `ThresholdMonitor` (threshold_monitor.py:30-54) tracks gold-trace/example counts but never invokes cascade; pipeline/orchestrator.py:1-150 has no cascade hook.
3. **Schema evolution not fed back into cascade** — best schema from schema-research doesn't regenerate cascade datasets.
4. **MOPD offline vs on-policy unclear** (cascade_scheduler.py:842-862) — teacher outputs not stored in filtered datasets; multi-teacher handling in `train_distillation()` unverified.
5. **`_train_with_remote_ssh()` wiring unverified** — branch points at trainer.py:485, 1278, 2107; `RemoteTrainer` (remote_trainer.py:63-200) exists with preflight checks but end-to-end integration unconfirmed by review (note: training-stack review found the orchestration in trainer.py:1726-1779 complete — see §below).
6. **No cascade resumption** — crash at stage 3/4 requires re-running all stages.
7. **No auto-deployment** — `auto_deploy_ollama`/`auto_push_hf` trainer flags exist but cascade never sets them.
8. **No job queue** for concurrent cascades; **validation continues past errors** (cascade_scheduler.py:681-692 logs warnings, no early exit); **no human-in-the-loop pause points**.

Estimated effort to close blockers 1–3: 1–2 weeks.

### Remote SSH training (reconciled view)
The training-stack review confirms the remote flow IS complete in `remote_trainer.py` + trainer.py:1726-1779: preflight (`PreflightResult`: ok, gpus, cuda_version, hostname, os_info) → script generation → SFTP upload to `{remote_work_dir}/{run_id}/` → `nohup bash -c '{venv python3 script}' > training.log 2>&1 & echo $!` → 2-second tail polling with `log_callback` → SFTP download of `/final` and `/merged` → pause/resume/cancel via SIGSTOP/SIGCONT/SIGTERM. The cascade-rl review's uncertainty applies only to whether all three strategy entry points exercise it; a smoke test should confirm.

---

## 4. Data Pipeline Status

### Trace → training-example flow
- **Classification**: Gold = `verification_passed==True` or quality≥0.7 & success_rate≥0.8; Failed = explicit False or success_rate<0.3; Pending = `verification_passed=None` treated as conditional gold (trace_processor.py:575-577 rejects only explicit False). Counts: 3,140 gold / 3,014 pending (2,927 subagent) / 458 failed.
- **Quality scoring** (`quality_calculator.py:calculate_quality_breakdown()`, total at lines 494-502): success_rate 25% + verification 20% + cognitive 15% + complexity 15% + tool_diversity 10% + efficiency 10% + length 5%. NaN-guarded, clamped [0,1].
- **Dedup**: content-hash `SHA256(canonical_json(task_prompt, commands))[:16]` (trace_processor.py:487-496) + semantic `EmbeddingDeduplicator` via NIM `nvidia/nv-embedqa-e5-v5`, cosine ≥0.95 (dedup.py:29); diversity metric at dedup.py:221-253.
- **Cognitive tags**: thinking/plan/reflection/decision_rationale injected via `build_tool_call_messages()` (data_factory.py:312-414); `DecisionExtractor` (decision_extractor.py) detects tool-selection, pivots, error recovery, commit points.
- **Outputs**: structured multi-turn OpenAI-format messages with tool_calls (preferred); legacy 3-message fallback (data_factory.py:216-225); trace-level DPO pairs (data_factory.py:237-259, 812-857); decision-level DPO (data_factory.py:859-942) — **exists but NOT wired into main factory flow**.

### Data Designer integration
- Targets `data-designer>=0.5.0`; imports `data_designer.config as dd` + `DataDesigner` from `data_designer.interface` (data_designer.py:26-48); hasattr feature detection (lines 34-48).
- Five pipelines in `designer_pipelines/__init__.py:PIPELINES`: coding_agent_sft, coding_agent_dpo (dual-temp 0.9/0.5 candidates + judge + ≥1.0 score gap), tool_use_sft, from_external, from_unstructured. Shared `build_base_config` (designer_pipelines/__init__.py:43-110) with multi-provider `ProviderSpec` (NVIDIA_API_KEY/ANTHROPIC_API_KEY/OPENAI_API_KEY/OPENROUTER_API_KEY env resolution).
- Uncommitted migration: `InferenceParameters` → `ChatCompletionInferenceParams` across all three modified builders (required for v0.5.x+).

### New untracked dataset layer
- `bashgym/datasets/contracts.py` — DatasetFormat schemas (DPO requires prompt/chosen_response/rejected_response; SFT requires messages; GRPO allows prompt as string or messages); `converters.py`; `validator.py`.
- `data-fixed/` (353 examples) vs `data-pipeline-fixed/` (5,476 examples) — output of the a2ea717 6-phase pipeline (high-fidelity capture → loss-targeted mining via `gemma_loader.ft_loss()` → DPO failure pairing via NIM embeddings → repo-domain Cascade RL → GemmaJudge schema evolution → TrainingTrigger micro-train).

### 20 weaknesses identified (abbreviated; full detail in data-pipeline review)
Extraction: (1) lossy tool normalization (data_factory.py:262-287); (2) shallow command→argument parsing (data_factory.py:290-309 — Edit gets empty old/new strings); (3) silent 2000-char output truncation (data_factory.py:350); (4) no cross-step semantic linking; (5) **~70% of traces have verification_passed=None**.
Data Designer: (6) no code-execution validation (judge reads, doesn't run); (7) implicit per-column provider assignment; (8) no incremental/streaming generation; (9) embedding dedup post-hoc, wastes generation budget; (10) no pipeline observability (no row-filter counts/timings/token costs).
Scoring: (11) binary per-step success; (12) pattern-based, non-semantic complexity (quality_calculator.py:128-201); (13) verbosity≈quality in cognitive scoring (361-453); (14) tool diversity capped at 4 (242-278); (15) length penalty kills >50-step traces (204-239, score 0.2); (16) no domain stratification; (17) static 0.95 dedup threshold; (18) cognitive data never coherence-checked (data_factory.py:356-388).
Training: (19) no difficulty curriculum in cascade ordering; (20) DPO from whole-trace comparison only (decision-level DPO dormant).

Critical path (from review): code-execution validation → wire decision-level DPO → bash AST parsing → curriculum ordering → cognitive coherence validation → per-domain thresholds → incremental generation.

---

## 5. DGX Spark State

**Hardware**: ascent-ponyo (192.168.50.173, user ponyo), GB10 Blackwell, 128GB unified memory, sm_121a.
**Current official stack (June 2026)**: DGX OS 7.5.0, driver 580.159.03, CUDA Toolkit 13.0.2, kernel 6.17.

### Local uncommitted trainer.py changes targeting the Spark
1. **Triton ptxas fix** (trainer.py:2159-2177): subprocess env sets `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (CUDA 13 ptxas with sm_121a; Triton bundles CUDA 12.8 ptxas without it) and `TORCH_CUDA_ARCH_LIST=12.1a`. Fixes https://github.com/triton-lang/triton/issues/9181. Caveat: hard-codes Linux path check at trainer.py:2164.
2. **Unsloth removed from GRPO** (trainer.py:2316-2356, 2464-2482): plain `AutoModelForCausalLM` + peft `get_peft_model()`, `dtype=torch.bfloat16`, `attn_implementation="sdpa"` (FlashAttention broken on sm_121), native gradient checkpointing, LoRA exclusions for vision/audio towers, plain `.merge_and_unload()` instead of `save_pretrained_merged()`.
3. **Gemma 4 PEFT patch** (trainer.py:2310-2341): monkey-patches `Gemma4ClippableLinear` (inherits nn.Module, invisible to PEFT) into an `nn.Linear` subclass preserving clipping via forward() override. Source: https://huggingface.co/google/gemma-4-31B/discussions/3.
4. **GRPOConfig**: `max_steps` param, `bf16=True`, LoRA-adapter-only save (trainer.py:2505-2526).

### Known environment issues — resolved vs still blocked (June 2026)
| Issue | Status |
|---|---|
| vLLM (transformers<5) vs Unsloth (>=5.5) venv conflict | **RESOLVED** — vLLM v0.20.0 (Apr 27) on transformers 5.5.3 (PR #30566) |
| GRPO needs vLLM (TRL ≥0.24) but vLLM conflicts | **RESOLVED as a conflict** — vLLM still required for TRL GRPO generation, but now coexists; single venv feasible with vLLM ≥0.20 + TRL ≥1.5 |
| vLLM Gemma 4 needs nightly cu129 wheels | **RESOLVED** — stable since v0.20.0; cu130 default wheels; official recipe page (https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html) still shows stale nightly install — ignore it, use `pip install vllm>=0.22.1` |
| GB10 bitsandbytes can't detect unified memory → 4-bit fails | **STILL BLOCKED** — keep bf16; aligns with Gemma 4 26B-A4B MoE guidance (QLoRA not recommended for MoE routing anyway) |
| Unsloth aarch64/GB10: kernels-community/vllm-flash-attn3 has no GB10 build | **STILL BLOCKED** — issue https://github.com/unslothai/unsloth/issues/4867 (Apr 6, Unsloth 2026.4.2/PyTorch 2.10/CUDA 13.0) open; 3 manual transformers patches OR official Docker (NVIDIA PyTorch 25.09 base, `TORCH_CUDA_ARCH_LIST="12.1"`): https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth, https://build.nvidia.com/spark/unsloth/instructions |
| Triton sm_121a ptxas | **WORKED AROUND locally** (uncommitted) via TRITON_PTXAS_PATH |
| tool_calls must be dicts not JSON strings (Gemma/Qwen3.5 templates) | **STILL REQUIRED** — and compounded by Gemma 4's `<|"|>` string-delimiter token (see §8) |

### DGX platform updates
- June 2026 release: NVIDIA Sync Cluster Assistant (3–4 Sparks, NCCL 2.30u1 3-system ring); OTA no longer auto-installed at setup.
- April 2026 release: DGX Dashboard release highlights; air-gapped deployment, USB repo, cloud-init.
- dgx-spark-playbooks (https://github.com/nvidia/dgx-spark-playbooks, 46+ playbooks): Unsloth, NeMo, LLaMA Factory, PyTorch FSDP+LoRA (2-Spark distributed up to 70B); inference: vLLM/TRT-LLM/SGLang/Ollama/NIM. No dedicated Gemma 4 or TRL playbook yet. Hermes Agent playbook: https://build.nvidia.com/spark/hermes-agent.
- Release notes: https://docs.nvidia.com/dgx/dgx-spark/release-notes.html

---

## 6. NVIDIA Ecosystem Updates (Apr–Jun 2026)

### NeMo Data Designer
- **v0.6.1 current** (PyPI 2026-06-01; Apache-2.0; Python ≥3.10). Repo: https://github.com/NVIDIA-NeMo/DataDesigner. Docs moved to Fern: https://docs.nvidia.com/nemo/datadesigner/ (llms.txt index: https://docs.nvidia.com/nemo/datadesigner/latest/llms.txt; MCP server: https://docs.nvidia.com/nemo/datadesigner/_mcp/server).
- Timeline: v0.5.7 (Apr 17, `skip_when` conditional generation), v0.5.8 (Apr 27, CVE fixes), v0.5.9 (Apr 28, VLM), **v0.6.0 (May 13: async engine DEFAULT, plugins production-grade + catalog CLI, deterministic config fingerprinting, run resumption)**, v0.6.1 (Jun 1: workflow chaining, AIMD ramp, audio/video context, reasoning-token tracking).
- **CLI exists**: `data-designer config providers|models|list|reset` (config in `~/.data-designer/`, override via `DATA_DESIGNER_HOME`; "change default provider" deprecated per issue #589 — set `provider=` per ModelConfig) and `data-designer plugin list|search|info|install|installed|uninstall` (catalog: https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json). No CLI generate command — generation stays programmatic.
- Eleven column types incl. `LLMCodeColumnConfig` (Bash among 15+ languages), validators: `CodeValidatorParams(code_lang=dd.CodeLang.PYTHON)` runs **Ruff** (is_valid + 0–10 linter score); SQL via SQLFluff; Local Callable + Remote HTTP validators. Custom columns via `@dd.custom_column_generator` with **row explosion**.
- **Local endpoints first-class**: `dd.ModelProvider(name="ollama", endpoint="http://localhost:11434/v1/")` — no NIM/NVIDIA key needed; vLLM/TGI/TensorRT-LLM all supported via provider_type="openai". For the Spark: `http://192.168.50.173:11434/v1/`. (Discussion #113: https://github.com/NVIDIA-NeMo/DataDesigner/discussions/113.) azure/bedrock/vertex_ai provider types dropped.
- **`AgentRolloutSeedSource`** (https://docs.nvidia.com/nemo/datadesigner/concepts/agent-rollout-ingestion): formats `CLAUDE_CODE` (defaults to `~/.claude/projects`, *.jsonl — exactly BashGym's importer source), `CODEX`, `HERMES_AGENT`, `PI_CODING_AGENT`, `ATIF`. Normalizes to trace_id, source_kind, root_session_id, agent_id, **is_sidechain** (Claude's isSidechain — maps to BashGym's subagent tagging), cwd, project_path, git_branch, messages, message_count, tool_call_count, final_assistant_message, source_meta. `path=`/`file_pattern=`/`recursive=` overrides allow pointing at `data/gold_traces/`.
- **Agent Rollout Trace Distillation recipe** (https://docs.nvidia.com/nemo/datadesigner/recipes/trace-ingestion/agent-rollout-trace-distillation; script: https://github.com/NVIDIA-NeMo/DataDesigner/blob/main/docs/assets/recipes/trace_ingestion/agent_rollout_distillation.py): seed → `AgentRolloutTraceDigest` structured digest (user_goal, task_type, training_value high/med/low) → `AgentRolloutFinetuningRecord` (instruction, response, skill_tags, difficulty) → 4-rubric judge (groundedness, standalone_task, response_quality, faithfulness, 0–4) → flatten to sft_instruction/sft_response. Run: `uv run agent_rollout_distillation.py --format claude_code --num-records 32 --preview` (supports --shuffle, --partition-index/--num-partitions).
- DPO: no dedicated column type; documented pattern = two candidate columns (diff aliases/temps) + judge + ExpressionColumnConfig threshold assembly — **exactly what BashGym's coding_agent_dpo already does**.
- Distribution: PyPI `data-designer` 0.6.1; `nemo-microservices[data-designer]` SDK (migration: https://docs.nvidia.com/nemo/microservices/latest/data-designer/migration.html); NGC container nvidia/nemo-microservices/data-designer; hosted https://build.nvidia.com/nemo/data-designer.

### NeMo RL / Gym / Microservices
- **NeMo RL v0.6.0 (Apr 30)** (https://github.com/NVIDIA-NeMo/RL/releases): GDPO (multi-reward decoupled normalization), ProRLv2 (GRPO + DAPO dynamic sampling + decoupled clipping), **LoRA for GRPO and DPO first-class** (Megatron + DTensor V2), SGLang backend, Eagle3/MTP speculative rollouts, Muon optimizer, YaRN 256K, chunked linear CE.
- **NeMo Gym v0.3.0 (Jun 4)** (https://github.com/NVIDIA-NeMo/Gym, https://docs.nvidia.com/nemo/gym/latest/about/index.html): 70+ new environments (coding, competitive programming, SQL, Tau2), **Claude Code and Hermes harnesses out-of-box**, OpenEnv + Harbor integrations, VeRL integration, `ng_reward_profile`. Powered all RLVR data for Nemotron 3 Nano. v0.2.1 (Apr 15) PyPI compat. `pip install nemo-gym`.
- **NeMo Microservices 26.3.0** (https://docs.nvidia.com/nemo/microservices/latest/about/release-notes/current-release.html): `nemo-platform` CLI for single-command local deploy **without Kubernetes** (relevant to single-box Spark); Customizer LoRA/full-SFT/DPO/GRPO; SDK 1.1.0 adds Auditor + Data Designer microservices. Fine-tune docs: https://docs.nvidia.com/nemo/microservices/latest/fine-tune/index.html.

### TRL / vLLM / transformers
- vLLM transformers-v5 support merged Apr 15 (PR https://github.com/vllm-project/vllm/pull/30566, 201 commits, pins 5.5.3). Tracking: #30466, #38379, #39216 (the Apr 7 "0.19.0 pins <5 but Gemma 4 needs >=5.5" bug — fixed by 0.20.0).
- vLLM v0.20.0 (Apr 27): first transformers≥5 release; **CUDA 13.0.2 default wheels aligned with PyTorch 2.11**; Blackwell CUTLASS/SM120; Gemma 4 fast prefill + quantized MoE + Eagle3. v0.21.0 (May 15): transformers v4 deprecated. v0.22.0 (May 29)/v0.22.1 (Jun 5): Gemma 4 fixes, FlashInfer MoE + FP4 GEMM, Model Runner V2.
- TRL 1.x: v1.1.0 (Apr 12) DistillationTrainer; v1.2.0 (Apr 17) tool-calling traces; v1.3.0 (Apr 26) Qwen 3.6; v1.4.0 (May 9) chunked NLL (−50% VRAM); v1.5.0 (May 25, requires vLLM ≥0.18); **v1.6.0 (Jun 11) AsyncGRPO in separate CPU-only process, A2PO trainer**. https://github.com/huggingface/trl/releases
- llm-compressor Gemma 4 quantization still lags (pinned transformers ≤4.57.6): https://github.com/vllm-project/llm-compressor/issues/2562.

### Unsloth
- Full Gemma 4 support: E2B/E4B/12B/26B-A4B/31B, text/vision/audio/RL (https://unsloth.ai/docs/models/gemma-4/train). VRAM: E2B LoRA 8–10GB, E4B 17GB, **31B QLoRA 22GB, 26B-A4B LoRA >40GB** (both fit 128GB Spark). Fixed upstream: KV-shared layer cache IndexError (31B/26B-A4B), use_cache=False corruption (E2B/E4B), fp16 audio attention overflow.
- GB10/aarch64 issue #4867 still open (see §5). Unsloth Studio rapid releases: v0.1.39-beta (May 5, local API + self-healing tool calling) → v0.1.451-beta (Jun 10, Gemma 4 MTP); v0.1.43-beta (May 31) adds CUDA 13.3.

### Nemotron / NIM
- **Nemotron 3 Super** (Mar 11): hybrid Mamba-Transformer MoE, ~120B total/12B active, 1M context, agentic coding cookbook: https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/OpenScaffoldingResources/README.html. Announcement: https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/
- **Nemotron 3 Nano Omni** (Apr 28): 30B-A3B MoE with vision+audio encoders, 9x throughput; NIM + HF + OpenRouter. https://blogs.nvidia.com/blog/nemotron-3-nano-omni-multimodal-ai-agents/
- Nemotron 3 Nano (text, 31.6B/~3.6B active, 1M ctx) — RL-trained via NeMo Gym, has DGX Spark playbook. Family: https://research.nvidia.com/labs/nemotron/Nemotron-3/, hub: https://github.com/NVIDIA-NeMo/Nemotron. Nemotron 3 Ultra unreleased (H1 2026 target).

---

## 7. Hermes Agent Assessment

**Identification**: "Hermes agent" = **Nous Research Hermes Agent** (https://github.com/nousresearch/hermes-agent) — MIT, released Feb 25, 2026, ~180k stars, v0.16.0 (Jun 6, 2026), most-used agent on OpenRouter. NVIDIA's role is partnership: RTX AI Garage blog (https://blogs.nvidia.com/blog/rtx-ai-garage-hermes-agent-dgx-spark/), DGX Spark playbook (https://build.nvidia.com/spark/hermes-agent), NemoClaw-for-Hermes blueprint (https://developer.nvidia.com/blog/deploy-self-evolving-agents-for-faster-more-secure-research-with-a-hermes-agent-and-nvidia-nemoclaw/). Distinct from the Hermes 4 model family (e.g. https://huggingface.co/NousResearch/Hermes-4-70B).

**What it is**: self-improving agent harness — writes/refines its own `SKILL.md` skills, persistent memory (FTS5 + Honcho), short-lived isolated sub-agents designed for local 30B-class models, cron, MCP, browser automation, sandboxed execution. Python 3.11 + React/Ink TUI; `hermes` CLI, dashboard (Profile Builder shipped Jun 11: https://www.marktechpost.com/2026/06/11/nous-research-ships-hermes-agent-profile-builder-identity-model-skills-and-mcp-servers-in-one-dashboard-flow/), desktop app, multi-platform gateway. Docs: https://hermes-agent.nousresearch.com/docs/

**Model requirements** (https://hermes-agent.nousresearch.com/docs/integrations/providers): any OpenAI-compatible endpoint; Ollama zero-config (`http://localhost:11434/v1`), vLLM (`--enable-auto-tool-choice --tool-call-parser hermes`), SGLang, llama.cpp (`--jinja`), LM Studio; NIM/Nemotron cloud. Hard requirements: **native tool calling + ≥64k context** (`OLLAMA_CONTEXT_LENGTH=64000` — Ollama defaults to 4,096 on <24GB VRAM).

**Trace storage** (https://hermes-agent.nousresearch.com/docs/developer-guide/session-storage): SQLite at `~/.hermes/state.db` (sessions + full message history: role, content, tool_calls, tool_name, token_count, FTS5 index) + gateway JSONL at `~/.hermes/sessions/`. Session IDs `YYYYMMDD_HHMMSS_<8-hex>`.

**Fit with BashGym**:
1. **Student harness (strongest fit)** — completes the flywheel deployment leg: Train on DGX → GGUF → Ollama → Hermes pointed at localhost. Sub-agent containment + curated skills compensate for small-model weaknesses. Engineering caveats: 64k context, Hermes-style tool-call emission (vLLM/SGLang ship a `hermes` parser; the existing Gemma/Qwen tool_calls sanitization issue applies).
2. **Trace source** — new importer `bashgym/trace_capture/importers/hermes_history.py` reading `~/.hermes/state.db` (structured SQLite, easier than Claude's heterogeneous JSONL) and/or session JSONL. Note Data Designer v0.6.1's `AgentRolloutSeedSource(format=HERMES_AGENT)` already parses this format. Bonus: Hermes `SKILL.md` artifacts as AutoCurriculum seed material.
3. **Teacher (weak fit, but principled variant)** — run Hermes with a frontier model (OpenRouter/NIM Nemotron 3 Super) to produce **on-policy harness-matched teacher traces**: identical tool schema/prompts for teacher and student = clean distillation setup.

**NemoClaw blueprint** (https://build.nvidia.com/nvidia/nemoclaw-for-hermes-agent/nemoclawcard): Nemotron 3 Super 120B + Hermes + NVIDIA OpenShell sandbox + **NeMo Relay observability emitting Agent Trajectory Format (ATF) traces** (Arize Phoenix debugging). ATF is a ready-made NVIDIA-standard trace schema BashGym could ingest (Data Designer's `AgentRolloutFormat.ATIF` reads Harbor trajectory format). The blueprint's "self-evolving" loop is skill-file persistence only — BashGym supplies the missing weight-update half.

---

## 8. Best Practices Findings (June 2026 survey)

### Trace filtering & scoring
- **Verifier-gated rejection-sampling SFT is the baseline**: rule-based filters → execution/verification gating. (Scaling Agentic Verifier: https://arxiv.org/abs/2602.04254; RFT overview: https://www.emergentmind.com/topics/rejection-sampling-fine-tuning-rft-ad4c417c-416b-40b6-bf9a-4653b83ddcfb)
- **SWE-Next** (https://arxiv.org/pdf/2603.20691): 102,582 commit pairs from 3,971 repos → 2,308 self-verifying instances; quality beats trajectory volume.
- **SERA / soft verification counterpoint** (Ai2, https://allenai.org/blog/open-coding-agents): "high workflow fidelity matters more than precise correctness"; Soft-Verified Generation stores per-sample verification thresholds as metadata instead of hard-binarizing; SERA-32B hit 54.2% SWE-Bench Verified **SFT-only** ($400–$12k data cost). **Directly applicable to BashGym's gold/pending/failed buckets: store continuous scores, don't binarize at ingest.**
- Reward robustness hierarchy (https://zylos.ai/en/research/2026-04-10-rl-posttraining-tool-using-agents-grpo-async-rl/): rule-based outcome > trained generative RM (~98.8% in-domain) > LLM-judge (gameable). PRMs give step-level credit; trajectory-level uniform credit degrades past ~20 tool calls (AgentRM: https://arxiv.org/pdf/2502.18407).
- **Loss masking: train on assistant action tokens only — never tool outputs/observations** (SWE-World: https://arxiv.org/pdf/2602.03419; mandatory in multi-turn GRPO).

### SFT vs DPO vs GRPO
- Consensus pipeline: SFT cold-start → optional DPO/SimPO/KTO → GRPO/DAPO with verifiable rewards (https://llm-stats.com/blog/research/post-training-techniques-2026). SFT+RL restores up to 99% of OOD performance lost during SFT. Counterexamples both ways: DeepSWE RL-only (https://www.together.ai/blog/deepswe, Qwen3-32B → 42.2%/59% TTS) and SERA SFT-only.
- DPO: trajectory pairs from same-instruction success/failure rollouts (Agent Q: https://multion-research.s3.us-east-2.amazonaws.com/AgentQ.pdf; hierarchical: https://arxiv.org/pdf/2510.03253); KTO when only binary verifier signal exists; DPO's role shrinking in 2026 recipes.
- GRPO recipe defaults: group 8–64 rollouts, **DAPO mods now standard** (clip-higher, dynamic sampling, token-level loss norm, overlong shaping); trend removes KL entirely; **async RL mandatory at agent horizons** (sync wastes 60–80% GPU). Frameworks: verl, OpenRLHF, TRL (7B–70B single-machine), SkyRL-Agent (https://arxiv.org/pdf/2511.16108), VerlTool (https://arxiv.org/pdf/2509.01055), NeMo Gym. MURPHY multi-turn GRPO with execution feedback: https://arxiv.org/pdf/2511.07833; reward calibration: https://arxiv.org/pdf/2604.02869. Cost: from-scratch agent RL $10k–50k; TRL GRPO 7B–14B domain specialization ~10–50 GPU-hours. Unsloth single-GPU RL envs: https://unsloth.ai/blog/rl-environments
- Difficulty proxy: empirical solve-rate from k rollouts, keep 0 < rate < 1.

### Dedup / decontamination / curriculum
- **Decontamination now standard**: zero 13-gram overlap + <0.7 3-gram + cosine <0.85 vs benchmarks (daVinci-Dev: https://arxiv.org/pdf/2601.18418; SWE-Bench++: https://arxiv.org/pdf/2512.17419); variant 4-gram + MinHash LSH. **BashGym has no decontamination step.**
- Dedup trajectories on (task embedding, repo, tool-call sequence shape), not raw text (SemDeDup: https://openreview.net/forum?id=IRSesTQUtb); trajectory reduction: https://arxiv.org/pdf/2509.23586.
- Curriculum: 3-stage — single-tool/4–8k/binary reward → 2–5 calls/8–32k/mixed → 20+ calls/32–64k/outcome+judge; C-GRPO, TCOD (https://arxiv.org/pdf/2604.24005).

### Key papers
SWE-Gym (https://github.com/SWE-Gym/SWE-Gym, https://arxiv.org/pdf/2412.21139); SWE-Universe (https://arxiv.org/pdf/2602.02361); SWE-Hub (https://arxiv.org/pdf/2603.00575); Self-Play SWE-RL (https://arxiv.org/pdf/2512.18552); Hybrid-Gym (https://arxiv.org/pdf/2602.16819); PC Agent-E single-step augmentation beats full-trajectory distillation 300× cheaper (https://arxiv.org/pdf/2505.13909); SAGE-32B (https://arxiv.org/pdf/2601.04237); reinforced distillation (https://arxiv.org/pdf/2509.14257); Structured Agent Distillation (https://arxiv.org/pdf/2506.14728); agentic RL landscape survey (https://arxiv.org/pdf/2509.02547); KAT-Coder-Pro V1 73.4% SBV (https://llm-stats.com/benchmarks/swe-bench-verified-(agentic-coding)).

### Gemma 4 fine-tuning specifics
- 31B dense: QLoRA at ~22GB OK. **26B-A4B MoE: QLoRA NOT recommended (4-bit × MoE routing); use bf16/16-bit LoRA (>40GB — fits Spark; conveniently sidesteps GB10 bitsandbytes issue)**. Start rank 16, short context. Hyperparams (Unsloth): lr 2e-4 (→2e-5 long runs), r=8–32, lora_alpha ≥ r, dropout 0, adamw_8bit, GA=4. Expected loss 1–3.
- Chat templates: `gemma-4` vs `gemma-4-thinking` (thinking recommended for 26B/31B); set `enable_thinking`; strip leading `<bos>`; keep ≥75% reasoning-bearing examples; multi-turn history keeps only final visible answers. **Template mismatch after GGUF export = top deploy bug** (https://cloudinsight.cc/en/blog/gemma-4-fine-tuning).
- **Tool-call format is NOT OpenAI JSON**: six special tokens `<|tool>…<tool|>`, `<|tool_call>…<tool_call|>`, `<|tool_result>…<tool_result|>`; calls look like `<|tool_call>call:fn{arg:<|"|>value<|"|>}<tool_call|>`; **the `<|"|>` delimiter wraps ALL string literals** — training data emitting plain `"` breaks parsing. Sanitize tool_calls to structured dicts, let the template render delimiters (extends BashGym's existing tool_calls sanitization fix). Parallel calls degrade beyond ~3/turn. Verify Ollama/vLLM versions parse the new tokens (langchaingo gap: https://github.com/tmc/langchaingo/issues/1490). Docs: https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4, https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4, https://www.analyticsvidhya.com/blog/2026/04/gemma-4-tool-calling/

---

## 9. Gap Analysis

### 9.1 Known blockers — resolved upstream vs still blocked

| Blocker (from MEMORY.md, April 2026) | June 2026 status | Action |
|---|---|---|
| vLLM (<5) vs Unsloth (>=5.5) transformers conflict → separate venvs | **RESOLVED**: vLLM ≥0.20.0 on transformers 5.5.3 | Collapse to single venv on DGX: vLLM ≥0.22.1 + TRL ≥1.5 + Unsloth + transformers 5.5.x |
| TRL ≥0.24 GRPO requires vLLM → blocked by venv conflict | **RESOLVED as conflict** (vLLM still required, now compatible) | Consider re-enabling vLLM-backed GRPO generation; TRL 1.6.0 AsyncGRPO runs in separate CPU process |
| vLLM Gemma 4 = nightly cu129 wheels | **RESOLVED**: stable since 0.20.0, cu130 default | Drop nightly install; recipe page is stale |
| GB10 bitsandbytes 4-bit failure | **STILL BLOCKED** | Keep bf16 LoRA (also the correct choice for 26B-A4B MoE per best practices) |
| Unsloth on GB10/aarch64 (flash-attn3 no GB10 build) | **STILL BLOCKED** (issue #4867 open) | Use official Unsloth Docker on Spark, or keep the 3 manual patches; local trainer.py plain-transformers GRPO path remains valid insurance |
| Triton sm_121a ptxas | Worked around locally (uncommitted) | Commit the TRITON_PTXAS_PATH fix; guard Linux path check |
| tool_calls dict sanitization for Gemma/Qwen | Still required, now expanded | Extend to Gemma 4 `<|"|>` delimiter + thinking-template rules before any Gemma 4 export |

### 9.2 Repo components outdated relative to current tooling

1. **GRPO trainer architecture vs TRL 1.6 / NeMo RL 0.6**: local GRPO removed vLLM entirely (and remote commits 750e9d2/d4b8442 went back to Unsloth — divergent!). Current best practice (TRL 1.5+/1.6 with vLLM ≥0.20, AsyncGRPO, DAPO mods) makes both approaches dated. The grpo_use_vllm test field should become real config once vLLM is back in the venv. NeMo RL v0.6 LoRA-GRPO and GDPO are credible alternatives to the homegrown loop.
2. **requirements-training.txt floors are stale**: trl>=0.15.0 (current 1.6.0 — major-version API breaks), transformers>=4.36.0 (current 5.5.x), torch>=2.1.0 (current 2.11). Needs an audited bump, not floors that silently allow ancient installs.
3. **Data Designer integration targets v0.5.x**: the `ChatCompletionInferenceParams` migration is correct, but v0.6.x brings async-default engine, run resumption (fixes weakness #8 "no incremental/streaming generation" partially via resumable runs), `skip_when`, workflow chaining, and plugin CLI. Feature-detection (hasattr) approach should survive the bump; verify against 0.6.1.
4. **Hand-rolled trace importers vs `AgentRolloutSeedSource`**: Data Designer now natively ingests `~/.claude/projects` (CLAUDE_CODE format) with `is_sidechain` preserved — overlapping `claude_history.py` and the subagent tagging work. The trace-distillation recipe (digest → finetuning record → 4-rubric judge) is a maintained, NVIDIA-supported version of what `example_generator.py` + `quality_calculator.py` do by hand.
5. **No code-execution validation in synthetic pipelines** (weakness #6): Data Designer v0.6 validators (Ruff `CodeValidatorParams`, Local Callable, Remote HTTP) close part of this — a Remote validator pointing at BashGym's own Verifier/sandbox would close it fully.
6. **No decontamination step anywhere in the pipeline**: 2026 standard is 13-gram/3-gram/cosine gates vs SWE-bench et al. before export.
7. **Hard gold/failed binarization at ingest** contradicts SERA-style soft verification — quality_calculator already computes continuous scores; the pipeline just needs to carry them as metadata instead of bucketing.
8. **No observation masking guarantee**: SFT exports include tool outputs in messages; must verify loss masking trains on assistant action tokens only (standard per SWE-World).
9. **Cascade curriculum**: orders by repo/loss only; best practice adds tool-count/context-length staging and solve-rate banding (0 < rate < 1) — GRPO dynamic sampling (DAPO) would enforce this automatically.
10. **NeMo Gym v0.3 overlap**: its Claude Code/Hermes harnesses + 70+ coding envs mirror BashGym's Arena/Judge; worth evaluating as the RL environment backend instead of maintaining `gym_env.py` rollout plumbing.

### 9.3 What Data Designer CLI/0.6 adoption would change

- **CLI config** (`data-designer config providers/models`) replaces hand-rolled ProviderSpec env-key plumbing for local dev; YAML in `~/.data-designer/` becomes the single source of provider truth.
- **Plugin system** (production-grade in 0.6.0) is the sanctioned extension point — BashGym's custom columns/validators could ship as a `data-designer-bashgym` plugin instead of forked builders.
- **`AgentRolloutSeedSource(path="data/gold_traces/")`** could replace the session→segment→example front half of `ExampleGenerator` for SFT distillation, with `--partition-index/--num-partitions` sharding for the 3,140 gold traces.
- **Run resumption + deterministic config fingerprinting** directly serve SchemaResearcher's two-stage eval (fast 25-row previews via `preview()`, resumable full runs).
- **Ollama/vLLM endpoints on the Spark** (`http://192.168.50.173:11434/v1/`) mean generation jobs can run free against local models — relevant to GemmaJudge and to cost-free DPO candidate generation.

### 9.4 Where Hermes fits

- **Deployment (highest value)**: the flywheel's DEPLOY leg currently ends at "Set as Student in router." Hermes gives the student a real-world, skill-augmented harness on the Spark via Ollama — the NVIDIA-blessed configuration. Prereqs: 64k context (`OLLAMA_CONTEXT_LENGTH=64000`), hermes tool-call parser compatibility, Gemma 4 template correctness post-GGUF.
- **Trace capture (clear, cheap win)**: `hermes_history.py` importer over `~/.hermes/state.db` SQLite — simpler than the Claude JSONL importer; Data Designer's `HERMES_AGENT` format provides a second ingestion path. Closes the loop: student runs in Hermes → traces flow back → train again.
- **On-policy teacher (optional)**: Hermes + Nemotron 3 Super/Qwen 3.6-Plus via OpenRouter/NIM = harness-matched teacher traces for distillation (identical tool schema for teacher and student).
- **ATF/NeMo Relay (watch)**: adopting Agent Trajectory Format as an interchange schema aligns with NVIDIA tooling and Data Designer's ATIF reader; not urgent.

---

## 10. Prioritized Roadmap

### A. Repo changes (no approval needed)

1. **Reconcile branch divergence FIRST**: fetch origin; resolve the GRPO direction conflict — remote (750e9d2, d4b8442) reverted GRPO to Unsloth; local uncommitted work removes Unsloth for sm_121. Decide: keep local plain-transformers path (works around still-open Unsloth #4867) and supersede remote commits, or gate per-platform. Then rebase/merge.
2. **Commit the finished work** (after ruff+black, per lessons): trainer.py sm_121 fixes, designer_pipelines `ChatCompletionInferenceParams` migration, `bashgym/datasets/`, `scripts/`. Fix `tests/gym/test_grpo_script.py` by adding `grpo_use_vllm: bool = False` to `TrainerConfig` (or deleting the tests). Consolidate the 4 `train_gemma4_*.py` scripts into one parametrized script. Add `data-fixed/`, `data-pipeline-fixed/` to `.gitignore`.
3. **Unstub MOPD**: replace cascade_routes.py:244-248 placeholder with `await distill_cascade(mopd_config, callback=mopd_callback)`. Smallest change unlocking the full cascade→merge flow.
4. **Wire auto-trigger**: ThresholdMonitor.should_train() → POST /api/cascade/start in pipeline/orchestrator.py; set `auto_deploy_ollama` in CascadeConfig for post-cascade deployment; raise on >10% dataset validation errors (cascade_scheduler.py:681-692); add stage-completion checkpointing for cascade resumption.
5. **Hermes trace importer**: `bashgym/trace_capture/importers/hermes_history.py` reading `~/.hermes/state.db` + `~/.hermes/sessions/` JSONL.
6. **Data pipeline upgrades (best-practices alignment)**: carry continuous verification scores as metadata (stop hard-binarizing); add decontamination gate (13-gram zero / 3-gram <0.7 / cosine <0.85 vs SWE-bench) to export; wire decision-level DPO (data_factory.py:859-942) into the main factory; verify observation masking in SFT exports; extend tool_calls sanitization for Gemma 4 `<|"|>` delimiters and thinking template.
7. **Data Designer 0.6.1 adoption** (repo side): verify pipelines against 0.6.1, prototype `AgentRolloutSeedSource` + trace-distillation recipe against `data/gold_traces/`, add a Remote validator hitting BashGym's Verifier for code-execution validation.
8. **Bump requirements-training.txt** floors with an upgrade audit (trl 1.x API breaks, transformers 5.x) — plan-first per user preference.

### B. DGX environment changes (REQUIRE USER APPROVAL — present plan first)

1. **Single-venv consolidation**: new venv with transformers 5.5.x + vLLM ≥0.22.1 (cu130 stable wheels — drop nightly cu129) + TRL ≥1.5 (ideally 1.6.0) + Unsloth latest. Retires the separate-venvs workaround. Verify Gemma 4 fine-tune + vLLM serve in one env; keep old venvs until proven.
2. **Unsloth path decision**: either official Unsloth DGX Spark Docker (NVIDIA PyTorch 25.09, TORCH_CUDA_ARCH_LIST="12.1") or maintain the 3 manual transformers patches for issue #4867; document choice.
3. **Re-enable vLLM-backed GRPO generation** on the Spark once (1) lands; benchmark vs the plain-transformers loop; consider DAPO mods (clip-higher, dynamic sampling, no-KL) per 2026 GRPO recipes.
4. **Hermes Agent install** on the Spark: point at Ollama with `OLLAMA_CONTEXT_LENGTH=64000`; deploy current student model; confirm hermes tool-call parsing of fine-tuned Gemma/Qwen GGUFs (template-mismatch is the #1 deploy bug).
5. **Evaluate NeMo Gym v0.3** (`pip install nemo-gym`, isolated env) — Claude Code/Hermes harnesses + 70+ coding environments — as a candidate RL environment backend; and NeMo RL v0.6 LoRA-GRPO/GDPO as an alternative trainer. Evaluation only; no pipeline change without results.
6. **Optional**: NeMo Microservices 26.3.0 `nemo-platform` CLI (no-Kubernetes single-box deploy) if the Customizer/Auditor stack becomes interesting; Data Designer generation jobs pointed at Spark Ollama (`http://192.168.50.173:11434/v1/`) for zero-cost synthesis.

---

*Report compiled June 11, 2026 from repo state at a2ea717 + 4 research reports. All version numbers, URLs, and file:line references preserved from source reports.*
