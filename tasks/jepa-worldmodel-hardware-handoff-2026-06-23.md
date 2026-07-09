# JEPA World-Model + Hardware-Discovery — Handoff

Compiled: 2026-06-23. Audience: Codex agents continuing this work.

This document hands off two threads landed over 2026-06-22/23, both built **hardware-agnostic** (no specific GPU baked in; reference the discovery feature) and **TDD** (test interpreter: `"C:/Program Files/Python314/python.exe" -m pytest <paths> -q -o addopts=`). All new code is pure/offline-testable; the GPU-bound consumption is explicitly deferred and marked below.

> Design rule followed throughout: implement the **math/data/contract layer before the trainer backend**, exactly like `bashgym/gym/dppo.py` ("implements the math/telemetry layer before any trainer backend"). Do not bolt these onto the single-turn code-gen TRL `GRPOTrainer` script — they belong to the multi-turn terminal-rollout RL path.

---

## Thread A — JEPA world-model objectives (ECHO + RWML)

Background: deep research (2026-06-20) chose **ECHO** (token-space observation-prediction aux loss, arXiv:2605.24517) and **RWML** (embedding-space world-model reward, arXiv:2602.05842) over CWM-style mid-train (infeasible at 32B/5T-token scale). They live on the terminal-rollout path (`environments/rollout.py` → `eval/dppo_replay.py` → external DPPO/verl backend), **not** the code-gen GRPO script.

### Landed (pure layers + wiring, tested)

| File | Contract | Tests |
|---|---|---|
| `bashgym/gym/echo.py` | `EchoConfig` (λ=0.05); `EchoSegment(role, token_ids)`; `build_echo_masks(segments)` → `EchoMasks(action_mask, observation_mask, total_observation_tokens)`; `environment_prediction_loss(kept_logprobs, total_obs_tokens)` = `-(Σlogp)/Z`, Z=\|O\|; `combine_echo_loss(grpo, env, λ)` | `tests/gym/test_echo.py` |
| `bashgym/gym/rwml.py` | `RWMLConfig`; `cosine_similarity`; `world_model_reward(pred, actual, embed_fn, *, distance_threshold)` (binary); `extract_transitions(steps, instruction, history_window)` → `WorldModelTransition`; `group_relative_advantages`; `keep_easy_sample(...,draw=...)` | `tests/gym/test_rwml.py` |
| `bashgym/gym/trainer.py` | `TrainerConfig` fields `echo_enabled/echo_aux_lambda/rwml_*`; `world_model_settings()`; validation in `_validate_world_model_settings` | `tests/gym/test_world_model_config.py` |
| `bashgym/providers/base.py` | non-abstract `async def embed(texts, *, model=None)` + `supports_embeddings` (default raises/False — existing providers unaffected) | `tests/providers/test_embeddings.py` |
| `bashgym/providers/embeddings.py` (new) | `parse_embeddings_response` (OpenAI), `parse_ollama_embeddings_response`, **`make_embed_fn(provider, model)`** → sync `Callable[[str], Sequence[float]]` = exactly rwml's `EmbedFn` | `tests/providers/test_embeddings.py` |
| `bashgym/providers/openai_compatible.py`, `ollama.py` | concrete `embed()` (httpx); model is a param, never hardcoded | `tests/providers/test_embeddings.py` |
| `bashgym/eval/dppo_replay.py` | `build_dppo_replay_records(..., include_world_model=False, history_window=...)` — additive `world_model` key, schema stays `bashgym.dppo_replay.v1` | `tests/eval/test_dppo_replay.py` |

### The `world_model` replay key (what the backend consumes)

Present only when `include_world_model=True`. Per record:
```json
"world_model": {
  "rwml_transitions": [
    {"instruction": "...", "prior": [["act","state"], ...], "action": "...", "next_state": "stdout\nstderr"}
  ],
  "echo": {
    "segments": [{"role":"action","text":"<cmd>"}, {"role":"observation","text":"<output>"}, ...],
    "n_action_chars": 0, "n_observation_chars": 0,
    "note": "<ECHO_DOWNSTREAM_NOTE: token masks are built downstream by the trainer's tokenizer>"
  }
}
```
- `CommandObservation` → `{action: command, state: stdout+stderr}`; empty-command observations skipped.
- `echo.segments` use `ACTION_ROLE`/`OBSERVATION_ROLE` from `echo.py`. **No token ids fabricated** — the backend tokenizes these text spans and calls `build_echo_masks` itself; char counts let it pre-size.

### Trainer-consumption layer — LANDED 2026-06-23 (the loss/reward/wiring the backend calls)

| File | Contract | Tests |
|---|---|---|
| `bashgym/gym/echo_trainer.py` (new, torch-lazy) | `environment_prediction_loss_from_logits(logits, input_ids, observation_mask, total_observation_tokens)` (next-token shift, `-(Σlogp)/Z`, Z=\|O\|) + `echo_augmented_loss(base_loss, …, aux_lambda)`. Module docstring spells out the `compute_loss` integration contract. | `tests/gym/test_echo_trainer.py` (torch, run on the **3.12 CUDA env** — `"C:/Users/Cade/AppData/Local/Programs/Python/Python312/python.exe"`; verified the loss == ln(vocab) on uniform logits + grad flows) |
| `bashgym/gym/rwml.py` | `build_world_model_reward_fn(embed_fn, *, distance_threshold)` → `reward_fn(predicted_states, actual_states) -> list[float]` (the pre-RL GRPO reward) | `tests/gym/test_rwml.py` |
| `bashgym/gym/dppo_launcher.py` | `DPPOSmokeLaunchConfig` gained `echo_enabled/echo_aux_lambda/rwml_*`; launch env exports `BASHGYM_DPPO_ECHO_*`/`BASHGYM_DPPO_RWML_*`; `to_dict()["world_model"]`; script exports all env vars | `tests/gym/test_dppo_launcher.py` |

So the **backend entrypoint** (verl/SkyRL/open-instruct) now receives the full world-model contract: the enriched replay (`include_world_model=True` → `world_model` key), the ECHO/RWML config via `BASHGYM_DPPO_*` env vars, and reference implementations to call — `echo_augmented_loss` inside its `compute_loss`, `build_world_model_reward_fn(make_embed_fn(ollama_provider, model))` for the RWML pre-RL stage.

### Still NOT done (the GPU/backend boundary — genuinely cannot be done here)

1. **The backend's training entrypoint wiring + a real run.** Inside verl/SkyRL/open-instruct's `compute_loss`, call `echo_augmented_loss(...)` with the tokenized observation mask (`build_echo_masks` with the model tokenizer); add the RWML pre-RL GRPO stage using `build_world_model_reward_fn`. Then a real multi-turn run to prove convergence. Gated on: the external backend being **installed/checked out** (none is, here) + DGX for real models (install is classifier-gated — hand Cade a `! ssh ...` one-liner). A tiny 1.5B local smoke on the 3080 Ti is possible once a backend is checked out.
2. Wire `include_world_model=True` into the rollout-export path when a world-model run is configured (currently opt-in at the `build_dppo_replay_records` call site).
3. RWML easy-sample filtering driver (`keep_easy_sample` exists; needs the K-attempt pass-rate computation around it).

> Why no TRL `GRPOTrainer` subclass shipped: TRL GRPO computes its own per-token logprobs internally and does not surface interleaved observation-token logits, so a tidy subclass would be misleading. The honest seam is the reference loss/reward helpers above, applied inside the external multi-turn backend — which is the path BashGym already targets via `dppo_launcher`.

### Do not
- Do not make `embed`/`supports_embeddings` `@abstractmethod` (breaks Anthropic/NIM).
- Do not hardcode an embedding model id — pass it from the live catalog.
- Do not fabricate ECHO token masks in `dppo_replay` (no tokenizer there).

---

## Thread B — Hardware discovery + HF model catalog + remote (ponyo) preflight

Goal: replace the hardcoded VRAM ladder + static base-model list with discovery, and fix remote unified-memory devices.

### Landed (tested)

| File | What | Tests |
|---|---|---|
| `bashgym/models/hardware_estimator.py` | pure VRAM math: `estimate_vram_gb`, `max_params_billions`, `model_fits`, `guess_params_billions_from_id`, `recommend_for_budget(vram, candidates)` → per-regime capacity + per-candidate `can_infer/qlora/lora/full` | `tests/models/test_hardware_estimator.py` |
| `bashgym/models/hf_catalog.py` | `discover_training_models()` (live `huggingface_hub.list_models`), `fetch_model_size()` (`get_safetensors_metadata`, no download), pure `normalize_training_models`/`total_and_dominant_dtype`/`params_billions` | `tests/models/test_hf_catalog.py` |
| `bashgym/gym/remote_trainer.py` | pure `parse_nvidia_smi_gpus` (GB10 `[N/A]`→None), `parse_meminfo_gb`, `remote_compute_budget_gb` (unified-memory→RAM); `PreflightResult.{to_dict,capabilities}` + new fields; `preflight_check(require_unsloth=True)` | `tests/gym/test_remote_preflight.py`, `test_remote_trainer.py` |
| `bashgym/api/device_routes.py` | preflight now calls `require_unsloth=False`, persists `result.capabilities()`, returns `result.to_dict()` | (route) |
| `bashgym/api/system_info.py` | `get_model_recommendations(vram_gb=None, *, unified_memory=False)` adds `regime_capacities` + `unified_memory` (additive) | `tests/api/test_model_recommendations.py` |
| `bashgym/api/schemas.py` | `ModelRecommendations` gained optional `regime_capacities`, `unified_memory` | — |
| `bashgym/api/routes.py` | `GET /api/system/recommendations?device_id=` (targets a registered device's `effective_vram_gb`); **new** `GET /api/models/discover?limit=&device_id=` (live catalog + per-model fit) | — |
| `frontend/src/services/api.ts` | `systemInfoApi.getRecommendations(deviceId?)`, `systemInfoApi.discoverModels(deviceId?)`, types `DiscoveredModel`/`ModelRecommendations.regime_capacities` | (typecheck) |
| `frontend/src/components/common/BaseModelSelect.tsx` | renders static `BASE_MODEL_GROUPS` + a live "Discovered (live from HuggingFace)" optgroup; discovery failure is non-fatal | (smoke) |

### Why Qwen3.6 was missing
Base-model catalog is hardcoded in two twinned spots: `frontend/src/components/common/baseModels.ts` and `bashgym/providers/detector.py:RECOMMENDED_TRAINING_MODELS`. `detector.get_available_models()` live-discovers *inference* providers but copies a *static* training list. `hf_catalog.discover_training_models()` is the live replacement; `/api/models/discover` exposes it; `BaseModelSelect` now merges it.

### NOT done / follow-ups
1. **Frontend Playwright smoke** of `BaseModelSelect` live group + a `SystemInfoPanel` view of `regime_capacities` (typecheck+lint pass; visual smoke pending).
2. Optionally retire/auto-refresh `detector.py:RECOMMENDED_TRAINING_MODELS` + `baseModels.ts` from `discover_training_models()` so there's one source of truth.
3. Pre-existing hardcodes to clean up (separate task): `trainer.py` `device_map="cuda:0"`, `batch_size=1 # 12GB` comments → discovery-driven.
4. `huggingface_hub` installed is 1.3.3 in the 3.14 test env while `requirements.txt` pins `>=1.10.0`; APIs used (`list_models`, `get_safetensors_metadata`) are present, but verify the training env matches the pin.

### Estimator regime formulas (planning numbers; exact = HF `accelerate estimate-memory`)
- inference: `params × bytes(dtype) × 1.2`
- qlora: `params × 1.0` · lora: `params × 3.0` · full_finetune: `params × 16`

---

## Verification snapshot (2026-06-23)
- `tests/models/ tests/api/test_model_recommendations.py tests/gym/test_remote_preflight.py tests/gym/test_remote_trainer.py tests/gym/test_echo.py tests/gym/test_rwml.py tests/gym/test_world_model_config.py tests/eval/test_dppo_replay.py tests/providers/test_embeddings.py` → **97 passed**
- `tests/providers/ tests/gym/test_rwml.py` → **185 passed**
- ruff clean across all 14 touched backend files; frontend `npm run typecheck`/`lint` clean.
