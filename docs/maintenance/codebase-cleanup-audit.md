# Codebase Cleanup Audit

**Scope:** confirmed cleanup candidates reviewed during the repository quality
refresh. This record is intentionally conservative: no code is removed without
reference proof.

## Removed frontend components

| Candidate | Evidence | Decision | Verification |
| --- | --- | --- | --- |
| `frontend/src/components/autoresearch/CampaignAgentControl.tsx` | Repository-wide import/reference search found only the component's own source. `frontend/src/components/autoresearch/index.ts` exports only `AutoResearchControlRoom`. The Control Room source test explicitly asserts that its primary journey contains no `CampaignAgentControl` or `campaignAgent` UI. | Removed as an unimported, unsupported UI component. The campaign-agent host/model modules remain because they have active importers and focused tests. | Frontend lint, typecheck, and control-room tests. |
| `frontend/src/components/autoresearch/ResearchNewsPanel.tsx` | Repository-wide import/reference search found only the component's own source; it is not exported from the AutoResearch component barrel or rendered by the Control Room. | Removed as an unimported, unsupported UI component. | Frontend lint and typecheck. |

## Retained legacy compatibility surface

The following remains deliberately hidden compatibility code, not the durable
AutoResearch campaign product:

| Surface | Evidence | Decision |
| --- | --- | --- |
| `/api/autoresearch/*` and `BASHGYM_ENABLE_LEGACY_AUTORESEARCH` | `bashgym/api/routes.py` imports the legacy router and includes it only when the flag is `1`, `true`, or `yes`. `tests/api/test_autoresearch_routes.py` verifies the routes are 404 by default and available when enabled. | Retain behind the feature gate. |
| `bashgym/gym/autoresearch.py` | Imported by `bashgym/api/autoresearch_routes.py`; legacy API tests exercise its router behavior. | Retain as feature-gated compatibility implementation. |
| `DataRecipeSearchSpace` and `EnvironmentRecipeSearchSpace` | Imported and instantiated by the legacy router, lazily exported through `bashgym/gym/__init__.py`, and covered by dedicated `tests/gym` modules. | Retain as feature-gated compatibility implementation. |
| SchemaResearcher and TraceResearcher flows | The legacy router exposes their start/status/control endpoints; `tests/api/test_autoresearch_routes.py` covers schema research, while `TraceResearcher` is imported and instantiated by the router. | Retain as feature-gated compatibility implementation. |

## Guardrail

No other deletions are authorized by this audit. Any future deletion must first
document production, public API/CLI, documentation, and test reference checks,
then add focused verification for the changed behavior.

## 2026-08-09 architecture and duplication pass

### Reference architecture

The pass used the framework maintainers' own guidance as the baseline:

- FastAPI's [Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
  guidance keeps the application entry point small, groups endpoints in
  `APIRouter` modules, and moves reusable dependencies outside routers.
- React's [state-structure guidance](https://react.dev/learn/choosing-the-state-structure)
  recommends avoiding redundant and duplicate state.
- Electron's [process model](https://www.electronjs.org/docs/latest/tutorial/process-model)
  keeps privileged operations in the main/preload boundary rather than the
  renderer.

The current codebase largely follows those boundaries: most backend domains
have dedicated router modules, frontend session data is normalized through the
`stores/*Resources.ts` layer, and production renderer code has no direct Node or
Electron imports. The main remaining concentration points are the legacy
`bashgym/api/routes.py` application module, `frontend/src/services/api.ts`, and
several large workflow components. Splitting those files mechanically would
create broad regression risk, so this pass did not change them without a
feature-scoped reason.

### Changes made

| Area | Evidence | Decision |
| --- | --- | --- |
| Model deployment boundary | `bashgym/gym/trainer.py` imported Ollama deployment from `bashgym/api/models_routes.py`, making the training domain depend on FastAPI. | Moved the operation to `bashgym/models/deployment.py`; both the API adapter and trainer now depend on the model domain. Added focused success, missing-file, and temporary-file-cleanup tests. |
| Preference validators | DPO and reward validators duplicated text extraction, metadata normalization, strict-level selection, and JSON/JSONL loading. | Extracted private primitives to `bashgym/preferences/_validation.py` while retaining domain-specific rules and error messages. Added JSON-container coverage for both public validators. |
| Hugging Face client imports | Repository-wide reference search and Vulture found three imported exception types and fallback aliases with no consumers. | Removed the unused imports and aliases; optional `huggingface_hub` availability behavior is unchanged. |
| Ollama temporary files | The deployment path removed its temporary Modelfile only after `subprocess.run` returned, leaking it when Ollama was missing or timed out. | Centralized cleanup in `finally` and covered the missing-executable path. |

### Conservative retain decisions

- A repository-wide JSCPD scan reported 1.15% duplicated lines. Most large
  matches are parallel provider/importer implementations, generated training
  scripts, or similar UI markup with different behavior. They were retained
  because a shared abstraction would couple independent workflows for little
  benefit.
- Knip was run without a repository-specific entrypoint configuration and
  therefore marked Electron entry modules, dynamically loaded native
  dependencies, tests, and public barrels as unused. None were removed from
  that heuristic alone.
- Vulture's remaining high-confidence findings are callback/protocol parameters
  or documented compatibility inputs. Renaming or removing public inputs does
  not reduce runtime code and could break callers, so they remain.
- Legacy orchestrator-to-WebSocket imports remain a known compatibility-layer
  inversion. Removing them belongs with the already documented orchestrator
  retirement decision, not a behavior-preserving cleanup pass.
