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
