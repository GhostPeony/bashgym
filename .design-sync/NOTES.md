# design-sync notes — BashGym

## Approach (read before re-syncing)
- **Off-script, hand-built bundle** — NOT the converter. BashGym's `frontend/` is an Electron **app**, not a React component library, so `package-build.mjs` would find zero components. Synced as a **CSS-class + token design system** instead.
- **Scope:** design primitives only (tokens + the Botanical Brutalist utility classes), not the 101 app components.
- **Project:** `BashGym — Botanical Brutalism` (`c74970f4-345c-4303-9452-4aceb7213c6f`).

## How the bundle is produced
- Source of truth: `frontend/src/styles/globals.css`.
- `ds-bundle/_ds_bundle.css` = `globals.css` with the 3 `@tailwind` lines stripped (`tail -n +4`). It's self-contained raw CSS (tokens + components), no `@apply`/`@screen`, no local `url()`.
- `ds-bundle/styles.css` = remote Google-Fonts `@import` (Playfair / Inter / JetBrains Mono) + `@import './_ds_bundle.css'`.
- `ds-bundle/_ds_bundle.js` = minimal stub (`window.BashGymDS = {}`); no importable components by design.
- 9 hand-authored preview cards under `components/{Foundations,Components}/<Name>/<Name>.html`, each with a `@dsCard` first line.
- README.md + `.design-sync/conventions.md` enumerate the class vocabulary (the design agent's reference).

## Re-sync risks / watch-list
- `_ds_bundle.css` is mechanically derived from `globals.css` — if `globals.css` ever adds `@apply`, `@screen`, `@variants`, or local `url(...)` refs, the raw `tail -n +4` strip is no longer safe; re-check before re-uploading.
- **No `_ds_sync.json` anchor** (off-script) → every re-sync re-verifies/re-uploads everything. That's fine, just not incremental.
- Fonts load via a **remote** Google-Fonts `@import` (families load at runtime); no woff2 shipped.
- **No machine render-check was run** (no chromium/playwright in the env). Verified instead that all 26 classes used in the cards exist in `_ds_bundle.css`. Visual confirmation is via the live project.
- New primitives added to `globals.css` won't appear until a matching preview card is authored.

## Magic Patterns track (separate, parallel)
- A Magic Patterns React component library IS a good fit for the converter's high-fidelity path. Route: export the MP lib to a repo → `/design-sync` that repo (real importable components). Feed BashGym's tokens into Magic Patterns first so its components are on-brand. Could be its own Claude Design project or merged here later.
