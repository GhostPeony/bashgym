# BashGym — Botanical Brutalism (design-agent conventions)

BashGym ships a **CSS-class + token** design system, not an importable React component library — there are no components on `window.BashGymDS`. Build with plain markup styled by the classes below; everything is defined in `styles.css` (it `@import`s the fonts, tokens, and `_ds_bundle.css`).

**Setup:** link `styles.css` once. For dark mode, put `class="dark"` on a root ancestor — all tokens shift.

**Tokens (use `var(--*)`, never hardcode):** `--bg-primary/-secondary/-card/-terminal`, `--text-primary/-secondary/-muted`, `--accent/-light/-dark`, `--status-success/-warning/-error`, `--border-color`, `--border-weight` (2px), `--shadow`/`--shadow-sm`/`--shadow-soft`/`--shadow-soft-lg`, `--radius` (8px, rounded). The accent is HSL-driven by `--accent-hue` (default 258, wisteria) — one variable re-tints everything.

**Type:** DM Serif Display (serif) for headings/brand (`<h1>`–`<h6>`, `.font-brand`); Inter (sans) for body/UI; JetBrains Mono for code/labels/tags (`<code>`, `.font-code`).

**Classes:** `.card` / `.card-elevated` / `.card-accent`; `.btn` + `.btn-primary`/`-secondary`/`-ghost`/`-icon`/`-cta` (press on `:active`); `.input`; `.tag` (inner `<span>` to counter-skew); `.status-dot` + `.status-success`/`-warning`/`-error`; `.prose-brutal` wrapper for brutalist `<table>`s; `.terminal-chrome`/`-header`/`-dot(-red/-yellow/-green)`/`-prompt`; plus `.menu-item`/`-active`, `.progress-bar`/`.progress-fill`, `.section-divider`.

**Idiom (neobrutalist):** rounded 8px corners with bold 2px borders + hard offset shadows (`--shadow`), plus soft layered shadows (`--shadow-soft`) and a faint grain texture (`.grain`) for depth. Monospace for function, serif for brand; warm parchment with charcoal ink in light, warm deep charcoal in dark (never pure white or pure black). Tags are rare functional labels — never decoration or data lists.

**Where the truth lives:** read `styles.css` and its `_ds_bundle.css` import for exact values; the preview cards under `components/` show each primitive in use.
