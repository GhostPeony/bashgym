# Botanical Brutalist Full Redesign

> Design document for overhauling the GhostGym frontend from Apple/NVIDIA-inspired to Botanical Brutalism.

**Date:** 2026-02-12
**Scope:** Full redesign — every component, every surface
**Palette:** Wisteria (#9B8EC4) with dynamic hue shifting

---

## 1. Design Philosophy

Botanical Brutalism fuses two opposing forces:

- **Brutalism** — structural honesty. Hard borders, offset shadows, monospace type, grid layouts, bold geometric shapes. The structure is the aesthetic.
- **Botanical** — organic warmth. Colors extracted from nature — parchment, wisteria, moss, stone. The harshness of brutalism tempered by nature's palette.

The tension is the point. A card has a hard 2px border and a geometric offset shadow, but the background is warm cream and the accent feels alive. A terminal block has developer-grade monospace text, but the prompt color is wisteria instead of neon green.

### Core Values

- **Structural clarity** over decorative polish
- **Warmth** over coldness — brutalism's edge without its hostility
- **Typography as identity** — serif for brand, mono for function, sans for content
- **Color from nature** — palettes should feel organic, never sterile
- **Bold geometry** — triangles, trapezoids, bold curves. Not afraid of expressive shapes
- **Honest interfaces** — components look like what they are

---

## 2. Color System — HSL-Based Dynamic Accents

The accent color is defined in HSL so a single hue slider can rotate the entire accent family in real-time. Changing `--accent-hue` via JavaScript instantly shifts every accent-colored element — buttons, tags, links, highlights, chart colors.

### Light Mode (`:root`)

```css
:root {
  /* Dynamic accent — hue can be shifted 0-360 */
  --accent-hue: 258;
  --accent-sat: 30%;
  --accent-lgt: 66%;

  --accent: hsl(var(--accent-hue), var(--accent-sat), var(--accent-lgt));
  --accent-light: hsl(var(--accent-hue), 35%, 80%);
  --accent-dark: hsl(var(--accent-hue), 30%, 47%);

  /* Botanical surfaces */
  --bg-primary: #F5F0EB;         /* Warm parchment */
  --bg-card: #FFFFFF;            /* Card/surface white */
  --bg-terminal: #1E1E1E;       /* Terminal dark */

  /* Text */
  --text-primary: #1B2040;       /* Deep navy */
  --text-secondary: #5A607A;     /* Muted navy */
  --text-muted: #8A8FA6;         /* Light muted */

  /* Brutalist structure */
  --border-color: var(--text-primary);
  --border: 2px solid var(--border-color);
  --shadow: 4px 4px 0px var(--border-color);
  --shadow-sm: 3px 3px 0px var(--border-color);
  --radius: 2px;                 /* Max 4px in light mode */

  /* Status */
  --status-success: #4A8C5C;     /* Botanical green */
  --status-warning: #C4923A;     /* Warm amber */
  --status-error: #B84A4A;       /* Muted red */
  --status-info: hsl(var(--accent-hue), var(--accent-sat), var(--accent-lgt));

  /* Charts — accent-derived */
  --chart-1: hsl(var(--accent-hue), var(--accent-sat), var(--accent-lgt));
  --chart-2: hsl(calc(var(--accent-hue) + 60), 25%, 55%);
  --chart-3: hsl(calc(var(--accent-hue) + 120), 25%, 55%);
  --chart-4: hsl(calc(var(--accent-hue) + 180), 25%, 55%);
  --chart-5: hsl(calc(var(--accent-hue) + 240), 25%, 55%);
}
```

### Dark Mode (`.dark`)

A mood shift, not a redesign. Same family, different hour.

```css
.dark {
  /* Accent warms up and desaturates slightly */
  --accent-sat: 25%;
  --accent-lgt: 70%;

  /* Deep warm backgrounds — rich soil, not void */
  --bg-primary: #0a0a0b;
  --bg-card: #111113;
  --bg-terminal: #0a0a0b;

  /* Text */
  --text-primary: #E8E4E0;       /* Warm off-white */
  --text-secondary: #9A9AA0;
  --text-muted: #6A6A72;

  /* Structure relaxes */
  --border-color: rgba(255, 255, 255, 0.1);
  --border: 1px solid var(--border-color);
  --shadow: 0 0 20px hsla(var(--accent-hue), 20%, 60%, 0.15);
  --shadow-sm: 0 0 12px hsla(var(--accent-hue), 20%, 60%, 0.1);
  --radius: 8px;                 /* Relaxes to 8-16px */

  /* Status — slightly desaturated for dark bg */
  --status-success: #5CAA6E;
  --status-warning: #D4A24A;
  --status-error: #C85A5A;
}
```

### What Changes Between Modes

| Property | Light | Dark |
|----------|-------|------|
| Borders | 2px solid, hard navy | 1px solid, subtle rgba |
| Shadows | Geometric offset, zero blur | Soft glow halos |
| Border radius | 0-4px | 8-16px |
| Backgrounds | Warm cream/white | Deep warm near-black |
| Accent | Full saturation | Slightly desaturated, warmer |

### What Stays the Same

- Three-font system
- Component structure
- Hover interactions (press-in, adapted per mode)
- Hue slider works identically
- Tag badge system

---

## 3. Typography Stack

Three fonts, three voices. Replaces SF Pro Display and SF Mono entirely.

```css
/* Google Fonts load */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
```

| Voice | Font | Weight | Use |
|-------|------|--------|-----|
| **Botanical** (brand) | Cormorant Garamond, serif | 300-400 | App title, page headers, section titles, card titles, large numerals |
| **Content** (body) | Inter, sans-serif | 400-700 | Navigation, labels, descriptions, body text, form content |
| **Structural** (code) | JetBrains Mono, monospace | 400-700 | Tags, badges, code blocks, terminal text, stat labels, buttons |

### Tag/Badge Treatment

All section tags (TRAINING, TRACES, FACTORY, MODELS, etc.):
- Font: JetBrains Mono
- Weight: 600
- Transform: uppercase
- Letter-spacing: 0.1-0.15em
- Background: `--accent-light`
- Border: `2px solid var(--border-color)`
- Transform: `skewX(-6deg)` (text counter-skewed back)

---

## 4. Structural Tokens

### Borders

- Light mode: `2px solid var(--text-primary)` — always. Softer variant: `2px solid`.
- Dark mode: `1px solid rgba(255,255,255,0.1)` — structure is there, just quieter.
- Never above `4px` border-radius in light mode. `8-16px` in dark mode.

### Shadows

- Light mode: `4px 4px 0px var(--text-primary)` — geometric, zero blur. Smaller: `3px 3px 0px`.
- Dark mode: `0 0 20px hsla(accent, 0.15)` — soft glow halos. No offset.
- **Never** use soft drop shadows (`box-shadow: 0 4px 6px rgba(...)`) in either mode.

### Hover Interaction

- Light mode: `transform: translate(2px, 2px)` + shadow shrinks to `2px 2px 0px`. Element "presses in."
- Dark mode: `transform: translate(1px, 1px)` + glow intensity reduces. Same press-in feel.
- Transition: `0.15s ease` for all interactions.

### Spacing

- Container max-width: `1200px`
- Section vertical padding: `80-100px` (for overlay views)
- Component gaps: `16px`, `24px`, `32px`
- Content line-height: `1.6-1.7`

---

## 5. Bold Geometry System

Not just rectangles. Triangles, trapezoids, and bold curves add expressive energy.

### Button Tiers

**Primary action buttons — Trapezoidal:**
```css
.btn-primary {
  clip-path: polygon(8% 0%, 100% 0%, 92% 100%, 0% 100%);
  background: var(--accent);
  border: var(--border);
  box-shadow: var(--shadow-sm);
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 12px 32px;
}

.btn-primary:hover {
  transform: translate(2px, 2px);
  box-shadow: 1px 1px 0px var(--border-color);
}
```

**Icon buttons — Circular with hard border:**
```css
.btn-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: var(--border);
  box-shadow: var(--shadow-sm);
  display: flex;
  align-items: center;
  justify-content: center;
}
```

**Directional / CTA buttons — Triangle-tipped:**
```css
.btn-cta {
  position: relative;
  padding-right: 40px;
  border: var(--border);
  box-shadow: var(--shadow-sm);
}

.btn-cta::after {
  content: '';
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  border-left: 10px solid var(--text-primary);
  border-top: 6px solid transparent;
  border-bottom: 6px solid transparent;
}
```

**Secondary buttons — Bordered rectangle:**
```css
.btn-secondary {
  background: var(--bg-card);
  border: var(--border);
  box-shadow: var(--shadow-sm);
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
}
```

### Triangular Elements

**Card corner accent:**
```css
.card::before {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  border-style: solid;
  border-width: 0 32px 32px 0;
  border-color: transparent var(--accent) transparent transparent;
}
```

**Navigation active indicator — Triangle pointer:**
```css
.menu-item-active::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  border-top: 6px solid transparent;
  border-bottom: 6px solid transparent;
  border-left: 8px solid var(--accent);
}
```

**Progress bar triangular end cap:**
```css
.progress-fill::after {
  content: '';
  position: absolute;
  right: -8px;
  top: 0;
  border-left: 8px solid var(--accent);
  border-top: 50% solid transparent;
  border-bottom: 50% solid transparent;
}
```

### Section Dividers — Zigzag/Sawtooth

```css
.section-divider {
  height: 12px;
  background: repeating-linear-gradient(
    90deg,
    transparent 0px,
    transparent 8px,
    var(--border-color) 8px,
    var(--border-color) 9px
  );
  clip-path: polygon(
    0% 100%, 2% 0%, 4% 100%, 6% 0%, 8% 100%, 10% 0%,
    /* ... repeating sawtooth pattern */
    100% 100%
  );
}
```

### Skewed Tag Badges

```css
.tag {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  padding: 4px 14px;
  background: var(--accent-light);
  border: 2px solid var(--border-color);
  transform: skewX(-6deg);
}

.tag > span {
  display: inline-block;
  transform: skewX(6deg); /* Counter-skew text */
}
```

---

## 6. Component-by-Component Overhaul

### Layout Shell

**NavigationBar:**
- Fixed top bar, `border-bottom: var(--border)`
- Logo: Cormorant Garamond serif, `/BashGym` with accent-colored slash
- Right-side icon buttons: circular with hard borders
- Background: solid `--bg-card`, no blur/transparency
- Breadcrumbs: JetBrains Mono

**Sidebar:**
- Hard right `border-right: var(--border)`
- Section headers: JetBrains Mono uppercase with letter-spacing
- Menu items: Inter, with triangular active indicator on left edge
- Status badges: small bordered monospace badges
- Dividers between sections: `border-top` lines

**StatusBar:**
- Bottom bar with `border-top: var(--border)`
- Connection status: small bordered badge with status dot
- Monospace text throughout

### Cards

All card components (ModelCard, StatCard, AchievementCard, feature cards):

```
┌────────────────────────────────▲┐
│                              ╱  │  <- triangular corner accent
│  Card Title (Garamond)         │
│  Body text (Inter)             │
│                                │
│  [TAG]  [TAG]  (skewed mono)  │
│                                │
│  [Action Button ▶]            │
└────────────────────────────────┘
  └── 4px offset shadow
```

- Border: `2px solid var(--text-primary)`
- Shadow: `4px 4px 0px var(--text-primary)`
- Corner accent: CSS triangle in `--accent` color, top-right
- Hover: translate(2px, 2px), shadow shrinks
- Dark mode: border becomes subtle, shadow becomes glow, radius relaxes to 8px

### Buttons (Summary)

| Tier | Shape | Use Case |
|------|-------|----------|
| Primary | Trapezoidal (`clip-path`) | Start Training, Save, Confirm |
| Secondary | Bordered rectangle | Cancel, Back, alternative actions |
| Icon | Circle with hard border | Settings, theme toggle, add terminal |
| CTA/Directional | Rectangle + triangle arrow | "Next Step >", "View Details >" |
| Ghost | No border, just text | Tertiary/inline actions |

### Inputs & Selects

- Hard `2px` border, `--bg-card` background (cream tint)
- No focus ring glow — focus state thickens border to `3px` and colors it accent
- Placeholder text in `--text-muted`
- Border-radius: `2px` light, `8px` dark
- Font: Inter for input text

### Modals

- Hard border, offset shadow, cream background
- No rounded corners (light mode)
- Overlay: `rgba(27, 32, 64, 0.5)` — warm navy tint, not pure black
- Modal title: Cormorant Garamond
- Close button: circular icon button, top-right

### Terminal Blocks

- Dark background `#1E1E1E`, hard `2px` border
- macOS window dots: red/yellow/green circles **with** `2px` borders around each
- Terminal title: JetBrains Mono in the header bar
- Prompt character: accent color (shifts with hue slider)
- Monospace throughout, no changes to xterm.js internals

---

## 7. Dashboard-Specific Designs

### Training Dashboard

- **Metrics grid:** Row of hard-bordered stat blocks. Large Cormorant Garamond number + JetBrains Mono label underneath.
- **Loss curve:** Recharts reskinned — grid lines use `--border-color`, stroke uses dynamic accent, tooltip is a bordered card with offset shadow.
- **Epoch progress:** Triangular end-cap progress bar.
- **Training logs:** Terminal block treatment — dark bg, bordered chrome, macOS dots, monospace text, prompt in accent color.
- **Controls:** Play/pause/stop as circular icon buttons with hard borders.

### Data Factory

- **Seeds panel:** Grid of bordered seed cards with triangular corner accents.
- **Tags:** Skewed monospace badges on each seed.
- **Generator progress:** Triangular progress bar.
- **Settings:** Hard-bordered fieldsets, Cormorant Garamond section headers, Inter body.
- **Code preview:** Terminal block treatment.

### Trace Browser

- **Session cards:** Bordered rectangles, tool-call count in bold JetBrains Mono stat block.
- **Quality indicators:** Triangular — green triangle up (gold), red triangle down (failed), neutral dash (pending).
- **Expand/collapse:** Triangular chevrons.

### Model Browser

- **ModelCards:** Full brutalist card with triangular corner bookmark.
- **Comparison view:** Side-by-side bordered columns with zigzag divider.
- **Lineage tree:** Thick curved connector lines with triangular arrowheads.

### Home Screen

- **Flywheel:** Bold curved arcs with triangular stage markers, accent-colored, slow rotation.
- **Space cards:** Trapezoidal CTA buttons.
- **Stats:** Large Cormorant Garamond numerals with mono labels.
- **Tutorial checklist:** Triangle checkmarks instead of standard checkboxes.

---

## 8. Accent Hue Slider — Global Color Changer

### Store

New `useAccentStore` (Zustand, persisted to localStorage):

```typescript
interface AccentState {
  accentHue: number        // 0-360, default 258 (wisteria)
  setAccentHue: (hue: number) => void
  randomizeHue: () => void
  resetHue: () => void
}
```

On every `setAccentHue`, update:
```typescript
document.documentElement.style.setProperty('--accent-hue', String(hue))
```

### UI — Settings Modal > Appearance Section

```
┌─────────────────────────────────────────────────┐
│ APPEARANCE                    [skewed tag badge] │
├─────────────────────────────────────────────────┤
│                                                 │
│  Theme                                          │
│  ┌────────┐ ┌────────┐ ┌────────┐              │
│  │ Light  │ │  Dark  │ │ System │              │
│  └────────┘ └────────┘ └────────┘              │
│  (bordered buttons, active = accent fill)      │
│                                                 │
│  Accent Color                                   │
│  ┌─────────────────────────────────────────┐    │
│  │ (🎲) ═══════════◆══════════════ (↺)    │    │
│  │      [rainbow gradient slider track]    │    │
│  └─────────────────────────────────────────┘    │
│                                                 │
│  Presets:                                       │
│  ┌──────────┐ ┌──────┐ ┌──────┐ ┌─────────┐   │
│  │ Wisteria │ │ Rose │ │ Moss │ │Marigold │   │
│  │   258    │ │ 350  │ │ 140  │ │   35    │   │
│  └──────────┘ └──────┘ └──────┘ └─────────┘   │
│  ┌──────────┐ ┌───────────┐                    │
│  │ Lavender │ │ Teal Leaf │                    │
│  │   270    │ │    175    │                    │
│  └──────────┘ └───────────┘                    │
│                                                 │
│  Live Preview:                                  │
│  [■ accent] [■ light] [■ dark]                 │
│  (bordered swatches, update in real-time)      │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Slider Details

- **Track:** Hard-bordered rectangle filled with rainbow gradient:
  `linear-gradient(to right, hsl(0,30%,66%), hsl(60,30%,66%), hsl(120,30%,66%), hsl(180,30%,66%), hsl(240,30%,66%), hsl(300,30%,66%), hsl(360,30%,66%))`
- **Thumb:** Diamond shape (rotated 45deg square) — brutalist and distinctive
- **Dice button:** Circular, hard border, rolls random 0-360
- **Reset button:** Circular, hard border, snaps to 258
- **Presets:** Bordered chips with color fill + botanical name in mono. Click to jump slider.
- **Swatches:** Three bordered rectangles showing current accent, accent-light, accent-dark.
- Persisted via `localStorage` key `bashgym-accent-hue`

### Preset Values

| Name | Hue | Botanical Inspiration |
|------|-----|----------------------|
| Wisteria | 258 | Wisteria vine (default) |
| Rose | 350 | Garden peony |
| Moss | 140 | Forest floor |
| Marigold | 35 | Calendula |
| Lavender | 270 | Lavender field |
| Teal Leaf | 175 | Eucalyptus |

---

## 9. What Gets Removed

Everything from the current Apple/NVIDIA design that conflicts:

- **Fonts:** SF Pro Display, SF Mono — replaced by Cormorant Garamond, Inter, JetBrains Mono
- **Colors:** NVIDIA green (#76B900), Apple blue (#0066CC), cyan (#00A6FF) — replaced by HSL accent system
- **Shadows:** All `box-shadow` with blur (`shadow-card`, `shadow-elevated`, `shadow-subtle`) — replaced by offset/glow
- **Effects:** All `backdrop-blur`, `glass` class, all `glow-*` classes — removed
- **Radius:** All `rounded-xl`, `rounded-2xl`, `rounded-3xl` — replaced by `rounded-none` (light) / `rounded-lg` (dark)
- **Animations:** `animate-pulse` glow, `glow` keyframes — removed or replaced with structural animations
- **Hover patterns:** `hover:opacity-90`, `hover:bg-*` — replaced by press-in translate

---

## 10. Implementation Strategy

### Phase 1: Foundation (Theme Layer)
1. Replace `globals.css` with botanical brutalist CSS custom properties
2. Update `tailwind.config.js` — new font stack, new color tokens, new shadows, new radius values
3. Load Google Fonts (Cormorant Garamond, Inter, JetBrains Mono)
4. Create `useAccentStore` in stores/
5. Extend `themeStore` dark mode to use botanical dark tokens

### Phase 2: Core Components
6. Rework `Button.tsx` — trapezoidal primary, circular icon, directional CTA, secondary bordered
7. Rework `Modal.tsx` — hard border, offset shadow, warm overlay
8. Rework cards pattern (shared CSS) — border, shadow, corner accent, hover
9. Create tag/badge component — skewed monospace
10. Create section divider component — zigzag/sawtooth
11. Create progress bar component — triangular end cap

### Phase 3: Layout Shell
12. Rework `NavigationBar.tsx` — serif logo, circular icon buttons, no blur
13. Rework `Sidebar.tsx` — triangle active indicators, mono section headers, bordered badges
14. Rework `StatusBar.tsx` — bordered badges, monospace
15. Rework `MainLayout.tsx` — cream background, structural borders

### Phase 4: Dashboard Views
16. `HomeScreen.tsx` — flywheel with bold geometry, Garamond stats, trapezoidal CTAs
17. `TrainingDashboard.tsx` — stat blocks, reskinned charts, terminal logs, triangle progress
18. `FactoryDashboard.tsx` — seed cards, skewed tags, terminal code blocks
19. `TraceBrowser.tsx` — bordered session cards, triangular quality indicators
20. `ModelBrowser.tsx` + `ModelCard.tsx` — brutalist cards, corner bookmarks
21. `RouterDashboard.tsx`, `EvaluatorDashboard.tsx`, `GuardrailsDashboard.tsx`, `ProfilerDashboard.tsx`
22. `HFDashboard.tsx` + sub-components

### Phase 5: Settings & Polish
23. `SettingsModal.tsx` — add Appearance section with accent hue slider
24. `OnboardingModal.tsx` — botanical brutalist welcome
25. `KeyboardShortcutsModal.tsx` — brutalist table styling
26. `FlywheelVisualization.tsx` + `FlywheelMini.tsx` — bold arcs, triangular markers
27. Achievement cards, tutorial checklist, tooltips

### Phase 6: Dark Mode Verification
28. Test every component in dark mode — verify border/shadow/radius adaptation
29. Verify accent hue slider works in both modes
30. Verify chart colors derive correctly from dynamic accent

---

## 11. Files Changed

| Category | Files | Count |
|----------|-------|-------|
| **Theme/Config** | `globals.css`, `tailwind.config.js`, `themeStore.ts` | 3 |
| **New Store** | `accentStore.ts` | 1 |
| **Layout** | `NavigationBar.tsx`, `Sidebar.tsx`, `StatusBar.tsx`, `MainLayout.tsx` | 4 |
| **Common** | `Button.tsx`, `Modal.tsx`, `SettingsModal.tsx`, `KeyboardShortcutsModal.tsx` | 4 |
| **Home** | `HomeScreen.tsx`, `AchievementSummary.tsx`, `TutorialChecklist.tsx`, `TutorialTooltip.tsx` | 4 |
| **Terminal** | `TerminalGrid.tsx`, `TerminalPane.tsx`, `PreviewPane.tsx`, `BrowserPane.tsx`, `MasterControlPanel.tsx`, `CanvasView.tsx`, `CollapsibleSection.tsx`, `ToolBreadcrumbs.tsx`, `FileDropZone.tsx` | 9 |
| **Training** | `TrainingDashboard.tsx`, `LossCurve.tsx`, `EpochProgress.tsx`, `MetricsGrid.tsx`, `TrainingLogs.tsx`, `SystemInfoPanel.tsx` | 6 |
| **Factory** | `FactoryDashboard.tsx`, `SyntheticGenerator.tsx`, `SeedEditor.tsx`, `SeedsPanel.tsx`, `SettingsPanel.tsx`, `TagEditor.tsx`, `codeHighlight.tsx` | 7 |
| **Models** | `ModelBrowser.tsx`, `ModelCard.tsx`, `ModelProfile.tsx`, `ModelComparison.tsx`, `ModelTrends.tsx`, `LineageTree.tsx` | 6 |
| **Other Views** | `TraceBrowser.tsx`, `RouterDashboard.tsx`, `EvaluatorDashboard.tsx`, `GuardrailsDashboard.tsx`, `ProfilerDashboard.tsx`, `IntegrationDashboard.tsx` | 6 |
| **HuggingFace** | `HFDashboard.tsx`, `CloudTraining.tsx`, `SpaceManager.tsx`, `DatasetBrowser.tsx`, `HFStatus.tsx` | 5 |
| **Flywheel** | `FlywheelVisualization.tsx`, `FlywheelMini.tsx` | 2 |
| **Achievements** | `AchievementsView.tsx`, `AchievementCard.tsx`, `StatCard.tsx` | 3 |
| **Files** | `FileBrowser.tsx`, `FileTreeItem.tsx` | 2 |
| **Settings** | `ModelsSection.tsx`, `HooksSection.tsx` | 2 |
| **Onboarding** | `OnboardingModal.tsx` | 1 |
| **Total** | | **~65 files** |

---

## 12. Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| Clip-path buttons may clip content | Test with long text, add adequate padding |
| Skewed tags may misalign in flex layouts | Counter-skew inner text, test in all containers |
| HSL `calc()` for chart colors has browser limits | Fallback to JS-computed hex values if needed |
| Cormorant Garamond may render poorly at small sizes | Only use at 18px+ for headlines/titles |
| Dark mode radius change may cause layout shifts | Use CSS transitions on border-radius |
| 65-file changeset is large | Phase the work, verify each phase before proceeding |
