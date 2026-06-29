# Lessons

Patterns captured from user corrections. Review at session start.

## 2026-06-17 — Do the exact action requested; no shortcuts

**Correction:** When the user directly asks for a specific action (e.g. "web search
the latest models"), do exactly that action, fully — do not substitute a shortcut
(reading a config file, the codebase, or a cached doc) and do not narrow it to your
own assumption (e.g. searching "Qwen3.6" when asked to survey *all* new open models).

**Why it matters:** Substituting a shortcut produces a stale or biased answer and
ignores the user's explicit method. Searching for a model I'd already guessed defeats
the purpose of surveying the field.

**How to apply:**
- If the user names a tool or method (web search, run X, read Y), use that exact tool/method.
- Keep searches broad and unbiased when asked to "survey" or find "all" of something —
  don't inject a specific name you assumed.
- When directions are not 100% clear, ASK a clarifying question instead of guessing.
- Related: [[feedback_do_exact_action_requested]], [[feedback_no_hardcoded_models]].

## 2026-06-24 — Stop PR mechanics when the user pivots back to product work

**Correction:** After fixing CI and opening/pushing the follow-up branch, the user
asked why we were back in GitHub when the intended focus was upgrades.

**Why it matters:** Shipping hygiene is useful, but it can become accidental
context switching if the user's current intent is to keep improving the product.

**How to apply:**
- Once the requested PR/CI unblock is handled, explicitly park GitHub work unless
  the user asks to merge, watch checks, or open another PR.
- Treat a user pivot like "we were supposed to be doing upgrades" as the new top
  priority immediately.
- Summarize any GitHub state briefly, then return to local implementation and
  verification.

## 2026-06-24 — Hover polish must verify depth, not just layout movement

**Correction:** The user pointed out that the dashboard still felt like every
container was pressing down on hover. The first pass moved hover translations to
active states, but left shadow compression and global card hover depth changes in
place.

**Why it matters:** A box-shadow change can look like physical movement even when
the bounding box does not move. Verifying only element rectangles misses the
actual visual bug the user is reporting.

**How to apply:**
- Audit hover interactions for transform, box-shadow, pseudo-elements, and
  Tailwind shadow utilities.
- Keep hover feedback to color, border, or brightness changes; reserve shadow
  compression and translation for `:active`.
- Use real browser hover checks (`locator.hover()` plus computed styles or
  screenshots), not only synthetic event dispatch or bounding-box comparisons.

## 2026-06-24 — Ask for plan approval before choosing upgrade scope

**Correction:** The user clarified that "upgrades" meant the training upgrades
around JEPA/ECHO/RWML, TMax terminal RL, DPPO, and related training/eval
effectiveness work, not a dashboard/UI pass.

**Why it matters:** A broad backlog has multiple plausible next moves. Choosing
one without checking can burn time on the wrong layer even if the work is useful.

**How to apply:**
- For broad upgrade requests, present the proposed implementation plan before
  making product edits.
- Name the intended subsystem explicitly: training loop, backend smoke, eval
  gates, docs/education, or UI.
- Wait for confirmation when the user asks to approve the plan beforehand.

## 2026-06-29 — Keep BashGym's original flywheel central

**Correction:** The user emphasized that BashGym began as a loop of coding,
trace extraction, dataset generation, and training methods. New RLHF, JEPA,
source-library, cloud, metric, and education work should be segmented as
supporting flywheels instead of blurring the core product concept.

**Why it matters:** If every capability is described as one broad training
platform, users and reviewers lose the simple product story: real coding work
becomes traces, traces become datasets, datasets train models, and better models
create new traces. Supporting loops should make that clearer, not compete with
it.

**How to apply:**
- Lead with the core trace-to-training loop before describing advanced methods.
- Name which flywheel a feature belongs to: trace/data, preference/reward,
  terminal RL, JEPA diagnostics, source library, AutoResearch, compute, or
  education.
- Keep world-model and reward metrics framed by their role in the loop, not as
  standalone proof of model quality.

## 2026-06-29 — Audit dirty files before committing session leftovers

**Correction:** The user asked to go through dirty files because they came from
the earlier session and still needed to be handled.

**Why it matters:** Session dirt can mix durable product context with local
tooling state. Committing it blindly can leak machine-specific paths or editor
permissions, while ignoring it can lose useful handoff material.

**How to apply:**
- Inspect every dirty and untracked file before staging.
- Promote durable scratchpad content into tracked `tasks/` or `docs/` files.
- Keep local permission/config changes out of product commits unless they are
  explicitly part of the repo contract.

## 2026-06-29 — Translate research docs into shipped behavior

**Correction:** The user asked how much of the documentation work was actually
translating into BashGym code.

**Why it matters:** Research docs and action boards are useful only if they drive
software, tests, or explicit human decisions. Otherwise the repo can start to
look complete while the product still lacks the promised workflows.

**How to apply:**
- For each roadmap pass, pick at least one item that becomes backend/API/CLI/UI
  behavior unless the item is genuinely blocked by GX10/runtime access or a
  human policy decision.
- When updating docs, mark whether the item is code-backed, runtime-blocked, or
  awaiting a human choice.
- Keep verification attached to the software surface, not only to the document
  that describes it.
