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
