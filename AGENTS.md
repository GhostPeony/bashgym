# BashGym Agent Route

Use the repository skills as the executable operating instructions. For durable
AutoResearch, read
`assistant/workspace/skills/bashgym-operator/SKILL.md` and the sibling
`training/SKILL.md` before changing campaign state or launching compute.

In a source clone, install and verify the reviewed public bundle for Codex:

```bash
bashgym operator skills install --host codex
bashgym operator skills check --host codex
```

This installs skills; it does not launch or register an agent. Start from
`campaign setup-context`, resume registered setup state, and ask only for a
missing or ambiguous model, data, evaluation, or compute choice. The initial
request authorizes preparation only. After creating a `READY` campaign, present
its exact contract and stop for a later explicit Start confirmation.

Keep real training on an explicitly registered local/private SSH target unless
the operator chooses another installed adapter. Never substitute a model,
download one implicitly, expose private transport details, or treat smoke
evidence as a quality result.
