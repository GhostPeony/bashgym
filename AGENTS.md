# BashGym Agent Route

## Start from the experiment loop

BashGym lets an agent run repeated model experiments:

1. evaluate the starting model on a fixed suite;
2. inspect failures and form one testable hypothesis;
3. change data, a training recipe, a reward, an evaluator, or approved code;
4. train one candidate;
5. evaluate it on the same suite and compare it with the current reference;
6. keep or discard the result, then iterate or report.

Explain and operate that loop before discussing setup, lifecycle state,
persistence, workers, execution targets, or UI. The canvas is an observation
surface for the same experiment record; it is not a separate research system.

Derive architecture claims from executable modules and tests. Distinguish the
durable campaign path from direct trainers, smoke tests, compatibility routes,
and planned integrations. Never turn a named model or deployment into the
platform architecture. Never place operator hostnames, hardware, paths, or
topology in tracked docs, UI copy, demos, diagrams, or fixtures.

## Use the agent platform

Keep scientific judgment in the host agent's native goal, plan, editor,
terminal, browser, and subagent tools. Do not create a second BashGym planner.
Use the repository skills as executable instructions. Before changing campaign
state or launching compute, read
`assistant/workspace/skills/bashgym-operator/SKILL.md` and the sibling
`training/SKILL.md`.

In a source clone, install and verify the reviewed public bundle for Codex:

```bash
bashgym operator skills install --host codex
bashgym operator skills check --host codex
```

This installs skills; it does not launch or register an agent. Start from
`research prepare`, resume registered setup state, and ask only for a
missing or ambiguous model, data, evaluation, or compute choice. The initial
request authorizes preparation only. After creating a `READY` campaign, present
its exact contract and stop for a later explicit Start confirmation.

Run training only through the execution adapter selected in the campaign.
Never substitute a model, download one implicitly, expose transport details, or
treat smoke evidence as a quality result.
