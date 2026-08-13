# BashGym development guide

## What the system does

BashGym runs an agent-guided model-improvement loop:

1. establish a baseline on a fixed evaluation suite;
2. inspect task-level failures and metrics;
3. propose one controlled change to data, training, reward, evaluation, or code;
4. build data when required and train a candidate;
5. evaluate the candidate on the same suite;
6. compare it with the current reference and keep or discard it;
7. repeat until a stop rule is met, then export the experiment report.

Codex, Cursor, Claude Code, or another compatible agent supplies the scientific
judgment. BashGym supplies composable actions, execution, experiment history,
comparisons, recovery, and the observation UI. Use the host's native goals,
plans, tools, and subagents; do not add a second planner inside BashGym.

## Executable architecture

The durable AutoResearch path is:

```text
agent goal and tools
  -> research CLI or MCP tools
  -> campaign API
  -> experiment repository
  -> CampaignWorker
  -> registered stage adapter
  -> data build / training / fixed evaluation
  -> CampaignEvaluationProjector
  -> KEEP or DISCARD
  -> next agent action or report
```

Use these modules as the authority:

| Responsibility                              | Implementation                                     |
| ------------------------------------------- | -------------------------------------------------- |
| Agent-facing research actions               | `bashgym/mcp/campaign_server.py`, `bashgym/cli.py` |
| Campaign API                                | `bashgym/api/campaign_routes.py`                   |
| Baseline, candidate, comparison, stop rules | `bashgym/campaigns/autoresearch.py`                |
| Stage selection                             | `bashgym/campaigns/runtime.py`                     |
| Leasing, launch, recovery                   | `bashgym/campaigns/worker.py`, `worker_service.py` |
| Registered SSH stage execution              | `bashgym/campaigns/remote.py`                      |
| Evaluation projection                       | `bashgym/campaigns/autoresearch_evidence.py`       |
| Mechanical reconciliation                   | `bashgym/campaigns/autoresearch_loop.py`           |
| Compact agent state                         | `bashgym/campaigns/agent_brief.py`                 |

The agent currently proposes the hypothesis and next intervention. The loop
coordinator handles mechanical work such as ingesting results, retrying bounded
failures, and enforcing stop rules; it is not an experiment proposer.

Keep these boundaries explicit:

- `bashgym/gym/trainer.py` is the direct-training path, not the durable campaign
  executor.
- fake executors and smoke templates prove wiring, not model improvement.
- standalone holdout, pass@k, regression, and release-gate utilities are not all
  part of the campaign KEEP/DISCARD decision yet.
- the canvas and experiment view render campaign state; they do not choose the
  next experiment.
- named models and training runs belong in examples and case studies, not in
  the generic architecture.

## Agent operating instructions

For AutoResearch, read
`assistant/workspace/skills/bashgym-operator/SKILL.md` and the sibling training
skill before mutating campaign state or launching compute.

Install the reviewed Claude Code bundle from a source clone:

```bash
bashgym operator skills install --host claude
bashgym operator skills check --host claude
```

Begin preparation with `research prepare` and resume any existing setup
record. Preparation and compute launch are separate operations. After a
campaign reaches `READY`, show its exact experiment contract and wait for an
explicit Start. Once started, the agent may iterate autonomously within the
selected model, data, evaluation, execution, budget, and stop conditions.

## Development

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\python -m pytest
cd frontend
npm install
npm run typecheck
npm test
```

Use the repository virtual environment for Python commands. Prefer focused tests
while iterating and run the relevant broader suite before completion.

Code standards:

- Python 3.10+ with type hints, Black, and Ruff.
- TypeScript strict mode; reusable state belongs in stores rather than
  fetch-on-mount component state.
- Keep model, dataset, evaluation, run, attempt, and artifact identities
  explicit across API and worker boundaries.
- Treat secrets, raw datasets, model weights, and transport paths according to
  the selected execution adapter; do not expose them in public projections.
- Make unsupported paths fail clearly instead of silently simulating success.
- Update tests and docs when an executable contract changes.

## Documentation rules

Public explanations start with the experiment loop. Use plain technical
language, not slogans or launch-status narration. Do not copy deployment-specific
hostnames, hardware, filesystem paths, or operator topology into public files.
Describe only behavior proven by code and tests; label smoke, compatibility, and
incomplete integrations accurately.

The primary public references are:

- `README.md`
- `docs/GETTING_STARTED.md`
- `docs/PLATFORM_OVERVIEW.md`
- `docs/TRAINING_DATA_GUIDE.md`
- `docs/training/strategy-guide.md`
- `docs/training/autoresearch-campaign.md`
