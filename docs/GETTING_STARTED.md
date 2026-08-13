# Getting started

This guide installs BashGym, opens the experiment workspace, and shows the two
execution paths:

- **AutoResearch** for a baseline, candidate comparisons, and repeated
  experiments driven by an agent;
- **direct training** for one known model, dataset, and recipe.

## Install

Requirements:

- Python 3.10 or newer;
- Node.js 22 LTS or newer for the frontend;
- Git;
- a compatible training environment when you run a real fine-tune.

```bash
git clone https://github.com/GhostPeony/bashgym.git
cd bashgym

python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

python -m pip install -e .
python -m pip install -e ".[training]"  # only when this checkout runs training

cd frontend
npm ci
cd ..

cp .env.example .env
```

The default configuration selects no model and contains no credential. Add only
what the features you use require.

Verify the install:

```bash
bashgym --help
bashgym research --help
bashgym training --help
```

## Open BashGym

```bash
# Windows
.\dev.ps1

# macOS/Linux
./dev.sh
```

Open `http://localhost:5173`. Generated API documentation is available at
`http://localhost:8003/api/docs`.

## Bring a dataset

BashGym is not limited to coding traces. A training dataset can come from a
researcher-provided JSONL file, imported traces, generated examples, preference
pairs, or executable environment tasks.

For SFT, the common format is one JSON object per line with a `messages` array:

```json
{
  "messages": [
    { "role": "user", "content": "Solve the task." },
    { "role": "assistant", "content": "The verified response." }
  ]
}
```

DPO requires a prompt with preferred and rejected responses. GRPO/RLVR requires
prompts that can produce multiple attempts plus a reward or deterministic
verifier that distinguishes them. See
[Training data](TRAINING_DATA_GUIDE.md) and
[Training strategy](training/strategy-guide.md) before converting data between
methods.

Keep a fixed held-out split out of the training dataset. It is the basis for
the baseline and every candidate comparison.

## Run a direct training job

Use direct training when the model, dataset, and method are already known.
Inspect the supported request before submitting it:

```bash
bashgym training plan --strategy sft --data custom_jsonl --json
bashgym training start --help
```

Example request shape:

```bash
bashgym training start \
  --strategy sft \
  --model <trainable-model-id-or-path> \
  --dataset-path <train.jsonl> \
  --compute-target <registered-target> \
  --checkpoint-limit 1 \
  --artifact-retention adapter_only \
  --json
```

For an official experiment, also supply `--tracking-context` with the exact
project, experiment, model-version, dataset-version, environment, and source
identities. BashGym does not infer them from the current chat.

After the run:

```bash
bashgym training analyze --run-id <run-id> --models-dir <models-directory> --json
```

Training loss describes optimization. It does not establish that the candidate
is better. Run the declared held-out evaluation and compare it with the starting
model before selecting the result.

## Run an AutoResearch experiment

AutoResearch is the repeated path:

```text
baseline → failure analysis → one intervention → training → fixed evaluation
        → comparison → next intervention or stop → report
```

The agent uses its own goal, plan, editor, terminal, browser, and subagents for
failure analysis and hypothesis work. BashGym exposes the experiment actions:

```bash
bashgym research prepare --help
bashgym research onboard --help
bashgym research state --help
bashgym research wait --help
bashgym research start --help
bashgym research submit-iteration --help
bashgym research report --help
```

Start preparation with the selected workspace and a secret-store reference:

```bash
bashgym research prepare \
  --workspace-id <workspace> \
  --credential-ref <secret-store-key> \
  --json
```

Preparation discovers registered model, dataset, evaluator, training, and
execution choices. It resumes its setup session instead of making the agent ask
for the same choice again.

After the exact choices have been reviewed, preview and apply one onboarding
contract:

```bash
bashgym research onboard --contract <onboarding.json> --json
bashgym research onboard --contract <onboarding.json> --apply --json
```

The first command returns the deterministic preparation plan without applying
it. `--apply` resumes or completes model registration/acquisition on the chosen
execution target, installation activation, resident API and worker startup,
registry sync, guided setup, and creation of one `READY` campaign. It does not
run `research start`.

Read the `READY` state, present its exact contract, and wait for a later
explicit Start. After Start, keep reading state and wait by durable event cursor
while BashGym is working:

```bash
bashgym research state \
  --workspace-id <workspace> \
  --credential-ref <secret-store-key> \
  --campaign <campaign-id> \
  --json

bashgym research wait \
  --workspace-id <workspace> \
  --credential-ref <secret-store-key> \
  --campaign <campaign-id> \
  --after-cursor <cursor> \
  --json
```

`research state` contains the objective, current work, latest comparison,
budget, next action, and resume identity. `research wait` is a read-only
long-poll; its `next_cursor` lets the host agent continue after a change,
requested action, timeout, or terminal state without inventing a second session
record. The agent submits the baseline first, then one candidate at a time
against the current reference. The onboarding contract fields and exact
proposal, start, and report commands are in
[AutoResearch campaigns](training/autoresearch-campaign.md).

## Use an Agent Skill

```bash
# Codex
bashgym operator skills install --host codex
bashgym operator skills check --host codex

# Claude Code
bashgym operator skills install --host claude
bashgym operator skills check --host claude

# Project-local Agent Skills, including Cursor-compatible discovery
bashgym operator skills install --host agents
bashgym operator skills check --host agents
```

The skill teaches the host agent to read and mutate BashGym experiments. It does
not replace the host's native goal or planning system.

## Use traces when they are useful

Trace capture is one data source, not an onboarding requirement.

```bash
python -m bashgym.trace_capture.setup
python -m bashgym.trace_capture.setup import-all --days 60
```

Review imported sessions before generating examples. Successful traces can
provide SFT demonstrations; failed or contrasting attempts may provide
preference or failure-analysis data when they are labeled correctly.

## Next

- [Architecture](PLATFORM_OVERVIEW.md)
- [Training data](TRAINING_DATA_GUIDE.md)
- [Training strategy](training/strategy-guide.md)
- [AutoResearch campaigns](training/autoresearch-campaign.md)
- [Metrics runbook](training/metrics-runbook.md)
