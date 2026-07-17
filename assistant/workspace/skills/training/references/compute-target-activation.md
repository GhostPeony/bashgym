# Compute Target Activation

Read this immediately before activating compute. A method recipe says **what to
train**; this reference says **where and how it actually runs**. Never report a
plan, queued request, or target label as an active job without an executable
submission receipt.

## Select one execution lane

| Lane | Executable surface | Durable job identity | Artifact authority |
|---|---|---|---|
| Same device | BashGym `training start` API/CLI | BashGym `run_id` | BashGym model/run directory; optional Hub push |
| Private SSH device | BashGym device preflight + `training start` with `ssh:<device_id>` | BashGym `run_id` plus remote PID/evidence | BashGym downloads the selected remote artifacts |
| NeMo Customizer | `/api/training/start` with `use_nemo_customizer: true` | BashGym `run_id` plus Customizer job id | Customizer result ingested by the BashGym run |
| Local NeMo RL | Registered private-compute campaign recipe (integration in progress) | BashGym attempt ID plus NeMo RL run/checkpoint identity | BashGym sealed artifacts and experiment ledger |
| Hugging Face Jobs | `hf jobs uv run` or an installed HF Jobs agent tool | Hugging Face job id | Hub model/dataset repo plus Trackio/log evidence |
| Managed fine-tune provider | `/api/training/managed/submit` | Provider job id | Provider model id; ingest comparison/report evidence separately |
| SkyPilot/dstack template | `bashgym compute launch --dry-run` | None | None; this is a plan only |

`compute_target` is operational for direct BashGym launches:

- `local` selects same-device execution.
- `ssh:<device_id>` automatically enables `use_remote_ssh` and selects that
  registered device.
- legacy `private` selects the default registered SSH device and is normalized
  to `ssh:remote`; new requests should use an explicit device id.
- `cloud:nemo-customizer` automatically enables `use_nemo_customizer`.
- legacy `cloud:nemo` and `use_nemo_gym` inputs remain compatibility aliases for
  Customizer only; they do not activate NeMo Gym or NeMo RL.
- `cloud`, `hf-jobs`, `managed:<provider>`, SkyPilot, and dstack must not be sent
  to `/api/training/start` as labels that pretend to activate a backend. Use the
  corresponding surface above.

## Common preflight contract

Before spending compute, verify and record:

1. exact model revision and license/access;
2. approved dataset revision, schema, sample inspection, split boundaries, and digest;
3. method config, seed, checkpoint/retention policy, stop rule, and eval gate;
4. target GPU/RAM/disk fit and the software/runtime versions;
5. credential **presence and verified identity** without printing secret values;
6. expected duration, timeout, cost/budget ceiling, and publication authority;
7. where checkpoints, final artifacts, metrics, logs, and the submission receipt survive.

Start with the environment-local ability check:

```bash
bashgym operator doctor
bashgym manifest --json
bashgym training capabilities --json
```

If the doctor says a lane is unavailable, do not substitute a documented command
from another checkout or device.

## Same-device BashGym

The BashGym API and trainer must be running on the device that will perform the
work, and the dataset path must be readable there.

```bash
bashgym api GET /api/health
bashgym api GET /api/system/info
bashgym api GET /api/system/recommendations
bashgym training start \
  --strategy sft \
  --model <model-id> \
  --dataset-path /path/on/that/device/train.jsonl \
  --compute-target local \
  --config run-config.json \
  --checkpoint-limit 1 \
  --artifact-retention adapter_only \
  --json
```

The returned `run_id` proves the request was accepted. Confirm `running` status
and a real process/metric update before saying training is active.

## Private SSH device

The BashGym controller owns the run. It generates the training script, uploads
the script and dataset, launches the registered remote environment, streams
logs, and retrieves the retained artifacts.

```bash
bashgym api GET /api/devices
bashgym api POST /api/devices/<device_id>/preflight
bashgym training start \
  --strategy dpo \
  --model <model-id> \
  --dataset-path /path/readable/by/controller/dpo.jsonl \
  --compute-target ssh:<device_id> \
  --config dpo-config.json \
  --checkpoint-limit 1 \
  --artifact-retention adapter_checkpoint \
  --json
```

Do not manually SSH-launch the same generated run in parallel. A successful
preflight must show the intended host, Python/runtime, disk, GPU or unified-memory
budget, and backend availability. Only stop unrelated model services for memory
with explicit authority.

## Hugging Face Jobs

This is separate from BashGym `/api/training/start`. Use it when managed HF
hardware is desired and the base model, dataset, training script, and final
destination can all be resolved from the job environment.

Required checks:

```bash
hf auth whoami
hf jobs --help
```

Requirements:

- a paid Hugging Face plan with Jobs access;
- a verified write token, passed to the job as the `HF_TOKEN` secret;
- a dataset on the Hub or otherwise downloadable by the job;
- an inline, URL-hosted, or CLI-uploaded training script with pinned dependencies;
- an explicit GPU flavor and timeout with setup/save buffer;
- `push_to_hub=True`, a destination `hub_model_id`, private visibility by default,
  and checkpoint pushes for runs that must survive interruption;
- Trackio (or equivalent persisted metrics) for the loss/metric curves.

Validate a custom dataset before GPU submission. SFT needs compatible `messages`
or `text`; DPO needs `prompt`, `chosen`, and `rejected`; GRPO needs prompts plus a
real reward function/verifier.

All CLI flags precede the script path or URL:

```bash
hf jobs uv run \
  --flavor <hardware-flavor> \
  --timeout <duration> \
  --secrets HF_TOKEN \
  "<training-script-path-or-url>" \
  <script arguments>
```

Capture the returned HF job id immediately. A submitted job is not yet running.
Use bounded checks at the session's agreed cadence:

```bash
hf jobs ps
hf jobs inspect <job-id>
hf jobs logs <job-id>
hf jobs cancel <job-id>
```

The environment is ephemeral. If the script does not push the adapter/model,
checkpoints, and required evidence before exit or timeout, those files are lost.
After completion, record the job id, model/dataset/script revisions, hardware,
timeout, Hub artifact commit, Trackio run, metrics, cost, and evaluation/report
references in the BashGym session evidence. Until a dedicated HF Jobs adapter is
installed, do not claim the HF job is a native BashGym training run.

When available to Codex, use the `hugging-face:huggingface-jobs` and
`hugging-face:huggingface-llm-trainer` skills. Hermes may use the verified local
`hf jobs` CLI. Both must follow the same BashGym session and evidence contract.

## NVIDIA NeMo and managed providers

For the existing hosted Customizer path, send
`compute_target: "cloud:nemo-customizer"` (or set
`use_nemo_customizer: true`) only after the NeMo Microservices client, endpoint,
and credentials pass preflight. This surface is not NeMo Gym or NeMo RL. The
BashGym run should expose both its own `run_id` and the upstream Customizer job
id.

NeMo RL belongs on operator-owned private compute behind a registered campaign
executor. Its container/source revision, platform image digest, recipe, model,
dataset, verifier, outputs, and budget must be installation-owned and pinned.
Until that executor is doctor-ready, do not substitute the Customizer path or
claim NeMo RL execution from a target label.

Managed fine-tune APIs use a separate endpoint. Save the request object as
`managed-submit.json` so the same command works in PowerShell, cmd.exe, and POSIX shells:

```bash
bashgym api POST /api/training/managed/submit --data-file managed-submit.json
```

```json
{
  "platform": "<connected-provider>",
  "base_model": "<provider-model-id>",
  "dataset_path": "/path/to/provider-compatible.jsonl",
  "n_epochs": 1,
  "learning_rate": 0.00001,
  "suffix": "<run-label>"
}
```

```text
bashgym api GET /api/training/managed/<provider>/<job-id>
```

This surface currently submits and polls. If cancellation is not exposed by the
configured BashGym provider adapter, use the provider's approved cancellation
surface and record that external action; never pretend a BashGym stop endpoint
controls it.

## Monitor, stop, and return evidence

Native BashGym runs:

```bash
bashgym api GET /api/training/<run_id>
bashgym api GET /api/training/<run_id>/log --query tail=200
bashgym api GET /api/training/runs/<run_id>/metrics
bashgym api POST /api/training/<run_id>/stop
```

Stopping/cancelling compute is destructive and requires the session's authority.
After any lane finishes, reconcile upstream status, final artifact identity,
metrics/log completeness, actual cost/time, and the required heldout/eval gate.
Only then generate reports or recommend promotion.
