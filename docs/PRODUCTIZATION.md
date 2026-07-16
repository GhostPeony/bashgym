# BashGym Productization and Fresh-Clone Contract

Status: tested control-plane path, July 2026.

## Verdict

BashGym's lightweight control plane is now fork-and-run productizable. A clean
wheel can install without a GPU or provider key, expose the CLI and packaged
training docs, load the packaged agent-skill manifests, create the API, and run
the durable AutoResearch control smoke.

A real training campaign now has one guided, replay-safe activation command,
but it is deliberately not zero-configuration. The operator must still supply
the actual model, dataset, evaluator, source, registered device, budget, and
credential references; BashGym refuses to invent or substitute them. The
command plans first, creates or verifies the installation-owned ledger and
worker records on `--apply`, and can optionally install the resident worker.

## Measured fresh-install path

The following historical snapshot was measured from a clean source overlay and
freshly built wheel on Windows with Python 3.14 before the guided-activation
documentation refresh. Dependency downloads used the local pip cache, so
network-cold time will vary; remeasure the complete table before a release claim.

| Step | Measured time | Evidence |
|---|---:|---|
| Create isolated virtual environment | 7.49 s | Fresh `venv` |
| Install wheel and declared core dependencies | 35.89 s | Clean environment |
| Render `bashgym --help` | 0.90 s | Installed console entry point |
| Read packaged training overview | 0.54 s | Packaged content returned successfully |
| Run durable AutoResearch control smoke | 3.64 s | One attempt, two sealed artifacts, four metric points |
| **Time to first working result** | **48.46 s** | Under the two-minute champion threshold |

The clean wheel contained 379 entries and excluded checkout tests, build output,
and local model compilation caches. It retained all eight packaged Hugging Face
skill manifests, the judge shell verifier, and the public training documents.

Run the equivalent checkout path with:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
bashgym --help
bashgym campaign control-smoke --json
```

The control smoke uses the production campaign repository, authenticated state
transition, scheduler, fake executor, budget ledger, metric ingestion, artifact
sealer, AutoResearch decision, and restart recovery. Its simulated result must
remain `ineligible` and the campaign must still request a real baseline.

## Three supported activation lanes

1. **No-GPU control plane.** Install, inspect docs, launch the API/canvas, and
   prove durable AutoResearch behavior without a model or API key.
2. **Local or private hardware over registered SSH.** Register the operator's
   actual model artifact, immutable revision, trainer,
   dataset/evaluator/project bindings, and SSH device. Same-machine hardware
   uses localhost SSH. This is BashGym's primary real-training lane; a native
   same-process campaign executor is not currently claimed.
3. **Optional hosted or NeMo adapters.** Enable only when an operator explicitly
   selects and pins that backend. Neither is a fallback for lane 2.

## Portable boundary

The repository owns:

- campaign contracts, state transitions, stop rules, policy math, environment
  contracts, evidence schemas, CLI/API/canvas projections, and operator skills;
- the no-GPU control smoke and public documentation;
- package metadata and CI checks that prove the installed artifact works outside
  the source checkout.

Each installation owns:

- secrets and credential references;
- local/private host identity, user, key path, runtime path, and capacity policy;
- the explicitly selected trainable model artifact and immutable revision;
- dataset version, evaluator suite, primary metric, ledger project, source
  repository, and compute profile;
- resident-worker configuration and retained run artifacts.

Those installation values must never be committed as repository defaults. A
model cache hit, adapter, GGUF/inference quant, process exit code, or hosted
backend availability cannot silently satisfy a real-training or quality claim.

## DX scorecard

| Dimension | Score | Method and evidence |
|---|---:|---|
| Getting started | 7/10 | Tested clean wheel to control result in 48.46 s; frontend installation remains a separate step. |
| CLI/API ergonomics | 8/10 | Tested installed help, docs, API construction, one-argument control smoke, and plan-first guided activation. Real campaigns retain explicit bindings by design. |
| Error guidance | 8/10 | Model inspection, guided activation, and campaign doctor fail closed with identity, source, runtime, and compute diagnostics. |
| Documentation | 8/10 | Public entry points link the durable campaign, training, portability, and contribution contracts; live hardware evidence remains operator-owned. |
| Upgrade path | 4/10 | Changelog exists, but migrations and compatibility policy need a release-grade guide. |
| Developer environment | 7/10 | One pyproject install contract and wheel smoke CI; frontend/native build and advertised Python matrix need broader CI. |
| Community | 5/10 | Contributing guide and issue URL exist; support/discussion workflow is still thin. |
| DX measurement | 6/10 | CI now measures installed-artifact behavior; recurring cold-install and hardware-lane telemetry are not yet automated. |
| **Overall** | **6.6/10** | The control plane and guided activation are usable; release-grade upgrades and live hardware proof remain. |

## Remaining productization milestones

1. Run the documented path on clean Windows and Linux hosts, then add macOS for
   the no-GPU lane. Test the supported Python minor versions in CI.
2. Add a hardware-gated local/private smoke that inspects an existing approved
   trainable model, runs a bounded real baseline, and ingests the authoritative
   evaluation without downloading or substituting a model.
3. Either package the frontend in Compose or keep Docker explicitly backend-only;
   make assistant services opt-in profiles with generated configuration.
4. Remove the obsolete multi-agent Orchestrator UI/backend/tests and its stale
   onboarding references while preserving the current canvas/campaign runtime.
5. Finish profiling the remaining active test suite so the default contributor
   command has a predictable completion time.

For the exact real campaign sequence, see
[Durable AutoResearch Campaigns](training/autoresearch-campaign.md).
