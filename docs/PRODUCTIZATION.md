# BashGym Productization and Fresh-Clone Contract

Status: tested control-plane path, July 2026.

## Verdict

BashGym's lightweight control plane is portable and fork-and-run for a no-GPU
control proof. A clean wheel can install without a GPU or provider key, expose
the CLI and packaged training docs, load the packaged agent-skill manifests,
create the API, and run the durable AutoResearch control smoke. A real training
installation is intentionally not a zero-configuration appliance.

The AutoResearch architecture is NVIDIA-informed but platform-native. BashGym's
campaign state, authority, evidence, Control Room, and primary local/private-
compute lane do not require NeMo services. NeMo RL and NeMo Gym remain optional,
explicitly selected adapters for trainers and environments that need them.

A real training campaign now has one guided, replay-safe activation command,
but it is deliberately not zero-configuration. The operator must still supply
the actual model, dataset, evaluator, source, registered device, budget, and
credential references; BashGym refuses to invent or substitute them. The
command plans first, creates or verifies the installation-owned ledger and
worker records on `--apply`, and can optionally install the resident worker.
Existing activation evidence can then be projected into guided setup with a
separate read-only plan and an explicit, atomic apply. Neither command chooses,
downloads, or substitutes a model.

A fresh installation also has a supported credential bootstrap:
`bashgym campaign provision-local-operator` creates one workspace-scoped local
human operator and stores the one-time raw refresh credential behind an opaque
secret reference. It never accepts or prints the raw token, refuses occupied
references, and revokes the database credential if secret storage fails.

## Measured fresh-install path

The following snapshot was measured from a clean source overlay and freshly
built wheel on Windows with Python 3.12. Dependency downloads used the local pip
cache, so network-cold time will vary; remeasure the complete table before a
release claim.

| Step | Measured time | Evidence |
|---|---:|---|
| Build wheel | 8.55 s | Clean publishable source overlay |
| Create isolated virtual environment | 7.43 s | Fresh Python 3.12 `venv` |
| Install wheel and declared core dependencies | 29.21 s | Clean environment; `pip check` passed |
| Render `bashgym --help` | 0.49 s | Installed console entry point |
| Render local-operator bootstrap help | 0.53 s | Installed campaign command |
| Run durable AutoResearch control smoke | 3.03 s | All seven control checks passed |
| **Source-to-first-working-result** | **48.71 s** | Build through bounded control proof |

The current clean-source wheel proof contains 409 entries (1,694,105 bytes) and
excludes checkout tests, frontend output, build artifacts, local model caches,
private plans, and machine profiles. It retains the eight packaged Hugging Face
skill manifests, the judge shell verifier, the public training documents, and
the explicit public workspace-skill bundle described below.

The wheel regression now also packages the seven source-managed workspace skills
(`bashgym`, `bashgym-operator`, `factory`, `models`, `system`, `traces`, and
`training`) through an explicit 17-file allowlist. It builds from the publishable
checkout file set, rejects every unrelated top-level wheel tree, installs the
wheel and its declared dependencies into an isolated virtual environment, and
proves both Skill Lab discovery and representative documented commands for all
seven skills from an arbitrary working directory. Ignored machine profiles, private operator
references, plans, `tasks/`, `.superpowers/`, and Python caches are not part of
that allowlist.

## Desktop distribution contract

The current Electron installer is intentionally a **thin desktop client**. It
does not embed a private Python runtime or a second copy of BashGym. Development
and unpacked source builds discover a nearby checkout. A packaged app with no
checkout starts the installed `bashgym` Python package through the interpreter
selected by `BASHGYM_PYTHON`, falling back to `python` on `PATH`.

Before first desktop launch, install the matching BashGym wheel into that
interpreter and run `bashgym operator doctor`. `BASHGYM_PROJECT_ROOT` is an
explicit source-development override only. If it points at a directory without
the BashGym backend marker, startup fails closed with configuration guidance;
the desktop does not search unrelated directories or silently change Python
environments. The Control Room shell remains visible and read-only while this
authority is unavailable.

All installed operator commands and the desktop now share
`BASHGYM_API_BASE`, defaulting to `http://127.0.0.1:8003/api`. Electron projects
that normalized value into the renderer before application modules load and
derives the matching WebSocket URL. `BASHGYM_DEV_SERVER_URL` is the independent
development-renderer origin. The legacy Electron `BASHGYM_API_URL` name remains
a validated compatibility alias; it is not a second source of truth.

The source-free resolution path has a renderer/main unit contract and the wheel
smoke imports and constructs the API from outside the checkout. CI now builds an
unsigned unpacked app natively on Windows, Linux, and macOS, launches it with the
backend deliberately unavailable, verifies the preload boundary, and creates,
lists, kills, and confirms removal of a real PTY. An existing Windows unpacked
artifact passed that probe locally. Retaining passing current-source results on
all three CI hosts remains a release gate; package construction by itself is not
counted as startup proof.

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

## Guided setup and activation registry

`bashgym campaign sync-autoresearch-registry` converts existing, applied
AutoResearch activation evidence into the public logical-binding registry used
by guided setup. Its default mode is read-only and secret-free. It accepts at
most 64 installed definitions, verifies exact registered training-executor,
dataset, evaluator, metric, and compute evidence, and reports unavailable
bindings as `unknown` with reason codes. `--apply` requires the reviewed
installation ID plus installation-owned controller and lease authority, then
writes the installation and bindings in one idempotent transaction.

The authenticated guided-setup backend exposes bounded public discovery: at
most 32 templates, 32 installations, and 32 bindings of each kind per
installation, with explicit truncation reason codes. A no-session context read
does not create setup state or require receipt-sealing authority. A resumable
session is scoped to one workspace and actor, advances through exactly six
ordered choices (template, installation, model, data, compute, evaluation), and
chains externally sealed, versioned receipts. Unknown, inaccessible,
out-of-order, stale, or out-of-contract choices fail closed. The desktop
renderer now presents those six registered choices, resumes the workspace
session, runs doctor and sealed validation, and creates the campaign without
starting it. The same structure stays visible and read-only when authority is
offline.

## Restart recovery and campaign-agent boundary

Recovery receipts and the mutable request lifecycle are sealed by external,
versioned campaign authority; the sealing key is not stored in the campaign
database. Accepted requests move through accepted, executing, and terminal
states under the resident worker's scheduler-leader fence. Resume and exact
single-attempt repair are restart-safe, stale claims can be reclaimed, and the
public projection reports whether a current recovery consumer is actually
ready before the UI enables a mutation.

The campaign-agent boundary is present, but it is deliberately fail-closed.
It supports human-issued, campaign-scoped Codex or Hermes grants, bounded
capabilities, per-action authorization, externally sealed audit/attachment
state, and one-time encrypted credential delivery. Host-session delivery uses
an ephemeral X25519 key agreement with HKDF and ChaCha20-Poly1305; the backend
stores ciphertext rather than the raw `bgag` token.

The current main-owned implementation adds four deliberately narrow pieces:

- Electron main has credential-bearing adapters for heartbeat and two fixed
  read-only campaign actions; those credential routes are not renderer-
  allowlisted.
- An isolated loopback MCP host exposes exactly `campaign_observe` and
  `campaign_artifacts`. It accepts bounded inputs and has no generic URL, body,
  filesystem, process, launch, pause, or proposal tool.
- The credential remains a main-process buffer at this boundary and is not part
  of the tool result, query string, request body, or renderer contract.
- Electron main directly launches one scope-bound Codex PTY, claims and activates
  the credential, maintains heartbeat authority, and tears down both MCP and
  backend authority on every terminal/reload/shutdown boundary. The renderer
  adopts only the already-running public PTY identity into Workspace/canvas.

The source implementation, renderer boundary, main/preload bundles, and
lifecycle regressions are verified. A clean packaged-desktop run must still
retain an end-to-end activation/non-leak receipt before release. Hermes launch
parity and all agent mutation actions, including training launch/pause and
artifact proposal, are not supported by this read-only slice.

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
- desktop host-session liveness, ephemeral private keys, and the agent credential
  activation lifecycle for one exact live process.

Those installation values must never be committed as repository defaults. A
model cache hit, adapter, GGUF/inference quant, process exit code, or hosted
backend availability cannot silently satisfy a real-training or quality claim.
This verdict applies to the current AutoResearch path, not every historical
utility in the repository. Legacy machine-specific scripts and environment
aliases elsewhere in the tree still need to be removed or generalized before
BashGym can claim repository-wide portability.

The public Git tree also still contains a project-specific experiment toolkit,
an external-feedback memo with author contact details, a legacy device bootstrap
script, and frozen v1 campaign identifiers used by historical records. They are
excluded from the workspace-skill allowlist, but wheel exclusion is not a
repository privacy boundary. The repository owner must explicitly retain,
generalize, move private, or remove those materials. Frozen durable identifiers
need a distinct generic v2 contract plus read compatibility; they must never be
renamed in place because their canonical JSON participates in hashes, sealed
artifact signatures, and idempotency keys.

## DX scorecard

| Dimension | Score | Method and evidence |
|---|---:|---|
| Getting started | 7/10 | Tested clean source to control result in 48.71 s; real hardware authority and the desktop frontend remain installation steps. |
| CLI/API ergonomics | 8/10 | Tested installed help, docs, API construction, one-argument control smoke, plan-first activation, registry synchronization, and bounded guided-setup APIs. Real campaigns retain explicit bindings by design. |
| Error guidance | 8/10 | Model inspection, guided activation, and campaign doctor fail closed with identity, source, runtime, and compute diagnostics. |
| Documentation | 8/10 | Public entry points link the durable campaign, training, portability, and contribution contracts; live hardware evidence remains operator-owned. |
| Upgrade path | 4/10 | Changelog exists, but migrations and compatibility policy need a release-grade guide. |
| Developer environment | 8/10 | CI defines Python 3.10–3.12/Linux plus Windows/macOS 3.12 packaging cells, the full Node 22 frontend gate, and native three-OS Electron startup/PTY smokes. Historical Black debt is hash-locked so new files and any touched legacy file cannot add formatting drift. Current-source hosted results still need to be retained. |
| Community | 5/10 | Contributing guide and issue URL exist; support/discussion workflow is still thin. |
| DX measurement | 6/10 | CI now measures installed-artifact behavior; recurring cold-install and hardware-lane telemetry are not yet automated. |
| **Overall** | **6.6/10** | The control plane and guided activation are usable; release-grade upgrades and live hardware proof remain. |

## Remaining productization milestones

1. Retain passing hosted results for the five-cell Python compatibility matrix,
   full frontend gate, and three native Electron startup/PTY smokes. Restore the
   existing whole-tree Black baseline before treating that check as a merge gate.
2. Add a hardware-gated local/private smoke that inspects an existing approved
   trainable model, runs a bounded real baseline, and ingests the authoritative
   evaluation without downloading or substituting a model.
3. Retain a clean packaged-desktop end-to-end proof for the implemented
   main-owned Codex/two-tool MCP lifecycle, including credential non-disclosure,
   Workspace adoption, heartbeat, and teardown. Add Hermes parity only as a
   separately verified launch adapter; add mutation tools only with their own
   fixed backend action adapters and human-authority gates.
4. Either package the frontend in Compose or keep Docker explicitly backend-only;
   make assistant services opt-in profiles with generated configuration.
5. Remove the obsolete multi-agent Orchestrator UI/backend/tests and its stale
   onboarding references while preserving the current canvas/campaign runtime.
6. Finish profiling the remaining active test suite so the default contributor
   command has a predictable completion time.
7. Remove or generalize legacy machine-specific scripts and compute aliases
   outside the current AutoResearch path.

For the exact real campaign sequence, see
[Durable AutoResearch Campaigns](training/autoresearch-campaign.md).
