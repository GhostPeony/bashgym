# Personal research studio

The studio follows one experiment record: evaluate, inspect failures, propose a
change, train, compare, then keep or discard. The browser and agent skills use
the existing campaign service and validator. Scientific judgment stays in the
research agent.

## Start without Electron

Install a browser release wheel, then run `bashgym init --agent-host hermes`
(or `codex` or `claude`). Open the reported browser address and enter its
single-use pairing code. `bashgym doctor --json` distinguishes service liveness
from recipe readiness. Preparation continues through `bashgym research prepare`
or the browser Setup view, using the same saved draft.

The agent host uses its own provider configuration. The learner, dataset,
evaluation, target and limits are selected independently during preparation.
Preparation stops at READY. Starting compute is a separate human action.

Initialization reuses a matching healthy service and reports wrong-state or
incompatible services before attempting setup. `--no-service` prepares local
access only and reports that service verification is still needed. The doctor
reports the check time, connection results and missing recipe inputs separately.
Registered model, dataset, evaluation and environment metadata appear in setup;
discovery does not prove that a target is reachable or compatible.

For an existing approved execution environment, `research prepare --template-id`
compiles the selected installed template and saved setup draft into the existing
onboarding contract. Supply its reviewed `--definition-digest`, draft
`--expected-version`, campaign identity and title, explicit `--stop-rules` file,
and existing `--controller-lease-key-ref`. Use `research prepare --help` for the
complete arguments. Missing or conflicting inputs are reported without choosing
replacements. The initial result is a plan; `--write-inputs` saves the private
contract and its exact input digests. Run the existing `research onboard` command
with that contract to review preparation, then apply preparation to reach READY.
The compiler does not register a new execution environment or acquire a model.

The new service defaults to loopback and requires authentication. Remote binds
require explicitly configured API authentication; configure HTTPS at the remote
deployment boundary. Browser sessions use HttpOnly cookies. No credentials are
placed in navigation URLs or browser storage. Revoking a paired credential also
invalidates its active WebSocket delivery.

## Data and learning tools

NeMo Data Designer is available through Resources → Data Designer and
`bashgym designer`. Install its optional `data-designer` extra in the environment
that performs generation. Existing trace curation, verified demonstrations,
preference validation and training tools remain available through the skills.
Document exports use the `reports` extra; `compat` retains the previous report
dependencies. Model training dependencies belong in execution environments.

Resources opens Data Designer directly. Its readiness panel reports the running
backend's optional imports, credential presence and check time. It does not
contact a generation provider to certify access. Recheck after updating the
backend; an unsuccessful refresh leaves prior evidence visibly stale. The browser
form currently uses NVIDIA NIM and its matching model catalog. Other providers
require an explicit provider/endpoint through the existing API. Generated
campaign data returns to inspection and registration before setup can select it.

JSON, Markdown and CSV campaign exports work in the base installation. Request
Word, PDF or charts explicitly when needed. Exports publish only after all
requested renderers finish; missing optional dependencies leave no partial
bundle and return an actionable error, so the same request can be retried after
repair. The direct Python export helper retains its full compatibility bundle
when no format selection is supplied.

Data Designer is an optional data-generation step in the research loop. Its
presence does not mean generated examples are verified demonstrations or that
generation automatically launches training. A proposed data change must retain
its source provenance, pass the relevant task verification and split checks,
and enter the same campaign review and evaluation path as other training data.
The complete failure-to-generation-to-training path still needs live validation.

Prepare the authored coding fixture suite without starting a model:

```bash
bashgym environments build-coding --output ./coding-tasks --json
bashgym environments inspect-coding --dataset ./coding-tasks --split dev --json
```

The versioned bundle contains separate train, development and confirmation
tasks, covering repository repair and failed-tool recovery. Its files and task
contents carry digests and authored-data provenance. These small fixtures check
the execution path; they do not establish personal coding quality. Personal
traces must be split by task or repository before creating a real baseline.

Personal trace export now enforces deterministic repository groups through
`POST /api/training/export`. The optional `split_group_by: "task"` requires
explicit task IDs. Shared source identities and duplicate training content stay
in one partition; insufficient groups or missing identity reject the export.
The result includes a content-addressed export ID and private manifest with
input/file digests, grouping method, seed, actual counts and group hashes.
Exact downloads verify that manifest and its data. Preserve this split for
development comparisons and use a separate final confirmation evaluation.

`bashgym environments export-nemo` creates the existing NeMo Gym bundle format.
It requires explicit immutable BashGym and NeMo Gym revisions and a locally
available Docker image digest. Use its `--archive` output with the existing
campaign NeMo RL setup adapter. Exporting a bundle does not execute it.
The coding adapter isolates episodes, disables networking, bounds time and
output, protects verifier material and records cleanup. It never pulls images
or falls back to a host shell. Its first NeMo integration accepts one bounded
command-list submission; the direct model adapter supports tool feedback.

The authored task runner and NeMo bundle do not yet supply the campaign's
personal-coding evaluation executable. That integration must run the selected
split and emit the campaign evaluation record with the exact learner, suite,
data and evaluator identities. A cached model also needs an accepted immutable
artifact receipt; observing its files or a working inference endpoint does not
make it a selected training checkpoint.

## Evidence and optimization

Training excludes automatically paired traces unless task and context match
and verification eligibility is present. Legacy pairs remain inspectable.
Diagnostics require checkpoint, recipe, data and probe provenance matching the
current proposal. Recovery intervals use a conservative paired Hoeffding bound
and record their sampling design. Exploratory evidence is labeled accordingly.

Structured pytest outcomes distinguish ordinary failures and execution errors;
they are not an independent trust boundary for adversarial generated code. The
direct trainer's verification reward runs candidate code in the test process.
Before certifying agent RL, establish and test a grader boundary that candidate
code cannot control. Protected test files and passing fixture checks alone do
not certify reward integrity.

`bashgym training compare-recipes` compares explicit measured baseline and
candidate records. Matching input, software, measurement and quality contracts
are required. It reports speed, memory, time, cost and quality guard results;
it does not run a benchmark or automatically choose an optimization.

## Release gates

Automated checks cover structured verification, generated tensor alignment,
pairing and authorization, campaign setup/restart fixtures, browser states,
task isolation contracts and packaged assets. The browser wheel CI job builds
the web assets and verifies their presence outside the source checkout.
An installed-release check exercises CLI initialization, Hermes skill integrity,
the HTTP API, single-use pairing, bundled assets and browser-to-agent setup
resume using disposable local state. A separate Windows check exercises isolated
autostart registration, process failure recovery, stop, resume and uninstall;
login/reboot persistence is a separate deployment check.
The registered-preparation integration fixture runs the concrete coordinator,
local activation and authenticated setup API through READY, then reopens the
saved state and verifies that no second campaign or training attempt is created.
SSH responses, OS supervisor operations and loopback health transport are
fixture boundaries; this is not certification of a live execution target.

Real model quality, a complete SFT keep/discard campaign, live NeMo RL checkpoint
behavior and Unsloth/Liger performance must still be certified on the selected
execution target. They require a prepared contract and explicit Start. Service
idle memory, time to READY and time to the first evaluated candidate are also
deployment measurements, not inferred from unit tests or build size.

### Next live validation

Keep the following gates open until their evidence is recorded:

1. **Selected inputs:** identify the research host/provider separately from the
   learner checkpoint or endpoint, the permitted personal data, fixed development
   and final confirmation suites, registered execution environment, supported
   recipe, and explicit compute limits. Do not infer missing selections from
   installed assets.
   Complete the personal-coding campaign evaluator and its evidence integration,
   register the required training and development-evaluation runner profiles,
   and verify the selected local sandbox image and model artifact receipts.
2. **Clean setup:** use a fresh installation and the selected assets to reach
   READY through the browser or skills. Interrupt and resume preparation; verify
   the saved contract and campaign identity are unchanged. Record which service,
   model and recipe checks actually ran. Review the exact contract before the
   separate human Start.
3. **First learning result:** after Start, record a reproducible learner baseline,
   verified SFT data, one candidate checkpoint, evaluation on the identical
   development suite, and a keep/discard decision. Exercise restart recovery and
   confirm it neither duplicates training nor accepts stale evidence. Preserve
   final confirmation tasks for the declared confirmation decision.
4. **Preference and RL execution:** certify same-context preference data and
   aligned distillation with real artifacts, then the existing NeMo path. Record
   actual sandbox isolation, timeout, cleanup, trajectories, verified rewards,
   and checkpoint recovery. Fixture coverage does not close this gate.
5. **Performance and release:** measure compatible recipes on the selected target
   with quality guards, memory, speed and cost. Record reservation behavior and
   restoration after success and failure for any explicitly configured service
   suspension. Measure installation footprint, idle memory, time to READY and
   time to the first evaluated candidate against a comparable baseline. Complete
   release review before distributing the branch.
