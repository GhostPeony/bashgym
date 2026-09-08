# Bounded coding benchmark campaigns

Evaluate a pinned starting checkpoint, inspect task failures, enable one bounded
SFT recipe, then evaluate the candidate on the identical development suite. The
host agent proposes and interprets the experiment; existing campaign workers
execute it. This path measures function completion. It does not replace the
separate repository-repair and failed-tool-recovery environment.

## Prepare fixed inputs

`bashgym.datasets.coding_benchmarks` provides pure Python preparation functions:

- `prepare_mbpp_sanitized`: explicit task IDs from the pinned sanitized MBPP
  split, with an explicit allowed-import list. It extracts the function signature
  and description, keeps reference answers separate, and binds the original
  assertions to the submitted function. Unsupported reference structures reject
  preparation rather than changing the task.
- `prepare_humaneval_plus`: explicit task IDs from the pinned Hugging Face
  HumanEval+ projection. This is not the native EvalPlus differential evaluator.
- `prepare_sft_examples`: complete instruction/response demonstrations with
  source row, seed and group hashes. Pass the actual learner tokenizer and a token
  limit to check eligibility. Selected overlong rows reject preparation; they
  are not truncated.

The functions do not download assets or execute source examples. Acquire reviewed
files at the revisions declared in the module, record their digests and licenses,
and freeze supported IDs before scoring a learner. Serialize with `encode_jsonl`
and register the exact bytes with the existing dataset binding. Keep canonical
answers in a separate canary artifact, outside model prompts and training data.
Test those answers through the selected grader before accepting the suite.

Keep development and confirmation IDs separate. Split personal traces by task or
repository; preserve source groups when sampling public SFT data. A source's
execution-filter claim and local token checks do not independently verify every
demonstration or rule out pretraining contamination. Report these limits.

## Register the evaluation runner

The installed module `bashgym.campaigns.first_party_coding_runner` accepts the
existing evaluator ABI:

```text
--context <evaluation-context.json>
--model-dir <full-local-checkpoint>
--dataset <fixed-tasks.jsonl>
--output <autoresearch_evaluation.json>
--config <coding-runner.json>
```

Configure every field in `CodingRunnerConfig` explicitly: source revision and
split, expected task count, primary metric, digest-pinned local sandbox image,
input/output token limits, generation/test timeouts, whole-evaluation
`max_seconds`, dtype, device, seed and completion protocol. The scope is `smoke`
or `development`; confirmation is not yet a supported campaign scope here.

`raw` grades the continuation unchanged. `humaneval_body_v1` ends a function-body
continuation at a new column-zero function, class, import, conditional, print,
comment or Markdown fence. Indented nested code remains intact. Choose and pin
the protocol before evaluation; changing it creates a different evaluator
contract. The selected protocol is recorded with task results.

The worker profile must collect both `autoresearch_evaluation.json` and
`coding_task_results.json`. The latter contains bounded failure categories and
is hash-bound into the evidence. The evaluator checks exact dataset bytes, row
count, unique IDs and provenance. It loads a full regular-file checkpoint
locally, with remote model code disabled. Adapter-only directories are rejected.

Generated code runs through `DockerCodingEpisode` and an installed HumanEval
checker in the pinned image. The runner never pulls an image or executes model
code on the host. Include the test suite's dependencies in the reviewed image;
some HumanEval+ tasks require NumPy. Network isolation and timeout tests establish
ordinary benchmark behavior, not an adversarial RL reward-integrity certificate.

Complete development evidence exits zero. Incomplete execution exits nonzero;
smoke evidence exits 3 and cannot establish a campaign quality result. A timed-out
test is a failed task in the fixed denominator. Infrastructure or generation
failure makes the evaluation incomplete instead of reducing the denominator.

## Register one SFT intervention

`bashgym.campaigns.sft_runner` wraps the existing generated trainer and accepts
`--config`, `--model-dir`, `--dataset`, `--output` and `--launch-manifest`. The
worker supplies the model binding and launch manifest. Pin the script, runtime,
configuration and input hashes in the existing execution profile.

`CampaignSFTConfig` supports explicit `plain` or `unsloth` backends, a full
unquantized local checkpoint, bounded steps and sequence length, LoRA settings,
batch/accumulation, learning rate, example limit and a whole-run time limit of
at most 1,200 seconds. It checks the complete chat and rendered-text token
representations before training. It rejects unsupported quantized bases,
preprocessed token rows and incompatible message formats.

The wrapper requires the requested actual steps, finite recorded training
metrics and a complete merged checkpoint before publishing `final`. Failure
preserves a failed run manifest rather than publishing a candidate. Training
dependencies remain in the execution environment; importing the configuration
does not import Torch or Unsloth.

In proposals, pin `training_recipe.seed` at the top level as required by the
campaign candidate gate, as well as in the runner configuration. Record the
same disabled recipe in the baseline and enable it as the declared candidate
change. Require the actual completed baseline study as its parent. A successful
training loss or model-load canary is not a keep/discard result.

## Activate and validate

Use the existing activation command's `--model-artifact-receipt` for the accepted
physical model receipt and repeat `--evaluation-output` for both evaluator
artifacts. Preparation validates the exact model revision and bindings; the
receipt must exist before applying activation. Inspect the resulting READY
contract and use the ordinary authorized Start path.

Validate in increasing cost order: pure input checks, focused behavioral tests,
installed-runtime imports, reference-answer sandbox canaries, then an actual
baseline and bounded candidate. A 32-task development suite is a useful pipeline
check, not a full published benchmark or broad coding-quality claim. Retain a
separate confirmation suite and report any incomplete live gate explicitly.
