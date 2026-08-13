# Training data

BashGym accepts structured datasets supplied by the researcher. Agent traces are
one possible source, not a prerequisite. Choose the record shape from the
training method, freeze evaluation data before training, and record the exact
dataset version used by each experiment.

## Dataset workflow

1. Define the evaluation suite and holdout split before creating training rows.
2. Normalize source records into the method-specific JSONL contract.
3. Remove invalid, duplicate, contaminated, and low-value rows.
4. Save a split manifest, row counts, source lineage, and a content digest.
5. Train one candidate, evaluate it on the unchanged suite, and keep or discard
   the result.
6. Use the failure analysis to propose one data change for the next iteration.

Do not select a candidate from training loss alone. The same fixed evaluation
suite must be used for the baseline and every candidate being compared.

## Common record metadata

The trainer consumes the method-specific content described below. For a serious
experiment, also keep enough metadata to reconstruct and audit the dataset:

```json
{
  "metadata": {
    "source_id": "source-record-42",
    "source_revision": "revision-id",
    "domain": "shell-debugging",
    "task_family": "dependency-recovery",
    "split": "train",
    "quality_score": 0.91,
    "verification_score": 1.0,
    "decontamination_status": "checked"
  }
}
```

For tracked runs, `TrainingTrackingContext` records the dataset ID and version,
source URI, SHA-256 content digest, split manifest, and row counts. See
[`TrainingTrackingContext`](../bashgym/api/schemas.py).

## SFT records

The direct SFT trainers read JSON or JSONL rows containing a `messages` list and
render it with the selected model's chat template. A minimal row is:

```json
{
  "messages": [
    { "role": "system", "content": "Follow the available tools and verify the result." },
    { "role": "user", "content": "Find and fix the failing test." },
    { "role": "assistant", "content": "I will inspect the failure first." }
  ],
  "metadata": {
    "source_id": "example-001",
    "split": "train"
  }
}
```

Tool-use examples may include assistant `tool_calls` and tool-result messages.
The trace serializer can also retain a top-level `tools` schema, but the direct
SFT formatter currently renders the `messages` column only. Put the tool context
needed for learning into the serving-matched conversation or system message.
Preserve the ordering used at inference time. Do not flatten a multi-step
interaction if the order of observations and actions is the behavior being
trained.

The trace factory can convert successful sessions into this format, but any
licensed and well-curated source can produce the same contract. The serializer
is defined by [`TrainingExample`](../bashgym/factory/data_factory.py), and the
trainer-side rendering is in [`trainer.py`](../bashgym/gym/trainer.py).

## DPO preference pairs

DPO needs two responses to the same prompt. The preferred response is `chosen`;
the comparison response is `rejected`.

```json
{
  "id": "pair-001",
  "prompt": "The command failed because the dependency is missing. What next?",
  "chosen": "Inspect the active environment, install the declared dependency, then rerun the test.",
  "rejected": "Repeat the same command and assume the failure is transient.",
  "metadata": {
    "chosen_trace_id": "candidate-a",
    "rejected_trace_id": "candidate-b",
    "chosen_verification_score": 1.0,
    "rejected_verification_score": 0.0,
    "prompt_hash": "saved-content-hash",
    "pair_generation_method": "verifier-ranked",
    "label_source": "deterministic-test",
    "domain": "dependency-recovery",
    "split": "train",
    "decontamination_status": "checked"
  }
}
```

The strict validator requires an ID, prompt, saved prompt hash, distinct chosen
and rejected text, source provenance for both responses, label strength or
source, quality or verification scores, domain or task family, split metadata,
and decontamination metadata. It also warns about extreme response-length
imbalance. The validator
accepts `chosen_response` and `rejected_response` aliases, but `chosen` and
`rejected` are the portable TRL field names used by the direct trainer. See
[`dpo_validation.py`](../bashgym/preferences/dpo_validation.py).

Good pairs isolate response quality. Avoid pairs where the preference can be
predicted from length, formatting, source identity, or leaked evaluation text.

## Reward examples

Reward-model records are separate from the direct SFT, DPO, GRPO, RLVR,
distillation, and Session Distillation strategies. Each row needs a prompt, a
response or trajectory, and either a scalar reward or step-level process
rewards.

```json
{
  "id": "reward-001",
  "reward_type": "outcome_reward",
  "prompt": "Repair the project and prove it passes its tests.",
  "trajectory": [{ "tool": "shell", "input": "pytest -q", "exit_code": 0 }],
  "reward": 1.0,
  "metadata": {
    "reward_scale": "0_to_1",
    "verifier_id": "project-tests-v1",
    "source_id": "task-001-attempt-2",
    "quality_score": 1.0,
    "task_family": "repository-repair",
    "split": "train",
    "decontamination_status": "checked"
  }
}
```

Supported validator labels include preference, outcome, and process rewards.
Process-reward records also need step rewards or scored steps. Strict validation
requires the reward scale, label source, source provenance, confidence or
quality, task family, split, and decontamination metadata. See
[`reward_validation.py`](../bashgym/preferences/reward_validation.py).

## GRPO and RLVR prompts

The direct GRPO/RLVR trainer reads prompt rows. Its built-in verification mode
can also consume test code:

```json
{
  "prompt": "Implement `parse_version` and return only the code.",
  "tests": "from solution import parse_version\n\ndef test_value():\n    assert parse_version('v2.4') == (2, 4)"
}
```

Use `strategy: "rlvr"` when the reward comes from an executable or
deterministic verifier. A prompt without a meaningful verifier is not an RLVR
example. The generated direct trainer currently provides syntax, execution, and
test-verification reward modes; more general terminal tasks use the environment
contract described next.

## Executable terminal environments

A terminal task is more than a prompt. `EnvironmentSpec` records:

- a stable task ID and instruction;
- domain, skills, and sampled axes;
- fixture files or services;
- build and setup commands;
- a verifier command or path, reward type, success threshold, and timeout;
- rollout limits such as steps, tool calls, prompt tokens, response tokens, and
  wall-clock timeout.

A task is not structurally verifiable until it has an instruction and verifier.
Verifier rewards can be binary, graded, or a weighted set of named components.
See [`contracts.py`](../bashgym/environments/contracts.py).

Keep environment training tasks and environment holdouts in separate groups.
For terminal-agent evaluation, compare baseline and candidate with verifier-
backed end-to-end attempts rather than tool-call imitation alone.

## Session Distillation records

Session Distillation is the trace-specific method in this guide. It converts a
failed or recoverable decision into two contexts for the same target action:
the original context and a context with a structured hint.

The current record contract includes `original_context`, `hinted_context`,
`hint_text`, `target_text`, a full-target `target_span`, a
`target_span_only` loss mask, reader confidence, verifier outcome, and source
metadata. Partial target spans are not implemented. The complete schema and
validator are in
[`session_distillation.py`](../bashgym/factory/session_distillation.py).

## Train, validation, and heldout separation

Use three roles when the dataset is large enough:

- **train** updates model parameters;
- **validation** selects training-time settings such as early stopping;
- **heldout evaluation** compares the frozen baseline and candidates and must
  not influence training examples.

For session-derived data, split by whole session or repository, not by row.
Rows from the same session share context and can leak across a row-level split.
[`make_holdout_split`](../bashgym/eval/split.py) performs grouped splitting,
saves holdout hashes, and exposes a contamination check.

For other datasets, select a grouping key that represents the underlying unit
of leakage: source document, repository, task template, user, problem family,
or generator seed. Save the grouping rule and random seed in the split manifest.

## Curation, deduplication, and decontamination

Apply these checks before training:

1. **Structural validation.** Parse every row and validate required fields for
   the selected method.
2. **Verification.** Retain the verifier result and distinguish execution
   success from stylistic or model-judge scores.
3. **Exact deduplication.** Hash a canonical representation and remove repeated
   rows.
4. **Near-duplicate detection.** Compare normalized text, code, task identity,
   or embeddings. BashGym's embedding deduplicator supports `messages` and
   prompt/response records; a skipped embedding service is reported rather than
   silently counted as successful deduplication. See
   [`dedup.py`](../bashgym/factory/dedup.py).
5. **Holdout decontamination.** Remove training rows that match or closely
   reproduce evaluation tasks, answers, fixtures, or repository snapshots.
6. **Distribution review.** Inspect domain, difficulty, source, length, tool,
   language, reward, and failure-mode distributions. Large counts do not replace
   coverage.
7. **Manual sampling.** Read examples from every major slice, especially rows
   with extreme scores or lengths.

Do not train on a source marked evaluation-only. The source catalog rejects
evaluation-only sources for training unless an explicit override is recorded;
such an override changes provenance, not the validity of the resulting claim.

## Data changes in AutoResearch

An AutoResearch candidate may reuse a versioned dataset or include a
`DATA_BUILD` stage before training. A data-building iteration can change one
declared variable, such as:

- source or domain mixture;
- quality threshold;
- deduplication or decontamination threshold;
- hard-example or failure-mode sampling;
- synthetic-data proportion;
- preference-pair construction;
- reward labels or verifier version;
- session-distillation confidence threshold.

Generated shards are represented by an `AutoResearchDatasetReceipt` containing
file digests, byte sizes, split names, row counts, generator metadata, and a
canonical content digest. The receipt carries metadata; it does not replace the
dataset rows. See
[`autoresearch_dataset.py`](../bashgym/campaigns/autoresearch_dataset.py).

Change one experimental variable at a time when attribution matters. Evaluate
each candidate on the same suite, keep or discard it, then let the next failure
analysis determine the following data hypothesis.

## Related references

- [Training strategy guide](training/strategy-guide.md)
- [Platform architecture](PLATFORM_OVERVIEW.md)
- [AutoResearch campaign](training/autoresearch-campaign.md)
