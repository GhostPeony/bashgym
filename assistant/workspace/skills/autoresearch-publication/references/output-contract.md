# Research-post output contract

## Purpose

`open_frontiers.research_post.v1` is a renderer-neutral exchange format. It carries evidence-backed editorial content and visual data, not HTML, CSS, filesystem locations, credentials, or execution topology.

## Required top-level fields

| Field | Purpose |
|---|---|
| `schema_version` | Exact value `open_frontiers.research_post.v1` |
| `publication` | Public slug, title, summary, and human approval state |
| `experiment` | Question, hypothesis, public model, method, controlled intervention, fixed evaluation, and optional method-selection rationale |
| `results` | Primary metric, optional secondary metrics, bounded failure-category comparison, and keep/discard/baseline decision |
| `narrative` | Simple explanation, technical explanation, judgment, limitations, and next experiment |
| `training_rungs` | Ordered sequence of training and evaluation work that actually ran |
| `visuals` | Typed data for downstream rendering |
| `claims` | Public claims linked to evidence and source identifiers |
| `sources` | Primary or authoritative contextual sources |
| `provenance` | Public evidence digests and generation timestamp |

## Approval

`publication.approval.status` is `draft` or `approved`. Its `feedback` array
contains `{ "note": "...", "status": "open|addressed|declined" }` objects.

- A draft uses `null` for `approved_by` and `approved_at`.
- An approved package requires a human-supplied reviewer label and ISO-8601 timestamp.
- A draft may contain open feedback. An approved package may not.
- An experiment's `KEEP` decision never implies publication approval.

## Metrics

The primary metric contains:

```json
{
  "name": "exact_accuracy",
  "unit": "fraction",
  "baseline": 0.1,
  "candidate": 0.7,
  "delta": 0.6,
  "direction": "higher_is_better"
}
```

All values must be finite. `delta` must equal `candidate - baseline`. Use `lower_is_better` when appropriate; do not invert the recorded values.

## Narrative

- `simple`: explain the measured change without assuming training expertise.
- `technical`: identify the training method, controlled variable, fixed evaluation, result, and decision rule.
- `judgement`: give the evidence-bounded editorial conclusion.
- `limitations`: list concrete reasons the result may not generalize.
- `next_experiment`: state one experiment that would reduce the most important uncertainty.

## Method selection

Optional `experiment.method_selection` records the method selected by the host
agent, the evidence-bounded rationale, and up to six alternatives. It is an
editorial explanation of a completed decision, not authority to run an
unsupported method. Alternative statuses are `eligible`, `not_selected`,
`blocked`, `diagnostic_needed`, or `unsupported_by_runner`.

## Failure analysis

Optional `results.failure_analysis` contains at most 12 evaluator-authored
behavior categories. Each item has a public category and summary plus baseline
count, candidate count, their exact delta, and `improved`, `regressed`, or
`unchanged` status. It must never contain raw prompts, targets, predictions,
example identifiers, dataset rows, or artifact previews.

## Training rungs

Use consecutive `order` values beginning at 1. Each rung contains `label`, `method`, `status`, and `summary`. Allowed statuses are `planned`, `completed`, `kept`, `discarded`, and `failed`.

Typical rungs are baseline evaluation, data construction, SFT, preference optimization or RLVR, and fixed candidate evaluation. Include only rungs supported by the evidence.

## Visuals

Visuals contain data and meaning, not presentation styling. Every package contains:

- `metric_comparison`: baseline and candidate values for the primary metric;
- `training_rungs`: ordered rung identifiers or order values.

Optional visual types include `slice_comparison`, `learning_curve`, and `evidence_map`. The downstream site chooses layout, color, typography, motion, and interaction.

## Claims and sources

Each claim has an `id`, public `text`, one or more `evidence_refs`, and zero or more `source_refs`.

- Use evidence references for measured results and decisions.
- Use source references for method background, scientific precedent, or interpretation.
- Do not use an external paper as evidence that the local experiment succeeded.

Each source contains `id`, `title`, `url`, `year`, and `role`. Prefer primary papers or authoritative technical sources.

## Public boundary

Do not include:

- absolute local or remote paths;
- hostnames, IP addresses, ports, usernames, or device labels;
- credentials, secret references, tokens, or environment values;
- raw launch requests, executor configurations, command lines, logs, or arbitrary artifact metadata;
- claims not traceable to a declared evidence or source reference.

Read raw campaign exports only as private source material. Construct this allowlisted contract field by field.
