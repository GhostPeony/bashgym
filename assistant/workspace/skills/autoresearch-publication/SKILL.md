---
name: autoresearch-publication
description: Create a human-approved, renderer-neutral publication package from a completed BashGym or comparable model experiment. Use when turning campaign evidence, evaluation results, training-rung history, research citations, and human judgment into structured JSON plus a Markdown review copy for Open Frontiers or another research website.
---

# AutoResearch publication

Produce an evidence-backed research-post package. Keep website implementation and visual styling outside this skill.

## Required inputs

Obtain:

- the canonical `campaign_evidence.json`, report, and export manifest;
- the experiment's fixed question, hypothesis, intervention, evaluation, result, and decision;
- the ordered training/evaluation rungs;
- bounded `research failures` comparisons and the decision packet's method-selection rationale, when present;
- primary sources used to explain the method or interpret the result;
- human editorial feedback and, only after review, explicit approval metadata.

Treat raw exports as private input. They may contain nested paths, executor details, internal identifiers, or other operator material. Extract only the fields defined in `references/output-contract.md`; never copy arbitrary export objects into the publication package.

## Workflow

1. Read `references/output-contract.md` completely.
2. Copy `assets/research-post.template.json` to a working output directory.
3. Reconcile every metric and decision against the canonical experiment evidence.
4. Replace every `REPLACE_...` marker. Keep the package in `draft` status.
5. Write both explanations:
   - `narrative.simple` for a reader unfamiliar with the training method;
   - `narrative.technical` for a researcher checking the intervention and evaluation.
6. Encode the experiment sequence in `training_rungs`. Describe what actually ran; do not turn compatibility checks or planned methods into completed rungs.
7. When evidence exists, record aggregate behavioral changes in
   `results.failure_analysis` and the completed method decision in
   `experiment.method_selection`. Never copy raw examples or an entire decision
   packet.
8. Add renderer-neutral visual data. Always include:
   - one `metric_comparison` visual;
   - one `training_rungs` visual.
9. Attach evidence references to quantitative claims and source references to contextual or methodological claims.
10. Validate the JSON:

   ```bash
   python scripts/research_post.py validate <research-post.json>
   ```

11. Render the review copy:

    ```bash
    python scripts/research_post.py render --output <research-post.md> <research-post.json>
    ```

12. Present the JSON and Markdown to a human. Record feedback in
    `publication.approval.feedback` and incorporate it while status remains
    `draft`.
13. Set every feedback item to `addressed` or `declined` with an explicit
    editorial decision. Set status to `approved` only when no feedback remains
    open and a human supplies `approved_by` and `approved_at`, then validate and
    render again.

## Editorial rules

- State what the experiment tested before describing infrastructure.
- Separate measured results from interpretation and future work.
- Describe uncertainty, suite size, contamination risk, missing protected metrics, and generalization limits explicitly.
- Use public model and method names when they matter to the study; do not expose deployment details.
- Prefer recent primary papers for method context. A source does not prove an experiment result; campaign evidence does.
- Preserve negative and discarded results when they affect the judgment.
- Keep prose direct and technical. Avoid slogans, launch language, and promotional claims.
- Do not generate HTML, CSS, React components, or website-specific layout. The downstream site owns presentation.

## Approval boundary

Never infer approval from a successful experiment, a `KEEP` decision, or the existence of an export. `KEEP` is a model-selection result. `approved` is a separate human publication decision.
