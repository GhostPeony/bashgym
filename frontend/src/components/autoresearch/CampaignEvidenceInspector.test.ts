import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type {
  CampaignArtifact,
  CampaignArtifactPreview,
  CampaignEventItem
} from '../../campaignVisibility'
import { CampaignEvidenceDetail, CampaignEvidenceRow } from './CampaignEvidenceInspector'

const artifact: CampaignArtifact = {
  schema_version: 'public_campaign_artifact.v1',
  workspace_id: 'workspace-a',
  campaign_id: 'campaign-1',
  artifact_id: 'artifact-training-metrics',
  producer_action_id: 'action-full-training',
  sha256: 'd'.repeat(64),
  size_bytes: 1_572_864,
  schema_name: 'training_metrics_jsonl.v1',
  sealed: true,
  valid: true,
  created_at: '2026-07-13T21:14:34Z'
}

const eventItem: CampaignEventItem = {
  cursor: 41,
  event: {
    schema_version: 'public_campaign_event.v1',
    event_id: 'event-metrics-appended',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1',
    sequence: 41,
    aggregate_version: 7,
    event_type: 'campaign:training-metrics-appended',
    summary: {
      schema_version: 'public_campaign_event_summary.v1',
      attempt_id: 'attempt-1'
    },
    actor_id: 'campaign-controller',
    credential_kind: 'controller',
    created_at: '2026-07-13T21:14:34Z'
  }
}

const preview: CampaignArtifactPreview = {
  schema_version: 'public_campaign_artifact_preview.v1',
  artifact_id: artifact.artifact_id,
  preview_kind: 'jsonl',
  content: '{"step":40,"loss":0.3321}\n{"step":41,"loss":0.3012}',
  truncated: true,
  redaction_count: 2,
  integrity_verified: true,
  unavailable_reason: null
}

test('shared evidence rows are explicit inspect buttons', () => {
  const artifactMarkup = renderToStaticMarkup(
    createElement(CampaignEvidenceRow, {
      selection: { kind: 'artifact', artifact },
      onInspect: () => undefined
    })
  )
  const eventMarkup = renderToStaticMarkup(
    createElement(CampaignEvidenceRow, {
      selection: { kind: 'event', item: eventItem },
      onInspect: () => undefined
    })
  )
  assert.match(artifactMarkup, /<button[^>]+aria-label="Inspect Training metrics captured"/)
  assert.match(artifactMarkup, /Inspect/)
  assert.match(eventMarkup, /<button[^>]+aria-label="Inspect Training Metrics Appended"/)
})

test('artifact detail exposes visible integrity metadata and bounded content', () => {
  const markup = renderToStaticMarkup(
    createElement(CampaignEvidenceDetail, {
      selection: { kind: 'artifact', artifact },
      preview,
      loading: false,
      error: null
    })
  )
  assert.match(markup, /artifact-training-metrics/)
  assert.match(markup, /training_metrics_jsonl\.v1/)
  assert.match(markup, /action-full-training/)
  assert.match(markup, new RegExp('d{64}'))
  assert.match(markup, /Sealed and valid/)
  assert.match(markup, /step/)
  assert.match(markup, /loss/)
  assert.match(markup, /Latest 500 lines shown|Preview truncated/)
  assert.match(markup, /2 sensitive values redacted/)
})

test('event detail exposes every public forensic field without raw payloads', () => {
  const markup = renderToStaticMarkup(
    createElement(CampaignEvidenceDetail, {
      selection: { kind: 'event', item: eventItem },
      loading: false,
      error: null
    })
  )
  assert.match(markup, /event-metrics-appended/)
  assert.match(markup, /campaign:training-metrics-appended/)
  assert.match(markup, /campaign-controller/)
  assert.match(markup, /Cursor/)
  assert.match(markup, />41</)
  assert.match(markup, /Aggregate version/)
  assert.match(markup, />7</)
  assert.match(markup, /attempt-1/)
  assert.doesNotMatch(markup, /raw payload/i)
})
