import assert from 'node:assert/strict'
import test from 'node:test'
import {
  toCampaignActivityFields,
  toCampaignArtifactPreview,
  toCampaignPublicArtifact,
} from './campaignVisibility'

function event(summary: Record<string, unknown> | null = null) {
  return {
    schema_version: 'public_campaign_event.v1',
    event_id: 'event-1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1',
    sequence: 4,
    aggregate_version: 3,
    event_type: 'campaign:action-blocked',
    summary,
    actor_id: 'campaign-controller',
    credential_kind: 'controller',
    created_at: '2026-07-16T00:00:00Z',
  }
}

function artifact() {
  return {
    schema_version: 'public_campaign_artifact.v1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1',
    artifact_id: 'artifact-1',
    producer_action_id: 'action-1',
    sha256: 'a'.repeat(64),
    size_bytes: 10,
    schema_name: 'training_metrics_jsonl.v1',
    sealed: true,
    valid: true,
    created_at: '2026-07-16T00:00:00Z',
  }
}

test('Activity projector explicitly picks approved public primitives', () => {
  const fields = toCampaignActivityFields(event({
    schema_version: 'public_campaign_event_summary.v1',
    action_id: 'action-1',
    stage: 'full_training',
    code: 'campaign_remote_profile_unavailable',
    stage_index: 2,
  }))

  assert.deepEqual(fields, {
    event_id: 'event-1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1',
    aggregate_version: 3,
    action_id: 'action-1',
    stage: 'full_training',
    code: 'campaign_remote_profile_unavailable',
    stage_index: 2,
  })
})

test('Activity projector rejects schema skew, runtime extras, nested values, and lists', () => {
  assert.equal(toCampaignActivityFields({ ...event(), schema_version: 'campaign_event.v1' }), null)
  assert.equal(toCampaignActivityFields({ ...event(), payload: { location: 'private-path-canary' } }), null)
  assert.equal(toCampaignActivityFields(event({
    schema_version: 'public_campaign_event_summary.v1',
    code: 'campaign_remote_profile_unavailable',
    location: 'private-path-canary',
  })), null)
  assert.equal(toCampaignActivityFields(event({
    schema_version: 'public_campaign_event_summary.v1',
    code: { nested: 'candidate-map-canary' },
  })), null)
  assert.equal(toCampaignActivityFields(event({
    schema_version: 'public_campaign_event_summary.v1',
    stage_index: Number.POSITIVE_INFINITY,
  })), null)
  assert.equal(toCampaignActivityFields(event({
    schema_version: 'public_campaign_event_summary.v1',
    metric_names: Array.from({ length: 101 }, (_, index) => `private-${index}`),
  })), null)
})

test('artifact projector accepts only the exact frozen public shape', () => {
  assert.deepEqual(toCampaignPublicArtifact(artifact()), artifact())
  assert.equal(toCampaignPublicArtifact({
    ...artifact(),
    uri: 'C:/operator/restricted-result.json',
  }), null)
  assert.equal(toCampaignPublicArtifact({
    ...artifact(),
    metadata: { reference: 'candidate-map-canary' },
  }), null)
  assert.equal(toCampaignPublicArtifact({
    ...artifact(),
    schema_version: 'campaign_artifact_record.v1',
  }), null)
  assert.equal(toCampaignPublicArtifact({
    ...artifact(),
    size_bytes: Number.POSITIVE_INFINITY,
  }), null)
  assert.equal(toCampaignPublicArtifact({
    ...artifact(),
    schema_name: 'candidate-map-canary',
  }), null)
})

test('artifact projector accepts the public query-format ablation manifest', () => {
  assert.notEqual(toCampaignPublicArtifact({
    ...artifact(),
    schema_name: 'query_format_ablation_manifest.v2',
  }), null)
})

test('artifact preview projector rejects extra private fields and oversized content', () => {
  const preview = {
    schema_version: 'public_campaign_artifact_preview.v1',
    artifact_id: 'artifact-1',
    preview_kind: 'text',
    content: 'step=40 loss=0.3',
    truncated: false,
    redaction_count: 0,
    integrity_verified: true,
    unavailable_reason: null,
  }
  assert.deepEqual(toCampaignArtifactPreview(preview), preview)
  assert.equal(toCampaignArtifactPreview({ ...preview, uri: 'C:/private/log' }), null)
  assert.equal(toCampaignArtifactPreview({ ...preview, content: 'x'.repeat(65_537) }), null)
  assert.equal(toCampaignArtifactPreview({ ...preview, integrity_verified: false }), null)
})
