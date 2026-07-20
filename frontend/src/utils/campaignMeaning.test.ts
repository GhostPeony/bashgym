import assert from 'node:assert/strict'
import test from 'node:test'

import type { CampaignArtifact, CampaignPublicEvent } from '../campaignVisibility'
import {
  describeCampaignArtifact,
  describeCampaignDecision,
  describeCampaignEvent,
  describeCampaignExecutor,
} from './campaignMeaning'

function artifact(overrides: Partial<CampaignArtifact> = {}): CampaignArtifact {
  return {
    schema_version: 'public_campaign_artifact.v1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1',
    artifact_id: 'artifact-1',
    producer_action_id: 'action-1',
    sha256: 'a'.repeat(64),
    size_bytes: 2048,
    schema_name: 'training_metrics_jsonl.v1',
    sealed: true,
    valid: true,
    created_at: '2026-07-16T18:02:00Z',
    ...overrides,
  }
}

function event(overrides: Partial<CampaignPublicEvent> = {}): CampaignPublicEvent {
  return {
    schema_version: 'public_campaign_event.v1',
    event_id: 'event-1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1',
    sequence: 12,
    aggregate_version: 7,
    event_type: 'campaign:attempt_completed',
    summary: {
      schema_version: 'public_campaign_event_summary.v1',
      attempt_id: 'attempt-1',
      stage: 'development_evaluation',
    },
    actor_id: 'campaign-controller',
    credential_kind: 'controller',
    created_at: '2026-07-16T18:01:00Z',
    ...overrides,
  }
}

test('describes sealed campaign evidence in operator language with forensics subordinate', () => {
  const presentation = describeCampaignArtifact(artifact())

  assert.equal(presentation.summary, 'Training metrics captured')
  assert.equal(presentation.detail, '2 KB · sealed and valid')
  assert.match(presentation.forensic, /artifact-1/)
  assert.match(presentation.forensic, new RegExp(`a{64}`))
})

test('decodes a successful remote exit artifact without treating the hash as the result', () => {
  const presentation = describeCampaignArtifact(artifact({
    schema_name: 'campaign_remote_exit_code.v1',
    // SHA-256 of the supervisor's canonical `0\n` exit-code file.
    sha256: '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa',
  }))

  assert.equal(presentation.summary, 'Remote run finished successfully')
  assert.equal(presentation.tone, 'success')
})

test('never presents a non-zero remote exit as success', () => {
  const presentation = describeCampaignArtifact(artifact({
    schema_name: 'campaign_remote_exit_code.v1',
    // SHA-256 of `1\n`.
    sha256: '4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865',
  }))

  assert.equal(presentation.summary, 'Remote run exit status sealed')
  assert.notEqual(presentation.tone, 'success')
})

test('keeps unknown artifacts inspectable without inventing meaning', () => {
  const presentation = describeCampaignArtifact(artifact({ schema_name: 'future_evidence.v9' }))

  assert.equal(presentation.summary, 'Artifact sealed for inspection')
  assert.match(presentation.detail, /Future Evidence V9/)
})

test('describes events and executors without leaking implementation labels', () => {
  const presentation = describeCampaignEvent(event())

  assert.equal(presentation.summary, 'Development evaluation completed')
  assert.match(presentation.detail, /Attempt attempt-1/)
  assert.equal(describeCampaignExecutor('ssh_remote'), 'Registered SSH compute')
  assert.equal(describeCampaignExecutor('development_evaluation'), 'Local evaluation')
  assert.equal(describeCampaignExecutor('fake'), 'Simulated executor')
})

test('makes the decision outcome and rationale primary while retaining evidence forensics', () => {
  const presentation = describeCampaignDecision({
    decision_id: 'decision-1',
    decision_type: 'retain',
    outcome: 'retain_current_champion',
    rationale: 'Development evidence is below the promotion threshold.',
    evidence_refs: ['eval-1'],
  })

  assert.equal(presentation.summary, 'Retain current champion')
  assert.equal(presentation.detail, 'Development evidence is below the promotion threshold.')
  assert.match(presentation.forensic, /decision-1/)
  assert.match(presentation.forensic, /eval-1/)
})
