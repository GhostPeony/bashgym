import assert from 'node:assert/strict'
import test from 'node:test'

import { campaignsMissingPanels } from './campaignCanvasLifecycle'
import type { CampaignRecord } from './campaignStore'
import type { Panel } from './terminalStore'

function campaign(id: string): CampaignRecord {
  return {
    schema_version: 'campaign.v1',
    campaign_id: id,
    workspace_id: 'workspace-a',
    title: id,
    kind: 'embedding_finetune',
    objective: 'Improve retrieval',
    target_model: {},
    owner_actor_id: 'codex-agent',
    manifest_revision: 1,
    status: 'active',
    version: 1,
    created_at: '2026-07-13T00:00:00Z',
    updated_at: '2026-07-13T00:00:00Z',
  }
}

test('campaign reload materializes every campaign exactly once', () => {
  const panels = [{
    id: 'panel-existing',
    type: 'campaign',
    title: 'Existing',
    adapterConfig: { campaignId: 'campaign-1' },
  }] as Panel[]

  assert.deepEqual(
    campaignsMissingPanels([campaign('campaign-1'), campaign('campaign-2')], panels)
      .map((item) => item.campaign_id),
    ['campaign-2'],
  )
})
