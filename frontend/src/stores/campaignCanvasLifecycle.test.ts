import assert from 'node:assert/strict'
import test from 'node:test'

import {
  campaignsForCanvasAutoMaterialization,
  campaignsMissingPanels,
  materializeCampaignPanel,
} from './campaignCanvasLifecycle'
import type { CampaignRecord, CampaignStatus } from './campaignStore'
import type { Panel } from './terminalStore'

function campaign(
  id: string,
  workspaceId = 'workspace-a',
  status: CampaignStatus = 'active',
): CampaignRecord {
  return {
    schema_version: 'campaign.v1',
    campaign_id: id,
    workspace_id: workspaceId,
    title: id,
    kind: 'embedding_finetune',
    objective: 'Improve retrieval',
    target_model: {},
    owner_actor_id: 'codex-agent',
    manifest_revision: 1,
    status,
    version: 1,
    created_at: '2026-07-13T00:00:00Z',
    updated_at: '2026-07-13T00:00:00Z',
  }
}

test('auto materialization includes only campaigns awaiting or doing work', () => {
  const eligible: CampaignStatus[] = [
    'draft',
    'validating',
    'ready',
    'active',
    'paused',
    'awaiting_authority',
  ]
  const excluded: CampaignStatus[] = [
    'cancelling',
    'cancelled',
    'completed',
    'failed',
    'exhausted',
  ]

  const campaigns = [...eligible, ...excluded].map((status) =>
    campaign(`campaign-${status}`, 'workspace-a', status),
  )
  assert.deepEqual(
    campaignsForCanvasAutoMaterialization(campaigns).map((item) => item.status),
    eligible,
  )
})

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

test('live campaign materialization rechecks current panels at insertion time', () => {
  const panels: Panel[] = []
  const state = {
    panels,
    activePanelId: null,
    canvasNodes: new Map(),
    addPanel(input: Omit<Panel, 'id'>) {
      const id = `panel-${panels.length + 1}`
      panels.push({ id, ...input } as Panel)
      return id
    },
    updateCanvasNode() {},
  }
  const getState = () => state

  assert.equal(materializeCampaignPanel(campaign('campaign-live'), getState), 'panel-1')
  assert.equal(materializeCampaignPanel(campaign('campaign-live'), getState), null)
  assert.equal(panels.length, 1)
})

test('same campaign ID materializes independently after a synchronous workspace canvas switch', () => {
  const canvasState = () => {
    const panels: Panel[] = []
    return {
      panels,
      activePanelId: null,
      canvasNodes: new Map(),
      addPanel(input: Omit<Panel, 'id'>) {
        const id = `panel-${panels.length + 1}`
        panels.push({ id, ...input } as Panel)
        return id
      },
      updateCanvasNode() {},
    }
  }
  const workspaceA = canvasState()
  const workspaceB = canvasState()

  assert.equal(materializeCampaignPanel(campaign('campaign-1', 'workspace-a'), () => workspaceA), 'panel-1')
  assert.equal(materializeCampaignPanel(campaign('campaign-1', 'workspace-b'), () => workspaceB), 'panel-1')
  assert.equal(workspaceA.panels.length, 1)
  assert.equal(workspaceB.panels.length, 1)
  assert.notEqual(workspaceA.panels, workspaceB.panels)
})
