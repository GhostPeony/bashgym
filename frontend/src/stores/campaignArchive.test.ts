import assert from 'node:assert/strict'
import test from 'node:test'

import {
  partitionCampaignsByArchive,
  pruneArchivedForCampaigns,
  readArchivedCampaignIds,
  useCampaignArchiveStore,
  writeArchivedCampaignIds
} from './campaignArchive'
import { campaignsForCanvasAutoMaterialization } from './campaignCanvasLifecycle'
import type { CampaignRecord, CampaignStatus } from './campaignStore'

function campaign(
  id: string,
  status: CampaignStatus = 'paused',
  workspaceId = 'workspace-a'
): CampaignRecord {
  return {
    schema_version: 'campaign.v1',
    campaign_id: id,
    workspace_id: workspaceId,
    title: id,
    kind: 'general',
    objective: 'Improve counting',
    target_model: {},
    owner_actor_id: 'operator',
    manifest_revision: 1,
    status,
    version: 1,
    created_at: '2026-07-17T00:00:00Z',
    updated_at: '2026-07-17T00:00:00Z'
  }
}

function fakeStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial))
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    removeItem: (key: string) => void data.delete(key),
    entries: () => Object.fromEntries(data)
  }
}

function resetStore() {
  useCampaignArchiveStore.setState({ archivedByWorkspace: {} })
}

test('partition splits campaigns into visible and archived without reordering', () => {
  const campaigns = [campaign('a'), campaign('b'), campaign('c')]
  const { visible, archived } = partitionCampaignsByArchive(campaigns, new Set(['b']))
  assert.deepEqual(
    visible.map((item) => item.campaign_id),
    ['a', 'c']
  )
  assert.deepEqual(
    archived.map((item) => item.campaign_id),
    ['b']
  )
})

test('prune drops archived ids whose campaign is active and keeps everything else', () => {
  const campaigns = [campaign('running', 'active'), campaign('parked', 'paused')]
  const pruned = pruneArchivedForCampaigns(['running', 'parked', 'unknown'], campaigns)
  assert.deepEqual(pruned, ['parked', 'unknown'])
})

test('storage round-trip is workspace-scoped and tolerates junk payloads', () => {
  const storage = fakeStorage()
  writeArchivedCampaignIds('ws-1', ['b', 'a', 'b'], storage)
  assert.deepEqual(readArchivedCampaignIds('ws-1', storage), ['a', 'b'])
  assert.deepEqual(readArchivedCampaignIds('ws-2', storage), [])

  const corrupted = fakeStorage({ bashgym_ws_bad_campaign_archive: '{not json' })
  assert.deepEqual(readArchivedCampaignIds('bad', corrupted), [])
  const wrongShape = fakeStorage({ bashgym_ws_bad_campaign_archive: '{"a":1}' })
  assert.deepEqual(readArchivedCampaignIds('bad', wrongShape), [])
})

test('store archive/unarchive updates the workspace set idempotently', () => {
  resetStore()
  const store = useCampaignArchiveStore.getState()
  store.archive('workspace-a', 'campaign-1')
  store.archive('workspace-a', 'campaign-1')
  store.archive('workspace-b', 'campaign-2')
  assert.deepEqual(useCampaignArchiveStore.getState().archivedByWorkspace['workspace-a'], [
    'campaign-1'
  ])
  assert.deepEqual(useCampaignArchiveStore.getState().archivedByWorkspace['workspace-b'], [
    'campaign-2'
  ])
  useCampaignArchiveStore.getState().unarchive('workspace-a', 'campaign-1')
  assert.deepEqual(useCampaignArchiveStore.getState().archivedByWorkspace['workspace-a'], [])
})

test('reconcile auto-unarchives campaigns that turned active', () => {
  resetStore()
  useCampaignArchiveStore.getState().archive('workspace-a', 'campaign-1')
  useCampaignArchiveStore.getState().archive('workspace-a', 'campaign-2')

  useCampaignArchiveStore
    .getState()
    .reconcile('workspace-a', [campaign('campaign-1', 'active'), campaign('campaign-2', 'paused')])

  assert.deepEqual(useCampaignArchiveStore.getState().archivedByWorkspace['workspace-a'], [
    'campaign-2'
  ])
})

test('reconcile without changes keeps the same array reference', () => {
  resetStore()
  useCampaignArchiveStore.getState().archive('workspace-a', 'campaign-2')
  const before = useCampaignArchiveStore.getState().archivedByWorkspace['workspace-a']
  useCampaignArchiveStore.getState().reconcile('workspace-a', [campaign('campaign-2', 'paused')])
  assert.equal(useCampaignArchiveStore.getState().archivedByWorkspace['workspace-a'], before)
})

test('auto materialization skips archived campaigns', () => {
  const campaigns = [campaign('keep', 'paused'), campaign('hidden', 'paused')]
  assert.deepEqual(
    campaignsForCanvasAutoMaterialization(campaigns, new Set(['hidden'])).map(
      (item) => item.campaign_id
    ),
    ['keep']
  )
  assert.deepEqual(
    campaignsForCanvasAutoMaterialization(campaigns).map((item) => item.campaign_id),
    ['keep', 'hidden']
  )
})
