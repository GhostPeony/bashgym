import { create } from 'zustand'

import type { CampaignRecord } from './campaignStore'
import { wsKey } from './workspacePersistence'

/**
 * Per-workspace "archived campaign" presentation preference. The durable
 * campaign record is never mutated; archiving only hides a campaign from the
 * default canvas materialization and Control Room list until unarchived.
 *
 * Persistence is a local seam: consumers only touch the store, so the
 * localStorage backing can later be swapped for a backend preference API.
 */

type ArchiveCampaign = Pick<CampaignRecord, 'campaign_id' | 'status'>

interface ArchiveStorageLike {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
}

function defaultStorage(): ArchiveStorageLike | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null
  }
}

export function readArchivedCampaignIds(
  workspaceId: string,
  storage: ArchiveStorageLike | null = defaultStorage(),
): string[] {
  if (!storage) return []
  try {
    const raw = storage.getItem(wsKey(workspaceId, 'campaign_archive'))
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return [...new Set(parsed.filter((item): item is string => typeof item === 'string'))].sort()
  } catch {
    return []
  }
}

export function writeArchivedCampaignIds(
  workspaceId: string,
  campaignIds: readonly string[],
  storage: ArchiveStorageLike | null = defaultStorage(),
): void {
  if (!storage) return
  try {
    storage.setItem(
      wsKey(workspaceId, 'campaign_archive'),
      JSON.stringify([...new Set(campaignIds)].sort()),
    )
  } catch {
    // Quota or privacy-mode failures degrade to session-only archiving.
  }
}

/** Drop archived ids whose campaign is actively running so live work always surfaces. */
export function pruneArchivedForCampaigns(
  archivedIds: readonly string[],
  campaigns: readonly ArchiveCampaign[],
): string[] {
  const active = new Set(
    campaigns.filter((item) => item.status === 'active').map((item) => item.campaign_id),
  )
  return archivedIds.filter((campaignId) => !active.has(campaignId))
}

export function partitionCampaignsByArchive<T extends Pick<CampaignRecord, 'campaign_id'>>(
  campaigns: readonly T[],
  archivedIds: ReadonlySet<string>,
): { visible: T[]; archived: T[] } {
  const visible: T[] = []
  const archived: T[] = []
  for (const item of campaigns) {
    ;(archivedIds.has(item.campaign_id) ? archived : visible).push(item)
  }
  return { visible, archived }
}

interface CampaignArchiveState {
  archivedByWorkspace: Record<string, string[]>
  ensureLoaded: (workspaceId: string) => void
  archive: (workspaceId: string, campaignId: string) => void
  unarchive: (workspaceId: string, campaignId: string) => void
  /** Apply auto-unarchive rules against the latest campaign list. */
  reconcile: (workspaceId: string, campaigns: readonly ArchiveCampaign[]) => void
}

function sameIds(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index])
}

export const useCampaignArchiveStore = create<CampaignArchiveState>((set, get) => {
  const commit = (workspaceId: string, next: string[]) => {
    const sorted = [...new Set(next)].sort()
    const current = get().archivedByWorkspace[workspaceId]
    if (current && sameIds(current, sorted)) return
    writeArchivedCampaignIds(workspaceId, sorted)
    set((state) => ({
      archivedByWorkspace: { ...state.archivedByWorkspace, [workspaceId]: sorted },
    }))
  }

  return {
    archivedByWorkspace: {},

    ensureLoaded: (workspaceId) => {
      if (get().archivedByWorkspace[workspaceId]) return
      const stored = readArchivedCampaignIds(workspaceId)
      set((state) => ({
        archivedByWorkspace: { ...state.archivedByWorkspace, [workspaceId]: stored },
      }))
    },

    archive: (workspaceId, campaignId) => {
      get().ensureLoaded(workspaceId)
      commit(workspaceId, [...(get().archivedByWorkspace[workspaceId] || []), campaignId])
    },

    unarchive: (workspaceId, campaignId) => {
      get().ensureLoaded(workspaceId)
      commit(
        workspaceId,
        (get().archivedByWorkspace[workspaceId] || []).filter((item) => item !== campaignId),
      )
    },

    reconcile: (workspaceId, campaigns) => {
      get().ensureLoaded(workspaceId)
      const current = get().archivedByWorkspace[workspaceId] || []
      commit(workspaceId, pruneArchivedForCampaigns(current, campaigns))
    },
  }
})

const EMPTY_ARCHIVED: readonly string[] = []

export function selectArchivedIds(
  state: Pick<CampaignArchiveState, 'archivedByWorkspace'>,
  workspaceId: string,
): readonly string[] {
  return state.archivedByWorkspace[workspaceId] || EMPTY_ARCHIVED
}
