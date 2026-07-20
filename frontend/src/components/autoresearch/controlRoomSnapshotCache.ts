import { create } from 'zustand'

import type { CampaignRecoveryPublicV1 } from './campaignRecoveryModel'
import type { GuidedSetupContext } from './guidedSetupModel'
import type { ParsedHumanWorkQueuePublicV1 } from './humanWorkModel'

/**
 * Session cache for Control Room snapshots (human oversight queue, recovery
 * state, guided setup context). The Control Room unmounts on navigation, so
 * without this cache every visit blanks those panels and refetches. Cached
 * snapshots seed the panels instantly on remount while the normal loaders
 * revalidate in the background. Local component state stays the source of
 * truth while mounted; this cache is written through on change and read only
 * when (re)mounting or switching campaigns.
 */

export const controlRoomSnapshotKey = (workspaceId: string, campaignId: string): string =>
  JSON.stringify([workspaceId, campaignId])

interface ControlRoomSnapshotCacheState {
  humanQueues: Record<string, ParsedHumanWorkQueuePublicV1>
  recoverySnapshots: Record<string, CampaignRecoveryPublicV1>
  setupContexts: Record<string, GuidedSetupContext>
  putHumanQueue: (key: string, queue: ParsedHumanWorkQueuePublicV1) => void
  putRecoverySnapshot: (key: string, snapshot: CampaignRecoveryPublicV1) => void
  putSetupContext: (workspaceId: string, context: GuidedSetupContext) => void
}

export const useControlRoomSnapshotCache = create<ControlRoomSnapshotCacheState>((set) => ({
  humanQueues: {},
  recoverySnapshots: {},
  setupContexts: {},

  putHumanQueue: (key, queue) =>
    set((state) => ({ humanQueues: { ...state.humanQueues, [key]: queue } })),

  putRecoverySnapshot: (key, snapshot) =>
    set((state) => ({ recoverySnapshots: { ...state.recoverySnapshots, [key]: snapshot } })),

  putSetupContext: (workspaceId, context) =>
    set((state) => ({ setupContexts: { ...state.setupContexts, [workspaceId]: context } })),
}))
