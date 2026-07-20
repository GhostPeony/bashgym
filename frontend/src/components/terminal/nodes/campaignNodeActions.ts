import { useUIStore } from '../../../stores/uiStore'
import type { CampaignDetailState } from '../../../stores/campaignStore'

type CampaignTransitionAction = 'start' | 'pause' | 'resume' | 'cancel'
type CampaignTransition = (
  workspaceId: string,
  campaignId: string,
  action: CampaignTransitionAction,
  expectedVersion: number,
  reason?: string
) => Promise<unknown>

export function openCampaignInAutoResearch(workspaceId: string, campaignId: string): void {
  useUIStore.getState().openTraining('autoresearch', { workspaceId, campaignId })
}

export function shouldLoadCampaignDrillDown(
  configOpen: boolean,
  snapshotVersion: number | null
): boolean {
  return configOpen && snapshotVersion !== null
}

export function hasLiveCampaignAuthority(
  detail: Pick<CampaignDetailState, 'freshness'> | null | undefined
): boolean {
  return detail?.freshness === 'live'
}

export async function submitCampaignTransitionIfLive(input: {
  detail: CampaignDetailState | null | undefined
  mutating: boolean
  action: CampaignTransitionAction
  workspaceId: string
  transition: CampaignTransition
}): Promise<boolean> {
  const { detail, mutating, action, workspaceId, transition } = input
  if (!hasLiveCampaignAuthority(detail) || !detail || mutating) return false

  const reason =
    action === 'pause'
      ? 'Paused from the BashGym campaign canvas.'
      : action === 'cancel'
        ? 'Cancelled from the BashGym campaign canvas.'
        : undefined
  await transition(
    workspaceId,
    detail.campaign.campaign_id,
    action,
    detail.campaign.version,
    reason
  )
  return true
}
