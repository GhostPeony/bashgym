import type { CampaignRecord } from '../../stores/campaignStore'

export function resolveControlRoomCampaignSelection(
  requestedCampaignId: string | null,
  selectedCampaignId: string | null | undefined,
): string | null {
  return requestedCampaignId || selectedCampaignId || null
}

export function shouldCanonicalizeControlRoomSelection(input: {
  requestedCampaignId: string | null
  selectedCampaignId: string | null | undefined
  campaigns: readonly CampaignRecord[]
  loading: boolean
  error: string | null | undefined
}): boolean {
  if (!input.selectedCampaignId) return false
  if (!input.requestedCampaignId || input.requestedCampaignId === input.selectedCampaignId) return true
  if (input.loading || input.error) return false
  return !input.campaigns.some((campaign) => campaign.campaign_id === input.requestedCampaignId)
}
