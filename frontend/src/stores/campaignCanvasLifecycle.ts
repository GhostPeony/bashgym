import type { CampaignRecord } from './campaignStore'
import type { Panel } from './terminalStore'

/** Return campaigns that do not yet have a durable workspace canvas projection. */
export function campaignsMissingPanels(
  campaigns: readonly CampaignRecord[],
  panels: readonly Panel[],
): CampaignRecord[] {
  const materialized = new Set(
    panels
      .filter((panel) => panel.type === 'campaign')
      .map((panel) => panel.adapterConfig?.campaignId)
      .filter((campaignId): campaignId is string => typeof campaignId === 'string'),
  )
  return campaigns.filter((campaign) => !materialized.has(campaign.campaign_id))
}
