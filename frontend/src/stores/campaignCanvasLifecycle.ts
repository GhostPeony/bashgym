import type { CampaignRecord } from './campaignStore'
import type { Panel } from './terminalStore'
import { useTerminalStore } from './terminalStore'
import { findDynamicNodePosition } from '../components/terminal/canvasPlacement'

type CampaignCanvasState = Pick<
  ReturnType<typeof useTerminalStore.getState>,
  'panels' | 'activePanelId' | 'canvasNodes' | 'addPanel' | 'updateCanvasNode'
>

type CampaignCanvasCampaign = Pick<CampaignRecord, 'campaign_id' | 'title'>

/**
 * Statuses that justify automatic canvas materialization on load. Terminal and
 * winding-down campaigns stay off the canvas unless a panel was already placed
 * for them while they ran.
 */
const AUTO_MATERIALIZE_STATUSES: ReadonlySet<CampaignRecord['status']> = new Set<
  CampaignRecord['status']
>(['draft', 'validating', 'ready', 'active', 'paused', 'awaiting_authority'])

/** Return campaigns whose current status warrants automatic canvas placement. */
export function campaignsForCanvasAutoMaterialization(
  campaigns: readonly CampaignRecord[],
): CampaignRecord[] {
  return campaigns.filter((campaign) => AUTO_MATERIALIZE_STATUSES.has(campaign.status))
}

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

/** Materialize from durable campaign data, re-reading the store before insert. */
export function materializeCampaignPanel(
  campaign: CampaignCanvasCampaign,
  getState: () => CampaignCanvasState = useTerminalStore.getState,
): string | null {
  const current = getState()
  const exists = current.panels.some((panel) => (
    panel.type === 'campaign' && panel.adapterConfig?.campaignId === campaign.campaign_id
  ))
  if (exists) return null

  const panelId = current.addPanel({
    type: 'campaign',
    title: campaign.title || 'Experiment Campaign',
    adapterConfig: { campaignId: campaign.campaign_id },
  })
  const latest = getState()
  const anchor = latest.activePanelId
    ? latest.canvasNodes.get(latest.activePanelId)?.position
    : undefined
  latest.updateCanvasNode(
    panelId,
    findDynamicNodePosition(
      'campaign',
      anchor,
      Array.from(latest.canvasNodes.values())
        .filter((node) => node.panelId !== panelId)
        .map((node) => node.position),
    ),
  )
  return panelId
}
