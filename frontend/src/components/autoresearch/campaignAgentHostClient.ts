import type { CampaignAgentCapability } from './campaignAgentModel'

type CampaignAgentHostResult =
  ({ success: true } & Record<string, unknown>) | { success: false; error: string }

export interface CampaignAgentHostRendererAPI {
  launch?: (request: { workspaceId: string; campaignId: string; cwd?: string }) => Promise<unknown>
  eligible: (request: {
    workspaceId: string
    campaignId: string
  }) => Promise<CampaignAgentHostResult & { sessions?: unknown }>
  attach: (request: {
    terminalId: string
    workspaceId: string
    campaignId: string
  }) => Promise<CampaignAgentHostResult>
  authorize: (request: {
    terminalId: string
    workspaceId: string
    campaignId: string
    requestedCapabilities: CampaignAgentCapability[]
    grantedCapabilities: CampaignAgentCapability[]
    idempotencyKey: string
  }) => Promise<CampaignAgentHostResult>
  activate: (request: {
    terminalId: string
    workspaceId: string
    campaignId: string
  }) => Promise<CampaignAgentHostResult>
  revoke: (request: {
    terminalId: string
    workspaceId: string
    campaignId: string
  }) => Promise<CampaignAgentHostResult>
}

export type CampaignAgentLaunchResult =
  { success: true; terminalId: string; cwd: string } | { success: false; error: string }

const safeTerminalId = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/

function exactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort()
  const expected = [...keys].sort()
  return actual.length === expected.length && actual.every((key, index) => key === expected[index])
}

export async function launchCodexCampaignAgent(
  host: Pick<CampaignAgentHostRendererAPI, 'launch'>,
  scope: { workspaceId: string; campaignId: string; cwd?: string }
): Promise<CampaignAgentLaunchResult> {
  if (!host.launch) return { success: false, error: 'Direct Codex launch is unavailable.' }
  let value: unknown
  try {
    value = await host.launch(
      scope.cwd === undefined
        ? {
            workspaceId: scope.workspaceId,
            campaignId: scope.campaignId
          }
        : {
            workspaceId: scope.workspaceId,
            campaignId: scope.campaignId,
            cwd: scope.cwd
          }
    )
  } catch {
    return { success: false, error: 'The desktop host could not launch Codex.' }
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return { success: false, error: 'The desktop host returned an invalid launch result.' }
  }
  try {
    const record = value as Record<string, unknown>
    if (
      record.success === false &&
      exactKeys(record, ['success', 'error']) &&
      typeof record.error === 'string'
    ) {
      return { success: false, error: 'The desktop host rejected the Codex launch.' }
    }
    if (
      record.success === true &&
      exactKeys(record, ['success', 'terminalId', 'cwd']) &&
      typeof record.terminalId === 'string' &&
      safeTerminalId.test(record.terminalId) &&
      typeof record.cwd === 'string' &&
      record.cwd.length > 0 &&
      record.cwd.length <= 1024
    ) {
      return { success: true, terminalId: record.terminalId, cwd: record.cwd }
    }
  } catch {
    return { success: false, error: 'The desktop host returned an invalid launch result.' }
  }
  return { success: false, error: 'The desktop host returned an invalid launch result.' }
}

export interface CampaignAgentHostActionBinding {
  terminalId: string
  workspaceId: string
  campaignId: string
  requestedCapabilities: readonly CampaignAgentCapability[]
  grantedCapabilities: readonly CampaignAgentCapability[]
  idempotencyKey: string
}

export type CampaignAgentHostAction = 'register' | 'approve' | 'activate' | 'revoke'

export function invokeCampaignAgentHostAction(
  host: CampaignAgentHostRendererAPI,
  action: CampaignAgentHostAction,
  binding: CampaignAgentHostActionBinding
): Promise<CampaignAgentHostResult> {
  if (action === 'register') {
    return host.attach({
      terminalId: binding.terminalId,
      workspaceId: binding.workspaceId,
      campaignId: binding.campaignId
    })
  }
  if (action === 'approve') {
    return host.authorize({
      terminalId: binding.terminalId,
      workspaceId: binding.workspaceId,
      campaignId: binding.campaignId,
      requestedCapabilities: [...binding.requestedCapabilities],
      grantedCapabilities: [...binding.grantedCapabilities],
      idempotencyKey: binding.idempotencyKey
    })
  }
  const terminalScope = {
    terminalId: binding.terminalId,
    workspaceId: binding.workspaceId,
    campaignId: binding.campaignId
  }
  if (action === 'activate') return host.activate(terminalScope)
  return host.revoke(terminalScope)
}
