import type { CampaignAgentEligibleSession } from './CampaignAgentSessionPanel'

const publicIdentifier = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/
const families = new Set(['codex', 'hermes'])
const states = new Set([
  'eligible',
  'registered',
  'authorized',
  'credential_ready',
  'active',
  'credential_consumed',
  'failed'
])

function exactDescriptor(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  if (prototype !== Object.prototype && prototype !== null) return false
  const keys = Object.keys(value).sort()
  return (
    keys.length === 3 && keys[0] === 'family' && keys[1] === 'state' && keys[2] === 'terminalId'
  )
}

export function parseCampaignAgentEligibleSessions(
  value: unknown
): CampaignAgentEligibleSession[] | null {
  if (!Array.isArray(value) || value.length > 32) return null
  const parsed: CampaignAgentEligibleSession[] = []
  const terminalIds = new Set<string>()
  for (const entry of value) {
    if (
      !exactDescriptor(entry) ||
      typeof entry.terminalId !== 'string' ||
      !publicIdentifier.test(entry.terminalId) ||
      typeof entry.family !== 'string' ||
      !families.has(entry.family) ||
      typeof entry.state !== 'string' ||
      !states.has(entry.state) ||
      terminalIds.has(entry.terminalId)
    )
      return null
    terminalIds.add(entry.terminalId)
    parsed.push(
      Object.freeze({
        terminalId: entry.terminalId,
        family: entry.family as CampaignAgentEligibleSession['family'],
        state: entry.state as CampaignAgentEligibleSession['state']
      })
    )
  }
  return parsed
}
