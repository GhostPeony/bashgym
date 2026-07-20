import { RefreshCw, ShieldCheck, TerminalSquare } from 'lucide-react'

import { Button } from '../common/Button'
import {
  isCampaignAgentMutationAllowed,
  launchableCampaignAgentCapabilities,
  type CampaignAgentCapability,
  type CampaignAgentFreshness,
  type CampaignAgentPublicView,
  type CampaignAgentScope
} from './campaignAgentModel'

export type CampaignAgentHostSessionState =
  | 'eligible'
  | 'registered'
  | 'authorized'
  | 'credential_ready'
  | 'active'
  | 'credential_consumed'
  | 'failed'

export interface CampaignAgentEligibleSession {
  terminalId: string
  family: 'codex' | 'hermes'
  state: CampaignAgentHostSessionState
}

export type CampaignAgentHostState = 'loading' | 'ready' | 'offline' | 'error'
export type CampaignAgentPendingAction =
  'launch' | 'refresh' | 'register' | 'approve' | 'activate' | 'revoke' | null

export interface CampaignAgentSessionPanelProps {
  scope: CampaignAgentScope
  freshness: CampaignAgentFreshness
  hostState: CampaignAgentHostState
  hostError: string | null
  eligibleSessions: readonly CampaignAgentEligibleSession[]
  selectedTerminalId: string
  requestedCapabilities: readonly CampaignAgentCapability[]
  grantedCapabilities: readonly CampaignAgentCapability[]
  publicView: CampaignAgentPublicView | null
  pendingAction: CampaignAgentPendingAction
  onSelectTerminal: (terminalId: string) => void
  onRequestedCapabilityChange: (capability: CampaignAgentCapability, checked: boolean) => void
  onGrantedCapabilityChange: (capability: CampaignAgentCapability, checked: boolean) => void
  onRefresh: () => void
  onLaunchCodex?: () => void
  onRegister: () => void
  onApproveAttachment: () => void
  onActivate: () => void
  onRevoke: () => void
}

const hostStateCopy: Record<Exclude<CampaignAgentHostState, 'ready'>, string> = {
  loading: 'Discovering the desktop campaign-agent host…',
  offline:
    'The desktop campaign-agent host is unavailable. Campaign evidence remains visible and all session actions stay disabled.',
  error:
    'The desktop campaign-agent host could not reconcile. Campaign evidence remains visible and all session actions stay disabled.'
}

function capabilityLabel(capability: CampaignAgentCapability): string {
  return capability.replace(/_/g, ' ')
}

function stateLabel(state: CampaignAgentHostSessionState): string {
  return state.replace(/_/g, ' ')
}

function isExactScope(left: CampaignAgentScope, right: CampaignAgentScope): boolean {
  return left.workspaceId === right.workspaceId && left.campaignId === right.campaignId
}

function validCapabilities(
  requested: readonly CampaignAgentCapability[],
  granted: readonly CampaignAgentCapability[]
): boolean {
  const supported = new Set<string>(launchableCampaignAgentCapabilities)
  return (
    requested.length > 0 &&
    requested.length === new Set(requested).size &&
    granted.length === new Set(granted).size &&
    requested.every((capability) => supported.has(capability)) &&
    granted.every((capability) => supported.has(capability) && requested.includes(capability))
  )
}

export function CampaignAgentSessionPanel({
  scope,
  freshness,
  hostState,
  hostError,
  eligibleSessions,
  selectedTerminalId,
  requestedCapabilities,
  grantedCapabilities,
  publicView,
  pendingAction,
  onSelectTerminal,
  onRequestedCapabilityChange,
  onGrantedCapabilityChange,
  onRefresh,
  onLaunchCodex,
  onRegister,
  onApproveAttachment,
  onActivate,
  onRevoke
}: CampaignAgentSessionPanelProps) {
  const safeSessions = eligibleSessions.filter(
    (session) =>
      (session.family === 'codex' || session.family === 'hermes') &&
      [
        'eligible',
        'registered',
        'authorized',
        'credential_ready',
        'active',
        'credential_consumed',
        'failed'
      ].includes(session.state)
  )
  const selected = safeSessions.find((session) => session.terminalId === selectedTerminalId) ?? null
  const scopeMatches = Boolean(publicView && isExactScope(publicView.scope, scope))
  const attachment = scopeMatches ? (publicView?.attachment ?? null) : null
  const activeAttachment = attachment?.status === 'attached'
  const attachmentMatchesFamily = Boolean(
    activeAttachment && selected && attachment?.provenance.agentFamily === selected.family
  )
  const liveAuthority = isCampaignAgentMutationAllowed(freshness)
  const pending = pendingAction !== null
  const capabilitiesValid = validCapabilities(requestedCapabilities, grantedCapabilities)
  const hostReady = hostState === 'ready'
  const viewReady = Boolean(publicView && scopeMatches)
  const selectedCodex = selected?.family === 'codex'
  const liveCodexSession = safeSessions.some(
    (session) => session.family === 'codex' && session.state !== 'failed'
  )
  const launchReady =
    Boolean(onLaunchCodex) &&
    hostReady &&
    liveAuthority &&
    viewReady &&
    !pending &&
    !activeAttachment &&
    !liveCodexSession
  const registerReady =
    hostReady &&
    liveAuthority &&
    viewReady &&
    !pending &&
    capabilitiesValid &&
    selectedCodex &&
    selected?.state === 'eligible' &&
    !activeAttachment
  const approveReady =
    hostReady &&
    liveAuthority &&
    viewReady &&
    !pending &&
    capabilitiesValid &&
    selectedCodex &&
    selected?.state === 'registered' &&
    !activeAttachment
  const activateReady =
    hostReady &&
    liveAuthority &&
    !pending &&
    attachmentMatchesFamily &&
    selectedCodex &&
    Boolean(selected && ['authorized', 'credential_ready'].includes(selected.state))
  const revokeReady =
    hostReady &&
    liveAuthority &&
    !pending &&
    attachmentMatchesFamily &&
    Boolean(
      selected &&
      ['authorized', 'credential_ready', 'active', 'credential_consumed'].includes(selected.state)
    )
  const declarationLocked = pending || Boolean(activeAttachment) || !hostReady || !liveAuthority

  const availabilityCopy =
    hostState === 'ready'
      ? safeSessions.length === 0
        ? 'No eligible Codex terminal is live. Launch one here, then continue through the visible approval steps.'
        : null
      : hostError || hostStateCopy[hostState]
  const credentialLifecycleCopy =
    selected?.state === 'credential_ready'
      ? 'Credential is ready inside the desktop host. Activate again to retry the authority heartbeat.'
      : selected?.state === 'active'
        ? 'Read-only campaign tools are active through the main-owned proxy.'
        : selected?.state === 'credential_consumed'
          ? 'Credential authority was consumed by a main-owned process.'
          : null

  return (
    <section className="card p-4" aria-labelledby="campaign-agent-session-title">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-border-subtle pb-3">
        <div className="min-w-0">
          <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-accent-dark">
            Scoped execution
          </p>
          <h2
            id="campaign-agent-session-title"
            className="mt-1 font-brand text-xl text-text-primary"
          >
            Campaign agent session
          </h2>
          <p className="mt-1 max-w-2xl text-xs leading-5 text-text-secondary">
            Launch and bind one read-only Codex session. Every trust boundary and approval step
            stays visible here.
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          variant="secondary"
          aria-label="Refresh eligible sessions"
          onClick={onRefresh}
          disabled={pendingAction === 'refresh'}
        >
          <RefreshCw
            className={`mr-2 h-3.5 w-3.5 ${pendingAction === 'refresh' ? 'animate-spin' : ''}`}
          />
          Refresh
        </Button>
      </header>

      {availabilityCopy ? (
        <p
          className="mt-3 border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs leading-5 text-text-secondary"
          role={hostState === 'error' || hostState === 'offline' ? 'alert' : 'status'}
        >
          {availabilityCopy}
        </p>
      ) : null}
      {hostState === 'ready' && hostError ? (
        <p
          className="mt-3 border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs leading-5 text-text-secondary"
          role="alert"
        >
          {hostError}
        </p>
      ) : null}
      {!liveAuthority ? (
        <p
          className="mt-3 border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs leading-5 text-text-secondary"
          role="status"
        >
          Campaign authority is {freshness}. The session plan remains visible, but registration,
          approval, activation, and revocation are disabled.
        </p>
      ) : null}
      {publicView && !scopeMatches ? (
        <p
          className="mt-3 border-l-2 border-status-error bg-status-error/10 px-3 py-2 text-xs leading-5 text-text-secondary"
          role="alert"
        >
          The authoritative attachment does not match the selected campaign. Session actions are
          blocked until reconciliation succeeds.
        </p>
      ) : null}

      <div className="mt-3 grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_15rem]">
        <div className="min-w-0 space-y-3">
          <label className="block">
            <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-text-muted">
              Eligible live terminal
            </span>
            <select
              className="input mt-1 w-full"
              aria-label="Eligible live terminal"
              value={selected?.terminalId ?? ''}
              disabled={!hostReady || pending || safeSessions.length === 0}
              onChange={(event) => onSelectTerminal(event.target.value)}
            >
              <option value="">Select a live terminal</option>
              {safeSessions.map((session) => (
                <option key={session.terminalId} value={session.terminalId}>
                  {session.family === 'codex' ? 'Codex' : 'Hermes'} · {session.terminalId} ·{' '}
                  {stateLabel(session.state)}
                </option>
              ))}
            </select>
          </label>

          <fieldset
            className="border-2 border-border-subtle bg-background-secondary/50 p-3"
            disabled={declarationLocked}
          >
            <legend className="px-1 font-mono text-[10px] font-bold uppercase tracking-widest text-text-muted">
              Campaign capabilities · read only
            </legend>
            <div className="mt-1 grid gap-x-4 gap-y-2 sm:grid-cols-2">
              {launchableCampaignAgentCapabilities.map((capability) => (
                <div key={capability} className="grid grid-cols-2 gap-2 text-xs text-text-primary">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      aria-label={`Request ${capabilityLabel(capability)}`}
                      checked={requestedCapabilities.includes(capability)}
                      onChange={(event) =>
                        onRequestedCapabilityChange(capability, event.target.checked)
                      }
                    />
                    Request {capabilityLabel(capability)}
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      aria-label={`Grant ${capabilityLabel(capability)}`}
                      checked={grantedCapabilities.includes(capability)}
                      disabled={!requestedCapabilities.includes(capability)}
                      onChange={(event) =>
                        onGrantedCapabilityChange(capability, event.target.checked)
                      }
                    />
                    Grant
                  </label>
                </div>
              ))}
            </div>
          </fieldset>
          <div
            className="grid gap-1 border-l-2 border-border-subtle pl-3 text-[11px] leading-4 text-text-muted"
            aria-label="Unavailable direct agent capabilities"
          >
            <p>
              <span className="font-semibold text-text-secondary">Training launch</span> ·
              Unavailable in this direct agent path
            </p>
            <p>
              <span className="font-semibold text-text-secondary">Pause training</span> ·
              Unavailable in this direct agent path
            </p>
            <p>
              <span className="font-semibold text-text-secondary">Propose artifacts</span> ·
              Unavailable in this direct agent path
            </p>
          </div>
          {!capabilitiesValid ? (
            <p className="text-xs text-status-warning">
              Choose at least one supported request and keep every grant within that request.
            </p>
          ) : null}

          <div
            className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4"
            aria-label="Campaign agent activation steps"
          >
            <Button
              type="button"
              size="sm"
              variant="primary"
              aria-label="Launch Codex campaign agent"
              disabled={!launchReady}
              onClick={onLaunchCodex}
            >
              1 · Launch Codex
            </Button>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              aria-label="Register selected session"
              disabled={!registerReady}
              onClick={onRegister}
            >
              2 · Register
            </Button>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              aria-label="Approve campaign attachment"
              disabled={!approveReady}
              onClick={onApproveAttachment}
            >
              3 · Approve scope
            </Button>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              aria-label="Activate campaign agent"
              disabled={!activateReady}
              onClick={onActivate}
            >
              4 · Activate
            </Button>
          </div>
          <p className="text-[10px] leading-4 text-text-muted">
            Direct Hermes launch is not available yet. Existing Hermes sessions remain visible for
            status and revocation only.
          </p>
        </div>

        <aside
          className="space-y-3 border-l-2 border-border-subtle pl-3"
          aria-label="Campaign agent authority status"
        >
          <div className="flex items-start gap-2">
            <TerminalSquare className="mt-0.5 h-4 w-4 shrink-0 text-accent-dark" />
            <div className="min-w-0 text-xs leading-5 text-text-secondary">
              <p className="font-semibold text-text-primary">Desktop host</p>
              <p>
                {selected
                  ? `${selected.family === 'codex' ? 'Codex' : 'Hermes'} · ${stateLabel(selected.state)}`
                  : 'No terminal selected'}
              </p>
              <p className="break-all font-mono text-[10px] text-text-muted">
                {selected?.terminalId ?? '—'}
              </p>
              {credentialLifecycleCopy ? (
                <p className="mt-1 text-text-primary">{credentialLifecycleCopy}</p>
              ) : null}
            </div>
          </div>
          <div className="flex items-start gap-2 border-t border-border-subtle pt-3">
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-accent-dark" />
            <div className="min-w-0 text-xs leading-5 text-text-secondary">
              <p className="font-semibold text-text-primary">Campaign authority</p>
              <p>
                {activeAttachment
                  ? `Attached · version ${attachment?.attachmentVersion}`
                  : 'No active attachment'}
              </p>
              <p>
                {attachment?.grantedCapabilities.length
                  ? `${attachment.grantedCapabilities.length} server-approved capabilities`
                  : 'Awaiting explicit approval'}
              </p>
            </div>
          </div>
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="w-full"
            aria-label="Revoke campaign attachment"
            disabled={!revokeReady}
            onClick={onRevoke}
          >
            Revoke attachment
          </Button>
          <p className="text-[10px] leading-4 text-text-muted">
            Credentials never enter this page, terminal input, renderer storage, logs, or command
            arguments.
          </p>
        </aside>
      </div>
    </section>
  )
}
