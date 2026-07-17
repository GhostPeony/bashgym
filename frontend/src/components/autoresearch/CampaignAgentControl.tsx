import { useCallback, useEffect, useRef, useState } from 'react'

import { campaignApi } from '../../services/api'
import { useWorkspaceStore } from '../../stores/workspaceStore'
import {
  CampaignAgentSessionPanel,
  type CampaignAgentEligibleSession,
  type CampaignAgentHostState,
  type CampaignAgentPendingAction,
} from './CampaignAgentSessionPanel'
import {
  invokeCampaignAgentHostAction,
  launchCodexCampaignAgent,
  type CampaignAgentHostAction,
  type CampaignAgentHostRendererAPI,
} from './campaignAgentHostClient'
import { parseCampaignAgentEligibleSessions } from './campaignAgentHostModel'
import {
  adoptCampaignAgentTerminal,
  finalizeSuccessfulCampaignAgentLaunch,
  preserveDurableCampaignAgentError,
} from './campaignAgentTerminalAdoption'
import type {
  CampaignAgentCapability,
  CampaignAgentFreshness,
  CampaignAgentPublicView,
  CampaignAgentScope,
} from './campaignAgentModel'

export interface CampaignAgentControlProps {
  scope: CampaignAgentScope
  freshness: CampaignAgentFreshness
}

function newAuthorizationIdempotencyKey(): string {
  const bytes = new Uint8Array(16)
  globalThis.crypto.getRandomValues(bytes)
  return `idem_${Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('')}`
}

function hostAPI(): CampaignAgentHostRendererAPI | null {
  const host = window.bashgym?.campaignAgentHost
  return host ? host as unknown as CampaignAgentHostRendererAPI : null
}

export function CampaignAgentControl({ scope, freshness }: CampaignAgentControlProps) {
  const [hostState, setHostState] = useState<CampaignAgentHostState>('loading')
  const [hostError, setHostError] = useState<string | null>(null)
  const [sessions, setSessions] = useState<CampaignAgentEligibleSession[]>([])
  const [selectedTerminalId, setSelectedTerminalId] = useState('')
  const [requestedCapabilities, setRequestedCapabilities] = useState<CampaignAgentCapability[]>(['campaign_observe', 'artifact_read'])
  const [grantedCapabilities, setGrantedCapabilities] = useState<CampaignAgentCapability[]>(['campaign_observe', 'artifact_read'])
  const [publicView, setPublicView] = useState<CampaignAgentPublicView | null>(null)
  const [pendingAction, setPendingAction] = useState<CampaignAgentPendingAction>(null)
  const [idempotencyKey, setIdempotencyKey] = useState(newAuthorizationIdempotencyKey)
  const loadGeneration = useRef(0)
  const activeReconciliations = useRef(0)
  const launchLifecycleActive = useRef(false)
  const durableHostError = useRef<string | null>(null)

  const reconcile = useCallback(async (showLoading = true): Promise<void> => {
    activeReconciliations.current += 1
    try {
      const setReconciledHostError = (error: string | null) => {
        setHostError(preserveDurableCampaignAgentError(durableHostError.current, error))
      }
      const generation = ++loadGeneration.current
      if (showLoading) setHostState('loading')
      setReconciledHostError(null)
      const host = hostAPI()
      const viewPromise = campaignApi.campaignAgentView(scope.workspaceId, scope.campaignId)
      if (!host) {
        const view = await viewPromise
        if (generation !== loadGeneration.current) return
        if (view.ok && view.data) setPublicView(view.data)
        setSessions([])
        setHostState('offline')
        setReconciledHostError('The desktop campaign-agent host is unavailable. Existing campaign evidence remains visible.')
        return
      }

      try {
        const [eligible, view] = await Promise.all([host.eligible({
          workspaceId: scope.workspaceId,
          campaignId: scope.campaignId,
        }), viewPromise])
        if (generation !== loadGeneration.current) return
        if (!eligible.success) {
          setSessions([])
          if (view.ok && view.data) setPublicView(view.data)
          setHostState('error')
          setReconciledHostError('The desktop host rejected terminal discovery. Session actions remain disabled.')
          return
        }
        const parsedSessions = parseCampaignAgentEligibleSessions(eligible.sessions)
        if (!parsedSessions) {
          setSessions([])
          setHostState('error')
          setReconciledHostError('The desktop host returned an invalid terminal projection. Session actions remain disabled.')
          return
        }
        if (!view.ok || !view.data) {
          setSessions(parsedSessions)
          setHostState(view.code === 'campaign_desktop_bridge_required' ? 'offline' : 'error')
          setReconciledHostError('Campaign attachment authority is unavailable. Existing Control Room evidence remains visible and session actions stay disabled.')
          return
        }
        setPublicView(view.data)
        setSessions(parsedSessions)
        setSelectedTerminalId((current) => (
          parsedSessions.some((session) => session.terminalId === current)
            ? current
            : parsedSessions.find((session) => session.state !== 'failed')?.terminalId ?? ''
        ))
        setHostState('ready')
        setReconciledHostError(null)
      } catch {
        if (generation !== loadGeneration.current) return
        setSessions([])
        setHostState('error')
        setReconciledHostError('The desktop host could not reconcile campaign-agent state. Session actions remain disabled.')
      }
    } finally {
      activeReconciliations.current = Math.max(0, activeReconciliations.current - 1)
    }
  }, [scope.campaignId, scope.workspaceId])

  useEffect(() => {
    loadGeneration.current += 1
    durableHostError.current = null
    setHostState('loading')
    setHostError(null)
    setSessions([])
    setSelectedTerminalId('')
    setPublicView(null)
    setPendingAction(null)
    setRequestedCapabilities(['campaign_observe', 'artifact_read'])
    setGrantedCapabilities(['campaign_observe', 'artifact_read'])
    setIdempotencyKey(newAuthorizationIdempotencyKey())
    let cancelled = false
    let timer: number | undefined
    const schedulePoll = () => {
      if (cancelled) return
      timer = window.setTimeout(() => {
        if (activeReconciliations.current > 0 || launchLifecycleActive.current) {
          schedulePoll()
          return
        }
        void reconcile(false).finally(schedulePoll)
      }, 3000)
    }
    void reconcile().finally(schedulePoll)
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
      loadGeneration.current += 1
    }
  }, [reconcile])

  const resetAuthorizationBinding = () => setIdempotencyKey(newAuthorizationIdempotencyKey())

  const runAction = (action: CampaignAgentHostAction) => {
    const host = hostAPI()
    if (!host || !selectedTerminalId || pendingAction) return
    void (async () => {
      setPendingAction(action)
      setHostError(null)
      let actionError: string | null = null
      try {
        const result = await invokeCampaignAgentHostAction(host, action, {
          terminalId: selectedTerminalId,
          workspaceId: scope.workspaceId,
          campaignId: scope.campaignId,
          requestedCapabilities,
          grantedCapabilities,
          idempotencyKey,
        })
        if (!result.success) {
          actionError = `The desktop host rejected the ${action} request. Authoritative state was reconciled before retry.`
        }
        await reconcile(false)
        if (result.success && (action === 'approve' || action === 'revoke')) resetAuthorizationBinding()
      } catch {
        actionError = `The ${action} request could not complete. Authoritative state was reconciled before retry.`
        await reconcile(false)
      } finally {
        if (actionError) setHostError(actionError)
        setPendingAction(null)
      }
    })()
  }

  const launchCodex = () => {
    const host = hostAPI()
    if (!host?.launch || pendingAction) return
    void (async () => {
      setPendingAction('launch')
      launchLifecycleActive.current = true
      durableHostError.current = null
      setHostError(null)
      try {
        const result = await launchCodexCampaignAgent(host, {
          workspaceId: scope.workspaceId,
          campaignId: scope.campaignId,
        })
        if (!result.success) {
          await reconcile(false)
          setHostError(result.error)
          return
        }

        const outcome = await finalizeSuccessfulCampaignAgentLaunch(result.terminalId, {
          adopt: () => adoptCampaignAgentTerminal(
            useWorkspaceStore.getState(),
            scope.workspaceId,
            result,
          ),
          kill: (terminalId) => (
            window.bashgym?.terminal.kill(terminalId) ?? Promise.resolve(false)
          ),
          selectTerminal: () => setSelectedTerminalId(result.terminalId),
          reconcile: () => reconcile(false),
        })
        if (outcome.status !== 'adopted') {
          durableHostError.current = outcome.durableError
          setHostError(outcome.durableError)
        }
      } finally {
        launchLifecycleActive.current = false
        setPendingAction(null)
      }
    })()
  }

  const handleRequestedCapabilityChange = (capability: CampaignAgentCapability, checked: boolean) => {
    setRequestedCapabilities((current) => checked
      ? [...new Set([...current, capability])]
      : current.filter((value) => value !== capability))
    if (!checked) setGrantedCapabilities((current) => current.filter((value) => value !== capability))
    resetAuthorizationBinding()
  }

  const handleGrantedCapabilityChange = (capability: CampaignAgentCapability, checked: boolean) => {
    if (checked && !requestedCapabilities.includes(capability)) return
    setGrantedCapabilities((current) => checked
      ? [...new Set([...current, capability])]
      : current.filter((value) => value !== capability))
    resetAuthorizationBinding()
  }

  return (
    <CampaignAgentSessionPanel
      scope={scope}
      freshness={freshness}
      hostState={hostState}
      hostError={hostError}
      eligibleSessions={sessions}
      selectedTerminalId={selectedTerminalId}
      requestedCapabilities={requestedCapabilities}
      grantedCapabilities={grantedCapabilities}
      publicView={publicView}
      pendingAction={pendingAction}
      onSelectTerminal={(terminalId) => {
        setSelectedTerminalId(terminalId)
        resetAuthorizationBinding()
      }}
      onRequestedCapabilityChange={handleRequestedCapabilityChange}
      onGrantedCapabilityChange={handleGrantedCapabilityChange}
      onRefresh={() => {
        if (pendingAction) return
        setPendingAction('refresh')
        void reconcile().finally(() => setPendingAction(null))
      }}
      onLaunchCodex={hostAPI()?.launch ? launchCodex : undefined}
      onRegister={() => runAction('register')}
      onApproveAttachment={() => runAction('approve')}
      onActivate={() => runAction('activate')}
      onRevoke={() => runAction('revoke')}
    />
  )
}
