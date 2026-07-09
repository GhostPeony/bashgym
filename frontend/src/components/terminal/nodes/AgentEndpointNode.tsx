import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  AlertCircle,
  Bot,
  Brain,
  ChevronDown,
  CheckCircle2,
  KeyRound,
  Loader2,
  Maximize2,
  MessageCircle,
  RefreshCw,
  Save,
  Send,
  Settings2,
  ShieldCheck,
  Wrench,
  Zap
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  API_BASE,
  agentEndpointApi,
  deviceApi,
  hermesSetupApi,
  type AgentEndpointDiscovery,
  type AgentEndpointProfile,
  type HermesSetupStatus,
  type HermesTunnelStatus,
  type SSHCandidate
} from '../../../services/api'
import { useTerminalStore, useTrainingStore } from '../../../stores'
import { Modal } from '../../common/Modal'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type AgentEndpointNodeType = Node<DataNodeData, 'agent'>

interface FormState {
  label: string
  baseUrl: string
  model: string
  modelOptions: string[]
  sessionKey: string
  apiKey: string
  enabled: boolean
}

interface ChatLine {
  role: 'user' | 'assistant'
  content: string
}

const DEFAULT_FORM: FormState = {
  label: 'Hermes',
  baseUrl: 'http://127.0.0.1:8642/v1',
  model: 'hermes-agent',
  modelOptions: ['hermes-agent'],
  sessionKey: '',
  apiKey: '',
  enabled: true
}

function profileToForm(profile: AgentEndpointProfile | null): FormState {
  if (!profile) return DEFAULT_FORM
  return {
    label: profile.label,
    baseUrl: profile.base_url,
    model: profile.model,
    modelOptions: profile.model_options?.length ? profile.model_options : [profile.model],
    sessionKey: profile.session_key || '',
    apiKey: '',
    enabled: profile.enabled
  }
}

function shortError(value?: string | null): string | null {
  if (!value) return null
  return value.length > 140 ? `${value.slice(0, 137)}...` : value
}

/** Model names reported by the endpoint's /v1/models probe ({data: []} or {models: []}) */
function discoveredModels(discovery: AgentEndpointDiscovery | null): string[] {
  const data = discovery?.probes?.models?.data
  if (!data || typeof data !== 'object') return []
  const record = data as Record<string, unknown>
  const items = record.data ?? record.models
  if (!Array.isArray(items)) return []
  const names: string[] = []
  for (const item of items) {
    if (typeof item === 'string') names.push(item)
    else if (item && typeof item === 'object' && typeof (item as { id?: unknown }).id === 'string') {
      names.push((item as { id: string }).id)
    }
  }
  return names
}

function uniqueModels(...groups: Array<Array<string | null | undefined>>): string[] {
  const models: string[] = []
  for (const group of groups) {
    for (const value of group) {
      const model = value?.trim()
      if (model && !models.includes(model)) models.push(model)
    }
  }
  return models
}

function parseModelOptions(value: string, currentModel: string): string[] {
  return uniqueModels(
    [currentModel],
    value.split(/\r?\n|,/).map((item) => item.trim())
  )
}

function buildAgentContext(
  data: DataNodeData,
  endpointId: string,
  form: FormState,
  discovery: AgentEndpointDiscovery | null,
  messages: ChatLine[]
): string {
  const terminalState = useTerminalStore.getState()
  const trainingState = useTrainingStore.getState()
  const panels = terminalState.panels
  const edges = terminalState.canvasEdges
  const currentPanel = panels.find((panel) => panel.id === data.panelId)
  const linkedIds = new Set<string>()

  for (const edge of edges) {
    if (edge.source === data.panelId) linkedIds.add(edge.target)
    if (edge.target === data.panelId) linkedIds.add(edge.source)
  }

  const linkedPanels = panels.filter((panel) => linkedIds.has(panel.id))
  const run = trainingState.currentRun
  const lines: string[] = [
    '## BashGym workspace handoff',
    '',
    `Generated: ${new Date().toISOString()}`,
    `Agent endpoint: ${form.label} (${endpointId})`,
    `Endpoint URL configured: ${form.baseUrl ? 'yes' : 'no'}`,
    `Requested model: ${form.model}`,
    `Memory session key set: ${form.sessionKey ? 'yes' : 'no'}`,
    `Panel: ${currentPanel?.title || data.title}`,
    '',
    '### Canvas',
    `- panels: ${panels.length}`,
    `- connected panels: ${linkedPanels.length || 0}`
  ]

  if (linkedPanels.length) {
    for (const panel of linkedPanels.slice(0, 8)) {
      lines.push(`- linked: ${panel.title} (${panel.type})`)
    }
  }

  if (discovery) {
    lines.push(
      '',
      '### Agent capability probe',
      `- reachable: ${discovery.ok ? 'yes' : 'no'}`,
      `- auth configured: ${discovery.auth_configured ? 'yes' : 'no'}`,
      `- models: ${discovery.summary.models}`,
      `- skills: ${discovery.summary.skills}`,
      `- toolsets: ${discovery.summary.toolsets}`
    )
    for (const warning of discovery.warnings.slice(0, 5)) {
      lines.push(`- warning: ${warning}`)
    }
  }

  if (run) {
    const metrics = run.currentMetrics
    lines.push(
      '',
      '### Current training run',
      `- id: ${run.id}`,
      `- status: ${run.status}`,
      `- strategy: ${run.config.strategy}`,
      `- base model: ${run.config.baseModel || '(unset)'}`,
      `- dataset: ${run.config.datasetPath || '(unset)'}`,
      `- started: ${new Date(run.startTime).toISOString()}`
    )
    if (metrics) {
      lines.push(
        `- step: ${metrics.step}/${metrics.totalSteps}`,
        `- loss: ${metrics.loss}`,
        `- learning rate: ${metrics.learningRate}`
      )
    }
  } else {
    lines.push('', '### Current training run', '- no active run in renderer state')
  }

  if (messages.length) {
    lines.push('', '### Recent Hermes node chat')
    for (const msg of messages.slice(-6)) {
      lines.push(`- ${msg.role}: ${msg.content.slice(0, 500)}`)
    }
  }

  lines.push('', '### BashGym handles')
  lines.push(`- agent endpoints: GET ${API_BASE}/agent/endpoints`)
  lines.push(`- training runs: GET ${API_BASE}/training/runs`)
  lines.push(`- heldout evals: GET ${API_BASE}/eval/heldout`)
  lines.push(`- huggingface inventory: GET ${API_BASE}/hf/inventory`)

  return lines.join('\n')
}

export const AgentEndpointNode = memo(function AgentEndpointNode({
  data,
  selected
}: NodeProps<AgentEndpointNodeType>) {
  const [profiles, setProfiles] = useState<AgentEndpointProfile[]>([])
  const [selectedId, setSelectedId] = useState('hermes')
  const [form, setForm] = useState<FormState>(DEFAULT_FORM)
  const [discovery, setDiscovery] = useState<AgentEndpointDiscovery | null>(null)
  const [messages, setMessages] = useState<ChatLine[]>([])
  const [prompt, setPrompt] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [sending, setSending] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [showConnectionDetails, setShowConnectionDetails] = useState(false)
  const [showTunnelDetails, setShowTunnelDetails] = useState(false)
  const [showChatModal, setShowChatModal] = useState(false)
  const [setupStatus, setSetupStatus] = useState<HermesSetupStatus | null>(null)
  const [setupBusy, setSetupBusy] = useState(false)
  const [setupActions, setSetupActions] = useState<string[]>([])
  const [sshCandidates, setSshCandidates] = useState<SSHCandidate[]>([])
  const [sshTarget, setSshTarget] = useState('')
  const [tunnelStatus, setTunnelStatus] = useState<HermesTunnelStatus | null>(null)
  const [tunnelBusy, setTunnelBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(true)

  const selectedProfile = useMemo(
    () => profiles.find((profile) => profile.id === selectedId) || null,
    [profiles, selectedId]
  )
  const endpointId = selectedId.trim() || 'hermes'
  const hasSavedKey = selectedProfile?.api_key_configured ?? false
  const models = useMemo(() => discoveredModels(discovery), [discovery])
  const modelOptions = useMemo(
    () => uniqueModels([form.model], form.modelOptions, models),
    [form.model, form.modelOptions, models]
  )
  const sshCandidateOptions = useMemo(
    () => sshCandidates.map((candidate) => candidate.ssh_alias).filter(Boolean),
    [sshCandidates]
  )
  const sshTargetInOptions = sshTarget ? sshCandidateOptions.includes(sshTarget) : true
  const tunnelLabel = tunnelStatus?.active
    ? tunnelStatus.healthy
      ? 'Tunnel connected'
      : 'Tunnel open'
    : 'No tunnel'

  const loadSetupStatus = useCallback(async (id: string, baseUrl: string) => {
    const res = await hermesSetupApi.status({ endpointId: id || 'hermes', baseUrl })
    if (!mountedRef.current) return
    if (res.ok && res.data) {
      setSetupStatus(res.data)
    }
  }, [])

  const loadTunnelStatus = useCallback(async (id: string) => {
    const res = await hermesSetupApi.tunnelStatus(id || 'hermes')
    if (!mountedRef.current) return
    if (res.ok && res.data) {
      setTunnelStatus(res.data)
    }
  }, [])

  const loadSshCandidates = useCallback(async () => {
    const res = await deviceApi.discover()
    if (!mountedRef.current) return
    if (res.ok && res.data) {
      setSshCandidates(res.data.candidates)
      setSshTarget((current) => current || res.data!.candidates[0]?.ssh_alias || '')
    }
  }, [])

  const probeEndpoint = useCallback(async (id: string) => {
    setTesting(true)
    try {
      const res = await agentEndpointApi.discover(id)
      if (!mountedRef.current) return
      if (res.ok && res.data) {
        setDiscovery(res.data)
        setError(res.data.ok ? null : res.data.warnings[0] || 'Endpoint did not respond')
      } else {
        setError(res.error || 'Unable to test endpoint')
      }
    } finally {
      if (mountedRef.current) setTesting(false)
    }
  }, [])

  // Fetch profiles once on mount; select the first saved profile and probe it.
  // Never re-runs while the user edits, so typed endpoint ids are not clobbered.
  useEffect(() => {
    mountedRef.current = true
    void (async () => {
      const res = await agentEndpointApi.list()
      if (!mountedRef.current) return
      setLoaded(true)
      if (res.ok && res.data) {
        setProfiles(res.data.endpoints)
        const first = res.data.endpoints.find((p) => p.id === 'hermes') || res.data.endpoints[0]
        if (first) {
          setSelectedId(first.id)
          setForm(profileToForm(first))
          void loadSetupStatus(first.id, first.base_url)
          void loadTunnelStatus(first.id)
          if (first.enabled && first.api_key_configured) void probeEndpoint(first.id)
        }
        setError(null)
      } else {
        setError(res.error || 'Unable to load agent endpoints')
      }
    })()
    return () => {
      mountedRef.current = false
    }
  }, [loadSetupStatus, loadTunnelStatus, probeEndpoint])

  useEffect(() => {
    if (!showConfig) return
    void loadTunnelStatus(endpointId)
    if (sshCandidates.length === 0) void loadSshCandidates()
  }, [endpointId, loadSshCandidates, loadTunnelStatus, showConfig, sshCandidates.length])

  const statusBarClass = error
    ? 'bg-status-error'
    : discovery?.ok
      ? 'bg-status-success'
      : hasSavedKey
        ? 'bg-status-warning'
        : 'bg-background-tertiary'

  const updateForm = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }, [])

  const buildContext = useCallback(
    () => buildAgentContext(data, endpointId, form, discovery, messages),
    [data, endpointId, form, discovery, messages]
  )

  const persistProfile = useCallback(async (next: FormState) => {
    if (!endpointId) return
    setSaving(true)
    setError(null)
    try {
      const res = await agentEndpointApi.save(endpointId, {
        label: next.label,
        kind: 'hermes',
        base_url: next.baseUrl,
        model: next.model,
        model_options: next.modelOptions,
        session_key: next.sessionKey || null,
        enabled: next.enabled,
        api_key: next.apiKey || null
      })
      if (!mountedRef.current) return
      if (res.ok && res.data) {
        setProfiles((prev) => {
          const rest = prev.filter((profile) => profile.id !== res.data!.id)
          return [...rest, res.data!].sort((a, b) => a.id.localeCompare(b.id))
        })
        setForm(profileToForm(res.data))
      } else {
        setError(res.error || 'Unable to save endpoint')
      }
    } finally {
      if (mountedRef.current) setSaving(false)
    }
  }, [endpointId])

  const saveProfile = useCallback(async () => {
    await persistProfile(form)
    setDiscovery(null)
    void loadSetupStatus(endpointId, form.baseUrl)
  }, [endpointId, form, loadSetupStatus, persistProfile])

  const runQuickSetup = useCallback(async () => {
    setSetupBusy(true)
    setError(null)
    setSetupActions([])
    try {
      const res = await hermesSetupApi.quickSetup({
        profile_id: endpointId || 'hermes',
        label: form.label || 'Hermes',
        base_url: form.baseUrl || 'http://127.0.0.1:8642/v1',
        model: form.model || 'hermes-agent',
        model_options: form.modelOptions,
        session_key: form.sessionKey || 'bashgym-canvas',
        api_key: form.apiKey || null,
        write_env: true,
        start_gateway: true
      })
      if (!mountedRef.current) return
      if (res.ok && res.data) {
        setSetupStatus(res.data.status)
        setSetupActions(res.data.actions)
        setProfiles((prev) => {
          const next = res.data!.status.profile
          const rest = prev.filter((profile) => profile.id !== next.id)
          return [...rest, next].sort((a, b) => a.id.localeCompare(b.id))
        })
        setSelectedId(res.data.status.profile.id)
        setForm(profileToForm(res.data.status.profile))
        setShowConfig(true)
        if (res.data.status.profile.api_key_configured) void probeEndpoint(res.data.status.profile.id)
      } else {
        setError(res.error || 'Hermes quick setup failed')
      }
    } finally {
      if (mountedRef.current) setSetupBusy(false)
    }
  }, [endpointId, form, probeEndpoint])

  const connectTunnel = useCallback(async () => {
    const target = sshTarget.trim()
    if (!target) {
      setError('Choose an SSH host or enter an SSH config alias')
      setShowTunnelDetails(true)
      return
    }
    setTunnelBusy(true)
    setError(null)
    try {
      const res = await hermesSetupApi.connectTunnel({
        endpoint_id: endpointId || 'hermes',
        label: form.label || 'Hermes',
        ssh_target: target,
        remote_host: '127.0.0.1',
        remote_port: 8642,
        model: form.model || 'hermes-agent',
        model_options: form.modelOptions,
        session_key: form.sessionKey || 'bashgym-canvas',
        api_key: form.apiKey || null,
        save_profile: true
      })
      if (!mountedRef.current) return
      if (res.ok && res.data) {
        setTunnelStatus(res.data)
        if (res.data.profile) {
          setProfiles((prev) => {
            const rest = prev.filter((profile) => profile.id !== res.data!.profile!.id)
            return [...rest, res.data!.profile!].sort((a, b) => a.id.localeCompare(b.id))
          })
          setSelectedId(res.data.profile.id)
          setForm(profileToForm(res.data.profile))
          setShowConnectionDetails(false)
          void loadSetupStatus(res.data.profile.id, res.data.profile.base_url)
          if (res.data.profile.api_key_configured) void probeEndpoint(res.data.profile.id)
        }
        if (!res.data.healthy) {
          setError(res.data.health_error || 'Tunnel opened, but remote Hermes health check failed')
        }
      } else {
        setError(res.error || 'Unable to connect remote Hermes tunnel')
      }
    } finally {
      if (mountedRef.current) setTunnelBusy(false)
    }
  }, [endpointId, form, loadSetupStatus, probeEndpoint, sshTarget])

  const disconnectTunnel = useCallback(async () => {
    setTunnelBusy(true)
    setError(null)
    try {
      const res = await hermesSetupApi.disconnectTunnel(endpointId || 'hermes')
      if (!mountedRef.current) return
      if (res.ok && res.data) {
        setTunnelStatus(res.data)
      } else {
        setError(res.error || 'Unable to disconnect remote Hermes tunnel')
      }
    } finally {
      if (mountedRef.current) setTunnelBusy(false)
    }
  }, [endpointId])

  /** Requested model name; the served LLM is configured in the Hermes app itself */
  const selectModel = useCallback(async (model: string) => {
    const next = { ...form, model, modelOptions: uniqueModels([model], form.modelOptions) }
    setForm(next)
    if (selectedProfile) await persistProfile(next)
  }, [form, persistProfile, selectedProfile])

  const sendPrompt = useCallback(async (text: string = prompt) => {
    const trimmed = text.trim()
    if (!trimmed || sending) return
    setPrompt('')
    setSending(true)
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }])
    try {
      const res = await agentEndpointApi.chat(endpointId, {
        message: trimmed,
        context: buildContext(),
        conversation: `bashgym-canvas-${data.panelId}`
      })
      if (res.ok && res.data) {
        setMessages((prev) => [...prev, { role: 'assistant', content: res.data!.response }])
      } else {
        const message = res.error || 'Hermes did not return a response'
        setError(message)
        setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${message}` }])
      }
    } finally {
      if (mountedRef.current) setSending(false)
    }
  }, [buildContext, data.panelId, endpointId, prompt, sending])

  const quickPrompts = [
    'Review active training and suggest next eval.',
    'Explain what context this canvas gives you.',
    'Help prepare a BashGym run handoff.'
  ]

  const connectedLabel = discovery?.ok
    ? `${form.label} connected`
    : hasSavedKey
      ? `${form.label} — not verified`
      : 'API key needed'
  const setupChecks = [
    { label: 'CLI', ok: setupStatus?.installed ?? false },
    { label: 'Config', ok: Boolean(setupStatus?.env_api_enabled && setupStatus?.env_key_present) },
    { label: 'Gateway', ok: setupStatus?.gateway_healthy ?? false },
    { label: 'Canvas', ok: setupStatus?.profile.api_key_configured ?? hasSavedKey }
  ]
  return (
    <>
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      icon={Bot}
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={buildContext}
      statusBarClass={statusBarClass}
      hue={hueFor('agent')}
      onFocus={data.onFocus}
      onClose={data.onClose}
      headerRight={
        <>
          {testing || saving ? (
            <Loader2 className="w-3 h-3 animate-spin text-text-muted" />
          ) : discovery?.ok ? (
            <CheckCircle2 className="w-3 h-3 text-status-success" />
          ) : error ? (
            <AlertCircle className="w-3 h-3 text-status-error" />
          ) : (
            <ShieldCheck className="w-3 h-3 text-text-muted" />
          )}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setShowConfig(true)
            }}
            className={clsx('nodrag node-btn', showConfig ? 'node-btn-success' : 'node-btn-accent')}
            title="Configure endpoint"
          >
            <Settings2 className="w-3 h-3" />
          </button>
        </>
      }
    >
      <div className="space-y-2">
        <div className="flex items-center gap-1.5 text-[10px] font-mono min-w-0">
          <span className={clsx(
            'w-1.5 h-1.5 rounded-full flex-shrink-0',
            discovery?.ok ? 'bg-status-success' : hasSavedKey ? 'bg-status-warning' : 'bg-text-muted'
          )} />
          <span className="truncate text-text-secondary">{connectedLabel}</span>
        </div>

        {modelOptions.length > 0 ? (
          <label className="nodrag node-field">
            <span className="node-field-label">Hermes model</span>
            <select
              className="input-brutal text-[11px] font-mono min-h-9"
              value={form.model}
              onChange={(event) => void selectModel(event.target.value)}
              title="Model used by this Hermes endpoint profile"
            >
              {modelOptions.map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </label>
        ) : null}

        <button
          type="button"
          className="nodrag node-btn node-btn-wide node-btn-accent w-full justify-center min-h-9"
          onClick={(event) => {
            event.stopPropagation()
            setShowChatModal(true)
          }}
          title="Open Hermes chat"
        >
          <MessageCircle className="w-3 h-3" />
          <span>Open Chat</span>
          <Maximize2 className="w-3 h-3 ml-auto" />
        </button>

        {!hasSavedKey && loaded && (
          <div className="text-[10px] font-mono text-text-muted">
            Connect your Hermes API server: set an API key in
            <button
              type="button"
              className="nodrag ml-1 text-accent hover:underline"
              onClick={(e) => {
                e.stopPropagation()
                setShowConfig(true)
              }}
            >
              endpoint config
            </button>
          </div>
        )}

        <NodeConfigModal
          isOpen={showConfig}
          onClose={() => setShowConfig(false)}
          title={`${form.label || 'Hermes'} Endpoint Config`}
          description={`Profile: ${endpointId}`}
          size="xl"
        >
          <div className="space-y-3">
            <div className="node-section-title">
              <Settings2 className="w-3 h-3" />
              <span>Endpoint</span>
            </div>
            <div className="nodrag flex items-center gap-2">
              <button
                type="button"
                className="node-btn node-btn-wide node-btn-accent flex-1 justify-center"
                onClick={(event) => {
                  event.stopPropagation()
                  void runQuickSetup()
                }}
                disabled={setupBusy}
                title="Save profile, verify Hermes API settings, and start the gateway if needed"
              >
                {setupBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
                <span>{setupStatus?.gateway_healthy ? 'Refresh Setup' : 'Setup'}</span>
              </button>
              <div className="flex flex-wrap gap-1 justify-end">
                {setupChecks.map((check) => (
                  <span
                    key={check.label}
                    className={clsx(
                      'border-brutal rounded-brutal px-1.5 py-1 text-[8px] font-mono uppercase',
                      check.ok
                        ? 'border-status-success/60 bg-status-success/10 text-status-success'
                        : 'border-border-subtle bg-background-card text-text-muted'
                    )}
                    title={check.label}
                  >
                    {check.label}
                  </span>
                ))}
              </div>
            </div>
            {(setupStatus?.setup_needed.length || setupActions.length) ? (
              <div className="text-[9px] font-mono text-text-muted truncate" title={[...(setupActions || []), ...(setupStatus?.setup_needed || [])].join('\n')}>
                {setupActions[0] || setupStatus?.setup_needed[0]}
              </div>
            ) : null}
            <div className="nodrag grid grid-cols-2 gap-2">
              <label className="node-field">
                <span className="node-field-label">Profile</span>
              <select
                className="input-brutal text-[11px] font-mono min-h-9"
                value={selectedProfile ? selectedId : ''}
                onChange={(event) => {
                  const nextId = event.target.value
                  if (!nextId) {
                    setSelectedId('')
                    setForm(DEFAULT_FORM)
                    setDiscovery(null)
                    return
                  }
                  const next = profiles.find((profile) => profile.id === nextId)
                  setSelectedId(nextId)
                  setForm(profileToForm(next || null))
                  setDiscovery(null)
                }}
                title="Saved endpoint profile"
              >
                <option value="">New profile</option>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>{profile.label}</option>
                ))}
              </select>
              </label>
              <label className="node-field">
                <span className="node-field-label">Id</span>
              <input
                className="input-brutal text-[11px] font-mono min-h-9"
                value={selectedId}
                onChange={(event) => {
                  setSelectedId(event.target.value.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64))
                  setDiscovery(null)
                }}
                placeholder="endpoint id"
                title="Endpoint id"
              />
              </label>
            </div>
            <label className="nodrag node-field">
              <span className="node-field-label">Label</span>
            <input
              className="input-brutal text-[11px] font-mono min-h-9"
              value={form.label}
              onChange={(event) => updateForm('label', event.target.value)}
              placeholder="Hermes"
              title="Endpoint label"
            />
            </label>
            <label className="nodrag node-field">
              <span className="node-field-label">Model options</span>
              <textarea
                className="input-brutal nowheel text-[11px] font-mono min-h-[72px]"
                value={modelOptions.join('\n')}
                onChange={(event) => {
                  const nextOptions = parseModelOptions(event.target.value, form.model)
                  updateForm('modelOptions', nextOptions)
                }}
                placeholder="One model id per line"
                title="Models you want this Hermes profile to cycle between"
              />
            </label>
            <div className="nodrag node-section !p-2 space-y-2">
              <button
                type="button"
                className="node-btn node-btn-wide w-full justify-between"
                onClick={(event) => {
                  event.stopPropagation()
                  setShowTunnelDetails((current) => !current)
                }}
                title={showTunnelDetails ? 'Hide remote tunnel' : 'Show remote tunnel'}
              >
                <span>Remote tunnel</span>
                <span className={clsx(
                  'ml-auto mr-2 text-[9px] font-mono',
                  tunnelStatus?.active
                    ? tunnelStatus.healthy
                      ? 'text-status-success'
                      : 'text-status-warning'
                    : 'text-text-muted'
                )}>
                  {tunnelLabel}
                </span>
                <ChevronDown
                  className={clsx(
                    'w-3 h-3 transition-transform',
                    showTunnelDetails && 'rotate-180'
                  )}
                />
              </button>
              {showTunnelDetails ? (
                <div className="space-y-2">
                  {sshCandidateOptions.length > 0 ? (
                    <label className="node-field">
                      <span className="node-field-label">SSH host</span>
                      <select
                        className="input-brutal text-[11px] font-mono min-h-9"
                        value={sshTargetInOptions ? sshTarget : ''}
                        onChange={(event) => setSshTarget(event.target.value)}
                        title="SSH config host to forward remote Hermes through"
                      >
                        <option value="">Custom</option>
                        {sshCandidateOptions.map((alias) => (
                          <option key={alias} value={alias}>{alias}</option>
                        ))}
                      </select>
                    </label>
                  ) : null}
                  <label className="node-field">
                    <span className="node-field-label">SSH alias</span>
                    <input
                      className="input-brutal text-[11px] font-mono min-h-9"
                      value={sshTarget}
                      onChange={(event) => setSshTarget(event.target.value)}
                      placeholder="ssh config host"
                      title="Any SSH config alias or user@host target"
                    />
                  </label>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      className="node-btn node-btn-wide node-btn-accent flex-1 justify-center"
                      onClick={() => void connectTunnel()}
                      disabled={tunnelBusy || !sshTarget.trim()}
                      title="Open an SSH port-forward to remote Hermes"
                    >
                      {tunnelBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
                      <span>Connect</span>
                    </button>
                    <button
                      type="button"
                      className="node-btn node-btn-wide flex-1 justify-center"
                      onClick={() => void disconnectTunnel()}
                      disabled={tunnelBusy || !tunnelStatus?.active}
                      title="Stop the BashGym-managed tunnel"
                    >
                      <span>Disconnect</span>
                    </button>
                    <button
                      type="button"
                      className="node-btn"
                      onClick={() => void loadTunnelStatus(endpointId)}
                      disabled={tunnelBusy}
                      title="Refresh tunnel status"
                    >
                      <RefreshCw className="w-3 h-3" />
                    </button>
                  </div>
                  {tunnelStatus?.local_base_url ? (
                    <div className="text-[9px] font-mono text-text-muted truncate" title={tunnelStatus.local_base_url}>
                      Forward: {tunnelStatus.local_base_url}
                    </div>
                  ) : null}
                  {tunnelStatus?.health_error ? (
                    <div className="text-[9px] font-mono text-status-warning truncate" title={tunnelStatus.health_error}>
                      {tunnelStatus.health_error}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
            <div className="nodrag node-section !p-2 space-y-2">
              <button
                type="button"
                className="node-btn node-btn-wide w-full justify-between"
                onClick={(event) => {
                  event.stopPropagation()
                  setShowConnectionDetails((current) => !current)
                }}
                title={showConnectionDetails ? 'Hide connection details' : 'Show connection details'}
              >
                <span>Connection</span>
                <ChevronDown
                  className={clsx(
                    'w-3 h-3 transition-transform',
                    showConnectionDetails && 'rotate-180'
                  )}
                />
              </button>
              {showConnectionDetails ? (
                <label className="node-field">
                  <span className="node-field-label">API URL</span>
                  <input
                    className="input-brutal text-[11px] font-mono min-h-9"
                    value={form.baseUrl}
                    onChange={(event) => updateForm('baseUrl', event.target.value)}
                    placeholder="http://127.0.0.1:8642/v1"
                    title="Hermes API server URL"
                  />
                </label>
              ) : null}
            </div>
            <div className="grid grid-cols-1 gap-2">
              <label className="nodrag node-field">
                <span className="node-field-label">Session key</span>
              <input
                className="input-brutal text-[11px] font-mono min-h-9"
                value={form.sessionKey}
                onChange={(event) => updateForm('sessionKey', event.target.value)}
                placeholder="memory session key"
                title="X-Hermes-Session-Key — scopes Hermes long-term memory for this canvas"
              />
              </label>
              <label className="nodrag node-field">
                <span className="node-field-label">API key</span>
              <input
                className="input-brutal text-[11px] font-mono min-h-9"
                type="password"
                value={form.apiKey}
                onChange={(event) => updateForm('apiKey', event.target.value)}
                placeholder={hasSavedKey ? 'key saved — blank keeps it' : 'API_SERVER_KEY'}
                title="Bearer token matching API_SERVER_KEY on the Hermes side. Stored by BashGym, never echoed."
              />
              </label>
            </div>
            <div className="nodrag flex items-center gap-2">
              <button
                type="button"
                className="node-btn node-btn-wide node-btn-accent flex-1 justify-center"
                onClick={() => void saveProfile()}
                disabled={saving || !endpointId}
                title="Save endpoint profile"
              >
                {saving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                <span>Save</span>
              </button>
              <button
                type="button"
                className="node-btn node-btn-wide flex-1 justify-center"
                onClick={() => void probeEndpoint(endpointId)}
                disabled={testing || !loaded || !endpointId}
                title="Probe health, models, skills, and toolsets"
              >
                {testing ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                <span>Test</span>
              </button>
            </div>
            {discovery && (
              <div className="grid grid-cols-3 gap-1.5">
                {[
                  { icon: Brain, label: 'Models', value: discovery.summary.models },
                  { icon: Wrench, label: 'Skills', value: discovery.summary.skills },
                  { icon: KeyRound, label: 'Auth', value: discovery.auth_configured ? 'set' : 'none' }
                ].map(({ icon: Icon, label, value }) => (
                  <div key={label} className="border-brutal border-border-subtle rounded-brutal px-2 py-1 min-w-0">
                    <div className="flex items-center gap-1 text-[8px] font-mono text-text-muted uppercase">
                      <Icon className="w-2.5 h-2.5" />
                      <span>{label}</span>
                    </div>
                    <div className="text-[11px] font-mono font-semibold text-text-primary">{value}</div>
                  </div>
                ))}
              </div>
            )}
            {discovery?.warnings.length ? (
              <div className="text-[9px] font-mono text-text-muted truncate" title={discovery.warnings.join('\n')}>
                {discovery.warnings[0]}
              </div>
            ) : null}
          </div>
        </NodeConfigModal>

        {shortError(error) && (
          <div className="border-brutal border-status-error/50 bg-status-error/10 rounded-brutal px-2 py-1 text-[10px] font-mono text-status-error">
            {shortError(error)}
          </div>
        )}

        {messages.length > 0 ? (
          <button
            type="button"
            className="nodrag w-full text-left border-t border-brutal border-border-subtle pt-2 text-[10px] font-mono text-text-muted hover:text-text-primary"
            onClick={(event) => {
              event.stopPropagation()
              setShowChatModal(true)
            }}
            title="Open latest Hermes reply"
          >
            <span className="text-accent">Latest:</span>{' '}
            <span className="break-words">
              {messages[messages.length - 1]?.content.slice(0, 120)}
            </span>
          </button>
        ) : null}
      </div>
    </DataNodeShell>
    <Modal
      isOpen={showChatModal}
      onClose={() => setShowChatModal(false)}
      title={`${form.label || 'Hermes'} Chat`}
      description={`Model: ${form.model}`}
      size="xl"
      footer={
        <div className="flex items-center gap-2 w-full">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              setMessages([])
              setError(null)
            }}
          >
            Clear
          </button>
          <button
            type="button"
            className="btn-primary ml-auto"
            onClick={() => void sendPrompt()}
            disabled={sending || !prompt.trim()}
          >
            {sending ? 'Sending...' : 'Send'}
          </button>
        </div>
      }
    >
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_220px] gap-3">
          <label className="node-field">
            <span className="node-field-label">Message</span>
            <textarea
              className="input-brutal nowheel text-sm font-mono min-h-[140px]"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault()
                  void sendPrompt()
                }
              }}
              placeholder="Ask Hermes..."
              title="Ctrl+Enter to send"
            />
          </label>
          <div className="space-y-2">
            <label className="node-field">
              <span className="node-field-label">Model</span>
              <select
                className="input-brutal text-[11px] font-mono min-h-9"
                value={form.model}
                onChange={(event) => void selectModel(event.target.value)}
              >
                {modelOptions.map((model) => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </label>
            {quickPrompts.map((quick) => (
              <button
                key={quick}
                type="button"
                className="node-btn node-btn-wide w-full justify-start"
                onClick={() => void sendPrompt(quick)}
                disabled={sending}
                title={quick}
              >
                <Send className="w-3 h-3" />
                <span>{quick}</span>
              </button>
            ))}
          </div>
        </div>

        {shortError(error) ? (
          <div className="border-brutal border-status-error/50 bg-status-error/10 rounded-brutal px-3 py-2 text-xs font-mono text-status-error">
            {shortError(error)}
          </div>
        ) : null}

        <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary min-h-[280px] max-h-[44vh] overflow-y-auto p-3 space-y-3">
          {messages.length === 0 ? (
            <div className="text-sm font-mono text-text-muted">
              No messages yet.
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={clsx(
                  'border-brutal rounded-brutal px-3 py-2 font-mono text-sm',
                  message.role === 'assistant'
                    ? 'border-accent/50 bg-accent/10 text-text-primary'
                    : 'border-border-subtle bg-background-card text-text-secondary'
                )}
              >
                <div className="text-[10px] uppercase text-text-muted mb-1">
                  {message.role === 'assistant' ? 'Hermes' : 'You'}
                </div>
                <div className="whitespace-pre-wrap break-words">{message.content}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </Modal>
    </>
  )
})
