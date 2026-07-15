import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertCircle,
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
  Square,
  Wrench,
  Zap
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  API_BASE,
  agentEndpointApi,
  deviceApi,
  hermesSetupApi,
  workspaceApi,
  type AgentEndpointDiscovery,
  type AgentEndpointProfile,
  type HermesSetupStatus,
  type HermesTunnelStatus,
  type SSHCandidate
} from '../../../services/api'
import { useTerminalStore, useWorkspaceStore } from '../../../stores'
import {
  registerHermesPromptSender,
  useAgentStreamStore
} from '../../../stores/agentStreamStore'
import { GhostPeonyIcon } from '../../common/GhostPeonyIcon'
import { Modal } from '../../common/Modal'
import { createAgentChatSurfaceActions } from './agentChatLifecycle'
import {
  composeAgentWorkspaceContext,
  hermesWorkspaceSessionKey,
} from './agentWorkspaceContext'
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

interface AgentStreamPayload {
  message: string
  context?: string
  conversation?: string
  session_key?: string | null
  workspace_id?: string
  origin?: {
    panel_id?: string
    terminal_id?: string
    agent?: string
  }
  history?: ChatLine[]
}

interface AgentStreamCallbacks {
  onDelta: (delta: string) => void
  onActivity: (label: string) => void
}

function createAgentSseParser(
  onEvent: (event: string, payload: Record<string, unknown>) => void
) {
  let buffer = ''

  const parseFrame = (frame: string) => {
    let eventName = 'message'
    const dataLines: string[] = []
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith(':')) continue
      if (line.startsWith('event:')) eventName = line.slice(6).trim() || 'message'
      if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
    }
    if (dataLines.length === 0) return
    try {
      const payload = JSON.parse(dataLines.join('\n'))
      if (payload && typeof payload === 'object') {
        onEvent(eventName, payload as Record<string, unknown>)
      }
    } catch {
      // A malformed upstream frame should not discard later valid stream events.
    }
  }

  const drain = (flush = false) => {
    while (true) {
      const boundary = /\r?\n\r?\n/.exec(buffer)
      if (!boundary || boundary.index === undefined) break
      parseFrame(buffer.slice(0, boundary.index))
      buffer = buffer.slice(boundary.index + boundary[0].length)
    }
    if (flush && buffer.trim()) {
      parseFrame(buffer)
      buffer = ''
    }
  }

  return {
    push(chunk: string) {
      buffer += chunk
      drain()
    },
    finish() {
      drain(true)
    }
  }
}

async function streamAgentChat(
  endpointId: string,
  payload: AgentStreamPayload,
  signal: AbortSignal,
  callbacks: AgentStreamCallbacks
): Promise<void> {
  const url = `${API_BASE}/agent/endpoints/${encodeURIComponent(endpointId)}/chat/stream`
  let streamError: string | null = null
  let streamDone = false
  const parser = createAgentSseParser((event, data) => {
    if (event === 'delta' && typeof data.delta === 'string') callbacks.onDelta(data.delta)
    if (event === 'activity' && typeof data.label === 'string') callbacks.onActivity(data.label)
    if (event === 'error') streamError = typeof data.error === 'string' ? data.error : 'Hermes stream failed'
    if (event === 'done') streamDone = true
  })
  const options: RequestInit = {
    method: 'POST',
    body: JSON.stringify(payload)
  }

  if (window.bashgym?.api?.stream) {
    await new Promise<void>((resolve, reject) => {
      let responseOk = true
      let responseStatus = 200
      let errorBody = ''
      let settled = false
      const settle = (error?: Error) => {
        if (settled) return
        settled = true
        signal.removeEventListener('abort', handleAbort)
        if (error) reject(error)
        else resolve()
      }
      const unsubscribe = window.bashgym.api.stream(url, options, (event) => {
        if (event.type === 'headers') {
          responseOk = event.ok
          responseStatus = event.status
        } else if (event.type === 'chunk') {
          if (responseOk) parser.push(event.data)
          else errorBody += event.data
          if (streamError) {
            unsubscribe()
            settle(new Error(streamError))
          }
        } else if (event.type === 'done') {
          parser.finish()
          if (!responseOk) {
            let detail = errorBody || `HTTP ${responseStatus}`
            try {
              const parsed = JSON.parse(errorBody)
              detail = parsed.detail || detail
            } catch {
              // Keep the response text when the backend did not return JSON.
            }
            settle(new Error(detail))
          } else if (streamError) {
            settle(new Error(streamError))
          } else if (!streamDone) {
            settle(new Error('Hermes stream ended before completion'))
          } else {
            settle()
          }
        } else if (event.type === 'aborted') {
          settle(new DOMException('The request was stopped', 'AbortError'))
        } else if (event.type === 'error') {
          settle(new Error(event.error))
        }
      })
      const handleAbort = () => {
        unsubscribe()
        settle(new DOMException('The request was stopped', 'AbortError'))
      }
      signal.addEventListener('abort', handleAbort, { once: true })
      if (signal.aborted) handleAbort()
    })
    return
  }

  const response = await fetch(url, {
    ...options,
    signal,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': localStorage.getItem('bashgym_api_key') || '',
      'X-Requested-With': 'XMLHttpRequest'
    }
  })
  if (!response.ok) {
    const body = await response.text()
    let detail = body || `HTTP ${response.status}`
    try {
      const parsed = JSON.parse(body)
      detail = parsed.detail || detail
    } catch {
      // Keep the response body when it is not JSON.
    }
    throw new Error(detail)
  }
  if (!response.body) throw new Error('Hermes stream returned no response body')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    parser.push(decoder.decode(value, { stream: true }))
    if (streamError) {
      await reader.cancel()
      throw new Error(streamError)
    }
  }
  parser.push(decoder.decode())
  parser.finish()
  if (streamError) throw new Error(streamError)
  if (!streamDone) throw new Error('Hermes stream ended before completion')
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

function buildAgentEndpointDetails(
  data: DataNodeData,
  endpointId: string,
  form: FormState,
  discovery: AgentEndpointDiscovery | null
): string {
  const terminalState = useTerminalStore.getState()
  const panels = terminalState.panels
  const edges = terminalState.canvasEdges
  const currentPanel = panels.find((panel) => panel.id === data.panelId)
  const linkedIds = new Set<string>()

  for (const edge of edges) {
    if (edge.source === data.panelId) linkedIds.add(edge.target)
    if (edge.target === data.panelId) linkedIds.add(edge.source)
  }

  const linkedPanels = panels.filter((panel) => linkedIds.has(panel.id))
  const lines: string[] = [
    '## Hermes endpoint and linked canvas details',
    '',
    `Generated: ${new Date().toISOString()}`,
    `Agent endpoint: ${form.label} (${endpointId})`,
    `Endpoint URL configured: ${form.baseUrl ? 'yes' : 'no'}`,
    `Requested model: ${form.model}`,
    `Memory session key set: ${form.sessionKey ? 'yes' : 'no'}`,
    `Panel: ${currentPanel?.title || data.title}`,
    '',
    '### Linked canvas details',
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

  lines.push('', '### BashGym handles')
  lines.push(`- workspace context: GET ${API_BASE}/workspace/context`)
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
  const workspaceId = useWorkspaceStore((state) => state.activeWorkspaceId)
  const publishHermesStream = useAgentStreamStore((state) => state.publishHermesStream)
  const removeHermesStream = useAgentStreamStore((state) => state.removeHermesStream)
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
  const [streamActivity, setStreamActivity] = useState<string | null>(null)
  const mountedRef = useRef(true)
  const promptRef = useRef<HTMLTextAreaElement>(null)
  const transcriptEndRef = useRef<HTMLDivElement>(null)
  const streamControllerRef = useRef<AbortController | null>(null)
  const streamTokenRef = useRef(0)
  const conversationRef = useRef(`bashgym-canvas-${data.panelId}`)

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
  const modelSelectWidth = `${Math.min(
    34,
    Math.max(18, ...modelOptions.map((model) => model.length + 4))
  )}ch`
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
      streamControllerRef.current?.abort()
    }
  }, [loadSetupStatus, loadTunnelStatus, probeEndpoint])

  useEffect(() => {
    if (!showChatModal) return
    const focusTimer = window.setTimeout(() => promptRef.current?.focus(), 80)
    return () => window.clearTimeout(focusTimer)
  }, [showChatModal])

  useEffect(() => {
    if (!showChatModal) return
    transcriptEndRef.current?.scrollIntoView({ behavior: sending ? 'auto' : 'smooth' })
  }, [messages, sending, showChatModal])

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

  const buildContext = useCallback(async () => {
    const endpointDetails = buildAgentEndpointDetails(data, endpointId, form, discovery)
    const response = await workspaceApi.getContext('markdown', workspaceId)
    return composeAgentWorkspaceContext(
      response.ok && typeof response.data === 'string' ? response.data : null,
      endpointDetails,
      response.error,
    )
  }, [data, discovery, endpointId, form, workspaceId])

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
        session_key: hermesWorkspaceSessionKey(form.sessionKey, workspaceId),
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
  }, [endpointId, form, probeEndpoint, workspaceId])

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
        session_key: hermesWorkspaceSessionKey(form.sessionKey, workspaceId),
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
  }, [endpointId, form, loadSetupStatus, probeEndpoint, sshTarget, workspaceId])

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

  const { dismiss: closeChat, stop: stopStreaming } = useMemo(
    () => createAgentChatSurfaceActions({
      abort: () => streamControllerRef.current?.abort(),
      hide: () => setShowChatModal(false),
    }),
    []
  )

  const clearChat = useCallback(() => {
    streamTokenRef.current += 1
    streamControllerRef.current?.abort()
    streamControllerRef.current = null
    conversationRef.current = `bashgym-canvas-${data.panelId}-${Date.now()}`
    setMessages([])
    setError(null)
    setStreamActivity(null)
    setSending(false)
  }, [data.panelId])

  const sendPrompt = useCallback(async (text: string = prompt) => {
    const trimmed = text.trim()
    if (!trimmed || sending) return
    const context = await buildContext()
    const history = messages.map((message) => ({ ...message }))
    const controller = new AbortController()
    const streamToken = streamTokenRef.current + 1
    streamTokenRef.current = streamToken
    streamControllerRef.current = controller
    setPrompt('')
    setSending(true)
    setError(null)
    setStreamActivity('Hermes is responding')
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: trimmed },
      { role: 'assistant', content: '' }
    ])
    try {
      await streamAgentChat(endpointId, {
        message: trimmed,
        context,
        conversation: conversationRef.current,
        session_key: hermesWorkspaceSessionKey(form.sessionKey, workspaceId),
        workspace_id: workspaceId,
        origin: {
          panel_id: data.panelId,
          agent: endpointId,
        },
        history
      }, controller.signal, {
        onDelta: (delta) => {
          if (streamTokenRef.current !== streamToken) return
          setStreamActivity(null)
          setMessages((prev) => {
            const last = prev[prev.length - 1]
            if (!last || last.role !== 'assistant') return prev
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + delta }
            ]
          })
        },
        onActivity: (label) => {
          if (streamTokenRef.current === streamToken) {
            setStreamActivity(`Using ${label}`)
          }
        }
      })
    } catch (streamError) {
      if (streamTokenRef.current !== streamToken) return
      if (streamError instanceof DOMException && streamError.name === 'AbortError') {
        setMessages((prev) => {
          const last = prev[prev.length - 1]
          return last?.role === 'assistant' && !last.content ? prev.slice(0, -1) : prev
        })
      } else {
        setError(streamError instanceof Error ? streamError.message : String(streamError))
      }
    } finally {
      if (mountedRef.current && streamTokenRef.current === streamToken) {
        streamControllerRef.current = null
        setStreamActivity(null)
        setSending(false)
        window.setTimeout(() => promptRef.current?.focus(), 0)
      }
    }
  }, [buildContext, data.panelId, endpointId, form.sessionKey, messages, prompt, sending, workspaceId])

  useEffect(() => registerHermesPromptSender(data.panelId, sendPrompt), [data.panelId, sendPrompt])

  useEffect(() => {
    publishHermesStream({
      panelId: data.panelId,
      label: form.label || 'Hermes',
      endpointId,
      messages,
      sending,
      activity: streamActivity,
      error: shortError(error)
    })
  }, [data.panelId, endpointId, error, form.label, messages, publishHermesStream, sending, streamActivity])

  useEffect(() => () => {
    removeHermesStream(data.panelId)
  }, [data.panelId, removeHermesStream])

  const quickPrompts = [
    { label: 'Evaluate a skill', prompt: 'Help me evaluate or improve a loaded skill in Skill Lab.' },
    { label: 'Training next step', prompt: 'Review active training and suggest next eval.' },
    { label: 'Canvas context', prompt: 'Explain what context this canvas gives you.' },
    { label: 'Run handoff', prompt: 'Help prepare a BashGym run handoff.' }
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
      flowerVariant="agent"
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={data.hasTerminalConnections ? buildContext : undefined}
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
          <div className="grid gap-2">
            <div className="node-section-title">
              <Settings2 className="w-3 h-3" />
              <span>Endpoint</span>
            </div>
            <div className="nodrag node-config-action-row">
              <button
                type="button"
                className="node-btn node-btn-wide node-btn-accent justify-center"
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
                className="node-config-toggle"
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
                  <div className="node-config-action-row">
                    <button
                      type="button"
                      className="node-btn node-btn-wide node-btn-accent justify-center"
                      onClick={() => void connectTunnel()}
                      disabled={tunnelBusy || !sshTarget.trim()}
                      title="Open an SSH port-forward to remote Hermes"
                    >
                      {tunnelBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
                      <span>Connect</span>
                    </button>
                    <button
                      type="button"
                      className="node-btn node-btn-wide justify-center"
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
                className="node-config-toggle"
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
            <div className="nodrag node-config-action-row">
              <button
                type="button"
                className="node-btn node-btn-wide node-btn-accent justify-center"
                onClick={() => void saveProfile()}
                disabled={saving || !endpointId}
                title="Save endpoint profile"
              >
                {saving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                <span>Save</span>
              </button>
              <button
                type="button"
                className="node-btn node-btn-wide justify-center"
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
            title={sending ? 'Hermes is responding — open chat to view progress' : 'Open latest Hermes reply'}
          >
            <span className="text-accent">{sending ? 'Responding:' : 'Latest:'}</span>{' '}
            <span className="break-words">
              {sending
                ? 'Hermes is continuing in the background…'
                : messages[messages.length - 1]?.content.slice(0, 120)}
            </span>
          </button>
        ) : null}
      </div>
    </DataNodeShell>
    <Modal
      isOpen={showChatModal}
      onClose={closeChat}
      title={`${form.label || 'Hermes'} Chat`}
      size="lg"
      variant="canvas"
    >
      <div className="flex h-[62vh] min-h-[420px] max-h-[620px] flex-col">
        <div className="flex flex-wrap items-center gap-2 border-b border-border-subtle px-1 pb-2">
          <label className="flex min-w-0 items-center gap-2">
            <span className="node-field-label flex-shrink-0">Model</span>
            <select
              className="input-brutal min-h-8 max-w-full py-1.5 text-[10px] font-mono"
              style={{ width: modelSelectWidth }}
              value={form.model}
              onChange={(event) => void selectModel(event.target.value)}
              disabled={sending}
            >
              {modelOptions.map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="node-btn node-btn-wide ml-auto"
            onClick={clearChat}
            disabled={messages.length === 0 && !error}
            title="Start a new conversation"
          >
            New chat
          </button>
        </div>

        <div
          className="min-h-0 flex-1 space-y-4 overflow-y-auto px-1 py-3"
          aria-live="polite"
        >
          {messages.length === 0 ? (
            <div className="flex min-h-full flex-col items-center justify-center gap-3 py-6 text-center">
              <div className="flex items-center gap-2 text-text-primary">
                <GhostPeonyIcon name="app" tone="color" size="sm" />
                <span className="font-mono text-xs font-bold uppercase tracking-wider">Ask Hermes</span>
              </div>
              <p className="max-w-md text-sm text-text-muted">
                Chat with the canvas context already attached.
              </p>
              <div className="flex max-w-2xl flex-wrap justify-center gap-2">
                {quickPrompts.map((quick) => (
                  <button
                    key={quick.label}
                    type="button"
                    className="node-btn node-btn-wide"
                    onClick={() => void sendPrompt(quick.prompt)}
                    disabled={sending}
                    title={quick.prompt}
                  >
                    {quick.label}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={clsx(
                  'flex items-start gap-2.5',
                  message.role === 'user' ? 'flex-row-reverse justify-start' : 'justify-start'
                )}
              >
                <GhostPeonyIcon
                  name="app"
                  tone="color"
                  size="lg"
                  className="mt-3 flex-shrink-0"
                  title={message.role === 'user' ? 'You' : 'Hermes'}
                />
                <div
                  className={clsx(
                    'text-sm leading-relaxed',
                    message.role === 'user'
                      ? 'max-w-[76%]'
                      : 'max-w-[92%]'
                  )}
                >
                  <div
                    className={clsx(
                      'mb-1 font-mono text-[9px] font-bold uppercase tracking-wider',
                      message.role === 'user' ? 'text-right text-text-muted' : 'text-accent-dark'
                    )}
                  >
                    {message.role === 'user' ? 'You' : 'Hermes'}
                  </div>
                  <div
                    className={clsx(
                      'px-3 py-2',
                      message.role === 'user'
                        ? 'bg-background-secondary text-text-primary'
                        : 'text-text-primary'
                    )}
                  >
                    <div className="prose-brutal">
                    {message.content ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                    ) : (
                      <div className="flex items-center gap-2 font-mono text-xs text-text-muted">
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />
                        <span>{streamActivity || 'Hermes is responding'}</span>
                      </div>
                    )}
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
          {sending && messages[messages.length - 1]?.content && streamActivity ? (
            <div className="flex items-center gap-2 pl-1 font-mono text-[10px] uppercase tracking-wider text-text-muted">
              <Loader2 className="h-3 w-3 animate-spin text-accent" />
              {streamActivity}
            </div>
          ) : null}
          <div ref={transcriptEndRef} />
        </div>

        {shortError(error) ? (
          <div className="mb-2 flex items-start gap-2 border-l-[3px] border-status-error bg-status-error/10 px-3 py-2 text-xs font-mono text-status-error">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
            <span>{shortError(error)}</span>
          </div>
        ) : null}

        <div className="border-t border-border-subtle pt-2">
          <div className="rounded-brutal border-brutal border-border bg-background-card px-2 py-1.5 shadow-brutal-sm transition-colors focus-within:border-text-primary">
          <textarea
            ref={promptRef}
            className="nowheel w-full resize-none bg-transparent px-1 py-1 text-sm leading-relaxed text-text-primary outline-none placeholder:text-text-muted"
            rows={2}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault()
                if (!sending) void sendPrompt()
              }
            }}
            placeholder="Message Hermes…"
            title="Enter to send · Shift+Enter for a new line"
          />
          <div className="flex items-center gap-3 px-1 pt-1">
            <span className="min-w-0 flex-1 truncate font-mono text-[9px] uppercase tracking-wider text-text-muted">
              {sending ? streamActivity || 'Streaming response' : 'Enter to send · Shift+Enter for new line'}
            </span>
            {sending ? (
              <button
                type="button"
                className="node-btn node-btn-wide node-btn-danger"
                onClick={stopStreaming}
                title="Stop the current response"
              >
                <Square className="h-3 w-3 fill-current" />
                Stop
              </button>
            ) : (
              <button
                type="button"
                className="node-btn node-btn-wide node-btn-accent"
                onClick={() => void sendPrompt()}
                disabled={!prompt.trim()}
                title="Send message"
              >
                <Send className="h-3 w-3" />
                Send
              </button>
            )}
          </div>
          </div>
        </div>
      </div>
    </Modal>
    </>
  )
})
