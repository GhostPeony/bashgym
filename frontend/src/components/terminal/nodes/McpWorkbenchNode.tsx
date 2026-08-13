import { memo, useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  AlertCircle,
  CheckCircle2,
  ClipboardCopy,
  FlaskConical,
  Loader2,
  Plug,
  RefreshCw,
  Search,
  Settings2,
  ShieldAlert,
  Square,
  Unplug,
  Wrench
} from 'lucide-react'
import { clsx } from 'clsx'
import { useTerminalStore } from '../../../stores'
import { DataNodeShell } from './DataNodeShell'
import type { NodeFlowerVariant } from '../nodeFlowerAssets'
import {
  ConfigPill,
  ConfigRow,
  ConfigRows,
  ConfigSection,
  NodeConfigModal
} from './NodeConfigModal'
import type { DataNodeData } from './types'
import {
  INITIAL_MCP_WORKBENCH_UI_STATE,
  buildMcpDiagnosticSummary,
  buildMcpTerminalContext,
  draftFromProfile,
  draftFromProfileInput,
  filterMcpTools,
  formatMcpVerificationResult,
  getConfiguredMcpWorkbenchApi,
  isMcpConnectionVerified,
  isMcpProfileDraftDirty,
  mcpServerIdentity,
  mcpTroubleshootingSteps,
  operationToConnectionState,
  parseManualToolArguments,
  parseMcpToolTestLimits,
  profileInputFromDraft,
  reduceMcpWorkbenchUi,
  sanitizeMcpAdapterConfig,
  type McpAdvancedSection,
  type McpCapabilitySnapshot,
  type McpClaudeImportCandidate,
  type McpConnectionState,
  type McpOperation,
  type McpOperationAccepted,
  type McpOAuthStatus,
  type McpProfile,
  type McpProfileDraft,
  type McpProfileInput,
  type McpTool
} from './mcpWorkbenchModel'

export type McpWorkbenchNodeType = Node<DataNodeData, 'mcp'>

const MCP_HUE = 258
const OPERATION_POLL_MS = 750
const MAX_OPERATION_POLLS = 240

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function operationDone(operation: McpOperation): boolean {
  return ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(operation.status)
}

function resultString(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function stateLabel(state: McpConnectionState): string {
  if (state === 'empty') return 'No profile'
  if (state === 'loading') return 'Working'
  if (state === 'connected') return 'Connected'
  if (state === 'stale') return 'Last known snapshot'
  if (state === 'error') return 'Needs attention'
  return 'Disconnected'
}

function stateTone(
  state: McpConnectionState
): 'neutral' | 'accent' | 'success' | 'warning' | 'error' {
  if (state === 'loading') return 'accent'
  if (state === 'connected') return 'success'
  if (state === 'stale') return 'warning'
  if (state === 'error') return 'error'
  return 'neutral'
}

function statusBarClass(state: McpConnectionState): string {
  if (state === 'connected') return 'bg-status-success'
  if (state === 'loading') return 'bg-accent canvas-state-strip-live'
  if (state === 'stale') return 'bg-status-warning'
  if (state === 'error') return 'bg-status-error'
  return 'bg-background-tertiary'
}

function profileValidationError(draft: McpProfileDraft): string | null {
  if (!draft.label.trim()) return 'Give this MCP connection a label.'
  if (draft.transport === 'streamable_http' && !draft.remoteUrl.trim()) {
    return 'Enter a Streamable HTTP MCP URL.'
  }
  if (
    draft.transport === 'streamable_http' &&
    draft.oauthClientSecretRef.trim() &&
    !draft.oauthClientId.trim()
  ) {
    return 'OAuth client secret reference requires a client ID.'
  }
  if (draft.transport === 'stdio' && !draft.command.trim()) {
    return 'Enter a local executable. Arguments stay in their own list.'
  }
  if (draft.transport === 'stdio' && draft.cwdPolicy === 'explicit' && !draft.explicitCwd.trim()) {
    return 'Enter the explicit working directory or choose a safer directory policy.'
  }
  return null
}

function MetricTile({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="min-w-0 border-brutal border-border-subtle rounded-brutal bg-background-card px-2 py-1.5">
      <div className="node-field-label">{label}</div>
      <div
        className="mt-1 truncate text-[11px] font-mono font-semibold text-text-primary"
        title={String(value)}
      >
        {value}
      </div>
    </div>
  )
}

function toolDisplayName(tool: McpTool): string {
  const title = tool.title?.trim()
  if (title) return title
  return tool.name
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toLocaleUpperCase()}${part.slice(1)}`)
    .join(' ')
}

function toolPolicyPresentation(policy: McpTool['policy']): {
  label: string
  description: string
  tone: 'success' | 'warning' | 'error' | 'neutral'
} {
  if (policy === 'allow') {
    return {
      label: 'Ready to run',
      description: 'Runs without asking for another confirmation.',
      tone: 'success'
    }
  }
  if (policy === 'ask') {
    return {
      label: 'Asks before running',
      description: 'You review and approve this tool before it can make a call.',
      tone: 'warning'
    }
  }
  if (policy === 'deny') {
    return {
      label: 'Blocked',
      description: 'This tool cannot run under the current server policy.',
      tone: 'error'
    }
  }
  return {
    label: 'Not classified',
    description: 'BashGym will pause for review before this tool runs.',
    tone: 'neutral'
  }
}

function formatToolLatency(latencyMs?: number | null): string {
  if (latencyMs == null) return 'Not measured'
  if (latencyMs < 1000) return `${Math.round(latencyMs)} ms`
  return `${(latencyMs / 1000).toFixed(1)} sec`
}

function formatToolReliability(errorRate?: number | null): string {
  if (errorRate == null) return 'Not measured'
  if (errorRate === 0) return 'No errors recorded'
  return `${(errorRate * 100).toFixed(1)}% errors`
}

function ToolListRow({
  tool,
  selected,
  onSelect
}: {
  tool: McpTool
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={clsx(
        'w-full min-w-0 p-2 text-left border-brutal rounded-brutal transition-colors',
        selected
          ? 'border-accent bg-accent/10 text-text-primary'
          : 'border-border-subtle bg-background-card text-text-secondary hover:border-border'
      )}
    >
      <div className="line-clamp-2 text-[10px] font-mono font-bold" title={toolDisplayName(tool)}>
        {toolDisplayName(tool)}
      </div>
      {tool.description ? (
        <div className="mt-1 line-clamp-2 text-[10px] leading-4 text-text-secondary">
          {tool.description}
        </div>
      ) : null}
    </button>
  )
}

export const McpWorkbenchNode = memo(function McpWorkbenchNode({
  data,
  selected
}: NodeProps<McpWorkbenchNodeType>) {
  const initialAdapterConfig = useMemo(
    () => sanitizeMcpAdapterConfig(data.adapterConfig),
    [data.adapterConfig]
  )
  const [ui, dispatchUi] = useReducer(reduceMcpWorkbenchUi, {
    ...INITIAL_MCP_WORKBENCH_UI_STATE,
    mode: initialAdapterConfig.view ?? 'simple',
    section: initialAdapterConfig.advanced_section ?? 'overview',
    selectedToolName: initialAdapterConfig.selected_tool ?? null
  })
  const [profiles, setProfiles] = useState<McpProfile[]>([])
  const [profileId, setProfileId] = useState(initialAdapterConfig.profile_id ?? '')
  const [draft, setDraft] = useState<McpProfileDraft>(() => draftFromProfile(null))
  const [snapshot, setSnapshot] = useState<McpCapabilitySnapshot | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [operation, setOperation] = useState<McpOperation | null>(null)
  const [loadingProfiles, setLoadingProfiles] = useState(true)
  const [savingProfile, setSavingProfile] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [quickTestResult, setQuickTestResult] = useState<string | null>(null)
  const [callResult, setCallResult] = useState<string>('')
  const [claudeImportJson, setClaudeImportJson] = useState('')
  const [claudeImportScope, setClaudeImportScope] = useState<'local' | 'project' | 'user'>(
    'project'
  )
  const [claudeImportCandidates, setClaudeImportCandidates] = useState<McpClaudeImportCandidate[]>(
    []
  )
  const [diagnosticsCopied, setDiagnosticsCopied] = useState(false)
  const [oauthStatus, setOauthStatus] = useState<McpOAuthStatus | null>(null)
  const contentScrollRef = useRef<HTMLDivElement>(null)
  const mountedRef = useRef(true)
  const adapterConfigRef = useRef(initialAdapterConfig)
  const initialProfileIdRef = useRef(initialAdapterConfig.profile_id ?? '')
  const updatePanelConfig = useTerminalStore((state) => state.updatePanelConfig)
  const renamePanel = useTerminalStore((state) => state.renamePanel)

  const profile = useMemo(
    () => profiles.find((candidate) => candidate.profile_id === profileId) ?? null,
    [profileId, profiles]
  )
  const selectedTool = useMemo(
    () => snapshot?.tools.find((tool) => tool.name === ui.selectedToolName) ?? null,
    [snapshot, ui.selectedToolName]
  )
  const filteredTools = useMemo(
    () => filterMcpTools(snapshot?.tools ?? [], ui.toolQuery),
    [snapshot, ui.toolQuery]
  )
  const selectedToolPolicy = toolPolicyPresentation(selectedTool?.policy)
  const selectedToolUsageCount = selectedTool?.usage_count ?? 0
  const connectionState = operationToConnectionState(
    operation,
    Boolean(profile),
    isMcpConnectionVerified(sessionId, snapshot),
    Boolean(snapshot?.stale)
  )
  const displayTitle = profile?.label.trim() || 'MCP Server'
  const connectionVerified = isMcpConnectionVerified(sessionId, snapshot)
  const serverIdentity = mcpServerIdentity(snapshot)
  const troubleshootingSteps = mcpTroubleshootingSteps(profile, operation)
  const draftDirty = isMcpProfileDraftDirty(draft, profile)

  useEffect(() => {
    if (data.title !== displayTitle) renamePanel(data.panelId, displayTitle)
  }, [data.panelId, data.title, displayTitle, renamePanel])

  const persistPresentation = useCallback(
    (updates: Record<string, unknown>) => {
      const safe = sanitizeMcpAdapterConfig({
        ...adapterConfigRef.current,
        ...updates
      })
      adapterConfigRef.current = safe
      updatePanelConfig(data.panelId, { ...safe })
    },
    [data.panelId, updatePanelConfig]
  )

  useEffect(() => {
    adapterConfigRef.current = sanitizeMcpAdapterConfig(data.adapterConfig)
  }, [data.adapterConfig])

  const loadSnapshot = useCallback(async (nextProfileId: string) => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api || !nextProfileId) {
      setSnapshot(null)
      return
    }
    const response = await api.snapshot(nextProfileId)
    if (!mountedRef.current) return
    if (response.ok && response.data) {
      setSnapshot(response.data)
      setError(null)
    } else {
      setSnapshot(null)
      if (response.error && !response.error.toLowerCase().includes('not found')) {
        setError(response.error)
      }
    }
  }, [])

  const loadOAuthStatus = useCallback(async (nextProfile: McpProfile | null) => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api || !nextProfile || nextProfile.transport !== 'streamable_http') {
      setOauthStatus(null)
      return
    }
    const response = await api.oauthStatus(nextProfile.profile_id)
    if (!mountedRef.current) return
    setOauthStatus(response.ok && response.data ? response.data : null)
  }, [])

  const loadProfiles = useCallback(async () => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api) {
      setLoadingProfiles(false)
      setError(
        'MCP API adapter is not configured. Start the BashGym backend and reload the canvas.'
      )
      return
    }
    setLoadingProfiles(true)
    const response = await api.listProfiles()
    if (!mountedRef.current) return
    setLoadingProfiles(false)
    if (!response.ok || !response.data) {
      setError(response.error || 'Unable to load MCP profiles.')
      return
    }
    setProfiles(response.data)
    const configured = response.data.find(
      (candidate) => candidate.profile_id === initialProfileIdRef.current
    )
    const nextProfile = configured ?? response.data[0] ?? null
    if (nextProfile) {
      setProfileId(nextProfile.profile_id)
      initialProfileIdRef.current = nextProfile.profile_id
      setDraft(draftFromProfile(nextProfile))
      setSessionId(nextProfile.active_session_id ?? null)
      persistPresentation({ profile_id: nextProfile.profile_id })
      await Promise.all([loadSnapshot(nextProfile.profile_id), loadOAuthStatus(nextProfile)])
    } else {
      setDraft(draftFromProfile(null))
      setSnapshot(null)
      setSessionId(null)
      setOauthStatus(null)
    }
  }, [loadOAuthStatus, loadSnapshot, persistPresentation])

  useEffect(() => {
    mountedRef.current = true
    void loadProfiles()
    return () => {
      mountedRef.current = false
    }
  }, [loadProfiles])

  useEffect(() => {
    if (!snapshot?.tools.length) return
    const stillExists = snapshot.tools.some((tool) => tool.name === ui.selectedToolName)
    if (!stillExists) dispatchUi({ type: 'select_tool', toolName: snapshot.tools[0].name })
  }, [snapshot, ui.selectedToolName])

  useEffect(() => {
    if (ui.mode !== 'advanced' || !contentScrollRef.current) return
    contentScrollRef.current.scrollTop =
      ui.section === 'overview' ? ui.overviewScrollTop : ui.toolsScrollTop
  }, [ui.mode, ui.overviewScrollTop, ui.section, ui.toolsScrollTop])

  const waitForOperation = useCallback(async (accepted: McpOperationAccepted) => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api) throw new Error('MCP API adapter is not configured.')
    let last: McpOperation = {
      operation_id: accepted.operation_id,
      status: accepted.status
    }
    setOperation(last)
    for (let poll = 0; poll < MAX_OPERATION_POLLS; poll += 1) {
      await delay(OPERATION_POLL_MS)
      if (!mountedRef.current) return last
      const response = await api.getOperation(accepted.operation_id)
      if (!response.ok || !response.data) {
        throw new Error(response.error || 'Unable to read MCP operation status.')
      }
      last = response.data
      setOperation(last)
      if (operationDone(last)) return last
    }
    throw new Error('MCP operation exceeded the local polling window. It may still be running.')
  }, [])

  const runManagedOperation = useCallback(
    async (
      start: () => Promise<{ ok: boolean; data?: McpOperationAccepted; error?: string }>,
      onSuccess: (completed: McpOperation) => void | Promise<void>
    ) => {
      setError(null)
      const accepted = await start()
      if (!accepted.ok || !accepted.data) {
        setError(accepted.error || 'MCP operation could not start.')
        return
      }
      try {
        const completed = await waitForOperation(accepted.data)
        if (completed.status === 'succeeded') {
          await onSuccess(completed)
        } else {
          setError(completed.error || `MCP operation ${completed.status}.`)
        }
      } catch (operationError) {
        if (mountedRef.current) {
          setError(
            operationError instanceof Error ? operationError.message : String(operationError)
          )
        }
      }
    },
    [waitForOperation]
  )

  const selectProfile = useCallback(
    (nextProfileId: string) => {
      const next = profiles.find((candidate) => candidate.profile_id === nextProfileId) ?? null
      setProfileId(nextProfileId)
      setDraft(draftFromProfile(next))
      setSessionId(next?.active_session_id ?? null)
      setOperation(null)
      setError(null)
      setQuickTestResult(null)
      setCallResult('')
      persistPresentation({ profile_id: nextProfileId })
      void loadSnapshot(nextProfileId)
      void loadOAuthStatus(next)
    },
    [loadOAuthStatus, loadSnapshot, persistPresentation, profiles]
  )

  const saveProfile = useCallback(async (): Promise<McpProfile | null> => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api) {
      setError('MCP API adapter is not configured.')
      return null
    }
    const validationError = profileValidationError(draft)
    if (validationError) {
      setError(validationError)
      return null
    }
    setSavingProfile(true)
    setError(null)
    let input: McpProfileInput
    try {
      input = profileInputFromDraft(draft)
    } catch (referenceError) {
      setSavingProfile(false)
      setError(referenceError instanceof Error ? referenceError.message : String(referenceError))
      return null
    }
    const response = profile
      ? await api.updateProfile(profile.profile_id, input, profile.profile_revision)
      : await api.createProfile(input)
    if (!mountedRef.current) return null
    setSavingProfile(false)
    if (!response.ok || !response.data) {
      setError(response.error || 'Unable to save MCP profile.')
      return null
    }
    const saved = response.data
    setProfiles((current) => {
      const remaining = current.filter((candidate) => candidate.profile_id !== saved.profile_id)
      return [...remaining, saved].sort((left, right) => left.label.localeCompare(right.label))
    })
    setProfileId(saved.profile_id)
    setDraft(draftFromProfile(saved))
    setSessionId(saved.active_session_id ?? null)
    persistPresentation({ profile_id: saved.profile_id })
    await loadOAuthStatus(saved)
    return saved
  }, [draft, loadOAuthStatus, persistPresentation, profile])

  const connectProfile = useCallback(
    (targetProfile: McpProfile) => {
      const api = getConfiguredMcpWorkbenchApi()
      if (!api) return
      void runManagedOperation(
        async () => {
          if (targetProfile.transport === 'stdio') {
            const preview = await api.previewStdio(
              targetProfile.profile_id,
              targetProfile.profile_revision
            )
            if (!preview.ok || !preview.data)
              return { ok: false, error: preview.error || 'Unable to preview the stdio launch.' }
            const accepted = window.confirm(
              [
                'Approve this local MCP process?',
                '',
                `Executable: ${preview.data.command}`,
                `Arguments: ${preview.data.args.join(' ') || '(none)'}`,
                `Working directory: ${preview.data.cwd_policy}`,
                `Environment names: ${preview.data.env_names.join(', ') || '(none)'}`,
                `Sandbox policy: ${preview.data.sandbox_policy}`,
                '',
                'Any change to this fingerprint requires approval again.'
              ].join('\n')
            )
            if (!accepted) return { ok: false, error: 'Local MCP launch was not approved.' }
            const approval = await api.approveStdio(
              targetProfile.profile_id,
              targetProfile.profile_revision,
              preview.data.executable.sha256,
              preview.data.launch_fingerprint
            )
            if (!approval.ok)
              return { ok: false, error: approval.error || 'Unable to save the launch approval.' }
          }
          return api.connect(targetProfile.profile_id, targetProfile.profile_revision)
        },
        async (completed) => {
          const nextSessionId = completed.result?.session_id
          if (typeof nextSessionId === 'string') setSessionId(nextSessionId)
          await Promise.all([
            loadSnapshot(targetProfile.profile_id),
            loadOAuthStatus(targetProfile)
          ])
        }
      )
    },
    [loadOAuthStatus, loadSnapshot, runManagedOperation]
  )

  const connect = useCallback(() => {
    if (!profile) return
    if (draftDirty) {
      setError(
        'Save these profile changes before connecting so the displayed URL and authentication settings are the ones actually tested.'
      )
      return
    }
    connectProfile(profile)
  }, [connectProfile, draftDirty, profile])

  const saveAndConnect = useCallback(async () => {
    const saved = await saveProfile()
    if (saved) connectProfile(saved)
  }, [connectProfile, saveProfile])

  const refresh = useCallback(() => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api || !profile) return
    void runManagedOperation(
      () => api.refresh(profile.profile_id),
      () => loadSnapshot(profile.profile_id)
    )
  }, [loadSnapshot, profile, runManagedOperation])

  const disconnect = useCallback(() => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api || !sessionId) return
    void runManagedOperation(
      () => api.disconnect(sessionId),
      () => {
        setSessionId(null)
        setSnapshot((current) => (current ? { ...current, stale: true } : current))
      }
    )
  }, [runManagedOperation, sessionId])

  const quickTest = useCallback(() => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api || !profile) return
    setQuickTestResult(null)
    void runManagedOperation(
      () => api.quickTest(profile.profile_id),
      async (completed) => {
        await loadSnapshot(profile.profile_id)
        setQuickTestResult(formatMcpVerificationResult(completed.result))
      }
    )
  }, [loadSnapshot, profile, runManagedOperation])

  const selfTest = useCallback(() => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api) return
    setQuickTestResult(null)
    void runManagedOperation(
      () => api.selfTest(),
      (completed) =>
        setQuickTestResult(resultString(completed.result) || 'Reference MCP self-test passed.')
    )
  }, [runManagedOperation])

  const previewClaudeImport = useCallback(async () => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api) return
    let config: Record<string, unknown>
    try {
      const parsed = JSON.parse(claudeImportJson) as unknown
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('Claude MCP config must be a JSON object.')
      }
      config = parsed as Record<string, unknown>
    } catch (parseError) {
      setError(parseError instanceof Error ? parseError.message : 'Invalid JSON config.')
      return
    }
    setError(null)
    const response = await api.previewClaudeConfig(config, claudeImportScope)
    if (!mountedRef.current) return
    if (!response.ok || !response.data) {
      setClaudeImportCandidates([])
      setError(response.error || 'Unable to preview Claude MCP config.')
      return
    }
    setClaudeImportCandidates(response.data)
  }, [claudeImportJson, claudeImportScope])

  const applyClaudeImportCandidate = useCallback(
    (candidate: McpClaudeImportCandidate) => {
      if (!candidate.supported || !candidate.profile_input) return
      setProfileId('')
      setSessionId(null)
      setSnapshot(null)
      setOperation(null)
      setDraft(draftFromProfileInput(candidate.profile_input))
      setError(null)
      persistPresentation({ profile_id: '' })
    },
    [persistPresentation]
  )

  const callTool = useCallback(() => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api || !sessionId || !selectedTool) return
    let argumentsValue: Record<string, unknown>
    try {
      argumentsValue = parseManualToolArguments(ui.manualInput)
      const limits = parseMcpToolTestLimits(ui.timeoutSeconds, ui.maxResultKilobytes)
      setError(null)
      setCallResult('')
      let approved = selectedTool.policy === 'allow'
      let typedConfirmation: string | undefined
      const destructive = selectedTool.annotations?.destructiveHint === true
      if (!approved && destructive) {
        typedConfirmation =
          window.prompt(`Type ${selectedTool.name} to approve this destructive call.`) || undefined
        approved = typedConfirmation === selectedTool.name
      } else if (!approved) {
        approved = window.confirm(`Allow ${selectedTool.name} once with the displayed arguments?`)
      }
      if (!approved) return
      void runManagedOperation(
        () =>
          api.callTool(sessionId, selectedTool.name, argumentsValue, {
            approved,
            typed_confirmation: typedConfirmation,
            ...limits
          }),
        (completed) => setCallResult(resultString(completed.result) || 'Tool returned no content.')
      )
    } catch (inputError) {
      setError(inputError instanceof Error ? inputError.message : String(inputError))
    }
  }, [
    runManagedOperation,
    selectedTool,
    sessionId,
    ui.manualInput,
    ui.maxResultKilobytes,
    ui.timeoutSeconds
  ])

  const cancelOperation = useCallback(async () => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api || !operation || operationDone(operation)) return
    const response = await api.cancelOperation(operation.operation_id)
    if (!mountedRef.current) return
    if (response.ok && response.data) setOperation(response.data)
    else setError(response.error || 'Unable to cancel MCP operation.')
  }, [operation])

  const copyDiagnostics = useCallback(async () => {
    const summary = buildMcpDiagnosticSummary(
      profile,
      snapshot,
      sessionId,
      operation,
      connectionState
    )
    try {
      await navigator.clipboard.writeText(summary)
      setDiagnosticsCopied(true)
      window.setTimeout(() => setDiagnosticsCopied(false), 1800)
    } catch {
      setError('Unable to copy diagnostics. Clipboard access may be disabled.')
    }
  }, [connectionState, operation, profile, sessionId, snapshot])

  const clearOAuth = useCallback(async () => {
    const api = getConfiguredMcpWorkbenchApi()
    if (!api || !profile || profile.transport !== 'streamable_http') return
    const response = await api.logoutOAuth(profile.profile_id)
    if (!mountedRef.current) return
    if (!response.ok) {
      setError(response.error || 'Unable to clear hosted MCP sign-in.')
      return
    }
    setSessionId(null)
    setSnapshot((current) => (current ? { ...current, stale: true } : current))
    await loadOAuthStatus(profile)
  }, [loadOAuthStatus, profile])

  const rememberScroll = useCallback(() => {
    if (!contentScrollRef.current) return
    dispatchUi({
      type: 'remember_scroll',
      section: ui.section,
      scrollTop: contentScrollRef.current.scrollTop
    })
  }, [ui.section])

  const openAdvanced = useCallback(
    (section: McpAdvancedSection = ui.section) => {
      dispatchUi({ type: 'open_advanced', section })
      persistPresentation({ view: 'advanced', advanced_section: section })
    },
    [persistPresentation, ui.section]
  )

  const closeAdvanced = useCallback(() => {
    rememberScroll()
    dispatchUi({ type: 'close_advanced' })
    persistPresentation({ view: 'simple' })
  }, [persistPresentation, rememberScroll])

  const selectSection = useCallback(
    (section: McpAdvancedSection) => {
      rememberScroll()
      dispatchUi({ type: 'select_section', section })
      persistPresentation({ advanced_section: section })
    },
    [persistPresentation, rememberScroll]
  )

  const selectTool = useCallback(
    (toolName: string) => {
      dispatchUi({ type: 'select_tool', toolName })
      persistPresentation({ selected_tool: toolName })
    },
    [persistPresentation]
  )

  const contextBuilder = useCallback(
    () => buildMcpTerminalContext(profile, snapshot, connectionState),
    [connectionState, profile, snapshot]
  )

  const warningCount = snapshot?.schema_warnings?.length ?? 0
  const isWorking = connectionState === 'loading'
  const profileTime = profile?.updated_at
    ? new Date(profile.updated_at).toLocaleString()
    : undefined

  return (
    <>
      <DataNodeShell
        panelId={data.panelId}
        title={displayTitle}
        flowerVariant={'mcp' as NodeFlowerVariant}
        selected={selected}
        hasConnections={data.hasConnections}
        buildContext={data.hasTerminalConnections ? contextBuilder : undefined}
        statusBarClass={statusBarClass(connectionState)}
        hue={MCP_HUE}
        visualPhase={isWorking ? 'running' : undefined}
        onFocus={data.onFocus}
        onClose={data.onClose}
        headerRight={
          <>
            {isWorking ? (
              <Loader2 className="w-3 h-3 animate-spin text-accent" />
            ) : connectionState === 'connected' ? (
              <CheckCircle2 className="w-3 h-3 text-status-success" />
            ) : connectionState === 'error' ? (
              <AlertCircle className="w-3 h-3 text-status-error" />
            ) : (
              <Plug className="w-3 h-3 text-text-muted" />
            )}
            <button
              type="button"
              className="nodrag node-btn node-btn-accent"
              onClick={(event) => {
                event.stopPropagation()
                openAdvanced('overview')
              }}
              title={`Configure ${displayTitle}`}
            >
              <Settings2 className="w-3 h-3" />
            </button>
          </>
        }
      >
        <div className="space-y-2">
          {loadingProfiles ? (
            <div className="flex items-center justify-center gap-2 py-5 text-[10px] font-mono text-text-muted">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading MCP profiles
            </div>
          ) : !profile ? (
            <div className="node-section text-center">
              <div className="mx-auto mb-2 flex h-9 w-9 items-center justify-center rounded-brutal border-brutal border-accent bg-accent/10 text-[11px] font-brand font-bold text-accent-dark">
                MCP
              </div>
              <div className="text-[11px] font-mono font-semibold text-text-primary">
                No MCP selected
              </div>
              <p className="mt-1 text-[10px] leading-4 text-text-muted">
                Add a public Streamable HTTP URL or an approved local command.
              </p>
              <button
                type="button"
                className="nodrag node-btn node-btn-wide node-btn-accent mt-2"
                onClick={(event) => {
                  event.stopPropagation()
                  openAdvanced('overview')
                }}
              >
                <Plug className="w-3 h-3" />
                Add connection
              </button>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-2 min-w-0">
                <ConfigPill tone={stateTone(connectionState)}>
                  {stateLabel(connectionState)}
                </ConfigPill>
                <span
                  className="min-w-0 flex-1 truncate text-[10px] font-mono text-text-secondary"
                  title={profile.label}
                >
                  {profile.label}
                </span>
                {snapshot?.stale ? (
                  <ShieldAlert className="w-3 h-3 flex-shrink-0 text-status-warning" />
                ) : null}
              </div>

              <div className="grid grid-cols-3 gap-1.5">
                <MetricTile
                  label="Transport"
                  value={profile.transport === 'stdio' ? 'stdio' : 'HTTP'}
                />
                <MetricTile label="Tools" value={snapshot?.tools.length ?? 0} />
                <MetricTile label="Warnings" value={warningCount} />
              </div>

              {connectionVerified ? (
                <div className="rounded-brutal border-brutal border-status-success bg-status-success/10 p-2">
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-status-success" />
                    <div className="min-w-0">
                      <div className="text-[10px] font-mono font-bold text-status-success">
                        Connection verified
                      </div>
                      <div
                        className="mt-0.5 truncate text-[10px] font-mono text-text-primary"
                        title={serverIdentity.name}
                      >
                        {serverIdentity.name}
                        {serverIdentity.version ? ` · ${serverIdentity.version}` : ''}
                      </div>
                      <div className="mt-1 text-[9px] font-mono text-text-muted">
                        MCP {snapshot?.negotiated_protocol_version} ·{' '}
                        {snapshot?.captured_at
                          ? new Date(snapshot.captured_at).toLocaleTimeString()
                          : 'just now'}
                      </div>
                      {oauthStatus?.has_tokens ? (
                        <div className="mt-1 text-[9px] font-mono font-bold text-accent-dark">
                          Hosted OAuth authenticated
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="rounded-brutal border-brutal border-border-subtle bg-background-card p-2 text-[10px] font-mono text-text-secondary">
                  <span className="font-bold text-text-primary">Not yet verified.</span> Connect to
                  complete a capability handshake and prove the server is responding.
                </div>
              )}

              <div className="node-section !p-2">
                <div className="node-field-label">Testing</div>
                <div className="mt-1 text-[10px] font-mono text-text-secondary">
                  {data.hasTerminalConnections
                    ? 'Terminal linked for shared server context. Agent eval suites arrive in Milestone 2.'
                    : 'Verify the server here, then open Tools for a real end-to-end call.'}
                </div>
              </div>

              {quickTestResult ? (
                <div className="max-h-16 overflow-y-auto rounded-brutal border-brutal border-status-success/60 bg-status-success/10 px-2 py-1 text-[9px] font-mono text-status-success">
                  {quickTestResult}
                </div>
              ) : null}
              {error ? (
                <div className="max-h-16 overflow-y-auto rounded-brutal border-brutal border-status-error/60 bg-status-error/10 px-2 py-1 text-[9px] font-mono text-status-error">
                  {error}
                </div>
              ) : null}

              <div className="nodrag flex items-center gap-2">
                <button
                  type="button"
                  className="node-btn node-btn-wide node-btn-accent flex-1 justify-center"
                  onClick={sessionId ? quickTest : connect}
                  disabled={isWorking}
                >
                  {isWorking ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : sessionId ? (
                    <FlaskConical className="w-3 h-3" />
                  ) : (
                    <Plug className="w-3 h-3" />
                  )}
                  {sessionId ? 'Verify server' : 'Connect'}
                </button>
                <button
                  type="button"
                  className="node-btn node-btn-wide flex-1 justify-center"
                  onClick={() => openAdvanced('overview')}
                >
                  <Wrench className="w-3 h-3" />
                  Open lab
                </button>
              </div>
            </>
          )}
        </div>
      </DataNodeShell>

      <NodeConfigModal
        isOpen={ui.mode === 'advanced'}
        onClose={closeAdvanced}
        title={`${displayTitle} Lab`}
        description="Inspect and safely call one MCP server"
        size="xl"
        layout="workspace"
      >
        <div className="mcp-lab-layout">
          <nav className="mcp-lab-nav" aria-label="MCP lab sections">
            {(['overview', 'tools'] as McpAdvancedSection[]).map((section) => (
              <button
                key={section}
                type="button"
                onClick={() => selectSection(section)}
                className={clsx(
                  'mcp-lab-nav-button',
                  ui.section === section
                    ? 'border-accent bg-accent/10 text-accent-dark'
                    : 'border-transparent text-text-secondary hover:border-border-subtle hover:text-text-primary'
                )}
              >
                {section}
              </button>
            ))}
          </nav>

          <div
            ref={contentScrollRef}
            className={clsx('mcp-lab-content', ui.section === 'tools' && 'mcp-lab-content-tools')}
          >
            {ui.section === 'overview' ? (
              <div className="space-y-3">
                <ConfigSection title="Connection">
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    <label className="node-field">
                      <span className="node-field-label">Saved profile</span>
                      <select
                        className="input-brutal min-h-9 text-[11px] font-mono"
                        value={profileId}
                        onChange={(event) => selectProfile(event.target.value)}
                      >
                        <option value="">New MCP connection</option>
                        {profiles.map((candidate) => (
                          <option key={candidate.profile_id} value={candidate.profile_id}>
                            {candidate.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="node-field">
                      <span className="node-field-label">Transport</span>
                      <select
                        className="input-brutal min-h-9 text-[11px] font-mono"
                        value={draft.transport}
                        onChange={(event) =>
                          setDraft((current) => ({
                            ...current,
                            transport: event.target.value === 'stdio' ? 'stdio' : 'streamable_http'
                          }))
                        }
                      >
                        <option value="streamable_http">Streamable HTTP</option>
                        <option value="stdio">Local stdio</option>
                      </select>
                    </label>
                    <label className="node-field md:col-span-2">
                      <span className="node-field-label">Label</span>
                      <input
                        className="input-brutal min-h-9 text-[11px] font-mono"
                        value={draft.label}
                        onChange={(event) =>
                          setDraft((current) => ({ ...current, label: event.target.value }))
                        }
                        placeholder="Public Search MCP"
                      />
                    </label>
                  </div>

                  {draft.transport === 'streamable_http' ? (
                    <div className="grid grid-cols-1 gap-3">
                      <label className="node-field">
                        <span className="node-field-label">MCP URL</span>
                        <input
                          className="input-brutal min-h-9 text-[11px] font-mono"
                          value={draft.remoteUrl}
                          onChange={(event) =>
                            setDraft((current) => ({ ...current, remoteUrl: event.target.value }))
                          }
                          placeholder="https://example.com/mcp"
                        />
                      </label>
                      <label className="node-field">
                        <span className="node-field-label">Authentication</span>
                        <select
                          className="input-brutal min-h-9 text-[11px] font-mono"
                          value={draft.authMode}
                          onChange={(event) =>
                            setDraft((current) => ({
                              ...current,
                              authMode: event.target.value as McpProfileDraft['authMode']
                            }))
                          }
                        >
                          <option value="auto">Automatic — OAuth when requested</option>
                          <option value="oauth">Hosted OAuth</option>
                          <option value="headers">Credential headers</option>
                          <option value="none">No authentication</option>
                        </select>
                        <span className="text-[10px] leading-4 text-text-muted">
                          Automatic is recommended for hosted MCPs. BashGym opens the provider login
                          page after an authentication challenge.
                        </span>
                      </label>
                      {draft.authMode === 'auto' || draft.authMode === 'oauth' ? (
                        <div className="grid grid-cols-1 gap-2 rounded-brutal border-brutal border-accent/50 bg-accent/5 p-2 md:grid-cols-2">
                          <div className="md:col-span-2 flex items-start gap-2 text-[10px] leading-4 text-text-secondary">
                            <ShieldAlert className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-accent-dark" />
                            OAuth uses protected-resource discovery, PKCE, and a loopback callback.
                            Tokens are kept in the OS credential store, never in this profile.
                          </div>
                          <label className="node-field md:col-span-2">
                            <span className="node-field-label">
                              Requested scopes — spaces or commas
                            </span>
                            <input
                              className="input-brutal min-h-9 text-[11px] font-mono"
                              value={draft.oauthScopes}
                              onChange={(event) =>
                                setDraft((current) => ({
                                  ...current,
                                  oauthScopes: event.target.value
                                }))
                              }
                              placeholder="Leave blank to use server defaults"
                            />
                          </label>
                          <label className="node-field">
                            <span className="node-field-label">Fixed callback port</span>
                            <input
                              className="input-brutal min-h-9 text-[11px] font-mono"
                              value={draft.oauthCallbackPort}
                              onChange={(event) =>
                                setDraft((current) => ({
                                  ...current,
                                  oauthCallbackPort: event.target.value
                                }))
                              }
                              inputMode="numeric"
                              placeholder="Random secure port"
                            />
                          </label>
                          <label className="node-field">
                            <span className="node-field-label">Pre-registered client ID</span>
                            <input
                              className="input-brutal min-h-9 text-[11px] font-mono"
                              value={draft.oauthClientId}
                              onChange={(event) =>
                                setDraft((current) => ({
                                  ...current,
                                  oauthClientId: event.target.value
                                }))
                              }
                              placeholder="Optional; dynamic registration is default"
                            />
                          </label>
                          <label className="node-field md:col-span-2">
                            <span className="node-field-label">Client secret reference</span>
                            <input
                              className="input-brutal min-h-9 text-[11px] font-mono"
                              value={draft.oauthClientSecretRef}
                              onChange={(event) =>
                                setDraft((current) => ({
                                  ...current,
                                  oauthClientSecretRef: event.target.value
                                }))
                              }
                              placeholder="MCP_OAUTH_CLIENT_SECRET"
                            />
                          </label>
                        </div>
                      ) : null}
                      <label className="node-field">
                        <span className="node-field-label">Header secret references</span>
                        <textarea
                          className="input-brutal min-h-[86px] text-[11px] font-mono"
                          value={draft.headerSecretRefs}
                          onChange={(event) =>
                            setDraft((current) => ({
                              ...current,
                              headerSecretRefs: event.target.value
                            }))
                          }
                          placeholder={'Authorization=MCP_TOKEN\nX-Workspace=MCP_WORKSPACE_ID'}
                        />
                        <span className="text-[10px] leading-4 text-text-muted">
                          Reference names only. Secret values belong in the OS credential store or
                          environment.
                        </span>
                      </label>
                      <label className="flex items-start gap-2 rounded-brutal border-brutal border-border-subtle bg-background-card p-2.5">
                        <input
                          type="checkbox"
                          className="mt-0.5 accent-accent"
                          checked={draft.allowPrivateNetwork}
                          onChange={(event) =>
                            setDraft((current) => ({
                              ...current,
                              allowPrivateNetwork: event.target.checked
                            }))
                          }
                        />
                        <span>
                          <span className="block text-[10px] font-mono font-bold text-text-primary">
                            Allow private-network addresses
                          </span>
                          <span className="mt-1 block text-[10px] leading-4 text-text-muted">
                            Off by default. Enable only for a trusted LAN or loopback MCP endpoint.
                          </span>
                        </span>
                      </label>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                      <label className="node-field md:col-span-2">
                        <span className="node-field-label">Executable</span>
                        <input
                          className="input-brutal min-h-9 text-[11px] font-mono"
                          value={draft.command}
                          onChange={(event) =>
                            setDraft((current) => ({ ...current, command: event.target.value }))
                          }
                          placeholder="npx"
                        />
                      </label>
                      <label className="node-field">
                        <span className="node-field-label">Arguments — one per line</span>
                        <textarea
                          className="input-brutal min-h-[112px] text-[11px] font-mono"
                          value={draft.args}
                          onChange={(event) =>
                            setDraft((current) => ({ ...current, args: event.target.value }))
                          }
                          placeholder={'-y\n@scope/public-mcp'}
                        />
                      </label>
                      <label className="node-field">
                        <span className="node-field-label">Environment secret references</span>
                        <textarea
                          className="input-brutal min-h-[112px] text-[11px] font-mono"
                          value={draft.envSecretRefs}
                          onChange={(event) =>
                            setDraft((current) => ({
                              ...current,
                              envSecretRefs: event.target.value
                            }))
                          }
                          placeholder="API_TOKEN=MCP_API_TOKEN"
                        />
                      </label>
                      <label className="node-field md:col-span-2">
                        <span className="node-field-label">Working-directory policy</span>
                        <select
                          className="input-brutal min-h-9 text-[11px] font-mono"
                          value={draft.cwdPolicy}
                          onChange={(event) =>
                            setDraft((current) => ({ ...current, cwdPolicy: event.target.value }))
                          }
                        >
                          <option value="workspace">Workspace</option>
                          <option value="isolated">Isolated temporary directory</option>
                          <option value="explicit">Explicit directory</option>
                        </select>
                      </label>
                      {draft.cwdPolicy === 'explicit' ? (
                        <label className="node-field md:col-span-2">
                          <span className="node-field-label">Explicit working directory</span>
                          <input
                            className="input-brutal min-h-9 text-[11px] font-mono"
                            value={draft.explicitCwd}
                            onChange={(event) =>
                              setDraft((current) => ({
                                ...current,
                                explicitCwd: event.target.value
                              }))
                            }
                            placeholder="C:\\path\\to\\trusted-workspace"
                          />
                        </label>
                      ) : null}
                      <label className="node-field md:col-span-2">
                        <span className="node-field-label">Process sandbox policy</span>
                        <select
                          className="input-brutal min-h-9 text-[11px] font-mono"
                          value={draft.sandboxPolicy}
                          onChange={(event) =>
                            setDraft((current) => ({
                              ...current,
                              sandboxPolicy: event.target.value as McpProfileDraft['sandboxPolicy']
                            }))
                          }
                        >
                          <option value="required">Required — fail closed if unavailable</option>
                          <option value="preferred">
                            Preferred — connect with a visible warning
                          </option>
                          <option value="disabled">Disabled — explicit trust only</option>
                        </select>
                      </label>
                    </div>
                  )}

                  <div className="node-config-action-row">
                    {draftDirty ? (
                      <div className="w-full rounded-brutal border-brutal border-status-warning/60 bg-status-warning/10 p-2 text-[10px] leading-4 text-text-secondary">
                        <strong className="text-text-primary">Unsaved configuration.</strong>{' '}
                        {sessionId
                          ? 'Disconnect first, then save so an active session cannot keep using the previous settings.'
                          : 'Save before testing so the URL, authentication, and safety settings shown here are the settings BashGym uses.'}
                      </div>
                    ) : null}
                    <button
                      type="button"
                      className="node-btn node-btn-wide node-btn-accent"
                      onClick={() => void saveProfile()}
                      disabled={savingProfile || !draftDirty || Boolean(sessionId)}
                    >
                      {savingProfile ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <CheckCircle2 className="w-3 h-3" />
                      )}
                      {profile ? 'Save changes' : 'Save profile'}
                    </button>
                    <div className="node-config-action-spacer" />
                    {sessionId ? (
                      <button
                        type="button"
                        className="node-btn node-btn-wide"
                        onClick={disconnect}
                        disabled={isWorking}
                      >
                        <Unplug className="w-3 h-3" />
                        Disconnect
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="node-btn node-btn-wide node-btn-accent"
                        onClick={draftDirty ? () => void saveAndConnect() : connect}
                        disabled={savingProfile || isWorking}
                      >
                        {savingProfile ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Plug className="w-3 h-3" />
                        )}
                        {draftDirty ? 'Save & connect' : 'Connect'}
                      </button>
                    )}
                    <button
                      type="button"
                      className="node-btn node-btn-wide"
                      onClick={refresh}
                      disabled={!profile || isWorking || draftDirty}
                    >
                      <RefreshCw className="w-3 h-3" />
                      Refresh
                    </button>
                  </div>
                </ConfigSection>

                <ConfigSection title="Import Claude MCP config">
                  <div className="grid grid-cols-1 gap-2 md:grid-cols-[minmax(0,1fr)_150px]">
                    <label className="node-field">
                      <span className="node-field-label">.mcp.json content</span>
                      <textarea
                        className="input-brutal min-h-[120px] text-[10px] font-mono"
                        value={claudeImportJson}
                        onChange={(event) => setClaudeImportJson(event.target.value)}
                        placeholder={
                          '{\n  "mcpServers": {\n    "my-server": { "type": "http", "url": "https://example.com/mcp" }\n  }\n}'
                        }
                        spellCheck={false}
                      />
                    </label>
                    <div className="space-y-2">
                      <label className="node-field">
                        <span className="node-field-label">Source scope</span>
                        <select
                          className="input-brutal min-h-9 text-[11px] font-mono"
                          value={claudeImportScope}
                          onChange={(event) =>
                            setClaudeImportScope(event.target.value as typeof claudeImportScope)
                          }
                        >
                          <option value="project">Project</option>
                          <option value="local">Local</option>
                          <option value="user">User</option>
                        </select>
                      </label>
                      <button
                        type="button"
                        className="node-btn node-btn-wide node-btn-accent w-full justify-center"
                        onClick={() => void previewClaudeImport()}
                      >
                        <Search className="w-3 h-3" />
                        Preview import
                      </button>
                    </div>
                  </div>
                  <p className="mt-2 text-[10px] leading-4 text-text-muted">
                    Use environment placeholders such as ${'{MCP_TOKEN}'}. Raw values are blocked
                    and never returned by the preview.
                  </p>
                  {claudeImportCandidates.length ? (
                    <div className="mt-2 space-y-2">
                      {claudeImportCandidates.map((candidate) => (
                        <div
                          key={candidate.server_name}
                          className="rounded-brutal border-brutal border-border-subtle bg-background-card p-2"
                        >
                          <div className="flex items-center gap-2">
                            <span className="min-w-0 flex-1 truncate text-[10px] font-mono font-bold text-text-primary">
                              {candidate.server_name}
                            </span>
                            <ConfigPill tone={candidate.supported ? 'success' : 'error'}>
                              {candidate.supported ? 'ready' : 'migration needed'}
                            </ConfigPill>
                            {candidate.supported ? (
                              <button
                                type="button"
                                className="node-btn"
                                onClick={() => applyClaudeImportCandidate(candidate)}
                              >
                                Use config
                              </button>
                            ) : null}
                          </div>
                          {candidate.issues.length ? (
                            <ul className="mt-1 space-y-1 text-[9px] font-mono text-text-secondary">
                              {candidate.issues.map((issue) => (
                                <li key={`${issue.code}:${issue.field ?? ''}`}>
                                  • {issue.message}
                                </li>
                              ))}
                            </ul>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </ConfigSection>

                <ConfigSection title="Connection verification and troubleshooting">
                  {connectionVerified ? (
                    <div className="rounded-brutal border-brutal border-status-success bg-status-success/10 p-3">
                      <div className="flex items-start gap-2">
                        <CheckCircle2 className="mt-0.5 h-5 w-5 flex-shrink-0 text-status-success" />
                        <div className="min-w-0">
                          <div className="text-xs font-mono font-bold text-status-success">
                            Verified capability handshake
                          </div>
                          <div className="mt-1 text-[11px] font-mono text-text-primary">
                            {serverIdentity.name}
                            {serverIdentity.version ? ` · ${serverIdentity.version}` : ''}
                          </div>
                          <p className="mt-1 text-[10px] leading-4 text-text-secondary">
                            BashGym has a live session and a fresh MCP inventory from this server
                            revision.
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-brutal border-brutal border-status-warning/70 bg-status-warning/10 p-3">
                      <div className="flex items-start gap-2">
                        <ShieldAlert className="mt-0.5 h-5 w-5 flex-shrink-0 text-status-warning" />
                        <div>
                          <div className="text-xs font-mono font-bold text-text-primary">
                            Connection not verified
                          </div>
                          <p className="mt-1 text-[10px] leading-4 text-text-secondary">
                            {operation?.error ||
                              'No fresh capability handshake is available for this profile.'}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  {profile?.transport === 'streamable_http' ? (
                    <div className="mt-2 flex flex-wrap items-center gap-2 rounded-brutal border-brutal border-border-subtle bg-background-card p-2">
                      <ConfigPill
                        tone={
                          oauthStatus?.has_tokens
                            ? 'success'
                            : oauthStatus?.interactive_oauth
                              ? 'accent'
                              : 'neutral'
                        }
                      >
                        {oauthStatus?.has_tokens
                          ? 'OAuth signed in'
                          : oauthStatus?.interactive_oauth
                            ? 'OAuth ready'
                            : draft.authMode === 'headers'
                              ? 'Header auth'
                              : 'No auth'}
                      </ConfigPill>
                      <span className="min-w-0 flex-1 text-[10px] leading-4 text-text-secondary">
                        {oauthStatus?.has_tokens
                          ? 'Tokens are stored in the OS credential store and refresh automatically.'
                          : oauthStatus?.interactive_oauth
                            ? isWorking
                              ? 'Waiting for the hosted provider sign-in to complete in your browser.'
                              : 'The provider login page opens automatically when this server requests authorization.'
                            : 'This profile does not use interactive hosted OAuth.'}
                      </span>
                      {oauthStatus?.has_tokens ? (
                        <button
                          type="button"
                          className="node-btn"
                          onClick={() => void clearOAuth()}
                          disabled={isWorking}
                        >
                          Clear sign-in
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                  <div className="flex flex-wrap gap-1.5">
                    <ConfigPill tone={stateTone(connectionState)}>
                      {stateLabel(connectionState)}
                    </ConfigPill>
                    {snapshot?.stale ? <ConfigPill tone="warning">stale</ConfigPill> : null}
                    {snapshot?.drifted ? (
                      <ConfigPill tone="warning">contract drift</ConfigPill>
                    ) : null}
                    {warningCount ? (
                      <ConfigPill tone="warning">{warningCount} warnings</ConfigPill>
                    ) : null}
                  </div>
                  <ConfigRows>
                    <ConfigRow label="Profile revision" value={profile?.profile_revision} />
                    <ConfigRow label="Profile updated" value={profileTime} />
                    <ConfigRow
                      label="Server"
                      value={`${serverIdentity.name}${serverIdentity.version ? ` · ${serverIdentity.version}` : ''}`}
                    />
                    <ConfigRow label="Session" value={sessionId} />
                    <ConfigRow label="Snapshot" value={snapshot?.snapshot_id} />
                    <ConfigRow
                      label="Captured"
                      value={
                        snapshot?.captured_at
                          ? new Date(snapshot.captured_at).toLocaleString()
                          : undefined
                      }
                    />
                    <ConfigRow label="Protocol" value={snapshot?.negotiated_protocol_version} />
                    <ConfigRow label="Contract hash" value={snapshot?.contract_hash} />
                    <ConfigRow
                      label="Contract drift"
                      value={
                        snapshot?.drifted ? 'Detected since previous snapshot' : 'None detected'
                      }
                    />
                    <ConfigRow label="Tools" value={snapshot?.tools.length ?? 0} />
                    <ConfigRow label="Resources" value={snapshot?.resources?.length ?? 0} />
                    <ConfigRow label="Prompts" value={snapshot?.prompts?.length ?? 0} />
                    <ConfigRow
                      label="Operation"
                      value={
                        operation
                          ? `${operation.status}${operation.phase ? ` — ${operation.phase}` : ''}`
                          : undefined
                      }
                    />
                    <ConfigRow label="Error code" value={operation?.error_code} />
                    <ConfigRow label="Error" value={error} />
                  </ConfigRows>
                  {snapshot?.schema_warnings?.length ? (
                    <div className="mt-2 rounded-brutal border-brutal border-status-warning/60 bg-status-warning/10 p-2">
                      <div className="node-field-label">Compatibility and schema warnings</div>
                      <ul className="mt-1 space-y-1 text-[9px] font-mono text-text-secondary">
                        {snapshot.schema_warnings.map((warning) => (
                          <li key={warning}>• {warning}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  {isWorking ? (
                    <button
                      type="button"
                      className="node-btn node-btn-wide node-btn-warning"
                      onClick={() => void cancelOperation()}
                    >
                      <Square className="w-3 h-3" />
                      Cancel operation
                    </button>
                  ) : null}
                  {!connectionVerified ? (
                    <div className="mt-2 rounded-brutal border-brutal border-border-subtle bg-background-secondary p-2">
                      <div className="node-field-label">Try these steps</div>
                      <ol className="mt-1 space-y-1 text-[10px] leading-4 text-text-secondary">
                        {troubleshootingSteps.map((step, index) => (
                          <li key={step}>
                            {index + 1}. {step}
                          </li>
                        ))}
                      </ol>
                    </div>
                  ) : null}
                  <div className="node-config-action-row mt-2">
                    {profile ? (
                      <button
                        type="button"
                        className="node-btn node-btn-wide node-btn-accent"
                        onClick={connectionVerified ? quickTest : sessionId ? refresh : connect}
                        disabled={isWorking}
                      >
                        {connectionVerified ? (
                          <RefreshCw className="w-3 h-3" />
                        ) : (
                          <Plug className="w-3 h-3" />
                        )}
                        {connectionVerified
                          ? 'Verify again'
                          : sessionId
                            ? 'Refresh handshake'
                            : 'Connect and verify'}
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="node-btn node-btn-wide"
                      onClick={selfTest}
                      disabled={isWorking}
                    >
                      <FlaskConical className="w-3 h-3" />
                      Test BashGym runtime
                    </button>
                    <button
                      type="button"
                      className="node-btn node-btn-wide"
                      onClick={() => void copyDiagnostics()}
                    >
                      <ClipboardCopy className="w-3 h-3" />
                      {diagnosticsCopied ? 'Copied' : 'Copy diagnostics'}
                    </button>
                  </div>
                  {quickTestResult ? (
                    <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap rounded-brutal border-brutal border-status-success/60 bg-status-success/10 p-2 text-[9px] font-mono text-status-success">
                      {quickTestResult}
                    </pre>
                  ) : null}
                </ConfigSection>
              </div>
            ) : (
              <div className="mcp-tools-view">
                <ConfigSection
                  title={
                    <span className="flex w-full min-w-0 items-center justify-between gap-2">
                      <span>Available tools</span>
                      <span className="font-normal normal-case tracking-normal text-text-secondary">
                        {filteredTools.length} of {snapshot?.tools.length ?? 0}
                      </span>
                    </span>
                  }
                  className="mcp-tools-section"
                >
                  <div className="mcp-tools-search">
                    <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-muted" />
                    <input
                      className="input-brutal min-h-9 !pl-8 text-[11px] font-mono"
                      value={ui.toolQuery}
                      onChange={(event) =>
                        dispatchUi({ type: 'set_tool_query', query: event.target.value })
                      }
                      placeholder="Search what you want the server to do"
                    />
                  </div>
                  <div className="mcp-tools-browser">
                    <div className="mcp-tools-list" aria-label="Advertised MCP tools">
                      {filteredTools.length ? (
                        filteredTools.map((tool) => (
                          <ToolListRow
                            key={tool.name}
                            tool={tool}
                            selected={selectedTool?.name === tool.name}
                            onSelect={() => selectTool(tool.name)}
                          />
                        ))
                      ) : (
                        <div className="py-8 text-center text-[10px] font-mono text-text-muted">
                          {snapshot
                            ? 'No tools match this search.'
                            : 'Connect and refresh to inspect tools.'}
                        </div>
                      )}
                    </div>

                    <div className="mcp-tool-detail">
                      {selectedTool ? (
                        <>
                          <div>
                            <div className="text-sm font-mono font-bold text-text-primary">
                              {toolDisplayName(selectedTool)}
                            </div>
                            <p className="mt-2 text-xs leading-5 text-text-secondary">
                              {selectedTool.description ||
                                'This server did not provide a tool description.'}
                            </p>
                          </div>
                          <div className="mcp-tool-insights">
                            <div className="mcp-tool-insight">
                              <div className="node-field-label">How it runs</div>
                              <div className="mt-2">
                                <ConfigPill tone={selectedToolPolicy.tone}>
                                  {selectedToolPolicy.label}
                                </ConfigPill>
                              </div>
                              <p className="mt-2 text-[10px] leading-4 text-text-secondary">
                                {selectedToolPolicy.description}
                              </p>
                            </div>
                            <div className="mcp-tool-insight">
                              <div className="node-field-label">Recent activity</div>
                              {selectedToolUsageCount > 0 ? (
                                <div className="mt-2 space-y-1 text-[10px] leading-4 text-text-secondary">
                                  <div>
                                    <strong className="text-text-primary">
                                      {selectedToolUsageCount}
                                    </strong>{' '}
                                    recorded {selectedToolUsageCount === 1 ? 'call' : 'calls'}
                                  </div>
                                  <div>
                                    Last response:{' '}
                                    <strong className="text-text-primary">
                                      {formatToolLatency(selectedTool.last_latency_ms)}
                                    </strong>
                                  </div>
                                  <div>
                                    Reliability:{' '}
                                    <strong className="text-text-primary">
                                      {formatToolReliability(selectedTool.error_rate)}
                                    </strong>
                                  </div>
                                </div>
                              ) : (
                                <p className="mt-2 text-[10px] leading-4 text-text-secondary">
                                  No calls recorded yet. Response time and reliability appear after
                                  this tool is tested.
                                </p>
                              )}
                            </div>
                          </div>
                          <details className="mcp-technical-details">
                            <summary>Technical details</summary>
                            <div className="mcp-technical-details-content">
                              <ConfigRows>
                                <ConfigRow label="Tool ID" value={selectedTool.name} />
                              </ConfigRows>
                              <div className="mcp-schema-grid">
                                <label className="node-field">
                                  <span className="node-field-label">Accepted input</span>
                                  <pre className="mcp-schema-panel">
                                    {resultString(selectedTool.input_schema)}
                                  </pre>
                                </label>
                                <label className="node-field">
                                  <span className="node-field-label">Returned output</span>
                                  <pre
                                    className={clsx(
                                      'mcp-schema-panel',
                                      !resultString(selectedTool.output_schema) &&
                                        'mcp-schema-panel-empty'
                                    )}
                                  >
                                    {resultString(selectedTool.output_schema) ||
                                      'The server did not describe its output.'}
                                  </pre>
                                </label>
                              </div>
                            </div>
                          </details>
                          <label className="node-field">
                            <span className="node-field-label">Test this tool — JSON input</span>
                            <textarea
                              className="input-brutal min-h-[116px] text-[11px] font-mono"
                              value={ui.manualInput}
                              onChange={(event) =>
                                dispatchUi({ type: 'set_manual_input', value: event.target.value })
                              }
                              spellCheck={false}
                            />
                            <span className="text-[10px] leading-4 text-text-muted">
                              Start with {'{}'} when no inputs are required. BashGym pauses when
                              approval is needed.
                            </span>
                          </label>
                          <details className="mcp-technical-details">
                            <summary>Test limits</summary>
                            <div className="mcp-technical-details-content">
                              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                                <label className="node-field">
                                  <span className="node-field-label">Timeout — seconds</span>
                                  <input
                                    className="input-brutal min-h-9 text-[11px] font-mono"
                                    type="number"
                                    min="1"
                                    max="300"
                                    step="1"
                                    value={ui.timeoutSeconds}
                                    onChange={(event) =>
                                      dispatchUi({
                                        type: 'set_timeout_seconds',
                                        value: event.target.value
                                      })
                                    }
                                  />
                                </label>
                                <label className="node-field">
                                  <span className="node-field-label">Maximum result — KB</span>
                                  <input
                                    className="input-brutal min-h-9 text-[11px] font-mono"
                                    type="number"
                                    min="1"
                                    max="8192"
                                    step="1"
                                    value={ui.maxResultKilobytes}
                                    onChange={(event) =>
                                      dispatchUi({
                                        type: 'set_max_result_kilobytes',
                                        value: event.target.value
                                      })
                                    }
                                  />
                                </label>
                              </div>
                              <p className="text-[10px] leading-4 text-text-muted">
                                Calls stop after 30 seconds and keep at most 1 MB by default. Raise
                                either limit only for a tool you trust.
                              </p>
                            </div>
                          </details>
                          {!connectionVerified ? (
                            <div className="rounded-brutal border-brutal border-status-warning/60 bg-status-warning/10 p-2 text-[10px] leading-4 text-text-secondary">
                              Connect and verify a fresh server inventory before running this tool.
                            </div>
                          ) : null}
                          <div className="node-config-action-row">
                            <button
                              type="button"
                              className="node-btn node-btn-wide node-btn-accent"
                              onClick={callTool}
                              disabled={!connectionVerified || isWorking}
                            >
                              {isWorking ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : (
                                <Wrench className="w-3 h-3" />
                              )}
                              Call tool
                            </button>
                            {isWorking ? (
                              <button
                                type="button"
                                className="node-btn node-btn-wide node-btn-warning"
                                onClick={() => void cancelOperation()}
                              >
                                <Square className="w-3 h-3" />
                                Cancel
                              </button>
                            ) : null}
                          </div>
                          {callResult ? (
                            <label className="node-field">
                              <span className="node-field-label">Persisted redacted result</span>
                              <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-brutal border-brutal border-border-subtle bg-background-secondary p-2 text-[10px] font-mono text-text-primary">
                                {callResult}
                              </pre>
                            </label>
                          ) : null}
                          {error ? (
                            <div className="rounded-brutal border-brutal border-status-error bg-status-error/10 p-2 text-[10px] font-mono text-status-error">
                              {error}
                            </div>
                          ) : null}
                        </>
                      ) : (
                        <div className="flex min-h-52 items-center justify-center text-center text-[10px] font-mono text-text-muted">
                          Select a tool to inspect its contract and prepare a safe manual call.
                        </div>
                      )}
                    </div>
                  </div>
                </ConfigSection>
              </div>
            )}
          </div>
        </div>
      </NodeConfigModal>
    </>
  )
})
