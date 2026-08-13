export type McpTransport = 'streamable_http' | 'stdio'

export type McpConnectionState = 'empty' | 'idle' | 'loading' | 'connected' | 'stale' | 'error'

export type McpOperationStatus =
  'queued' | 'running' | 'awaiting_approval' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'

export interface McpProfile {
  profile_id: string
  profile_revision: number
  label: string
  transport: McpTransport
  remote?: {
    url: string
    header_secret_refs: Record<string, string>
    allow_private_network?: boolean
    auth_mode?: 'auto' | 'oauth' | 'headers' | 'none'
    oauth_scopes?: string[]
    oauth_callback_port?: number | null
    oauth_client_id?: string | null
    oauth_client_secret_ref?: string | null
  } | null
  stdio?: {
    command: string
    args: string[]
    cwd_policy?: string | null
    cwd?: string | null
    env_secret_refs: Record<string, string>
    sandbox_policy?: 'required' | 'preferred' | 'disabled'
  } | null
  active_session_id?: string | null
  enabled: boolean
  updated_at?: string
}

export interface McpTool {
  name: string
  title?: string | null
  description?: string | null
  input_schema: Record<string, unknown>
  output_schema?: Record<string, unknown> | null
  annotations?: Record<string, unknown> | null
  _meta?: Record<string, unknown> | null
  policy?: 'allow' | 'ask' | 'deny' | 'unknown'
  last_latency_ms?: number | null
  error_rate?: number | null
  usage_count?: number
}

export interface McpCapabilitySnapshot {
  snapshot_id: string
  profile_id: string
  profile_revision: number
  captured_at: string
  negotiated_protocol_version?: string | null
  contract_hash: string
  upstream_version?: string | null
  server_info?: Record<string, unknown>
  capabilities?: Record<string, unknown>
  instructions?: string | null
  tools: McpTool[]
  resources?: unknown[]
  resource_templates?: unknown[]
  prompts?: unknown[]
  schema_warnings?: string[]
  stale?: boolean
  drifted?: boolean
}

export interface McpOperation {
  operation_id: string
  correlation_id?: string
  kind?: string
  status: McpOperationStatus
  phase?: string | null
  error?: string | null
  error_code?: string | null
  result?: Record<string, unknown> | null
  created_at?: string
  updated_at?: string
}

export interface McpOperationAccepted {
  operation_id: string
  status: 'queued' | 'running'
}

export interface McpOAuthStatus {
  profile_id: string
  auth_mode: 'auto' | 'oauth' | 'headers' | 'none'
  interactive_oauth: boolean
  has_tokens: boolean
  storage: 'credential_store'
}

export interface McpClaudeImportIssue {
  severity: 'info' | 'warning' | 'blocked'
  code: string
  message: string
  field?: string | null
}

export interface McpClaudeImportCandidate {
  server_name: string
  supported: boolean
  source_scope: 'local' | 'project' | 'user'
  profile_input?: McpProfileInput | null
  issues: McpClaudeImportIssue[]
  preserved_fields: Record<string, unknown>
}

export interface McpStdioLaunchPreview {
  profile_id: string
  profile_revision: number
  command: string
  args: string[]
  cwd_policy: string
  env_names: string[]
  sandbox_policy: string
  executable: { path: string; sha256: string; size: number; modified_ns: number }
  launch_fingerprint: string
}

export interface McpProfileInput {
  label: string
  transport: McpTransport
  enabled: boolean
  remote?: {
    url: string
    header_secret_refs: Record<string, string>
    allow_private_network: boolean
    auth_mode: 'auto' | 'oauth' | 'headers' | 'none'
    oauth_scopes: string[]
    oauth_callback_port?: number
    oauth_client_id?: string
    oauth_client_secret_ref?: string
  }
  stdio?: {
    command: string
    args: string[]
    cwd_policy?: string
    cwd?: string
    env_secret_refs: Record<string, string>
    sandbox_policy?: 'required' | 'preferred' | 'disabled'
  }
}

export interface ApiResult<T> {
  ok: boolean
  data?: T
  error?: string
}

export interface McpWorkbenchApi {
  listProfiles(): Promise<ApiResult<McpProfile[]>>
  createProfile(input: McpProfileInput): Promise<ApiResult<McpProfile>>
  updateProfile(
    profileId: string,
    input: McpProfileInput,
    expectedRevision: number
  ): Promise<ApiResult<McpProfile>>
  previewStdio(
    profileId: string,
    profileRevision: number
  ): Promise<ApiResult<McpStdioLaunchPreview>>
  approveStdio(
    profileId: string,
    profileRevision: number,
    executableSha256: string,
    launchFingerprint: string
  ): Promise<ApiResult<Record<string, unknown>>>
  connect(profileId: string, profileRevision: number): Promise<ApiResult<McpOperationAccepted>>
  refresh(profileId: string): Promise<ApiResult<McpOperationAccepted>>
  quickTest(profileId: string): Promise<ApiResult<McpOperationAccepted>>
  selfTest(): Promise<ApiResult<McpOperationAccepted>>
  previewClaudeConfig(
    config: Record<string, unknown>,
    sourceScope: 'local' | 'project' | 'user'
  ): Promise<ApiResult<McpClaudeImportCandidate[]>>
  oauthStatus(profileId: string): Promise<ApiResult<McpOAuthStatus>>
  logoutOAuth(profileId: string): Promise<ApiResult<Record<string, unknown>>>
  snapshot(profileId: string): Promise<ApiResult<McpCapabilitySnapshot>>
  disconnect(sessionId: string): Promise<ApiResult<McpOperationAccepted>>
  callTool(
    sessionId: string,
    toolName: string,
    argumentsValue: Record<string, unknown>,
    options?: {
      approved: boolean
      typed_confirmation?: string
      timeout_seconds?: number
      max_result_bytes?: number
    }
  ): Promise<ApiResult<McpOperationAccepted>>
  getOperation(operationId: string): Promise<ApiResult<McpOperation>>
  cancelOperation(operationId: string): Promise<ApiResult<McpOperation>>
}

let configuredMcpWorkbenchApi: McpWorkbenchApi | null = null

/** Configure the native MCP service adapter once during canvas integration. */
export function configureMcpWorkbenchApi(api: McpWorkbenchApi): void {
  configuredMcpWorkbenchApi = api
}

export function getConfiguredMcpWorkbenchApi(): McpWorkbenchApi | null {
  return configuredMcpWorkbenchApi
}

export interface McpAdapterConfig {
  profile_id?: string
  view?: 'simple' | 'advanced'
  advanced_section?: 'overview' | 'tools'
  selected_tool?: string
  selected_version?: string
  selected_eval_suite?: string
}

const ADAPTER_STRING_FIELDS = [
  'profile_id',
  'selected_tool',
  'selected_version',
  'selected_eval_suite'
] as const

export function sanitizeMcpAdapterConfig(value: unknown): McpAdapterConfig {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const input = value as Record<string, unknown>
  const output: McpAdapterConfig = {}

  for (const key of ADAPTER_STRING_FIELDS) {
    const field = input[key]
    if (typeof field === 'string' && field.trim()) output[key] = field.trim()
  }
  if (input.view === 'simple' || input.view === 'advanced') output.view = input.view
  if (input.advanced_section === 'overview' || input.advanced_section === 'tools') {
    output.advanced_section = input.advanced_section
  }
  return output
}

export interface McpProfileDraft {
  label: string
  transport: McpTransport
  remoteUrl: string
  headerSecretRefs: string
  allowPrivateNetwork: boolean
  authMode: 'auto' | 'oauth' | 'headers' | 'none'
  oauthScopes: string
  oauthCallbackPort: string
  oauthClientId: string
  oauthClientSecretRef: string
  command: string
  args: string
  cwdPolicy: string
  explicitCwd: string
  envSecretRefs: string
  sandboxPolicy: 'required' | 'preferred' | 'disabled'
  enabled: boolean
}

export const EMPTY_MCP_PROFILE_DRAFT: McpProfileDraft = {
  label: '',
  transport: 'streamable_http',
  remoteUrl: '',
  headerSecretRefs: '',
  allowPrivateNetwork: false,
  authMode: 'auto',
  oauthScopes: '',
  oauthCallbackPort: '',
  oauthClientId: '',
  oauthClientSecretRef: '',
  command: '',
  args: '',
  cwdPolicy: 'workspace',
  explicitCwd: '',
  envSecretRefs: '',
  sandboxPolicy: 'preferred',
  enabled: true
}

function parseReferenceMap(value: string): Record<string, string> {
  const lines = value
    .split(/[\n,]/)
    .map((entry) => entry.trim())
    .filter(Boolean)
  const entries = lines.map((entry) => {
    const separator = entry.indexOf('=')
    if (separator < 1) {
      throw new Error('Secret references must use NAME=REFERENCE_NAME, never a secret value.')
    }
    const name = entry.slice(0, separator).trim()
    const reference = entry.slice(separator + 1).trim()
    if (!name || !/^[A-Z][A-Z0-9_]{1,127}$/.test(reference)) {
      throw new Error('Reference names must be uppercase identifiers such as MCP_API_TOKEN.')
    }
    return [name, reference] as const
  })
  return Object.fromEntries(entries)
}

function parseArgs(value: string): string[] {
  return value
    .split('\n')
    .map((argument) => argument.trim())
    .filter(Boolean)
}

export function profileInputFromDraft(draft: McpProfileDraft): McpProfileInput {
  const base = {
    label: draft.label.trim() || 'MCP Server',
    transport: draft.transport,
    enabled: draft.enabled
  }
  if (draft.transport === 'stdio') {
    return {
      ...base,
      stdio: {
        command: draft.command.trim(),
        args: parseArgs(draft.args),
        cwd_policy: draft.cwdPolicy.trim() || 'workspace',
        ...(draft.cwdPolicy === 'explicit' ? { cwd: draft.explicitCwd.trim() } : {}),
        env_secret_refs: parseReferenceMap(draft.envSecretRefs),
        sandbox_policy: draft.sandboxPolicy
      }
    }
  }
  const callbackPort = draft.oauthCallbackPort.trim() ? Number(draft.oauthCallbackPort) : undefined
  if (
    callbackPort !== undefined &&
    (!Number.isInteger(callbackPort) || callbackPort < 1024 || callbackPort > 65535)
  ) {
    throw new Error('OAuth callback port must be an integer from 1024 to 65535.')
  }
  const clientSecretRef = draft.oauthClientSecretRef.trim()
  if (clientSecretRef && !/^[A-Z][A-Z0-9_]{1,127}$/.test(clientSecretRef)) {
    throw new Error('OAuth client secret reference must be an uppercase identifier.')
  }
  return {
    ...base,
    remote: {
      url: draft.remoteUrl.trim(),
      header_secret_refs: parseReferenceMap(draft.headerSecretRefs),
      allow_private_network: draft.allowPrivateNetwork,
      auth_mode: draft.authMode,
      oauth_scopes: draft.oauthScopes
        .split(/[\s,]+/)
        .map((scope) => scope.trim())
        .filter(Boolean),
      ...(callbackPort !== undefined ? { oauth_callback_port: callbackPort } : {}),
      ...(draft.oauthClientId.trim() ? { oauth_client_id: draft.oauthClientId.trim() } : {}),
      ...(clientSecretRef ? { oauth_client_secret_ref: clientSecretRef } : {})
    }
  }
}

export function draftFromProfile(profile?: McpProfile | null): McpProfileDraft {
  if (!profile) return { ...EMPTY_MCP_PROFILE_DRAFT }
  const mapToLines = (value?: Record<string, string>) =>
    Object.entries(value ?? {})
      .map(([name, reference]) => `${name}=${reference}`)
      .join('\n')
  return {
    label: profile.label,
    transport: profile.transport,
    remoteUrl: profile.remote?.url ?? '',
    headerSecretRefs: mapToLines(profile.remote?.header_secret_refs),
    allowPrivateNetwork: profile.remote?.allow_private_network ?? false,
    authMode: profile.remote?.auth_mode ?? 'auto',
    oauthScopes: profile.remote?.oauth_scopes?.join(' ') ?? '',
    oauthCallbackPort:
      profile.remote?.oauth_callback_port == null ? '' : String(profile.remote.oauth_callback_port),
    oauthClientId: profile.remote?.oauth_client_id ?? '',
    oauthClientSecretRef: profile.remote?.oauth_client_secret_ref ?? '',
    command: profile.stdio?.command ?? '',
    args: profile.stdio?.args.join('\n') ?? '',
    cwdPolicy: profile.stdio?.cwd_policy ?? 'workspace',
    explicitCwd: profile.stdio?.cwd ?? '',
    envSecretRefs: mapToLines(profile.stdio?.env_secret_refs),
    sandboxPolicy: profile.stdio?.sandbox_policy ?? 'preferred',
    enabled: profile.enabled
  }
}

export function draftFromProfileInput(input: McpProfileInput): McpProfileDraft {
  return draftFromProfile({
    profile_id: 'import-preview',
    profile_revision: 1,
    label: input.label,
    transport: input.transport,
    remote: input.remote ?? null,
    stdio: input.stdio ?? null,
    enabled: input.enabled
  })
}

export function isMcpProfileDraftDirty(
  draft: McpProfileDraft,
  profile?: McpProfile | null
): boolean {
  if (!profile) return true
  try {
    return (
      JSON.stringify(profileInputFromDraft(draft)) !==
      JSON.stringify(profileInputFromDraft(draftFromProfile(profile)))
    )
  } catch {
    return true
  }
}

export function filterMcpTools(tools: McpTool[], query: string): McpTool[] {
  const normalized = query.trim().toLocaleLowerCase()
  const seen = new Set<string>()
  const uniqueTools = tools.filter((tool) => {
    const identity = tool.name.trim().toLocaleLowerCase()
    if (seen.has(identity)) return false
    seen.add(identity)
    return true
  })
  if (!normalized) return uniqueTools
  return uniqueTools.filter((tool) =>
    [tool.name, tool.title, tool.description]
      .filter((part): part is string => typeof part === 'string')
      .some((part) => part.toLocaleLowerCase().includes(normalized))
  )
}

export function operationToConnectionState(
  operation: McpOperation | null,
  hasProfile: boolean,
  hasSession: boolean,
  snapshotStale: boolean
): McpConnectionState {
  if (!hasProfile) return 'empty'
  if (operation?.status === 'failed' || operation?.status === 'interrupted') return 'error'
  if (
    operation?.status === 'queued' ||
    operation?.status === 'running' ||
    operation?.status === 'awaiting_approval'
  )
    return 'loading'
  if (hasSession) return snapshotStale ? 'stale' : 'connected'
  if (snapshotStale) return 'stale'
  return 'idle'
}

export function isMcpConnectionVerified(
  sessionId: string | null,
  snapshot: McpCapabilitySnapshot | null
): boolean {
  return Boolean(sessionId && snapshot && !snapshot.stale)
}

export function mcpServerIdentity(snapshot: McpCapabilitySnapshot | null): {
  name: string
  version?: string
} {
  const name = snapshot?.server_info?.name
  const version = snapshot?.server_info?.version ?? snapshot?.upstream_version
  return {
    name: typeof name === 'string' && name.trim() ? name : 'Unnamed MCP server',
    ...(typeof version === 'string' && version.trim() ? { version } : {})
  }
}

export function mcpTroubleshootingSteps(
  profile: McpProfile | null,
  operation: McpOperation | null
): string[] {
  const errorCode = operation?.error_code
  const tailored: Record<string, string[]> = {
    auth_required: [
      'Confirm every configured header or environment reference exists in the credential store.',
      'For a hosted server, choose Automatic or Hosted OAuth, reconnect, and complete the provider sign-in in your browser.'
    ],
    oauth_timeout: [
      'Start the connection again and complete the provider login before the five-minute callback window closes.'
    ],
    oauth_declined: [
      'Connect again and approve the requested access on the hosted provider sign-in page.'
    ],
    oauth_browser_unavailable: [
      'Check that the operating system has a default browser, then retry the hosted MCP connection.'
    ],
    oauth_failed: [
      'Verify the hosted MCP publishes protected-resource and authorization-server metadata.',
      'If dynamic client registration is unavailable, configure the provider client ID, callback port, and client secret reference.'
    ],
    policy_denied: [
      'Review the target host and enable private-network access only when this is a trusted LAN or loopback server.',
      'Confirm the endpoint uses HTTP or HTTPS and does not redirect to another host.'
    ],
    approval_required: [
      'Reconnect and review the exact executable, arguments, working directory, environment names, and sandbox fingerprint.'
    ],
    launch_changed: [
      'Preview and approve the changed local-process fingerprint before reconnecting.'
    ],
    sandbox_unavailable: [
      'Use Preferred only when degraded local-process isolation is acceptable; otherwise keep Required and install a supported sandbox.'
    ],
    session_not_found: ['Reconnect the server; the previous live session is no longer available.'],
    session_closed: ['Reconnect, then verify the connection before running another tool call.'],
    tool_timeout: [
      'Increase the tool budget only after confirming the server reports progress and is not stuck.'
    ],
    result_too_large: ['Lower the requested result size or use a narrower MCP tool query.']
  }
  const steps = tailored[errorCode ?? ''] ?? [
    profile?.transport === 'stdio'
      ? 'Verify the executable, one-argument-per-line list, working directory, and sandbox policy.'
      : 'Verify the MCP URL, credential references, and whether the host requires private-network access.',
    'Reconnect and verify to capture a fresh capability handshake.'
  ]
  return [
    ...steps,
    'Run the reference self-test to separate a BashGym runtime problem from a server-specific problem.'
  ]
}

export function buildMcpDiagnosticSummary(
  profile: McpProfile | null,
  snapshot: McpCapabilitySnapshot | null,
  sessionId: string | null,
  operation: McpOperation | null,
  connectionState: McpConnectionState
): string {
  const identity = mcpServerIdentity(snapshot)
  const lines = [
    'BashGym MCP diagnostics',
    `profile: ${profile ? redactText(profile.label) : 'none'}`,
    `profile_id: ${profile?.profile_id ?? 'none'}`,
    `profile_revision: ${profile?.profile_revision ?? 'none'}`,
    `transport: ${profile?.transport ?? 'none'}`,
    `auth_mode: ${profile?.remote?.auth_mode ?? (profile?.transport === 'stdio' ? 'process_environment' : 'none')}`,
    `connection: ${connectionState}`,
    `verified: ${isMcpConnectionVerified(sessionId, snapshot) ? 'yes' : 'no'}`,
    `server: ${redactText(identity.name)}${identity.version ? ` ${redactText(identity.version)}` : ''}`,
    `session_id: ${sessionId ?? 'none'}`,
    `snapshot_id: ${snapshot?.snapshot_id ?? 'none'}`,
    `verified_at: ${snapshot?.captured_at ?? 'none'}`,
    `protocol: ${snapshot?.negotiated_protocol_version ?? 'none'}`,
    `contract_hash: ${snapshot?.contract_hash ?? 'none'}`,
    `contract_drift: ${snapshot?.drifted ? 'yes' : 'no'}`,
    `tools: ${snapshot?.tools.length ?? 0}`,
    `resources: ${snapshot?.resources?.length ?? 0}`,
    `prompts: ${snapshot?.prompts?.length ?? 0}`,
    `warnings: ${snapshot?.schema_warnings?.length ?? 0}`,
    `operation_id: ${operation?.operation_id ?? 'none'}`,
    `correlation_id: ${operation?.correlation_id ?? 'none'}`,
    `operation_kind: ${operation?.kind ?? 'none'}`,
    `operation_status: ${operation?.status ?? 'none'}`,
    `error_code: ${operation?.error_code ?? 'none'}`,
    `safe_error: ${operation?.error ? redactText(operation.error) : 'none'}`
  ]
  return lines.join('\n')
}

export function formatMcpVerificationResult(value: unknown): string {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    return 'Server verification passed.'
  const result = value as Record<string, unknown>
  const count = (key: string) => (typeof result[key] === 'number' ? (result[key] as number) : 0)
  const warnings = count('schema_warning_count')
  return [
    'Server verification passed.',
    `${count('tool_count')} tools · ${count('resource_count')} resources · ${count('prompt_count')} prompts`,
    warnings
      ? `${warnings} compatibility warning${warnings === 1 ? '' : 's'} to review`
      : 'No compatibility warnings reported'
  ].join('\n')
}

const SENSITIVE_VALUE_RE =
  /(bearer\s+[a-z0-9._-]+|sk-[a-z0-9_-]+|ghp_[a-z0-9]+|xox[bp]-[a-z0-9-]+)/gi

function redactText(value: string): string {
  return value.replace(SENSITIVE_VALUE_RE, '[redacted]')
}

export function buildMcpTerminalContext(
  profile: McpProfile | null,
  snapshot: McpCapabilitySnapshot | null,
  connectionState: McpConnectionState
): string {
  const lines = ['## BashGym MCP Workbench', '']
  if (!profile) {
    lines.push('No MCP profile is selected. Configure a public MCP profile in BashGym first.')
    return lines.join('\n')
  }

  lines.push(
    `- profile: ${redactText(profile.label)} (${profile.profile_id})`,
    `- transport: ${profile.transport}`,
    `- connection: ${connectionState}`,
    `- profile revision: ${profile.profile_revision}`
  )
  if (!snapshot) {
    lines.push('- capability snapshot: unavailable')
    return lines.join('\n')
  }

  lines.push(
    `- snapshot: ${snapshot.snapshot_id}${snapshot.stale ? ' (stale)' : ''}`,
    `- contract: ${snapshot.contract_hash}`,
    `- tools: ${snapshot.tools.length}`,
    `- resources: ${snapshot.resources?.length ?? 0}`,
    `- prompts: ${snapshot.prompts?.length ?? 0}`,
    `- schema warnings: ${snapshot.schema_warnings?.length ?? 0}`,
    '',
    '### Advertised tools'
  )
  for (const tool of snapshot.tools.slice(0, 30)) {
    const description = tool.description ? `: ${redactText(tool.description)}` : ''
    lines.push(`- ${redactText(tool.name)}${description}`)
  }
  if (snapshot.tools.length > 30) lines.push(`- ...and ${snapshot.tools.length - 30} more`)
  lines.push(
    '',
    'Use the MCP Workbench UI for calls. Secret values and raw payloads are intentionally excluded.'
  )
  return lines.join('\n')
}

export type McpAdvancedSection = 'overview' | 'tools'

export interface McpWorkbenchUiState {
  mode: 'simple' | 'advanced'
  section: McpAdvancedSection
  selectedToolName: string | null
  toolQuery: string
  manualInput: string
  timeoutSeconds: string
  maxResultKilobytes: string
  overviewScrollTop: number
  toolsScrollTop: number
}

export type McpWorkbenchUiAction =
  | { type: 'open_advanced'; section?: McpAdvancedSection }
  | { type: 'close_advanced' }
  | { type: 'select_section'; section: McpAdvancedSection }
  | { type: 'select_tool'; toolName: string | null }
  | { type: 'set_tool_query'; query: string }
  | { type: 'set_manual_input'; value: string }
  | { type: 'set_timeout_seconds'; value: string }
  | { type: 'set_max_result_kilobytes'; value: string }
  | { type: 'remember_scroll'; section: McpAdvancedSection; scrollTop: number }

export const INITIAL_MCP_WORKBENCH_UI_STATE: McpWorkbenchUiState = {
  mode: 'simple',
  section: 'overview',
  selectedToolName: null,
  toolQuery: '',
  manualInput: '{}',
  timeoutSeconds: '30',
  maxResultKilobytes: '1024',
  overviewScrollTop: 0,
  toolsScrollTop: 0
}

export function reduceMcpWorkbenchUi(
  state: McpWorkbenchUiState,
  action: McpWorkbenchUiAction
): McpWorkbenchUiState {
  switch (action.type) {
    case 'open_advanced':
      return { ...state, mode: 'advanced', section: action.section ?? state.section }
    case 'close_advanced':
      return { ...state, mode: 'simple' }
    case 'select_section':
      return { ...state, section: action.section }
    case 'select_tool':
      return { ...state, selectedToolName: action.toolName }
    case 'set_tool_query':
      return { ...state, toolQuery: action.query }
    case 'set_manual_input':
      return { ...state, manualInput: action.value }
    case 'set_timeout_seconds':
      return { ...state, timeoutSeconds: action.value }
    case 'set_max_result_kilobytes':
      return { ...state, maxResultKilobytes: action.value }
    case 'remember_scroll':
      return action.section === 'overview'
        ? { ...state, overviewScrollTop: action.scrollTop }
        : { ...state, toolsScrollTop: action.scrollTop }
  }
}

export function parseManualToolArguments(value: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(value)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Tool input must be a JSON object.')
  }
  return parsed as Record<string, unknown>
}

export function parseMcpToolTestLimits(
  timeoutSecondsValue: string,
  maxResultKilobytesValue: string
): { timeout_seconds: number; max_result_bytes: number } {
  const timeoutSeconds = Number(timeoutSecondsValue)
  if (!Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0 || timeoutSeconds > 300) {
    throw new Error('Tool timeout must be between 1 and 300 seconds.')
  }
  const maxResultKilobytes = Number(maxResultKilobytesValue)
  if (
    !Number.isInteger(maxResultKilobytes) ||
    maxResultKilobytes < 1 ||
    maxResultKilobytes > 8192
  ) {
    throw new Error('Result limit must be a whole number from 1 to 8192 KB.')
  }
  return {
    timeout_seconds: timeoutSeconds,
    max_result_bytes: maxResultKilobytes * 1024
  }
}
