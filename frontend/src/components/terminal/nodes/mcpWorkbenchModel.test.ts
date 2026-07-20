import assert from 'node:assert/strict'
import test from 'node:test'
import {
  EMPTY_MCP_PROFILE_DRAFT,
  INITIAL_MCP_WORKBENCH_UI_STATE,
  buildMcpDiagnosticSummary,
  buildMcpTerminalContext,
  filterMcpTools,
  formatMcpVerificationResult,
  isMcpConnectionVerified,
  isMcpProfileDraftDirty,
  mcpTroubleshootingSteps,
  operationToConnectionState,
  profileInputFromDraft,
  parseMcpToolTestLimits,
  reduceMcpWorkbenchUi,
  sanitizeMcpAdapterConfig,
  type McpCapabilitySnapshot,
  type McpProfile,
  type McpTool
} from './mcpWorkbenchModel'

const profile: McpProfile = {
  profile_id: 'profile-1',
  profile_revision: 3,
  label: 'Public Search MCP',
  transport: 'streamable_http',
  remote: {
    url: 'https://example.test/mcp',
    header_secret_refs: { Authorization: 'MCP_SEARCH_TOKEN' }
  },
  active_session_id: 'session-1',
  enabled: true
}

const tools: McpTool[] = [
  {
    name: 'search_library',
    title: 'Search Library',
    description: 'Find public documents',
    input_schema: { type: 'object' }
  },
  {
    name: 'save_note',
    description: 'Create a note',
    input_schema: { type: 'object' }
  }
]

const snapshot: McpCapabilitySnapshot = {
  snapshot_id: 'snapshot-1',
  profile_id: profile.profile_id,
  profile_revision: profile.profile_revision,
  captured_at: '2026-07-10T00:00:00Z',
  contract_hash: 'sha256:contract',
  tools,
  resources: [],
  prompts: [],
  schema_warnings: []
}

test('adapter config keeps only harmless MCP presentation references', () => {
  const sanitized = sanitizeMcpAdapterConfig({
    profile_id: ' profile-1 ',
    view: 'advanced',
    advanced_section: 'tools',
    selected_tool: 'search_library',
    selected_version: 'version-2',
    selected_eval_suite: 'suite-1',
    url: 'https://private.test/mcp',
    authorization: 'Bearer secret-token',
    api_key: 'sk-never-store-this',
    nested: { password: 'never' }
  })

  assert.deepEqual(sanitized, {
    profile_id: 'profile-1',
    view: 'advanced',
    advanced_section: 'tools',
    selected_tool: 'search_library',
    selected_version: 'version-2',
    selected_eval_suite: 'suite-1'
  })
  assert.equal(JSON.stringify(sanitized).includes('secret-token'), false)
})

test('profile input accepts reference names and refuses token-shaped values', () => {
  const safe = profileInputFromDraft({
    ...EMPTY_MCP_PROFILE_DRAFT,
    label: 'Search',
    transport: 'streamable_http',
    remoteUrl: 'https://example.test/mcp',
    headerSecretRefs: 'Authorization=MCP_SEARCH_TOKEN',
    command: '',
    args: '',
    cwdPolicy: 'workspace',
    envSecretRefs: '',
    enabled: true
  })
  assert.deepEqual(safe.remote?.header_secret_refs, { Authorization: 'MCP_SEARCH_TOKEN' })

  assert.throws(
    () =>
      profileInputFromDraft({
        ...EMPTY_MCP_PROFILE_DRAFT,
        label: 'Unsafe',
        transport: 'streamable_http',
        remoteUrl: 'https://example.test/mcp',
        headerSecretRefs: 'Authorization=sk-live-secret',
        command: '',
        args: '',
        cwdPolicy: 'workspace',
        envSecretRefs: '',
        enabled: true
      }),
    /Reference names must be uppercase identifiers/
  )
})

test('profile draft changes are detected before a connection can test stale settings', () => {
  const savedDraft = {
    ...EMPTY_MCP_PROFILE_DRAFT,
    label: profile.label,
    transport: profile.transport,
    remoteUrl: profile.remote?.url ?? '',
    headerSecretRefs: 'Authorization=MCP_SEARCH_TOKEN'
  }
  assert.equal(isMcpProfileDraftDirty(savedDraft, profile), false)
  assert.equal(
    isMcpProfileDraftDirty({ ...savedDraft, remoteUrl: 'https://other.test/mcp' }, profile),
    true
  )
  assert.equal(isMcpProfileDraftDirty(savedDraft, null), true)
})

test('manual tool test limits map UI values to bounded API controls', () => {
  assert.deepEqual(parseMcpToolTestLimits('45', '2048'), {
    timeout_seconds: 45,
    max_result_bytes: 2 * 1024 * 1024
  })
  assert.throws(() => parseMcpToolTestLimits('301', '1024'), /between 1 and 300/)
  assert.throws(() => parseMcpToolTestLimits('30', '8193'), /from 1 to 8192 KB/)
})

test('server verification results explain capability proof without exposing internal IDs', () => {
  const result = formatMcpVerificationResult({
    session_id: 'session-secret',
    snapshot_id: 'snapshot-secret',
    tool_count: 33,
    resource_count: 59,
    prompt_count: 6,
    schema_warning_count: 1
  })
  assert.match(result, /33 tools · 59 resources · 6 prompts/)
  assert.match(result, /1 compatibility warning/)
  assert.doesNotMatch(result, /session-secret|snapshot-secret/)
})

test('terminal context excludes endpoints and secret reference values and redacts token-shaped text', () => {
  const unsafeSnapshot: McpCapabilitySnapshot = {
    ...snapshot,
    tools: [{ ...tools[0], description: 'Uses Bearer abc.def-secret internally' }]
  }
  const context = buildMcpTerminalContext(profile, unsafeSnapshot, 'connected')

  assert.match(context, /Public Search MCP/)
  assert.match(context, /search_library/)
  assert.match(context, /\[redacted\]/)
  assert.doesNotMatch(context, /example\.test/)
  assert.doesNotMatch(context, /MCP_SEARCH_TOKEN/)
  assert.doesNotMatch(context, /abc\.def-secret/)
})

test('tool filtering searches names, titles, and descriptions and removes duplicate IDs', () => {
  assert.deepEqual(filterMcpTools(tools, 'PUBLIC'), [tools[0]])
  assert.deepEqual(filterMcpTools(tools, 'save_'), [tools[1]])
  assert.deepEqual(filterMcpTools(tools, ''), tools)
  assert.deepEqual(filterMcpTools([...tools, { ...tools[0], title: 'Duplicate title' }], ''), tools)
})

test('operation state mapping distinguishes empty, loading, connected, stale, and error', () => {
  assert.equal(operationToConnectionState(null, false, false, false), 'empty')
  assert.equal(
    operationToConnectionState({ operation_id: '1', status: 'queued' }, true, false, false),
    'loading'
  )
  assert.equal(
    operationToConnectionState({ operation_id: '1', status: 'succeeded' }, true, true, false),
    'connected'
  )
  assert.equal(operationToConnectionState(null, true, true, true), 'stale')
  assert.equal(
    operationToConnectionState({ operation_id: '1', status: 'failed' }, true, false, false),
    'error'
  )
})

test('verified connection requires both a live session and a fresh handshake snapshot', () => {
  assert.equal(isMcpConnectionVerified('session-1', snapshot), true)
  assert.equal(isMcpConnectionVerified(null, snapshot), false)
  assert.equal(isMcpConnectionVerified('session-1', { ...snapshot, stale: true }), false)
})

test('OAuth failures produce actionable recovery without leaking endpoint or credentials', () => {
  const authRequiredSteps = mcpTroubleshootingSteps(profile, {
    operation_id: 'operation-auth',
    status: 'failed',
    error_code: 'auth_required'
  })
  assert.match(authRequiredSteps.join(' '), /Automatic or Hosted OAuth/i)
  assert.doesNotMatch(authRequiredSteps.join(' '), /wait for the OAuth lifecycle/i)

  const steps = mcpTroubleshootingSteps(profile, {
    operation_id: 'operation-1',
    status: 'failed',
    error_code: 'oauth_failed',
    error: 'Bearer sk-never-copy'
  })
  assert.match(steps.join(' '), /dynamic client registration/i)

  const diagnostics = buildMcpDiagnosticSummary(
    profile,
    { ...snapshot, server_info: { name: 'Hosted Search', version: '2.0' } },
    'session-1',
    {
      operation_id: 'operation-1',
      correlation_id: 'correlation-1',
      kind: 'connect',
      status: 'failed',
      error_code: 'oauth_failed',
      error: 'Bearer sk-never-copy'
    },
    'error'
  )
  assert.match(diagnostics, /verified: yes/)
  assert.match(diagnostics, /server: Hosted Search 2\.0/)
  assert.match(diagnostics, /oauth_failed/)
  assert.doesNotMatch(diagnostics, /example\.test|sk-never-copy/)
})

test('opening and closing Advanced preserves selected tool, draft input, query, section, and scroll state', () => {
  let state = reduceMcpWorkbenchUi(INITIAL_MCP_WORKBENCH_UI_STATE, {
    type: 'open_advanced',
    section: 'tools'
  })
  state = reduceMcpWorkbenchUi(state, { type: 'select_tool', toolName: 'search_library' })
  state = reduceMcpWorkbenchUi(state, { type: 'set_tool_query', query: 'library' })
  state = reduceMcpWorkbenchUi(state, { type: 'set_manual_input', value: '{"query":"mcp"}' })
  state = reduceMcpWorkbenchUi(state, { type: 'set_timeout_seconds', value: '45' })
  state = reduceMcpWorkbenchUi(state, { type: 'set_max_result_kilobytes', value: '2048' })
  state = reduceMcpWorkbenchUi(state, { type: 'remember_scroll', section: 'tools', scrollTop: 240 })
  state = reduceMcpWorkbenchUi(state, { type: 'close_advanced' })
  state = reduceMcpWorkbenchUi(state, { type: 'open_advanced' })

  assert.deepEqual(state, {
    mode: 'advanced',
    section: 'tools',
    selectedToolName: 'search_library',
    toolQuery: 'library',
    manualInput: '{"query":"mcp"}',
    timeoutSeconds: '45',
    maxResultKilobytes: '2048',
    overviewScrollTop: 0,
    toolsScrollTop: 240
  })
})
