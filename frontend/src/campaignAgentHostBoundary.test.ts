import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const preloadPath = fileURLToPath(new URL('../electron/preload.ts', import.meta.url))
const mainPath = fileURLToPath(new URL('../electron/main.ts', import.meta.url))
const campaignAgentRoutesPath = fileURLToPath(new URL('../../bashgym/api/campaign_agent_routes.py', import.meta.url))
const campaignAgentServicePath = fileURLToPath(new URL('../../bashgym/campaigns/campaign_agents.py', import.meta.url))
const campaignClientPath = fileURLToPath(new URL('../../bashgym/campaigns/client.py', import.meta.url))
const campaignMcpPath = fileURLToPath(new URL('../../bashgym/mcp/campaign_server.py', import.meta.url))
const preload = readFileSync(preloadPath, 'utf8')
const main = readFileSync(mainPath, 'utf8')
const campaignAgentRoutes = readFileSync(campaignAgentRoutesPath, 'utf8')
const campaignAgentService = readFileSync(campaignAgentServicePath, 'utf8')
const campaignClient = readFileSync(campaignClientPath, 'utf8')
const campaignMcp = readFileSync(campaignMcpPath, 'utf8')

test('preload exposes only high-level campaign-agent host operations', () => {
  for (const operation of ['launch', 'eligible', 'attach', 'authorize', 'activate', 'revoke']) {
    assert.match(preload, new RegExp(`campaign-agent-host:${operation}`))
  }
  assert.doesNotMatch(preload, /campaign-agent-host:(?:claim|status)/)
  assert.doesNotMatch(preload, /\/api\/campaign-agent\/sessions/)
  assert.doesNotMatch(preload, /raw_token|rawToken|ciphertext|ephemeralPublicKey/)
  assert.doesNotMatch(preload, /agentOrigin|agentPrincipalId/)
})

test('main reads only attested PTY identity state and tears authority down on every lifecycle boundary', () => {
  assert.match(main, /session\.campaignAgentIdentity/)
  assert.match(main, /campaignAgentHostController\.eligibleSessions/)
  for (const boundary of ['pty_exit', 'pty_replacement', 'renderer_reload', 'app_shutdown']) {
    assert.match(main, new RegExp(`['"]${boundary}['"]`))
  }
  const revokeStart = main.indexOf("ipcMain.handle('campaign-agent-host:revoke'")
  const revokeEnd = main.indexOf('function safeBridgeArgument', revokeStart)
  assert.ok(revokeStart >= 0 && revokeEnd > revokeStart)
  assert.match(main.slice(revokeStart, revokeEnd), /revokeCampaignAgentTerminal\(terminalId, 'explicit_revoke'\)/)
  assert.doesNotMatch(main, /pty\.write\([^)]*bgag|process\.argv[^\n]*bgag|writeFileSync\([^\n]*bgag/)
})

test('renderer-triggered command preparation cannot attest a campaign agent identity', () => {
  const start = main.indexOf("ipcMain.handle('agent-bridge:prepare-launch'")
  const end = main.indexOf('// Get fresh environment', start)
  assert.ok(start >= 0 && end > start)
  const prepareLaunchHandler = main.slice(start, end)
  assert.doesNotMatch(prepareLaunchHandler, /campaignAgentIdentity|bindMainOwnedCampaignAgentIdentity/)
  assert.doesNotMatch(preload, /campaign-agent-host:claim/)
})

test('desktop launch is a main-owned direct Codex PTY behind a fixed read-only MCP proxy', () => {
  assert.match(main, /ipcMain\.handle\(['"]campaign-agent-host:launch['"]/)
  assert.match(main, /new CampaignAgentMcpHost\(/)
  assert.match(main, /observe:\s*\(\)\s*=>\s*campaignAgentHostController\.observe\(terminalId\)/)
  assert.match(main, /artifacts:\s*\(args\)\s*=>\s*campaignAgentHostController\.artifacts\(terminalId, args\)/)
  assert.match(main, /pty\.spawn\(launch\.executable,\s*\[\.\.\.launch\.args\]/)
  assert.match(main, /registerPtySession\([\s\S]*?launch\.identity/)
  assert.doesNotMatch(main, /env\s*:\s*\{[^}]*bgag|BASHGYM_CAMPAIGN_AGENT_(?:TOKEN|CREDENTIAL)/is)
})

test('renderer launch contract exposes only campaign scope and a public terminal result', () => {
  assert.match(preload, /launch:\s*\(request\)\s*=>\s*ipcRenderer\.invoke\(['"]campaign-agent-host:launch['"],\s*request\)/)
  assert.match(preload, /workspaceId:\s*string[\s\S]*campaignId:\s*string[\s\S]*cwd\?:\s*string/)
  const start = main.indexOf("ipcMain.handle('campaign-agent-host:launch'")
  const end = main.indexOf("ipcMain.handle('campaign-agent-host:eligible'", start)
  assert.ok(start >= 0 && end > start)
  const launchHandler = main.slice(start, end)
  assert.match(launchHandler, /assertCampaignAgentLaunchRequest\(request\)/)
  assert.match(launchHandler, /return \{ success: true, terminalId, cwd: launch\.cwd \}/)
  assert.match(launchHandler, /launchedTerminalId[\s\S]*?closeCampaignAgentMcpHost\(launchedTerminalId\)/)
  assert.doesNotMatch(
    launchHandler,
    /return\s*\{[^}]*\b(?:credential|token|url|headers|args|env|identity|origin|principal|session|generation)\b/is,
  )
})

test('eligible campaign-agent sessions are filtered to the selected campaign scope', () => {
  assert.match(preload, /eligible:\s*\(request\)\s*=>\s*ipcRenderer\.invoke\(['"]campaign-agent-host:eligible['"],\s*request\)/)
  const start = main.indexOf("ipcMain.handle('campaign-agent-host:eligible'")
  const end = main.indexOf("ipcMain.handle('campaign-agent-host:attach'", start)
  assert.ok(start >= 0 && end > start)
  const eligibleHandler = main.slice(start, end)
  assert.match(eligibleHandler, /assertCampaignAgentScopeRequest\(request\)/)
  assert.match(eligibleHandler, /campaignAgentLaunchScopes\.get\(identity\.terminalId\)/)
  assert.match(eligibleHandler, /scope\.workspaceId === selectedScope\.workspaceId/)
  assert.match(eligibleHandler, /scope\.campaignId === selectedScope\.campaignId/)
})

test('main binds heartbeat and fixed actions while lifecycle closure locks the MCP host', () => {
  assert.match(main, /heartbeat:\s*\(credential, body\)\s*=>\s*campaignBridgeClient\.heartbeatCampaignAgent\(credential, body\)/)
  assert.match(main, /observe:\s*\(credential\)\s*=>\s*campaignBridgeClient\.observeCampaignAsAgent\(credential\)/)
  assert.match(main, /artifacts:\s*\(credential, query\)\s*=>\s*campaignBridgeClient\.listCampaignArtifactsAsAgent\(credential, query\)/)
  assert.match(main, /onLifecycle:[\s\S]*?host\.lock\(\)[\s\S]*?host\.close\(\)/)
  assert.match(main, /campaignAgentMcpHosts\.delete\(event\.terminalId\)/)
})

test('activation retries do not attempt to claim a delivered credential twice', () => {
  const start = main.indexOf("ipcMain.handle('campaign-agent-host:activate'")
  const end = main.indexOf("ipcMain.handle('campaign-agent-host:authorize'", start)
  assert.ok(start >= 0 && end > start)
  const activateHandler = main.slice(start, end)
  assert.match(activateHandler, /assertCampaignAgentTerminalScopeRequest\(request\)/)
  assert.match(activateHandler, /assertActiveCampaignAgentRequest\(\{ terminalId, workspaceId, campaignId \}\)/)
  assert.match(activateHandler, /campaignAgentHostController\.status\(terminalId\)/)
  assert.match(activateHandler, /state !== 'credential_ready'/)
  assert.match(activateHandler, /campaignAgentHostController\.claim\(terminalId\)/)
  assert.match(activateHandler, /campaignAgentHostController\.activate\(terminalId\)/)
})

test('heartbeat timeout locks MCP immediately and revokes controller authority', () => {
  const start = main.indexOf('onLifecycle: (event) =>')
  const end = main.indexOf('let backendProcess', start)
  assert.ok(start >= 0 && end > start)
  const lifecycle = main.slice(start, end)
  assert.match(lifecycle, /event\.kind === 'actions_locked'/)
  assert.match(lifecycle, /host\?\.lock\(\)/)
  assert.match(lifecycle, /revokeCampaignAgentTerminal\(event\.terminalId, 'explicit_revoke'\)/)
})

test('registration and authorization cannot outlive the main-owned MCP host', () => {
  for (const [operation, next] of [['attach', 'activate'], ['authorize', 'revoke']] as const) {
    const start = main.indexOf(`ipcMain.handle('campaign-agent-host:${operation}'`)
    const end = main.indexOf(`ipcMain.handle('campaign-agent-host:${next}'`, start)
    assert.ok(start >= 0 && end > start)
    assert.match(main.slice(start, end), /assertActiveCampaignAgentRequest\(request\)/)
  }
})

test('bgag authorization is limited to fixed, read-only campaign-agent action adapters', () => {
  assert.match(campaignAgentService, /def authorize_action\([\s\S]*?required_capability:/)
  assert.match(campaignAgentService, /def authorize_bearer_action\([\s\S]*?required_capability:/)
  const credentialRoutes = [...campaignAgentRoutes.matchAll(
    /@campaign_agent_credential_router\.(?:get|post)\("([^"]+)"/g,
  )].map((match) => match[1])
  assert.deepEqual(credentialRoutes, [
    '/heartbeat',
    '/actions/observe',
    '/actions/artifacts',
    '/sessions',
    '/sessions/{registration_id}/deliveries/claim',
    '/sessions/{registration_id}/revoke',
  ])
  assert.match(campaignAgentRoutes, /required_capability=CampaignAgentCapability\.CAMPAIGN_OBSERVE/)
  assert.match(campaignAgentRoutes, /required_capability=CampaignAgentCapability\.ARTIFACT_READ/)
  assert.doesNotMatch(
    campaignAgentRoutes,
    /@campaign_agent_credential_router\.(?:get|post)\([^\n]*(?:launch|pause|propose|forward|proxy)/,
  )
  assert.doesNotMatch(campaignAgentRoutes, /raw_token|rawToken|credential(?:_token|Token)\s*:/)
})

test('operator campaign MCP credentials cannot be repurposed for campaign-agent delivery', () => {
  assert.match(campaignMcp, /--credential-ref/)
  assert.match(campaignClient, /pass a credential reference, never a raw campaign token/)
  assert.match(campaignClient, /\/campaign-auth\/exchange/)
  assert.match(campaignClient, /raw_token\.startswith\("bgca\."\)/)
  assert.doesNotMatch(campaignMcp, /bgag\./)
})
