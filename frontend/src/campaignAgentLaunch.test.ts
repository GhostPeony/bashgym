import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildCodexCampaignAgentLaunch,
  type CodexCampaignAgentLaunchDependencies,
  type CodexCampaignAgentLaunchInput
} from '../electron/campaignAgentLaunch'

const CWD = process.platform === 'win32' ? 'C:\\workspaces\\bashgym' : '/workspaces/bashgym'
const CODEX = process.platform === 'win32' ? 'C:\\tools\\codex.exe' : '/usr/local/bin/codex'
const LAUNCH_SECRET = 'L'.repeat(43)

function input(
  overrides: Partial<CodexCampaignAgentLaunchInput> = {}
): CodexCampaignAgentLaunchInput {
  return {
    intent: {
      workspaceId: 'workspace-1',
      campaignId: 'campaign-1',
      cwd: CWD
    },
    terminalId: 'terminal-1',
    generation: 'ptygen_1',
    hostInstanceId: 'desktop-host-instance',
    mcpLaunch: {
      url: `http://127.0.0.1:43123/${'r'.repeat(43)}`,
      headers: { 'X-BashGym-MCP-Launch': LAUNCH_SECRET },
      terminalId: 'terminal-1',
      generation: 'ptygen_1'
    },
    ...overrides
  }
}

function dependencies(
  overrides: Partial<CodexCampaignAgentLaunchDependencies> = {}
): CodexCampaignAgentLaunchDependencies {
  return {
    resolveCodexExecutable: () => CODEX,
    pathExists: (candidate) => candidate === CWD || candidate === CODEX,
    defaultCwd: CWD,
    sourceEnv: {
      PATH: process.platform === 'win32' ? 'C:\\Windows\\System32' : '/usr/bin',
      HOME: process.platform === 'win32' ? undefined : '/home/operator',
      USERPROFILE: process.platform === 'win32' ? 'C:\\Users\\operator' : undefined,
      TEMP: process.platform === 'win32' ? 'C:\\Temp' : '/tmp',
      TERM: 'xterm-256color',
      OPENAI_API_KEY: 'must-not-cross-launch-boundary',
      BASHGYM_DESKTOP_BOOTSTRAP_SECRET: 'must-not-cross-launch-boundary',
      CAMPAIGN_AGENT_CREDENTIAL: `bgag.credential.${'s'.repeat(48)}`
    },
    ...overrides
  }
}

test('builds a direct node-pty Codex launch with launch-scoped MCP configuration', () => {
  const launch = buildCodexCampaignAgentLaunch(input(), dependencies())

  assert.equal(launch.executableFamily, 'codex')
  assert.equal(launch.executable, CODEX)
  assert.deepEqual(launch.args, [
    '-c',
    `mcp_servers.bashgym_campaign.url="http://127.0.0.1:43123/${'r'.repeat(43)}"`,
    '-c',
    `mcp_servers.bashgym_campaign.http_headers={"X-BashGym-MCP-Launch"="${LAUNCH_SECRET}"}`
  ])
  assert.equal(launch.cwd, CWD)

  // This is the exact shape consumed by node-pty.spawn(executable, args, options):
  // there is no shell command, wrapper, or renderer-authored argument to parse.
  const nodePtyInvocation = [
    launch.executable,
    launch.args,
    { cwd: launch.cwd, env: launch.env }
  ] as const
  assert.equal(nodePtyInvocation[0], CODEX)
  assert.ok(Array.isArray(nodePtyInvocation[1]))
  assert.equal(typeof nodePtyInvocation[2].cwd, 'string')
})

test('derives the main-owned identity from the actual terminal generation', () => {
  const first = buildCodexCampaignAgentLaunch(input(), dependencies()).identity
  const replacement = buildCodexCampaignAgentLaunch(
    input({
      generation: 'ptygen_2',
      mcpLaunch: {
        ...input().mcpLaunch,
        generation: 'ptygen_2'
      }
    }),
    dependencies()
  ).identity

  assert.equal(first.terminalId, 'terminal-1')
  assert.equal(first.generation, 'ptygen_1')
  assert.equal(first.family, 'codex')
  assert.equal(first.live, true)
  assert.notEqual(first.principalId, replacement.principalId)
  assert.notEqual(first.sessionId, replacement.sessionId)
})

test('uses a fresh allowlisted environment and carries no ambient credential', () => {
  const launch = buildCodexCampaignAgentLaunch(input(), dependencies())

  assert.deepEqual(
    Object.keys(launch.env).sort(),
    (process.platform === 'win32'
      ? ['PATH', 'TEMP', 'TERM', 'USERPROFILE']
      : ['HOME', 'PATH', 'TEMP', 'TERM']
    ).sort()
  )
  assert.doesNotMatch(JSON.stringify(launch.env), /bgag\.|API_KEY|BOOTSTRAP|credential/i)

  const serialized = JSON.stringify(launch)
  assert.equal(serialized.includes(LAUNCH_SECRET), true)
  assert.doesNotMatch(serialized, /bgag\./)
  assert.equal((serialized.match(new RegExp(LAUNCH_SECRET, 'g')) ?? []).length, 1)
})

test('uses and validates the injected default cwd when the product intent omits cwd', () => {
  const withoutCwd = input({
    intent: { workspaceId: 'workspace-1', campaignId: 'campaign-1' }
  })
  assert.equal(buildCodexCampaignAgentLaunch(withoutCwd, dependencies()).cwd, CWD)

  assert.throws(
    () => buildCodexCampaignAgentLaunch(withoutCwd, dependencies({ pathExists: () => false })),
    /working directory.*does not exist/i
  )
})

test('rejects cwd values that are relative, multiline, or missing', () => {
  for (const cwd of ['relative/path', `CWD\n--dangerously-bypass`, CWD]) {
    const deps = cwd === CWD ? dependencies({ pathExists: () => false }) : dependencies()
    assert.throws(
      () =>
        buildCodexCampaignAgentLaunch(
          input({
            intent: { workspaceId: 'workspace-1', campaignId: 'campaign-1', cwd }
          }),
          deps
        ),
      /working directory/i
    )
  }
})

test('requires an exact loopback MCP launch and the one launch header', () => {
  const invalidUrls = [
    `https://127.0.0.1:43123/${'r'.repeat(43)}`,
    `http://localhost:43123/${'r'.repeat(43)}`,
    `http://0.0.0.0:43123/${'r'.repeat(43)}`,
    `http://127.0.0.1:43123/${'r'.repeat(43)}?next=evil`,
    `http://127.0.0.1:43123/${'r'.repeat(42)};x`,
    `http://127.0.0.1:43123/${'r'.repeat(43)}\n-c`
  ]
  for (const url of invalidUrls) {
    assert.throws(
      () =>
        buildCodexCampaignAgentLaunch(
          input({
            mcpLaunch: { ...input().mcpLaunch, url }
          }),
          dependencies()
        ),
      /MCP loopback URL/i
    )
  }

  for (const headers of [
    {},
    { Authorization: LAUNCH_SECRET },
    { 'X-BashGym-MCP-Launch': LAUNCH_SECRET, Extra: 'x' },
    { 'X-BashGym-MCP-Launch': `${'x'.repeat(42)};` }
  ]) {
    assert.throws(
      () =>
        buildCodexCampaignAgentLaunch(
          input({
            mcpLaunch: {
              ...input().mcpLaunch,
              headers
            } as CodexCampaignAgentLaunchInput['mcpLaunch']
          }),
          dependencies()
        ),
      /MCP launch header/i
    )
  }
})

test('requires the MCP launch to match the main-owned terminal generation', () => {
  assert.throws(
    () =>
      buildCodexCampaignAgentLaunch(
        input({
          mcpLaunch: { ...input().mcpLaunch, terminalId: 'terminal-other' }
        }),
        dependencies()
      ),
    /terminal identity does not match/i
  )
  assert.throws(
    () =>
      buildCodexCampaignAgentLaunch(
        input({
          mcpLaunch: { ...input().mcpLaunch, generation: 'ptygen_other' }
        }),
        dependencies()
      ),
    /terminal identity does not match/i
  )
})

test('rejects invalid identifiers and runtime fields that could spoof main ownership', () => {
  for (const [field, value] of [
    ['workspaceId', 'workspace; rm -rf'],
    ['campaignId', 'campaign\nnext']
  ] as const) {
    assert.throws(
      () =>
        buildCodexCampaignAgentLaunch(
          input({
            intent: { ...input().intent, [field]: value }
          }),
          dependencies()
        ),
      /identifier/i
    )
  }

  for (const extra of [
    { family: 'hermes' },
    { command: 'powershell.exe' },
    { args: ['--dangerously-bypass-approvals-and-sandbox'] },
    { origin: 'renderer-origin' },
    { principalId: 'renderer-principal' },
    { sessionId: 'renderer-session' },
    { executable: 'malware.exe' }
  ]) {
    const untrusted = { ...input(), ...extra } as CodexCampaignAgentLaunchInput
    assert.throws(
      () => buildCodexCampaignAgentLaunch(untrusted, dependencies()),
      'family' in extra ? /Hermes.*not supported/i : /unsupported launch field/i
    )
  }
})

test('accepts only an exact resolved Codex executable basename or path', () => {
  const accepted =
    process.platform === 'win32'
      ? ['codex', 'codex.exe', 'C:\\Program Files\\Codex\\codex.exe']
      : ['codex', '/opt/codex/bin/codex']
  for (const executable of accepted) {
    const launch = buildCodexCampaignAgentLaunch(
      input(),
      dependencies({
        resolveCodexExecutable: () => executable,
        pathExists: (candidate) => candidate === CWD || candidate === executable
      })
    )
    assert.equal(launch.executable, executable)
  }

  for (const executable of [
    'powershell.exe',
    'codex-wrapper.exe',
    `codex\n--danger`,
    process.platform === 'win32' ? 'C:\\tools\\not-codex.exe' : '/tmp/not-codex'
  ]) {
    assert.throws(
      () =>
        buildCodexCampaignAgentLaunch(
          input(),
          dependencies({
            resolveCodexExecutable: () => executable,
            pathExists: () => true
          })
        ),
      /Codex executable/i
    )
  }
})
