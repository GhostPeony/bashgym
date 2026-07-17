import path from 'node:path'
import {
  createMainOwnedCampaignAgentIdentity,
  type MainOwnedCampaignAgentIdentity,
} from './campaignAgentHost'
import type { CampaignAgentMcpLaunch } from './campaignAgentMcpHost'

const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/
const HOST_INSTANCE_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{7,255}$/
const MCP_ROUTE = /^\/[A-Za-z0-9_-]{43}$/
const MCP_LAUNCH_SECRET = /^[A-Za-z0-9_-]{43}$/
const MCP_LAUNCH_HEADER = 'X-BashGym-MCP-Launch'
const CONTROL_CHARACTER = /[\0\r\n]/

const SAFE_ENVIRONMENT_KEYS = process.platform === 'win32'
  ? [
      'APPDATA',
      'COLORTERM',
      'COMSPEC',
      'HOMEDRIVE',
      'HOMEPATH',
      'LANG',
      'LC_ALL',
      'LOCALAPPDATA',
      'NO_COLOR',
      'PATHEXT',
      'TEMP',
      'TERM',
      'TMP',
      'USERPROFILE',
      'WINDIR',
    ]
  : [
      'COLORTERM',
      'HOME',
      'LANG',
      'LC_ALL',
      'NO_COLOR',
      'TEMP',
      'TERM',
      'TMP',
      'TMPDIR',
    ]

export interface CodexCampaignAgentLaunchIntent {
  workspaceId: string
  campaignId: string
  cwd?: string
}

export interface CodexCampaignAgentLaunchInput {
  intent: CodexCampaignAgentLaunchIntent
  terminalId: string
  generation: string
  hostInstanceId: string
  mcpLaunch: CampaignAgentMcpLaunch
}

export interface CodexCampaignAgentLaunchDependencies {
  resolveCodexExecutable(): string
  pathExists(absolutePath: string): boolean
  defaultCwd?: string
  sourceEnv?: Readonly<Record<string, string | undefined>>
}

export interface CodexCampaignAgentLaunch {
  executableFamily: 'codex'
  executable: string
  args: readonly string[]
  cwd: string
  env: Readonly<Record<string, string>>
  identity: MainOwnedCampaignAgentIdentity
}

function isRecord(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function hasExactKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[] = [],
): boolean {
  const allowed = new Set([...required, ...optional])
  const keys = Object.keys(value)
  return required.every((key) => key in value) && keys.every((key) => allowed.has(key))
}

function identifier(value: unknown, label: string): string {
  if (typeof value !== 'string' || !IDENTIFIER.test(value)) {
    throw new Error(`Codex campaign launch ${label} identifier is invalid`)
  }
  return value
}

function validateInputShape(value: unknown): asserts value is CodexCampaignAgentLaunchInput {
  if (!isRecord(value)) throw new Error('Codex campaign launch input is invalid')
  if (value.family === 'hermes') {
    throw new Error('Hermes campaign-agent launch is not supported')
  }
  if (!hasExactKeys(value, ['intent', 'terminalId', 'generation', 'hostInstanceId', 'mcpLaunch'])) {
    throw new Error('Codex campaign launch contains an unsupported launch field')
  }
  if (!isRecord(value.intent)
    || !hasExactKeys(value.intent, ['workspaceId', 'campaignId'], ['cwd'])) {
    throw new Error('Codex campaign launch intent contains an unsupported launch field')
  }
  if (!isRecord(value.mcpLaunch)
    || !hasExactKeys(value.mcpLaunch, ['url', 'headers', 'terminalId', 'generation'])) {
    throw new Error('Codex campaign launch MCP input is invalid')
  }
}

function validateMcpLaunch(
  launch: CampaignAgentMcpLaunch,
  terminalId: string,
  generation: string,
): { url: string; launchSecret: string } {
  if (launch.terminalId !== terminalId || launch.generation !== generation) {
    throw new Error('Campaign agent MCP terminal identity does not match the PTY generation')
  }
  if (typeof launch.url !== 'string' || CONTROL_CHARACTER.test(launch.url)) {
    throw new Error('Campaign agent MCP loopback URL is invalid')
  }
  let url: URL
  try {
    url = new URL(launch.url)
  } catch {
    throw new Error('Campaign agent MCP loopback URL is invalid')
  }
  if (url.protocol !== 'http:'
    || url.hostname !== '127.0.0.1'
    || !url.port
    || Number(url.port) < 1
    || Number(url.port) > 65_535
    || url.username
    || url.password
    || url.search
    || url.hash
    || !MCP_ROUTE.test(url.pathname)
    || url.href !== launch.url) {
    throw new Error('Campaign agent MCP loopback URL is invalid')
  }
  if (!isRecord(launch.headers)
    || !hasExactKeys(launch.headers, [MCP_LAUNCH_HEADER])) {
    throw new Error('Campaign agent MCP launch header is invalid')
  }
  const launchSecret = launch.headers[MCP_LAUNCH_HEADER]
  if (typeof launchSecret !== 'string' || !MCP_LAUNCH_SECRET.test(launchSecret)) {
    throw new Error('Campaign agent MCP launch header is invalid')
  }
  return { url: launch.url, launchSecret }
}

function resolveWorkingDirectory(
  requested: string | undefined,
  dependencies: CodexCampaignAgentLaunchDependencies,
): string {
  const cwd = requested ?? dependencies.defaultCwd ?? process.cwd()
  if (typeof cwd !== 'string'
    || CONTROL_CHARACTER.test(cwd)
    || !path.isAbsolute(cwd)) {
    throw new Error('Codex campaign launch working directory must be absolute')
  }
  if (!dependencies.pathExists(cwd)) {
    throw new Error('Codex campaign launch working directory does not exist')
  }
  return cwd
}

function resolveExecutable(dependencies: CodexCampaignAgentLaunchDependencies): string {
  const executable = dependencies.resolveCodexExecutable()
  if (typeof executable !== 'string' || CONTROL_CHARACTER.test(executable)) {
    throw new Error('Resolved Codex executable is invalid')
  }
  const acceptedBasenames = process.platform === 'win32'
    ? new Set(['codex', 'codex.exe'])
    : new Set(['codex'])
  const basename = path.basename(executable).toLowerCase()
  const isBare = executable === path.basename(executable)
  if (!acceptedBasenames.has(basename)
    || (!isBare && !path.isAbsolute(executable))
    || (!isBare && !dependencies.pathExists(executable))) {
    throw new Error('Resolved Codex executable is invalid')
  }
  return executable
}

function safeEnvironment(source: Readonly<Record<string, string | undefined>>): Readonly<Record<string, string>> {
  const environment: Record<string, string> = {}
  const pathValue = source.PATH ?? source.Path
  if (pathValue !== undefined) {
    if (CONTROL_CHARACTER.test(pathValue)) throw new Error('Codex launch PATH is invalid')
    environment.PATH = pathValue
  }
  for (const key of SAFE_ENVIRONMENT_KEYS) {
    const value = source[key]
    if (value === undefined || key === 'PATH') continue
    if (CONTROL_CHARACTER.test(value)) {
      throw new Error(`Codex launch environment variable ${key} is invalid`)
    }
    environment[key] = value
  }
  return Object.freeze(environment)
}

/**
 * Produces the arguments for a main-owned Codex PTY without invoking a shell.
 * The sole launch secret is scoped to the ephemeral loopback MCP listener.
 */
export function buildCodexCampaignAgentLaunch(
  input: CodexCampaignAgentLaunchInput,
  dependencies: CodexCampaignAgentLaunchDependencies,
): CodexCampaignAgentLaunch {
  validateInputShape(input)
  const workspaceId = identifier(input.intent.workspaceId, 'workspace')
  const campaignId = identifier(input.intent.campaignId, 'campaign')
  const terminalId = identifier(input.terminalId, 'terminal')
  const generation = identifier(input.generation, 'PTY generation')
  if (!HOST_INSTANCE_ID.test(input.hostInstanceId)) {
    throw new Error('Codex campaign launch desktop host identifier is invalid')
  }
  // Validate the scope even though it is deliberately not serialized into argv
  // or the environment. The loopback host remains the authority for this scope.
  void workspaceId
  void campaignId

  const mcp = validateMcpLaunch(input.mcpLaunch, terminalId, generation)
  const cwd = resolveWorkingDirectory(input.intent.cwd, dependencies)
  const executable = resolveExecutable(dependencies)
  const env = safeEnvironment(dependencies.sourceEnv ?? process.env)
  const identity = createMainOwnedCampaignAgentIdentity({
    terminalId,
    generation,
    family: 'codex',
    hostInstanceId: input.hostInstanceId,
  })
  const args = Object.freeze([
    '-c',
    `mcp_servers.bashgym_campaign.url="${mcp.url}"`,
    '-c',
    `mcp_servers.bashgym_campaign.http_headers={"${MCP_LAUNCH_HEADER}"="${mcp.launchSecret}"}`,
  ])
  return Object.freeze({
    executableFamily: 'codex' as const,
    executable,
    args,
    cwd,
    env,
    identity: Object.freeze(identity),
  })
}
