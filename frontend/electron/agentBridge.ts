export type AgentBridgeKind = 'claude' | 'codex'

export interface AgentBridgeLaunchRequest {
  kind: AgentBridgeKind
  workspaceId: string
  terminalId: string
  panelId?: string
  apiBase?: string
  pythonCommand?: string
  serverCommand?: string
  serverArgs?: string[]
  claudeConfigPath?: string
}

function shellQuote(value: string, platform: NodeJS.Platform): string {
  if (platform === 'win32') return `'${value.replaceAll("'", "''")}'`
  return `'${value.replaceAll("'", `'"'"'`)}'`
}

function serverArgs(request: AgentBridgeLaunchRequest): string[] {
  const args = [
    '-m',
    'bashgym.mcp.skill_lab_server',
    '--workspace-id',
    request.workspaceId || 'default',
    '--origin-terminal-id',
    request.terminalId,
    '--agent',
    request.kind,
  ]
  if (request.panelId) args.push('--origin-panel-id', request.panelId)
  if (request.apiBase) args.push('--api-base', request.apiBase)
  return args
}

/** Build a launch-only MCP attachment without modifying global Claude/Codex config. */
export function buildAgentBridgeLaunchCommand(
  request: AgentBridgeLaunchRequest,
  platform: NodeJS.Platform = process.platform,
): string {
  const pythonCommand = request.pythonCommand || 'python'
  const command = request.serverCommand || pythonCommand
  const args = request.serverArgs || serverArgs(request)

  if (request.kind === 'claude') {
    if (request.claudeConfigPath) {
      return `claude --mcp-config ${shellQuote(request.claudeConfigPath, platform)}`
    }
    const config = JSON.stringify({
      mcpServers: {
        bashgym: {
          command,
          args,
        },
      },
    })
    return `claude --mcp-config ${shellQuote(config, platform)}`
  }

  const commandOverride = `mcp_servers.bashgym.command=${JSON.stringify(command)}`
  const argsOverride = `mcp_servers.bashgym.args=${JSON.stringify(args)}`
  return [
    'codex',
    '-c',
    shellQuote(commandOverride, platform),
    '-c',
    shellQuote(argsOverride, platform),
  ].join(' ')
}
