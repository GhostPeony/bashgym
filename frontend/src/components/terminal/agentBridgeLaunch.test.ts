import assert from 'node:assert/strict'
import test from 'node:test'
import { buildAgentBridgeLaunchCommand } from '../../../electron/agentBridge'

const request = {
  workspaceId: 'workspace-main',
  terminalId: 'terminal-1',
  panelId: 'panel-1',
  apiBase: 'http://127.0.0.1:8003',
  pythonCommand: 'python'
} as const

test('attaches BashGym MCP to a Claude launch without changing global config', () => {
  const command = buildAgentBridgeLaunchCommand(
    {
      ...request,
      kind: 'claude',
      claudeConfigPath: 'C:\\Users\\developer\\.bashgym\\agent_bridge\\terminal-1.claude-mcp.json'
    },
    'win32'
  )

  assert.match(command, /^claude --mcp-config /)
  assert.match(command, /terminal-1\.claude-mcp\.json/)
  assert.doesNotMatch(command, /mcpServers/)
  assert.doesNotMatch(command, /mcp add/)
})

test('layers a workspace-scoped BashGym MCP override onto Codex', () => {
  const command = buildAgentBridgeLaunchCommand(
    {
      ...request,
      kind: 'codex',
      serverCommand: 'C:\\Users\\developer\\.bashgym\\agent_bridge\\terminal-1.cmd',
      serverArgs: []
    },
    'win32'
  )

  assert.match(command, /^codex -c /)
  assert.match(command, /mcp_servers\.bashgym\.command/)
  assert.match(command, /mcp_servers\.bashgym\.args/)
  assert.match(command, /agent_bridge/)
  assert.match(command, /mcp_servers\.bashgym\.args=\[\]/)
  assert.doesNotMatch(command, /codex mcp add/)
})
