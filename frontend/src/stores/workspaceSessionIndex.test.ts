import assert from 'node:assert/strict'
import test from 'node:test'
import type { Panel, TerminalSession } from './terminalStore'
import type { WorkspaceMeta } from './workspacePersistence'
import { appendPanelToWorkspace, wsKey } from './workspacePersistence'
import { buildWorkspaceSessionIndex } from './workspaceSessionIndex'

class MemoryStorage {
  private values = new Map<string, string>()
  failKey: string | null = null

  get length() { return this.values.size }
  clear() { this.values.clear() }
  getItem(key: string) { return this.values.get(key) ?? null }
  key(index: number) { return Array.from(this.values.keys())[index] ?? null }
  removeItem(key: string) { this.values.delete(key) }
  setItem(key: string, value: string) {
    if (key === this.failKey) {
      this.failKey = null
      throw new Error('quota exceeded')
    }
    this.values.set(key, value)
  }
}

const storage = new MemoryStorage()
Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true })

const workspaces: WorkspaceMeta[] = [
  { id: 'main', name: 'MAIN', createdAt: 1, lastActiveAt: 2 },
  { id: 'background', name: 'RESEARCH', createdAt: 1, lastActiveAt: 1 }
]

function terminalSession(id: string): TerminalSession {
  return {
    id,
    title: 'Claude',
    cwd: 'C:\\repo',
    isActive: false,
    attention: 'waiting',
    showBanner: false,
    status: 'waiting_input',
    lastActivity: 123,
    agentKind: 'claude'
  }
}

test('indexes active and background sessions under their owning workspace', () => {
  storage.clear()
  storage.setItem(wsKey('background', 'panels'), JSON.stringify([{
    id: 'panel-background',
    type: 'terminal',
    title: 'Codex',
    terminalId: 'terminal-background',
    cwd: 'C:\\research',
    agentKind: 'codex',
    sessionState: {
      status: 'running',
      attention: 'none',
      lastActivity: 99,
      taskSummary: 'Indexing papers'
    }
  }]))

  const activePanel: Panel = {
    id: 'panel-main',
    type: 'terminal',
    title: 'Claude',
    terminalId: 'terminal-main'
  }
  const groups = buildWorkspaceSessionIndex({
    workspaces,
    activeWorkspaceId: 'main',
    activePanels: [activePanel],
    activeSessions: new Map([['terminal-main', terminalSession('terminal-main')]]),
    liveTerminalIds: new Set(['terminal-main', 'terminal-background'])
  })

  assert.equal(groups[0].workspace.name, 'MAIN')
  assert.equal(groups[0].waitingCount, 1)
  assert.equal(groups[0].sessions[0].runtimeState, 'live')
  assert.equal(groups[1].workspace.name, 'RESEARCH')
  assert.equal(groups[1].runningCount, 1)
  assert.equal(groups[1].sessions[0].session.taskSummary, 'Indexing papers')
  assert.equal(groups[1].sessions[0].runtimeState, 'live')
})

test('rolls back a destination snapshot when a transactional move write fails', () => {
  storage.clear()
  const original = JSON.stringify([{ id: 'existing', type: 'preview', title: 'Existing' }])
  storage.setItem(wsKey('background', 'panels'), original)
  storage.failKey = wsKey('background', 'positions')

  const moved = appendPanelToWorkspace('background', {
    id: 'moving',
    type: 'terminal',
    title: 'Moving',
    terminalId: 'terminal-moving'
  })

  assert.equal(moved, false)
  assert.equal(storage.getItem(wsKey('background', 'panels')), original)
})
