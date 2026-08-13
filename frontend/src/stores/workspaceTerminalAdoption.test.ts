import assert from 'node:assert/strict'
import test, { before } from 'node:test'

import type { Panel, TerminalSession } from './terminalStore'
import type { WorkspaceMeta } from './workspacePersistence'

class MemoryStorage {
  private values = new Map<string, string>()

  get length() {
    return this.values.size
  }
  clear() {
    this.values.clear()
  }
  getItem(key: string) {
    return this.values.get(key) ?? null
  }
  key(index: number) {
    return Array.from(this.values.keys())[index] ?? null
  }
  removeItem(key: string) {
    this.values.delete(key)
  }
  setItem(key: string, value: string) {
    this.values.set(key, value)
  }
}

const storage = new MemoryStorage()
let loadWorkspaceSnapshot: (typeof import('./workspacePersistence'))['loadWorkspaceSnapshot']
let saveRegistry: (typeof import('./workspacePersistence'))['saveRegistry']
let useTerminalStore: (typeof import('./terminalStore'))['useTerminalStore']
let useWorkspaceStore: (typeof import('./workspaceStore'))['useWorkspaceStore']

before(async () => {
  Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true })
  const persistence = await import('./workspacePersistence')
  loadWorkspaceSnapshot = persistence.loadWorkspaceSnapshot
  saveRegistry = persistence.saveRegistry
  useTerminalStore = (await import('./terminalStore')).useTerminalStore
  useWorkspaceStore = (await import('./workspaceStore')).useWorkspaceStore
})

const workspaces: WorkspaceMeta[] = [
  { id: 'workspace-a', name: 'A', createdAt: 1, lastActiveAt: 1 },
  { id: 'workspace-b', name: 'B', createdAt: 2, lastActiveAt: 2 }
]

function terminalSession(id: string, title: string, cwd: string): TerminalSession {
  return {
    id,
    title,
    cwd,
    isActive: true,
    attention: 'none',
    showBanner: false,
    status: 'idle',
    lastActivity: 1
  }
}

function resetToWorkspaceB(): void {
  storage.clear()
  saveRegistry({ version: 1, activeWorkspaceId: 'workspace-b', workspaces })
  useWorkspaceStore.setState({
    workspaces,
    activeWorkspaceId: 'workspace-b',
    switching: false,
    sessionIndexVersion: 0
  })
  const panel: Panel = {
    id: 'panel-b',
    type: 'terminal',
    title: 'Workspace B terminal',
    terminalId: 'terminal-b'
  }
  useTerminalStore.setState({
    panels: [panel],
    sessions: new Map([
      ['terminal-b', terminalSession('terminal-b', panel.title, 'C:\\workspace-b')]
    ]),
    activePanelId: panel.id,
    activeSessionId: 'terminal-b',
    canvasNodes: new Map(),
    canvasEdges: [],
    sessionsVersion: 0
  })
}

type WorkspaceTerminalAdoption = (
  workspaceId: string,
  terminal: { terminalId: string; title: string; cwd: string }
) => boolean

function adoptionAction(): WorkspaceTerminalAdoption {
  const action = (
    useWorkspaceStore.getState() as unknown as {
      adoptTerminalIntoWorkspace?: WorkspaceTerminalAdoption
    }
  ).adoptTerminalIntoWorkspace
  assert.equal(typeof action, 'function')
  return action as WorkspaceTerminalAdoption
}

test('persists a late launch only to its original background workspace and materializes it on return', () => {
  resetToWorkspaceB()
  const adopt = adoptionAction()
  const terminal = {
    terminalId: 'terminal-codex-race',
    title: 'AutoResearch Codex',
    cwd: 'C:\\workspace-a'
  }

  assert.equal(adopt('workspace-a', terminal), true)
  assert.equal(useWorkspaceStore.getState().activeWorkspaceId, 'workspace-b')
  assert.equal(useWorkspaceStore.getState().sessionIndexVersion, 1)
  assert.deepEqual(
    useTerminalStore.getState().panels.map((panel) => panel.terminalId),
    ['terminal-b']
  )
  assert.equal(useTerminalStore.getState().sessions.has(terminal.terminalId), false)

  const backgroundSnapshot = loadWorkspaceSnapshot('workspace-a')
  assert.equal(backgroundSnapshot.panels.length, 1)
  assert.deepEqual(backgroundSnapshot.panels[0], {
    id: backgroundSnapshot.panels[0].id,
    type: 'terminal',
    title: terminal.title,
    terminalId: terminal.terminalId,
    cwd: terminal.cwd,
    sessionState: {
      status: 'idle',
      attention: 'none',
      lastActivity: backgroundSnapshot.panels[0].sessionState?.lastActivity
    }
  })
  assert.equal(
    loadWorkspaceSnapshot('workspace-b').panels.some(
      (panel) => panel.terminalId === terminal.terminalId
    ),
    false
  )

  assert.equal(adopt('workspace-a', terminal), true)
  assert.equal(
    loadWorkspaceSnapshot('workspace-a').panels.filter(
      (panel) => panel.terminalId === terminal.terminalId
    ).length,
    1
  )
  assert.equal(useWorkspaceStore.getState().sessionIndexVersion, 1)

  useWorkspaceStore.getState().switchWorkspace('workspace-a')
  const materialized = useTerminalStore.getState()
  const session = materialized.sessions.get(terminal.terminalId)
  assert.equal(useWorkspaceStore.getState().activeWorkspaceId, 'workspace-a')
  assert.equal(
    materialized.panels.some((panel) => panel.terminalId === terminal.terminalId),
    true
  )
  assert.equal(session?.title, terminal.title)
  assert.equal(session?.cwd, terminal.cwd)
  assert.equal(session?.launchCommand, undefined)
  assert.equal(
    loadWorkspaceSnapshot('workspace-b').panels.some(
      (panel) => panel.terminalId === terminal.terminalId
    ),
    false
  )
})

test('adopts normally in the active workspace and focuses the real existing panel on replay', () => {
  resetToWorkspaceB()
  useWorkspaceStore.getState().switchWorkspace('workspace-a')
  const adopt = adoptionAction()
  const terminal = {
    terminalId: 'terminal-codex-active',
    title: 'AutoResearch Codex',
    cwd: 'C:\\workspace-a'
  }

  assert.equal(adopt('workspace-a', terminal), true)
  const createdPanel = useTerminalStore
    .getState()
    .panels.find((panel) => panel.terminalId === terminal.terminalId)
  assert.ok(createdPanel)
  assert.equal(useTerminalStore.getState().sessions.get(terminal.terminalId)?.cwd, terminal.cwd)
  assert.equal(
    useTerminalStore.getState().sessions.get(terminal.terminalId)?.launchCommand,
    undefined
  )

  useTerminalStore.setState({ activePanelId: null, activeSessionId: null })
  assert.equal(adopt('workspace-a', terminal), true)
  assert.equal(
    useTerminalStore.getState().panels.filter((panel) => panel.terminalId === terminal.terminalId)
      .length,
    1
  )
  assert.equal(useTerminalStore.getState().activePanelId, createdPanel.id)
  assert.equal(useTerminalStore.getState().activeSessionId, terminal.terminalId)
})

test('fails closed when the launch workspace was deleted before adoption completes', () => {
  resetToWorkspaceB()
  const adopt = adoptionAction()

  assert.equal(
    adopt('workspace-deleted', {
      terminalId: 'terminal-codex-deleted-workspace',
      title: 'AutoResearch Codex',
      cwd: 'C:\\deleted-workspace'
    }),
    false
  )
  assert.equal(useWorkspaceStore.getState().sessionIndexVersion, 0)
  assert.deepEqual(
    useTerminalStore.getState().panels.map((panel) => panel.terminalId),
    ['terminal-b']
  )
  assert.equal(loadWorkspaceSnapshot('workspace-a').panels.length, 0)
  assert.equal(loadWorkspaceSnapshot('workspace-b').panels.length, 0)
})
