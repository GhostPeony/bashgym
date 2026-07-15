import type { Panel, TerminalSession } from './terminalStore'
import type { WorkspaceMeta } from './workspacePersistence'
import { loadWorkspaceSnapshot, terminalSessionFromPersistedPanel } from './workspacePersistence'

export type SessionRuntimeState = 'live' | 'saved' | 'unknown'

export interface WorkspaceSessionRecord {
  workspaceId: string
  workspaceName: string
  isActiveWorkspace: boolean
  panel: Panel
  session: TerminalSession
  runtimeState: SessionRuntimeState
}

export interface WorkspaceSessionGroup {
  workspace: WorkspaceMeta
  isActive: boolean
  sessions: WorkspaceSessionRecord[]
  runningCount: number
  waitingCount: number
  errorCount: number
  liveCount: number
}

interface BuildWorkspaceSessionIndexInput {
  workspaces: WorkspaceMeta[]
  activeWorkspaceId: string
  activePanels: Panel[]
  activeSessions: Map<string, TerminalSession>
  liveTerminalIds?: Set<string>
}

function runtimeStateFor(
  terminalId: string,
  liveTerminalIds: Set<string> | undefined
): SessionRuntimeState {
  if (!liveTerminalIds) return 'unknown'
  return liveTerminalIds.has(terminalId) ? 'live' : 'saved'
}

export function buildWorkspaceSessionIndex({
  workspaces,
  activeWorkspaceId,
  activePanels,
  activeSessions,
  liveTerminalIds
}: BuildWorkspaceSessionIndexInput): WorkspaceSessionGroup[] {
  return workspaces.map((workspace) => {
    const isActive = workspace.id === activeWorkspaceId
    const records: WorkspaceSessionRecord[] = []

    if (isActive) {
      for (const panel of activePanels) {
        if (panel.type !== 'terminal' || !panel.terminalId) continue
        const session = activeSessions.get(panel.terminalId)
        if (!session) continue
        records.push({
          workspaceId: workspace.id,
          workspaceName: workspace.name,
          isActiveWorkspace: true,
          panel,
          session,
          runtimeState: runtimeStateFor(panel.terminalId, liveTerminalIds)
        })
      }
    } else {
      const snapshot = loadWorkspaceSnapshot(workspace.id)
      for (const persisted of snapshot.panels) {
        if (persisted.type !== 'terminal' || !persisted.terminalId) continue
        records.push({
          workspaceId: workspace.id,
          workspaceName: workspace.name,
          isActiveWorkspace: false,
          panel: {
            id: persisted.id,
            type: persisted.type,
            title: persisted.title,
            terminalId: persisted.terminalId
          },
          session: terminalSessionFromPersistedPanel(persisted),
          runtimeState: runtimeStateFor(persisted.terminalId, liveTerminalIds)
        })
      }
    }

    records.sort((a, b) => b.session.lastActivity - a.session.lastActivity)
    return {
      workspace,
      isActive,
      sessions: records,
      runningCount: records.filter(({ session }) => session.status === 'running' || session.status === 'tool_calling').length,
      waitingCount: records.filter(({ session }) => session.status === 'waiting_input').length,
      errorCount: records.filter(({ session }) => session.attention === 'error' || Boolean(session.errorMessage)).length,
      liveCount: records.filter(({ runtimeState }) => runtimeState === 'live').length
    }
  })
}
