/**
 * Monitor edge routing.
 *
 * A terminal→terminal canvas edge is a monitor edge: source = WATCHED,
 * target = WATCHER. Snapshots of the watched terminal's scrollback are
 * written to a temp markdown file and the path is typed into the watcher's
 * input — prefilled (no Enter) or submitted, depending on mode.
 *
 * Sends are edge-directional, so this deliberately does not reuse
 * routeToLinkedTerminals (which fans out to every linked terminal).
 */

import { useTerminalStore } from '../stores'
import type { CanvasEdge, Panel } from '../stores'
import { stripAnsi } from './ansi'

export const AUTO_MIN_INTERVAL_MS = 20_000
const SNAPSHOT_MAX_BYTES = 64_000
const SNAPSHOT_MAX_LINES = 200

// edgeId → timestamp of last auto-send (module-level; no store churn)
const lastAutoSentAt = new Map<string, number>()

/** True when both edge endpoints resolve to terminal panels */
export function isMonitorEdge(edge: { source: string; target: string }, panels: Panel[]): boolean {
  const source = panels.find((p) => p.id === edge.source)
  const target = panels.find((p) => p.id === edge.target)
  return source?.type === 'terminal' && target?.type === 'terminal' && source.id !== target.id
}

export interface MonitorInfo {
  /** This panel is the watched end of at least one monitor edge */
  isWatched: boolean
  /** Title of the watched terminal when this panel is a watcher */
  watchingTitle?: string
}

export function getMonitorInfo(panelId: string, edges: CanvasEdge[], panels: Panel[]): MonitorInfo {
  let isWatched = false
  const watchedTitles: string[] = []
  for (const edge of edges) {
    if (!isMonitorEdge(edge, panels)) continue
    if (edge.source === panelId) isWatched = true
    if (edge.target === panelId) {
      const watched = panels.find((p) => p.id === edge.source)
      if (watched) watchedTitles.push(watched.title)
    }
  }
  return {
    isWatched,
    watchingTitle:
      watchedTitles.length === 0
        ? undefined
        : watchedTitles.length === 1
          ? watchedTitles[0]
          : `${watchedTitles.length} terminals`
  }
}

export interface SnapshotResult {
  sent: boolean
  error?: string
}

function buildSnapshotDoc(watched: Panel, tail: string): string {
  const session = watched.terminalId
    ? useTerminalStore.getState().sessions.get(watched.terminalId)
    : undefined
  const agent = session?.agentKind ?? 'shell'
  const status = session?.status ?? 'unknown'
  const cwd = session?.cwd ?? ''
  return [
    `# Terminal snapshot: ${watched.title}`,
    '',
    `- Agent: ${agent}`,
    `- Status: ${status}`,
    cwd ? `- Working dir: ${cwd}` : null,
    `- Captured: ${new Date().toISOString()}`,
    '',
    'You are the WATCHER for this terminal. Review the recent output below,',
    'summarize progress, and flag anything that conflicts with your own work.',
    '',
    `## Recent output (last ${SNAPSHOT_MAX_LINES} lines max)`,
    '',
    '```text',
    tail,
    '```',
    ''
  ]
    .filter((line) => line !== null)
    .join('\n')
}

/**
 * Snapshot the watched terminal of a monitor edge and type the resulting
 * file path into the watcher's input. `submit` appends Enter.
 */
export async function sendMonitorSnapshot(edgeId: string, submit = false): Promise<SnapshotResult> {
  const { canvasEdges, panels } = useTerminalStore.getState()
  const edge = canvasEdges.find((e) => e.id === edgeId)
  if (!edge) return { sent: false, error: 'Edge not found' }

  const watched = panels.find((p) => p.id === edge.source)
  const watcher = panels.find((p) => p.id === edge.target)
  if (!watched?.terminalId || !watcher?.terminalId || watched.id === watcher.id) {
    return { sent: false, error: 'Both ends must be terminals' }
  }

  if (typeof window.bashgym?.terminal.snapshot !== 'function') {
    return { sent: false, error: 'Snapshot bridge unavailable — restart the app' }
  }

  const snap = await window.bashgym.terminal.snapshot(watched.terminalId, SNAPSHOT_MAX_BYTES)
  if (!snap?.success || !snap.data) {
    return { sent: false, error: snap?.error ?? 'No output captured yet' }
  }

  // Drop the first line: the byte-capped tail may start mid-line/mid-sequence
  const lines = stripAnsi(snap.data).split('\n').slice(1)
  const tail = lines.slice(-SNAPSHOT_MAX_LINES).join('\n').trimEnd()
  if (!tail) return { sent: false, error: 'No output captured yet' }

  const doc = buildSnapshotDoc(watched, tail)
  const base64 = btoa(unescape(encodeURIComponent(doc)))
  const written = await window.bashgym?.files.writeTempFile(
    `data:text/plain;base64,${base64}`,
    'md',
    'bashgym_monitor'
  )
  if (!written?.success || !written.path) {
    return { sent: false, error: written?.error ?? 'Failed to write snapshot file' }
  }

  window.bashgym?.terminal.write(watcher.terminalId, written.path + (submit ? '\r' : ''))
  return { sent: true }
}

/**
 * Auto-mode trigger, called when a watched terminal's agent finishes a step
 * (status transition running/tool_calling → waiting_input/idle).
 * Zero-cost when no canvas edges exist.
 */
export function maybeAutoSnapshot(watchedTerminalId: string): void {
  const { canvasEdges, panels, sessions } = useTerminalStore.getState()
  if (canvasEdges.length === 0) return

  const watchedPanel = panels.find((p) => p.terminalId === watchedTerminalId)
  if (!watchedPanel) return

  for (const edge of canvasEdges) {
    if (edge.type !== 'monitor') continue
    const mode = edge.data?.auto ?? 'off'
    if (mode === 'off' || edge.source !== watchedPanel.id) continue

    const last = lastAutoSentAt.get(edge.id) ?? 0
    if (Date.now() - last < AUTO_MIN_INTERVAL_MS) continue

    if (mode === 'send') {
      // Don't submit into a watcher that is mid-task; retry on the next transition
      const watcherPanel = panels.find((p) => p.id === edge.target)
      const watcherSession = watcherPanel?.terminalId
        ? sessions.get(watcherPanel.terminalId)
        : undefined
      if (watcherSession && (watcherSession.status === 'running' || watcherSession.status === 'tool_calling')) {
        continue
      }
    }

    lastAutoSentAt.set(edge.id, Date.now())
    void sendMonitorSnapshot(edge.id, mode === 'send')
  }
}
