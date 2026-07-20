/**
 * Agent Sessions store — orchestrates polling of local agent-CLI session
 * journals while the Agent Sessions rail is open.
 *
 * Zero cost when closed: the poll timer exists only between startPolling()
 * (rail mount) and stopPolling() (rail unmount); ticks pause while the
 * document is hidden and skip when a previous tick is still in flight.
 */

import { create } from 'zustand'
import type { AgentSessionSnapshot, SessionMatch } from '../services/agentSessions/types'
import type { SessionAccountInfo, SessionFileInfo } from '../../electron/preload'
import {
  ingestClaudeFile,
  resetClaudeFile,
  encodeClaudeProjectDir
} from '../services/agentSessions/claudeSessionAdapter'
import { ingestCodexFile, resetCodexFile } from '../services/agentSessions/codexSessionAdapter'
import { matchSessions, type MatchableTerminal } from '../services/agentSessions/matching'
import { useTerminalStore } from './terminalStore'
import { loadRegistry } from './workspacePersistence'
import { buildWorkspaceSessionIndex } from './workspaceSessionIndex'

const POLL_INTERVAL_MS = 15_000
const MAX_INGESTS_PER_TICK = 10
const SCAN_LOOKBACK_DAYS = 14
const PINS_KEY = 'bashgym_session_pins'
const ACCOUNT_OPTIN_KEY = 'bashgym_sessions_account_optin'

const loadPins = (): Record<string, string> => {
  try {
    const stored = localStorage.getItem(PINS_KEY)
    if (stored) return JSON.parse(stored)
  } catch {
    // Ignore
  }
  return {}
}

const savePins = (pins: Record<string, string>) => {
  try {
    localStorage.setItem(PINS_KEY, JSON.stringify(pins))
  } catch {
    // Ignore storage errors
  }
}

// Snapshot cache: parsed journal intel persisted across app sessions so the
// feed fills instantly on open — only files whose size changed re-read.
const SNAPSHOT_CACHE_KEY = 'bashgym_session_snapshots_v1'
const SNAPSHOT_CACHE_MAX = 400

const loadSnapshotCache = (): Map<string, AgentSessionSnapshot> => {
  try {
    const stored = localStorage.getItem(SNAPSHOT_CACHE_KEY)
    if (stored) {
      const arr: AgentSessionSnapshot[] = JSON.parse(stored)
      return new Map(arr.filter((s) => s && s.filePath).map((s) => [s.filePath, s]))
    }
  } catch {
    // Ignore
  }
  return new Map()
}

const saveSnapshotCache = (snapshots: Map<string, AgentSessionSnapshot>) => {
  try {
    const arr = Array.from(snapshots.values())
      .sort((a, b) => b.fileMtime - a.fileMtime)
      .slice(0, SNAPSHOT_CACHE_MAX)
    localStorage.setItem(SNAPSHOT_CACHE_KEY, JSON.stringify(arr))
  } catch {
    // Ignore storage errors
  }
}

// Module-level bookkeeping — not reactive state
let pollTimer: ReturnType<typeof setInterval> | null = null
let pollInFlight = false
let firstPollDone = false
const knownSizes = new Map<string, number>()

const initialSnapshots = loadSnapshotCache()
for (const [path, snap] of initialSnapshots) {
  knownSizes.set(path, snap.fileSize)
}

interface AgentSessionsState {
  isPolling: boolean
  /** Bumped once per completed poll — subscribe to this for cheap re-renders */
  version: number
  lastScanAt: number | null
  snapshots: Map<string, AgentSessionSnapshot>
  matches: Map<string, SessionMatch>
  liveTerminalIds: Set<string>
  hasLiveTerminalSnapshot: boolean
  pins: Record<string, string>
  accountOptIn: boolean
  account: SessionAccountInfo | null
  error: string | null

  startPolling: () => void
  stopPolling: () => void
  pollOnce: () => Promise<void>
  pinSession: (panelId: string, filePath: string | null) => void
  setAccountOptIn: (optIn: boolean) => void
}

export const useAgentSessionsStore = create<AgentSessionsState>((set, get) => ({
  isPolling: false,
  version: 0,
  lastScanAt: null,
  snapshots: initialSnapshots,
  matches: new Map(),
  liveTerminalIds: new Set(),
  hasLiveTerminalSnapshot: false,
  pins: loadPins(),
  accountOptIn: localStorage.getItem(ACCOUNT_OPTIN_KEY) === 'true',
  account: null,
  error: null,

  startPolling: () => {
    if (pollTimer) return
    set({ isPolling: true })
    void get().pollOnce()
    pollTimer = setInterval(() => {
      if (document.hidden) return
      void get().pollOnce()
    }, POLL_INTERVAL_MS)
  },

  stopPolling: () => {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
    set({ isPolling: false })
  },

  pollOnce: async () => {
    const api = window.bashgym?.sessions
    if (!api || pollInFlight) return
    pollInFlight = true
    try {
      const terminalState = useTerminalStore.getState()
      const [scan, terminalInfos] = await Promise.all([
        api.scan(SCAN_LOOKBACK_DAYS),
        window.bashgym?.terminal.list().catch(() => []) ?? Promise.resolve([])
      ])
      if (!scan.success) {
        set({ error: scan.error ?? 'Session scan failed' })
        return
      }
      const liveTerminalIds = new Set(
        terminalInfos.filter((info) => !info.exited).map((info) => info.id)
      )
      const registry = loadRegistry()
      const workspaceGroups = buildWorkspaceSessionIndex({
        workspaces: registry.workspaces,
        activeWorkspaceId: registry.activeWorkspaceId,
        activePanels: terminalState.panels,
        activeSessions: terminalState.sessions,
        liveTerminalIds
      })
      const terminals: MatchableTerminal[] = workspaceGroups.flatMap((group) =>
        group.sessions
          .filter((record) => record.runtimeState !== 'saved')
          .map((record) => ({
            terminalId: record.session.id,
            panelId: record.panel.id,
            cwd: record.session.cwd,
            agentKind: record.session.agentKind,
            lastActivity: record.session.lastActivity
          }))
      )

      const files: Array<SessionFileInfo & { kind: 'claude' | 'codex' }> = [
        ...(scan.claude ?? []).map((f) => ({ ...f, kind: 'claude' as const })),
        ...(scan.codex ?? []).map((f) => ({ ...f, kind: 'codex' as const }))
      ]

      // Drop state for files that disappeared (session cleared/aged out)
      const present = new Set(files.map((f) => f.path))
      const snapshots = new Map(get().snapshots)
      for (const path of Array.from(snapshots.keys())) {
        if (!present.has(path)) {
          snapshots.delete(path)
          knownSizes.delete(path)
          resetClaudeFile(path)
          resetCodexFile(path)
        }
      }

      // Journals in a live terminal's project (or pinned) get exact ingest;
      // historical journals get the cheap head+tail read. Case-insensitive —
      // Windows prompt casing can differ from the journal dir encoding.
      const hotClaudeDirs = new Set(
        terminals
          .filter((t) => t.cwd && t.cwd !== '~')
          .map((t) => encodeClaudeProjectDir(t.cwd).toLowerCase())
      )
      const pinnedPaths = new Set(Object.values(get().pins))
      const isHot = (f: SessionFileInfo & { kind: 'claude' | 'codex' }) => {
        if (pinnedPaths.has(f.path)) return true
        if (f.kind !== 'claude') return true // codex ingest is always cheap
        const dir = f.path.replace(/\\/g, '/').split('/').slice(-2, -1)[0]?.toLowerCase() ?? ''
        return hotClaudeDirs.has(dir)
      }

      // Ingest changed files: hot first, then freshest, bounded per tick.
      // The first poll after open bursts higher so an uncached feed fills fast.
      const tickCap = firstPollDone ? MAX_INGESTS_PER_TICK : 24
      firstPollDone = true
      const changed = files
        .filter((f) => knownSizes.get(f.path) !== f.size)
        .sort((a, b) => Number(isHot(b)) - Number(isHot(a)) || b.modified - a.modified)
        .slice(0, tickCap)

      for (const file of changed) {
        const snapshot =
          file.kind === 'claude'
            ? await ingestClaudeFile(file, isHot(file))
            : await ingestCodexFile(file)
        if (snapshot) {
          snapshots.set(file.path, snapshot)
          knownSizes.set(file.path, file.size)
        }
      }

      const matches = matchSessions(terminals, snapshots, get().pins)

      // Backfill matched intel into terminal sessions (gitBranch/model/metrics
      // don't bump sessionsVersion, so this causes no canvas re-render churn)
      for (const [terminalId, match] of matches) {
        const snap = snapshots.get(match.filePath)
        if (!snap || match.confidence === 'none') continue
        const updates: Record<string, unknown> = {}
        if (snap.gitBranch) updates.gitBranch = snap.gitBranch
        if (snap.model) updates.model = snap.model
        if (snap.totals.input + snap.totals.output > 0) {
          updates.metrics = {
            inputTokens: snap.totals.input + snap.totals.cacheRead + snap.totals.cacheCreate,
            outputTokens: snap.totals.output,
            cost: snap.estCostUsd ?? 0
          }
        }
        if (Object.keys(updates).length > 0) terminalState.updateSession(terminalId, updates)
      }

      set((state) => ({
        snapshots,
        matches,
        liveTerminalIds,
        hasLiveTerminalSnapshot: true,
        version: state.version + 1,
        lastScanAt: Date.now(),
        error: null
      }))
      if (changed.length > 0) saveSnapshotCache(snapshots)

      if (get().accountOptIn && !get().account) {
        const result = await api.readAccount()
        if (result.success && result.account) set({ account: result.account })
      }
    } catch (err) {
      const msg = String(err)
      if (/No handler registered/i.test(msg)) {
        // Running app predates the sessions IPC (main process only loads at
        // launch) — stop polling instead of erroring every tick
        get().stopPolling()
        set({ error: 'App update pending — restart BashGym to enable session intel.' })
      } else {
        set({ error: msg })
      }
    } finally {
      pollInFlight = false
    }
  },

  pinSession: (panelId: string, filePath: string | null) => {
    const pins = { ...get().pins }
    if (filePath) {
      pins[panelId] = filePath
    } else {
      delete pins[panelId]
    }
    savePins(pins)
    set({ pins })
    void get().pollOnce()
  },

  setAccountOptIn: (optIn: boolean) => {
    try {
      localStorage.setItem(ACCOUNT_OPTIN_KEY, String(optIn))
    } catch {
      // Ignore storage errors
    }
    set({ accountOptIn: optIn, account: optIn ? get().account : null })
    if (optIn) {
      void window.bashgym?.sessions.readAccount().then((result) => {
        if (result.success && result.account) set({ account: result.account })
      })
    }
  }
}))
