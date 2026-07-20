/**
 * Incremental parser for Codex CLI rollout journals
 * (~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl).
 *
 * The first line is a session_meta payload (session id, cwd, cli version,
 * model provider, git info). token_count events carry CUMULATIVE totals plus
 * the model's context window and rate limits, so tail-only reads suffice —
 * no full-file bootstrap is ever needed.
 */

import type { AgentSessionSnapshot, TailState, TokenTotals } from './types'
import { emptyTotals } from './types'

interface FileState {
  tail: TailState
  snapshot: AgentSessionSnapshot
  metaLoaded: boolean
}

const fileStates = new Map<string, FileState>()

// The session_meta first line embeds base_instructions, dynamic_tools, and
// other unbounded fields — recent Codex CLI versions push it well past 16KB
// (40KB+ observed). Read up to the IPC hard cap so the whole first line is
// captured; a truncated line fails JSON.parse and the session loses its cwd,
// which drops it from the rail's project history entirely.
const META_HEAD_BYTES = 262_144

export function resetCodexFile(filePath: string): void {
  fileStates.delete(filePath)
}

function newSnapshot(filePath: string): AgentSessionSnapshot {
  return {
    kind: 'codex',
    filePath,
    contextWindowApprox: false,
    totals: emptyTotals(),
    perModel: {},
    recentFiles: [],
    totalsApprox: false,
    fileSize: 0,
    fileMtime: 0
  }
}

function readUsage(usage: Record<string, unknown> | undefined): TokenTotals | null {
  if (!usage || typeof usage !== 'object') return null
  return {
    input: Number(usage.input_tokens) || 0,
    output: Number(usage.output_tokens) || 0,
    cacheRead: Number(usage.cached_input_tokens) || 0,
    cacheCreate: 0
  }
}

function applyCwd(snap: AgentSessionSnapshot, cwd: unknown): void {
  if (snap.cwd !== undefined || typeof cwd !== 'string' || !cwd) return
  snap.cwd = cwd
  const base = cwd
    .replace(/[\\/]+$/, '')
    .split(/[\\/]/)
    .pop()
  if (base) snap.title = base
}

function applyMeta(snap: AgentSessionSnapshot, payload: Record<string, unknown>): void {
  if (typeof payload.session_id === 'string') snap.sessionId = payload.session_id
  applyCwd(snap, payload.cwd)
  const git = payload.git as Record<string, unknown> | undefined
  if (git && typeof git.branch === 'string') snap.gitBranch = git.branch
  if (typeof payload.model_provider === 'string' && !snap.model) snap.model = payload.model_provider
}

function applyLine(snap: AgentSessionSnapshot, line: string): void {
  let event: Record<string, unknown>
  try {
    event = JSON.parse(line)
  } catch {
    return
  }
  if (!event || typeof event !== 'object') return

  if (typeof event.timestamp === 'string') {
    const ts = Date.parse(event.timestamp)
    if (!Number.isNaN(ts)) snap.lastEventAt = ts
  }

  const payload = event.payload as Record<string, unknown> | undefined
  if (!payload || typeof payload !== 'object') return

  if (event.type === 'session_meta' || payload.type === 'session_meta') {
    applyMeta(snap, payload)
    return
  }

  // turn_context / world_state events also carry cwd and recur through the
  // session, so a session whose meta line outgrew even the head read still
  // recovers its cwd (and thus its place in the project history) from the tail.
  applyCwd(snap, payload.cwd)

  if (payload.type === 'user_message' && snap.topic === undefined) {
    const text = typeof payload.message === 'string' ? payload.message.trim() : undefined
    if (text && !/^[<[]/.test(text)) {
      snap.topic = text.split('\n')[0].slice(0, 120)
    }
    return
  }

  if (payload.type === 'token_count' || event.type === 'token_count') {
    const info = payload.info as Record<string, unknown> | undefined
    if (info) {
      const total = readUsage(info.total_token_usage as Record<string, unknown>)
      if (total) snap.totals = total // cumulative — replace, don't add
      const last = readUsage(info.last_token_usage as Record<string, unknown>)
      if (last) snap.contextTokens = last.input + last.cacheRead
      const win = Number(info.model_context_window)
      if (win > 0) snap.contextWindow = win
    }
    const rateLimits = payload.rate_limits
    if (rateLimits && typeof rateLimits === 'object') {
      snap.rateLimits = rateLimits as Record<string, unknown>
    }
  }
}

function finalize(snap: AgentSessionSnapshot, size: number, mtime: number): AgentSessionSnapshot {
  snap.fileSize = size
  snap.fileMtime = mtime
  return { ...snap, totals: { ...snap.totals }, recentFiles: [...snap.recentFiles] }
}

export async function ingestCodexFile(fileInfo: {
  path: string
  size: number
  modified: number
}): Promise<AgentSessionSnapshot | null> {
  const api = window.bashgym?.sessions
  if (!api) return null

  let state = fileStates.get(fileInfo.path)
  if (!state) {
    state = {
      tail: { offset: Math.max(0, fileInfo.size - 262_144), carry: '', bootstrapped: false },
      snapshot: newSnapshot(fileInfo.path),
      metaLoaded: false
    }
    fileStates.set(fileInfo.path, state)
  }

  if (!state.metaLoaded) {
    const head = await api.readHead(fileInfo.path, META_HEAD_BYTES)
    if (head.success && head.data) {
      const firstLine = head.data.split('\n')[0]
      if (firstLine) applyLine(state.snapshot, firstLine)
    }
    // Only latch once the meta line actually parsed; otherwise retry next poll
    // rather than permanently stranding the session without a cwd.
    if (state.snapshot.cwd !== undefined) state.metaLoaded = true
  }

  for (let guard = 0; guard < 100; guard++) {
    const result = await api.readTail(fileInfo.path, state.tail.offset)
    if (!result.success) return finalize(state.snapshot, fileInfo.size, fileInfo.modified)

    if (result.reset) state.tail.carry = ''

    if (result.data) {
      const lines = (state.tail.carry + result.data).split('\n')
      state.tail.carry = lines.pop() ?? ''
      for (const line of lines) if (line) applyLine(state.snapshot, line)
    }

    const newOffset = result.newOffset ?? state.tail.offset
    const caughtUp = newOffset >= (result.size ?? fileInfo.size) || newOffset === state.tail.offset
    state.tail.offset = newOffset
    if (caughtUp) break
  }

  state.tail.bootstrapped = true
  return finalize(state.snapshot, fileInfo.size, fileInfo.modified)
}
