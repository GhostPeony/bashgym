/**
 * Incremental parser for Claude Code session journals
 * (~/.claude/projects/<encoded-cwd>/<session>.jsonl).
 *
 * Field knowledge mirrors bashgym/trace_capture/importers/claude_history.py:
 * first-occurrence session fields, per-model usage accumulation, and the four
 * usage token classes. Parsing is defensive — unknown events are skipped.
 */

import type { AgentSessionSnapshot, TailState } from './types'
import { emptyTotals } from './types'
import { estimateCostUsd, contextWindowFor } from './pricing'

const RECENT_FILES_MAX = 8
const BOOTSTRAP_FULL_SCAN_MAX = 20 * 1024 * 1024
const BOOTSTRAP_TAIL_BYTES = 2 * 1024 * 1024
const FILE_TOOLS = new Set(['Read', 'Edit', 'Write', 'NotebookEdit'])

interface FileState {
  tail: TailState
  snapshot: AgentSessionSnapshot
}

const fileStates = new Map<string, FileState>()

export function resetClaudeFile(filePath: string): void {
  fileStates.delete(filePath)
}

/** Encode an absolute cwd the way Claude Code names its project directories */
export function encodeClaudeProjectDir(cwd: string): string {
  return cwd.replace(/[^a-zA-Z0-9]/g, '-')
}

function newSnapshot(filePath: string): AgentSessionSnapshot {
  return {
    kind: 'claude',
    filePath,
    contextWindowApprox: true,
    totals: emptyTotals(),
    perModel: {},
    recentFiles: [],
    totalsApprox: false,
    fileSize: 0,
    fileMtime: 0
  }
}

function applyLine(snap: AgentSessionSnapshot, line: string): void {
  let event: Record<string, unknown>
  try {
    event = JSON.parse(line)
  } catch {
    return
  }
  if (!event || typeof event !== 'object') return
  if (event.isSidechain === true) return

  // First-occurrence session identity fields
  if (snap.cwd === undefined && typeof event.cwd === 'string') snap.cwd = event.cwd
  if (snap.gitBranch === undefined && typeof event.gitBranch === 'string' && event.gitBranch) {
    snap.gitBranch = event.gitBranch
  }
  if (snap.title === undefined && typeof event.slug === 'string' && event.slug)
    snap.title = event.slug
  if (snap.sessionId === undefined && typeof event.sessionId === 'string')
    snap.sessionId = event.sessionId

  if (typeof event.timestamp === 'string') {
    const ts = Date.parse(event.timestamp)
    if (!Number.isNaN(ts)) snap.lastEventAt = ts
  }

  // Conversation summaries are the best topic when present (last one wins)
  if (event.type === 'summary' && typeof event.summary === 'string' && event.summary) {
    snap.topic = event.summary
    return
  }

  // Fallback topic: the first real user prompt (skip command/caveat/tool noise)
  if (event.type === 'user' && snap.topic === undefined) {
    const message = event.message as Record<string, unknown> | undefined
    const content = message?.content
    let text: string | undefined
    if (typeof content === 'string') {
      text = content
    } else if (Array.isArray(content)) {
      const block = content.find((b) => b?.type === 'text' && typeof b.text === 'string')
      text = block?.text
    }
    const clean = text?.trim()
    if (clean && !/^[<[]/.test(clean)) {
      snap.topic = clean.split('\n')[0].slice(0, 120)
    }
    return
  }

  if (event.type !== 'assistant') return
  const message = event.message as Record<string, unknown> | undefined
  if (!message) return

  const model = typeof message.model === 'string' ? message.model : undefined
  if (model) snap.model = model

  const usage = message.usage as Record<string, unknown> | undefined
  if (usage) {
    const input = Number(usage.input_tokens) || 0
    const output = Number(usage.output_tokens) || 0
    const cacheCreate = Number(usage.cache_creation_input_tokens) || 0
    const cacheRead = Number(usage.cache_read_input_tokens) || 0

    snap.totals.input += input
    snap.totals.output += output
    snap.totals.cacheCreate += cacheCreate
    snap.totals.cacheRead += cacheRead

    if (model) {
      const pm = snap.perModel[model] ?? (snap.perModel[model] = emptyTotals())
      pm.input += input
      pm.output += output
      pm.cacheCreate += cacheCreate
      pm.cacheRead += cacheRead
    }

    // Latest assistant turn's prompt-side tokens ≈ current context occupancy
    snap.contextTokens = input + cacheRead + cacheCreate
  }

  const content = message.content
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block?.type !== 'tool_use' || !FILE_TOOLS.has(block?.name)) continue
      const filePath = block?.input?.file_path
      if (typeof filePath !== 'string' || !filePath) continue
      const existing = snap.recentFiles.indexOf(filePath)
      if (existing !== -1) snap.recentFiles.splice(existing, 1)
      snap.recentFiles.push(filePath)
      if (snap.recentFiles.length > RECENT_FILES_MAX) snap.recentFiles.shift()
    }
  }
}

function finalize(snap: AgentSessionSnapshot, size: number, mtime: number): AgentSessionSnapshot {
  snap.fileSize = size
  snap.fileMtime = mtime
  snap.contextWindow = contextWindowFor(snap.contextTokens)
  snap.estCostUsd = Object.entries(snap.perModel).reduce(
    (sum, [model, totals]) => sum + estimateCostUsd(model, totals),
    0
  )
  // Return a fresh reference so store consumers see the update
  return { ...snap, totals: { ...snap.totals }, recentFiles: [...snap.recentFiles] }
}

/**
 * Ingest new journal content for a file.
 *
 * `exact` (journals bound to a live terminal): bootstraps exact totals via a
 * chunked full read for files up to 20MB; beyond that, head for identity +
 * last 2MB tail, totals approximate.
 *
 * Cheap mode (historical journals): head for identity + last 256KB tail —
 * enough for slug/branch/model, latest context occupancy, and last activity,
 * without paying a full read per old session.
 */
export async function ingestClaudeFile(
  fileInfo: { path: string; size: number; modified: number },
  exact = true
): Promise<AgentSessionSnapshot | null> {
  const api = window.bashgym?.sessions
  if (!api) return null

  let state = fileStates.get(fileInfo.path)
  if (!state) {
    state = {
      tail: { offset: 0, carry: '', bootstrapped: false },
      snapshot: newSnapshot(fileInfo.path)
    }
    const cheapTailStart = Math.max(0, fileInfo.size - 262_144)
    const fullScanCap = exact ? BOOTSTRAP_FULL_SCAN_MAX : 262_144
    if (fileInfo.size > fullScanCap) {
      const head = await api.readHead(fileInfo.path, 65_536)
      if (head.success && head.data) {
        for (const line of head.data.split('\n')) if (line) applyLine(state.snapshot, line)
      }
      state.tail.offset = exact ? Math.max(0, fileInfo.size - BOOTSTRAP_TAIL_BYTES) : cheapTailStart
      state.snapshot.totalsApprox = true
    }
    fileStates.set(fileInfo.path, state)
  } else if (
    exact &&
    state.snapshot.totalsApprox &&
    state.snapshot.fileSize <= BOOTSTRAP_FULL_SCAN_MAX
  ) {
    // Upgraded from cheap to exact (journal became live-bound): re-bootstrap
    fileStates.delete(fileInfo.path)
    return ingestClaudeFile(fileInfo, true)
  }

  // Chunked catch-up loop; bounded by file size / read cap
  for (let guard = 0; guard < 400; guard++) {
    const result = await api.readTail(fileInfo.path, state.tail.offset)
    if (!result.success) return finalize(state.snapshot, fileInfo.size, fileInfo.modified)

    if (result.reset) {
      // File truncated/rotated: start over from the restart offset
      state.snapshot = newSnapshot(fileInfo.path)
      state.snapshot.totalsApprox = (result.newOffset ?? 0) > 0
      state.tail.carry = ''
    }

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
