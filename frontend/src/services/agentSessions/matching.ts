/**
 * Terminal ↔ session-file matching.
 *
 * Pure functions: given the live PTY terminals and the parsed journal
 * snapshots, decide which journal belongs to which terminal. Manual pins
 * always win; otherwise match on cwd (Claude via the encoded project dir
 * name, Codex via session_meta.cwd) with recency as the tiebreaker.
 */

import type { AgentSessionSnapshot, MatchConfidence, SessionMatch } from './types'
import { encodeClaudeProjectDir } from './claudeSessionAdapter'

export interface MatchableTerminal {
  terminalId: string
  panelId: string
  cwd: string
  agentKind?: 'claude' | 'codex'
  lastActivity: number
}

/** Normalize a path for comparison: forward slashes, lowercase drive, no trailing slash */
export function normPath(p: string): string {
  let out = p.replace(/\\/g, '/').replace(/\/+$/, '')
  if (/^[A-Za-z]:/.test(out)) out = out[0].toLowerCase() + out.slice(1)
  return out
}

// Lowercased for comparison: Windows paths are case-insensitive and the
// prompt-scraped cwd casing can differ from the journal dir's encoding
function claudeDirOf(filePath: string): string {
  const parts = filePath.replace(/\\/g, '/').split('/')
  return (parts[parts.length - 2] ?? '').toLowerCase()
}

function encodedDirFor(cwd: string): string {
  return encodeClaudeProjectDir(cwd).toLowerCase()
}

function freshness(s: AgentSessionSnapshot): number {
  return s.lastEventAt ?? s.fileMtime
}

export function matchSessions(
  terminals: MatchableTerminal[],
  snapshots: Map<string, AgentSessionSnapshot>,
  pins: Record<string, string>
): Map<string, SessionMatch> {
  const matches = new Map<string, SessionMatch>()
  const claimed = new Set<string>()
  const all = Array.from(snapshots.values())

  const ordered = [...terminals].sort((a, b) => b.lastActivity - a.lastActivity)

  for (const term of ordered) {
    const pinned = pins[term.panelId]
    if (pinned && snapshots.has(pinned)) {
      matches.set(term.terminalId, { filePath: pinned, confidence: 'manual' })
      claimed.add(pinned)
      continue
    }

    const cwd = normPath(term.cwd === '~' ? '' : term.cwd).toLowerCase()
    if (!cwd) continue
    const encodedDir = encodedDirFor(term.cwd)

    const cwdMatches = all.filter((s) => {
      if (claimed.has(s.filePath)) return false
      if (s.kind === 'claude') return claudeDirOf(s.filePath) === encodedDir
      return s.cwd !== undefined && normPath(s.cwd).toLowerCase() === cwd
    })

    // Prefer journals matching the detected CLI; agentKind clears at the shell
    // prompt, so an idle terminal falls back to all cwd matches
    let candidates = term.agentKind
      ? cwdMatches.filter((s) => s.kind === term.agentKind)
      : cwdMatches
    if (candidates.length === 0) candidates = cwdMatches

    let capped: MatchConfidence | null = null
    if (candidates.length === 0) {
      // Prefix fallback: journal cwd is an ancestor of the terminal cwd
      candidates = all.filter(
        (s) =>
          !claimed.has(s.filePath) &&
          s.cwd !== undefined &&
          s.cwd.length > 3 &&
          cwd.startsWith(normPath(s.cwd).toLowerCase())
      )
      capped = 'probable'
    }
    if (candidates.length === 0) continue

    const sorted = [...candidates].sort((a, b) => freshness(b) - freshness(a))
    const best = sorted[0]
    const runnerUp = sorted[1]

    let confidence: MatchConfidence
    if (term.agentKind === best.kind && Math.abs(best.fileMtime - term.lastActivity) < 120_000) {
      confidence = 'exact'
    } else if (!runnerUp || freshness(best) - freshness(runnerUp) > 60_000) {
      confidence = 'probable'
    } else {
      confidence = 'none'
    }
    if (capped && confidence === 'exact') confidence = capped

    if (confidence === 'none') continue
    matches.set(term.terminalId, { filePath: best.filePath, confidence })
    claimed.add(best.filePath)
  }

  return matches
}

/** Candidate journals for the manual pin picker, freshest first */
export function candidatesForTerminal(
  term: MatchableTerminal,
  snapshots: Map<string, AgentSessionSnapshot>
): AgentSessionSnapshot[] {
  const cwd = normPath(term.cwd === '~' ? '' : term.cwd).toLowerCase()
  const encodedDir = encodedDirFor(term.cwd)
  return Array.from(snapshots.values())
    .filter((s) => {
      if (s.kind === 'claude') return claudeDirOf(s.filePath) === encodedDir
      const sCwd = s.cwd !== undefined ? normPath(s.cwd).toLowerCase() : undefined
      return sCwd !== undefined && (sCwd === cwd || (cwd !== '' && cwd.startsWith(sCwd)))
    })
    .sort((a, b) => freshness(b) - freshness(a))
    .slice(0, 8)
}
