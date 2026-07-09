/**
 * Agent session intel — shared types.
 *
 * Snapshots are built by tailing the local session journals that agent CLIs
 * write (Claude Code: ~/.claude/projects, Codex: ~/.codex/sessions). All data
 * stays on this machine; these files are read via the sessions:* IPC only.
 */

export type AgentSessionKind = 'claude' | 'codex'

export interface TokenTotals {
  input: number
  output: number
  cacheRead: number
  cacheCreate: number
}

export const emptyTotals = (): TokenTotals => ({ input: 0, output: 0, cacheRead: 0, cacheCreate: 0 })

export interface AgentSessionSnapshot {
  kind: AgentSessionKind
  filePath: string
  sessionId?: string
  /** Claude: session slug; Codex: cwd basename */
  title?: string
  /** What the session is about: conversation summary, else the first real user prompt */
  topic?: string
  cwd?: string
  gitBranch?: string
  /** Most recent model seen in the journal */
  model?: string
  /** Approximate tokens currently occupying the context window */
  contextTokens?: number
  /** Codex reports this explicitly; Claude uses a per-model table */
  contextWindow?: number
  contextWindowApprox: boolean
  totals: TokenTotals
  /** Per-model accumulation so cost survives mid-session model switches */
  perModel: Record<string, TokenTotals>
  /** Estimated API-equivalent cost (Claude only) */
  estCostUsd?: number
  /** Codex rate-limit payload, passed through best-effort */
  rateLimits?: Record<string, unknown>
  /** Recently touched file paths from tool calls (newest last, max 8) */
  recentFiles: string[]
  /** True when totals were bootstrapped from a partial read of a huge file */
  totalsApprox: boolean
  lastEventAt?: number
  fileSize: number
  fileMtime: number
}

export type MatchConfidence = 'manual' | 'exact' | 'probable' | 'none'

export interface SessionMatch {
  filePath: string
  confidence: MatchConfidence
}

/** Incremental tail-read bookkeeping per journal file (module-level, not store state) */
export interface TailState {
  offset: number
  carry: string
  bootstrapped: boolean
}
