export type TerminalAgentKind = 'claude' | 'codex'
export type TerminalObservedStatus = 'running' | 'tool_calling' | 'waiting_input' | 'idle'

export interface TerminalActivityObservation {
  status?: TerminalObservedStatus
  currentTool?: string
  summary?: string
  target?: string
  shellCwd?: string
}

export interface TerminalOutputTimerApi {
  set: (callback: () => void, delayMs: number) => unknown
  clear: (handle: unknown) => void
}

export interface TerminalOutputBatcherOptions {
  quietMs?: number
  maxWaitMs?: number
  maxBufferChars?: number
  timerApi?: TerminalOutputTimerApi
}

const CLAUDE_BANNER_LINE = /(?:^|\n)\s*(?:[|│]\s*)?(?:(?:✻\s*)?Welcome(?:\s+to)?\s+Claude\s+Code!?|Claude\s+Code\s+v?\d[\w.-]*)\s*(?:[|│])?\s*$/im
const CODEX_BANNER_LINE = /(?:^|\n)\s*(?:[|│]\s*)?(?:>[_ ]\s*)?OpenAI\s+Codex(?:\s+\(v[\w.-]+\)|\s+v?\d[\w.-]*)?\s*(?:[|│])?\s*$/im
const ACTIVITY_MARK_RE = /^[\s│┃┆┊]*(?:[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏●•⏺✻✢✽✶])\s*/
const COMPLETION_MARK_RE = /^[\s│┃┆┊]*[✓✗✔✘]/
const INPUT_PROMPT_RE = /^\s*[>›❯]\s*$/
const TOOL_RE = /^(Apply\s+Patch|NotebookEdit|WebSearch|WebFetch|Read|Edit|Write|Bash|Glob|Grep|Task|Shell|Command|Run|Ran|Search|Searching|Explore|Explored|Patch|Edited?)\b/i
const ACTIVE_RE = /^(Thinking|Planning|Working|Reasoning|Analyzing|Searching|Reading|Writing|Editing|Running)\b/i

const DEFAULT_TIMER_API: TerminalOutputTimerApi = {
  set: (callback, delayMs) => globalThis.setTimeout(callback, delayMs),
  clear: (handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
}

/**
 * Combines split PTY chunks without allowing a continuous spinner stream to
 * postpone observation forever. Quiet output flushes quickly; sustained output
 * flushes at least once per maxWaitMs interval.
 */
export class TerminalOutputBatcher {
  private buffer = ''
  private quietTimer: unknown
  private maxWaitTimer: unknown
  private readonly quietMs: number
  private readonly maxWaitMs: number
  private readonly maxBufferChars: number
  private readonly timerApi: TerminalOutputTimerApi

  constructor(
    private readonly onFlush: (output: string) => void,
    options: TerminalOutputBatcherOptions = {},
  ) {
    this.quietMs = options.quietMs ?? 80
    this.maxWaitMs = options.maxWaitMs ?? 250
    this.maxBufferChars = options.maxBufferChars ?? 32_768
    this.timerApi = options.timerApi ?? DEFAULT_TIMER_API
  }

  push(data: string): void {
    this.buffer += data
    if (this.buffer.length > this.maxBufferChars) {
      this.buffer = this.buffer.slice(-Math.floor(this.maxBufferChars / 4))
    }

    if (this.quietTimer !== undefined) this.timerApi.clear(this.quietTimer)
    this.quietTimer = this.timerApi.set(() => this.flush(), this.quietMs)
    if (this.maxWaitTimer === undefined) {
      this.maxWaitTimer = this.timerApi.set(() => this.flush(), this.maxWaitMs)
    }
  }

  flush(): void {
    const output = this.buffer
    this.buffer = ''
    this.clearTimers()
    if (output) this.onFlush(output)
  }

  dispose(): void {
    this.buffer = ''
    this.clearTimers()
  }

  private clearTimers(): void {
    if (this.quietTimer !== undefined) this.timerApi.clear(this.quietTimer)
    if (this.maxWaitTimer !== undefined) this.timerApi.clear(this.maxWaitTimer)
    this.quietTimer = undefined
    this.maxWaitTimer = undefined
  }
}

function normalizeActivityLine(line: string): string {
  return line
    .replace(ACTIVITY_MARK_RE, '')
    .replace(COMPLETION_MARK_RE, '')
    .replace(/\s*\(\s*\d+(?:\.\d+)?s(?:\s*[·•|-]\s*(?:esc|ctrl\+c)[^)]+)?\)\s*$/i, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 160)
}

function normalizeTool(raw: string): string {
  const key = raw.toLowerCase().replace(/\s+/g, ' ')
  if (['shell', 'command', 'run', 'ran'].includes(key)) return 'Bash'
  if (['search', 'searching', 'explore', 'explored'].includes(key)) return 'Search'
  if (['patch', 'apply patch', 'edit', 'edited'].includes(key)) return 'Edit'
  return raw.replace(/\s+/g, '')
}

function meaningfulLines(output: string): string[] {
  return output
    .split(/\r\n|\n|\r/)
    // eslint-disable-next-line no-control-regex
    .map((line) => line.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '').trim())
    .filter((line) => line.length > 0 && !/^[╭╰╮╯─━┄┈│┃┆┊\s]+$/.test(line))
}

/** Classifies the latest observable terminal state, favoring the newest line. */
export function detectTerminalActivity(output: string): TerminalActivityObservation {
  const lines = meaningfulLines(output)
  let fallbackSummary: string | undefined
  let sawActivityMark = false

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const rawLine = lines[index]
    const summary = normalizeActivityLine(rawLine)
    if (!summary) continue
    fallbackSummary ??= summary

    const psPrompt = rawLine.match(/^PS\s+([A-Z]:\\[^>]*?)>\s*$/i)
    if (psPrompt) return { status: 'idle', shellCwd: psPrompt[1].trim() }
    if (/^(?:[\w.-]+@[\w.-]+(?::[^$#%]+)?|[~/.\\A-Za-z0-9:_-]+)?[$#%]\s*$/.test(rawLine)) {
      return { status: 'idle' }
    }
    if (INPUT_PROMPT_RE.test(rawLine)) return { status: 'waiting_input' }

    if (COMPLETION_MARK_RE.test(rawLine)) {
      return { status: 'running', summary }
    }

    const hadActivityMark = ACTIVITY_MARK_RE.test(rawLine)
    sawActivityMark ||= hadActivityMark
    const toolMatch = summary.match(TOOL_RE)
    if (toolMatch && hadActivityMark) {
      const target = summary.slice(toolMatch[0].length).replace(/^[\s(:-]+|\)\s*$/g, '').trim()
      return {
        status: 'tool_calling',
        currentTool: normalizeTool(toolMatch[1]),
        summary,
        ...(target ? { target: target.slice(0, 120) } : {}),
      }
    }

    if (ACTIVE_RE.test(summary) && (hadActivityMark || lines.length === 1)) {
      return { status: 'running', summary }
    }
  }

  if (sawActivityMark) return { status: 'running', summary: fallbackSummary }
  return { summary: fallbackSummary }
}

/** Lines worth retaining for context handoff, excluding redraw-only UI chrome. */
export function terminalOutputLines(output: string): string[] {
  return meaningfulLines(output)
    .filter((line) => !ACTIVITY_MARK_RE.test(line) && !INPUT_PROMPT_RE.test(line))
    .map(normalizeActivityLine)
    .filter(Boolean)
}

/**
 * Returns a provider only when the terminal receives a dedicated CLI banner.
 * Shell output can mention provider names without an agent actually running.
 */
export function detectTerminalAgentKind(output: string): TerminalAgentKind | undefined {
  if (CLAUDE_BANNER_LINE.test(output)) return 'claude'
  if (CODEX_BANNER_LINE.test(output)) return 'codex'
  return undefined
}

export function terminalRuntimeLabel(agentKind?: TerminalAgentKind): string {
  if (agentKind === 'claude') return 'Claude Code'
  if (agentKind === 'codex') return 'Codex'
  return 'Bash shell'
}
