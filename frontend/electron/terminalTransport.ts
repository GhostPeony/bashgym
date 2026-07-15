export interface TerminalTimerApi {
  set(callback: () => void, delayMs: number): unknown
  clear(handle: unknown): void
}

const defaultTimerApi: TerminalTimerApi = {
  set: (callback, delayMs) => setTimeout(callback, delayMs),
  clear: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
}

/**
 * PSReadLine repaints and syntax-highlights the full command line on every key.
 * Inside ConPTY that can turn one character into dozens of ANSI bytes and add
 * hundreds of milliseconds of echo latency. Keep the advanced editor opt-in.
 */
export function buildPowerShellArgs(enablePsReadLine: boolean): string[] {
  if (enablePsReadLine) return ['-NoLogo']
  return [
    '-NoLogo',
    '-NoProfile',
    '-NoExit',
    '-Command',
    'Remove-Module PSReadLine -ErrorAction SilentlyContinue',
  ]
}

/** Tracks the last ConPTY dimensions so duplicate layout notifications do not redraw a TUI. */
export class TerminalSizeTracker {
  constructor(
    private cols: number,
    private rows: number,
  ) {}

  update(cols: number, rows: number): boolean {
    if (!Number.isInteger(cols) || !Number.isInteger(rows) || cols < 1 || rows < 1) return false
    if (this.cols === cols && this.rows === rows) return false
    this.cols = cols
    this.rows = rows
    return true
  }
}

/**
 * Bounded chunk buffer with an advancing head index. Removing old terminal
 * redraw chunks stays amortized O(1); Array.shift() would copy the remaining
 * array on every write once a long-running session reached its cap.
 */
export class TerminalScrollbackBuffer {
  private chunks: string[] = []
  private head = 0
  private retainedLength = 0

  constructor(private readonly maxLength: number) {}

  append(data: string): void {
    if (!data) return
    this.chunks.push(data)
    this.retainedLength += data.length

    while (this.retainedLength > this.maxLength && this.chunks.length - this.head > 1) {
      this.retainedLength -= this.chunks[this.head].length
      this.head += 1
    }

    if (this.head >= 1_024 && this.head * 2 >= this.chunks.length) {
      this.chunks = this.chunks.slice(this.head)
      this.head = 0
    }
  }

  toString(): string {
    return this.chunks.slice(this.head).join('')
  }

  tail(maxLength: number): string {
    return this.toString().slice(-maxLength)
  }

  get length(): number {
    return this.retainedLength
  }
}

/** Coalesces noisy PTY redraw chunks into a single ordered renderer message. */
export class TerminalIpcBatcher {
  private pending = ''
  private timer: unknown

  constructor(
    private readonly onFlush: (data: string) => void,
    private readonly delayMs = 8,
    private readonly timerApi: TerminalTimerApi = defaultTimerApi,
  ) {}

  push(data: string): void {
    if (!data) return
    // PowerShell's low-latency path echoes one character at a time. Forward it
    // synchronously so a busy timer queue cannot turn an 8ms frame into a
    // visible typing delay. Multi-byte ANSI/TUI redraws still batch below.
    if (data.length === 1 && !this.pending && this.timer === undefined) {
      this.onFlush(data)
      return
    }
    this.pending += data
    if (this.timer !== undefined) return
    this.timer = this.timerApi.set(() => {
      this.timer = undefined
      this.flush()
    }, this.delayMs)
  }

  flush(): void {
    if (!this.pending) return
    const data = this.pending
    this.pending = ''
    this.onFlush(data)
  }

  dispose(flush = false): void {
    if (this.timer !== undefined) {
      this.timerApi.clear(this.timer)
      this.timer = undefined
    }
    if (flush) this.flush()
    else this.pending = ''
  }
}
