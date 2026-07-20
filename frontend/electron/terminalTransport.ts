export interface TerminalTimerApi {
  set(callback: () => void, delayMs: number): unknown
  clear(handle: unknown): void
}

const defaultTimerApi: TerminalTimerApi = {
  set: (callback, delayMs) => setTimeout(callback, delayMs),
  clear: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>)
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
    'Remove-Module PSReadLine -ErrorAction SilentlyContinue'
  ]
}

/** Tracks the last ConPTY dimensions so duplicate layout notifications do not redraw a TUI. */
export class TerminalSizeTracker {
  constructor(
    private cols: number,
    private rows: number
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
    private readonly timerApi: TerminalTimerApi = defaultTimerApi
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

export interface TerminalOutputFlowOptions {
  highWatermarkChars?: number
  lowWatermarkChars?: number
}

/**
 * Keeps Electron and xterm from accumulating an unbounded redraw queue.
 * Only one renderer frame is in flight. Later PTY bytes stay ordered in a
 * bounded main-process buffer while xterm parses the current frame.
 */
export class TerminalOutputFlowController {
  private enabled = false
  private awaitingAcknowledgement = false
  private currentFrameId: number | undefined
  private nextFrameId = 1
  private pending = ''
  private paused = false
  private readonly highWatermarkChars: number
  private readonly lowWatermarkChars: number

  constructor(
    private readonly onDeliver: (data: string, frameId?: number) => void,
    private readonly onPauseChange: (paused: boolean) => void,
    options: TerminalOutputFlowOptions = {}
  ) {
    this.highWatermarkChars = Math.max(1, options.highWatermarkChars ?? 64 * 1024)
    this.lowWatermarkChars = Math.min(
      this.highWatermarkChars,
      Math.max(0, options.lowWatermarkChars ?? 16 * 1024)
    )
  }

  enable(): void {
    if (this.enabled) return
    this.enabled = true
    this.awaitingAcknowledgement = false
    this.currentFrameId = undefined
    this.pending = ''
    this.setPaused(false)
  }

  disable(): void {
    if (!this.enabled && !this.paused) return
    this.enabled = false
    this.awaitingAcknowledgement = false
    this.currentFrameId = undefined
    // Output is still retained by TerminalScrollbackBuffer. A remounted pane
    // replays it, so dropping renderer-only pending bytes here cannot lose PTY data.
    this.pending = ''
    this.setPaused(false)
  }

  push(data: string): void {
    if (!data) return
    if (!this.enabled) {
      this.onDeliver(data)
      return
    }

    if (!this.awaitingAcknowledgement) {
      this.awaitingAcknowledgement = true
      this.deliverNext(data)
      return
    }

    this.pending += data
    this.updateBackpressure()
  }

  acknowledge(frameId: number | undefined): void {
    if (!this.enabled || !this.awaitingAcknowledgement || frameId !== this.currentFrameId) return
    if (this.pending) {
      const next = this.pending.slice(0, this.highWatermarkChars)
      this.pending = this.pending.slice(next.length)
      this.deliverControlled(next)
      this.updateBackpressure()
      return
    }

    this.awaitingAcknowledgement = false
    this.currentFrameId = undefined
    this.setPaused(false)
  }

  dispose(): void {
    this.disable()
  }

  get inFlight(): boolean {
    return this.awaitingAcknowledgement
  }

  get pendingChars(): number {
    return this.pending.length
  }

  get frameId(): number | undefined {
    return this.currentFrameId
  }

  private deliverNext(data: string): void {
    const next = data.slice(0, this.highWatermarkChars)
    this.pending = data.slice(next.length)
    this.deliverControlled(next)
    this.updateBackpressure()
  }

  private deliverControlled(data: string): void {
    const frameId = this.nextFrameId
    this.nextFrameId += 1
    this.currentFrameId = frameId
    this.onDeliver(data, frameId)
  }

  private updateBackpressure(): void {
    if (this.pending.length >= this.highWatermarkChars) {
      this.setPaused(true)
    } else if (this.pending.length <= this.lowWatermarkChars) {
      this.setPaused(false)
    }
  }

  private setPaused(paused: boolean): void {
    if (this.paused === paused) return
    this.paused = paused
    this.onPauseChange(paused)
  }
}

/**
 * Arbitrates which mounted xterm may control and acknowledge a PTY's output.
 * A stale pane can neither release nor advance a replacement pane's frames.
 */
export class TerminalOutputFlowOwnership {
  private owner: string | undefined

  constructor(private readonly flow: TerminalOutputFlowController) {}

  acquire(owner: string): void {
    if (this.owner === owner) return
    this.flow.disable()
    this.owner = owner
    this.flow.enable()
  }

  release(owner: string): void {
    if (this.owner !== owner) return
    this.flow.disable()
    this.owner = undefined
  }

  acknowledge(owner: string, frameId: number): void {
    if (this.owner === owner) this.flow.acknowledge(frameId)
  }
}
