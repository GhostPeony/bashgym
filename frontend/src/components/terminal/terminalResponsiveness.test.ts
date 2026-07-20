import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import {
  buildPowerShellArgs,
  TerminalIpcBatcher,
  TerminalOutputFlowController,
  TerminalOutputFlowOwnership,
  TerminalSizeTracker,
  TerminalScrollbackBuffer,
  type TerminalTimerApi
} from '../../../electron/terminalTransport'

const frontendRoot = fileURLToPath(new URL('../../../', import.meta.url))
const readFrontendFile = (path: string) => readFileSync(`${frontendRoot}${path}`, 'utf8')

function cssRuleBody(css: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = css.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`))
  assert.ok(match, `Expected CSS rule for ${selector}`)
  return match[1]
}

test('terminal status surfaces do not run full-area paint animations', () => {
  const css = readFrontendFile('src/styles/globals.css')

  for (const selector of [
    '.terminal-attention-waiting',
    '.terminal-status-running',
    '.terminal-status-tool-calling',
    '.terminal-status-waiting-input'
  ]) {
    assert.doesNotMatch(cssRuleBody(css, selector), /\banimation\s*:/)
  }
})

test('terminal input uses fire-and-forget IPC while retaining reload compatibility', () => {
  const preload = readFrontendFile('electron/preload.ts')
  const main = readFrontendFile('electron/main.ts')

  assert.match(preload, /write:[\s\S]*?ipcRenderer\.send\('terminal:write'/)
  assert.match(preload, /resize:[\s\S]*?ipcRenderer\.invoke\('terminal:resize'/)
  assert.match(main, /ipcMain\.on\('terminal:write'/)
  assert.match(main, /ipcMain\.handle\('terminal:write'/)
  assert.match(main, /ipcMain\.handle\('terminal:resize'/)
  assert.match(main, /ipcMain\.on\('terminal:output-ack'/)
  assert.match(preload, /ackOutput:[\s\S]*?ipcRenderer\.send\('terminal:output-ack'/)
  assert.match(
    readFrontendFile('src/components/terminal/TerminalPane.tsx'),
    /terminal\.write\(\s*data,\s*frameId === undefined\s*\? undefined\s*:\s*\(\) =>\s*window\.bashgym\?\.terminal\.ackOutput\?\.\(id, outputFlowOwner, frameId\)/
  )
})

test('terminal remount lifecycle cannot activate flow control after disposal', () => {
  const pane = readFrontendFile('src/components/terminal/TerminalPane.tsx')

  assert.match(
    pane,
    /let disposed = false[\s\S]*?\.then\(async \(result\) => \{\s*if \(disposed\) return/
  )
  assert.match(
    pane,
    /if \(!removeDataListener\)[\s\S]*?setOutputFlowControl\?\.\(id, outputFlowOwner, true\)/
  )
  assert.match(
    pane,
    /return \(\) => \{\s*disposed = true\s*window\.bashgym\?\.terminal\.setOutputFlowControl\?\.\(id, outputFlowOwner, false\)/
  )
})

test('hidden terminal layouts never trigger PTY fitting or resize churn', () => {
  const pane = readFrontendFile('src/components/terminal/TerminalPane.tsx')

  assert.match(pane, /getComputedStyle\(surface\)/)
  assert.match(pane, /style\.visibility !== 'hidden'/)
  assert.match(pane, /requestAnimationFrame/)
  assert.doesNotMatch(pane, /\[10, 50, 150\]/)
})

test('reopening a canvas terminal reactivates xterm focus without popup gradient chrome', () => {
  const grid = readFrontendFile('src/components/terminal/TerminalGrid.tsx')
  const css = readFrontendFile('src/styles/globals.css')

  assert.match(grid, /const isPresented = viewMode === 'canvas' \? isInPopup : isActive/)
  assert.match(grid, /renderPanel\(panel, isPresented,/)
  assert.match(grid, /panel\.type === 'terminal'[\s\S]*?'canvas-popup-panel-terminal'/)
  assert.match(cssRuleBody(css, '.canvas-popup-panel-terminal::before'), /display:\s*none/)
})

test('development launches do not attach DevTools unless explicitly requested', () => {
  const main = readFrontendFile('electron/main.ts')

  assert.match(main, /process\.env\.BASHGYM_OPEN_DEVTOOLS === '1'/)
  assert.doesNotMatch(main, /mainWindow\.webContents\.openDevTools\(\)\s*\/\/ TEMP/)
})

test('embedded PowerShell defaults to the low-latency line editor path', () => {
  assert.deepEqual(buildPowerShellArgs(false), [
    '-NoLogo',
    '-NoProfile',
    '-NoExit',
    '-Command',
    'Remove-Module PSReadLine -ErrorAction SilentlyContinue'
  ])
  assert.deepEqual(buildPowerShellArgs(true), ['-NoLogo'])
})

test('a fresh PTY receives its visible size before a coding agent launches', () => {
  const pane = readFrontendFile('src/components/terminal/TerminalPane.tsx')
  const createSuccess = pane.indexOf('if (!result.success)')
  const sizeSync = pane.indexOf('await syncPtySize()', createSuccess)
  const pendingLaunch = pane.indexOf('const pending =', createSuccess)

  assert.ok(createSuccess >= 0)
  assert.ok(sizeSync > createSuccess)
  assert.ok(pendingLaunch > sizeSync)
  assert.match(pane, /fitAddon\.fit\(\)[\s\S]*?syncTerminalSize\(true\)/)
})

test('long-running scrollback stays bounded without front-shifting chunks', () => {
  const scrollback = new TerminalScrollbackBuffer(1_024)
  for (let index = 0; index < 50_000; index += 1) {
    scrollback.append(String(index % 10))
  }

  assert.equal(scrollback.length, 1_024)
  assert.equal(scrollback.toString().length, 1_024)
  assert.equal(scrollback.tail(24), '678901234567890123456789')
  assert.doesNotMatch(readFrontendFile('electron/terminalTransport.ts'), /chunks\.shift\(/)
})

test('identical PTY dimensions are applied only once', () => {
  const sizes = new TerminalSizeTracker(80, 24)

  assert.equal(sizes.update(80, 24), false)
  assert.equal(sizes.update(140, 48), true)
  assert.equal(sizes.update(140, 48), false)
  assert.equal(sizes.update(141, 48), true)
})

test('PTY redraws are sent as one ordered short-frame batch', () => {
  let scheduled: (() => void) | undefined
  const timerApi: TerminalTimerApi = {
    set: (callback) => {
      scheduled = callback
      return 1
    },
    clear: () => {
      scheduled = undefined
    }
  }
  const flushed: string[] = []
  const batcher = new TerminalIpcBatcher((data) => flushed.push(data), 8, timerApi)

  batcher.push('\x1b[?25l')
  batcher.push('\x1b[31m')
  batcher.push('bulk')
  assert.deepEqual(flushed, [])

  scheduled?.()
  assert.deepEqual(flushed, ['\x1b[?25l\x1b[31mbulk'])
})

test('single-character PTY echoes bypass redraw batching', () => {
  const flushed: string[] = []
  const batcher = new TerminalIpcBatcher((data) => flushed.push(data))

  batcher.push('x')

  assert.deepEqual(flushed, ['x'])
})

test('Claude redraws wait for xterm parser acknowledgement and apply bounded PTY backpressure', () => {
  const delivered: string[] = []
  const pauseStates: boolean[] = []
  const flow = new TerminalOutputFlowController(
    (data) => delivered.push(data),
    (paused) => pauseStates.push(paused),
    { highWatermarkChars: 12, lowWatermarkChars: 4 }
  )

  flow.enable()
  flow.push('frame-1')
  flow.push('abcdefgh')
  flow.push('ijklmnop')

  assert.deepEqual(delivered, ['frame-1'])
  assert.deepEqual(pauseStates, [true])

  flow.acknowledge(flow.frameId)
  assert.deepEqual(delivered, ['frame-1', 'abcdefghijkl'])
  assert.deepEqual(pauseStates, [true, false])

  flow.push('qrst')
  assert.deepEqual(delivered, ['frame-1', 'abcdefghijkl'])

  flow.acknowledge(flow.frameId)
  assert.deepEqual(delivered, ['frame-1', 'abcdefghijkl', 'mnopqrst'])
  flow.acknowledge(flow.frameId)

  assert.deepEqual(delivered.join(''), 'frame-1abcdefghijklmnopqrst')
  assert.equal(flow.inFlight, false)
  assert.equal(flow.pendingChars, 0)
})

test('large redraws are chunked and disabling a paused flow releases the PTY', () => {
  const delivered: string[] = []
  const pauseStates: boolean[] = []
  const flow = new TerminalOutputFlowController(
    (data) => delivered.push(data),
    (paused) => pauseStates.push(paused),
    { highWatermarkChars: 12, lowWatermarkChars: 4 }
  )

  flow.enable()
  flow.acknowledge(flow.frameId)
  flow.push('abcdefghijklmnopqrstuvwxyz')

  assert.deepEqual(delivered, ['abcdefghijkl'])
  assert.equal(flow.pendingChars, 14)
  assert.deepEqual(pauseStates, [true])

  flow.disable()
  assert.equal(flow.inFlight, false)
  assert.equal(flow.pendingChars, 0)
  assert.deepEqual(pauseStates, [true, false])

  flow.push('passthrough')
  assert.deepEqual(delivered, ['abcdefghijkl', 'passthrough'])
})

test('stale terminal owners cannot acknowledge or release replacement panes', () => {
  const delivered: string[] = []
  const pauseStates: boolean[] = []
  const flow = new TerminalOutputFlowController(
    (data) => delivered.push(data),
    (paused) => pauseStates.push(paused),
    { highWatermarkChars: 4, lowWatermarkChars: 0 }
  )
  const ownership = new TerminalOutputFlowOwnership(flow)

  ownership.acquire('pane-a')
  flow.push('aaaa')
  flow.push('bbbb')
  ownership.acquire('pane-b')
  flow.push('cccc')
  flow.push('dddd')

  const paneBFrame = flow.frameId
  ownership.acknowledge('pane-a', paneBFrame ?? -1)
  ownership.release('pane-a')
  assert.deepEqual(delivered, ['aaaa', 'cccc'])
  assert.equal(flow.inFlight, true)

  ownership.acknowledge('pane-b', paneBFrame ?? -1)
  ownership.release('pane-b')
  assert.deepEqual(delivered, ['aaaa', 'cccc', 'dddd'])
  assert.equal(flow.inFlight, false)
  assert.deepEqual(pauseStates, [true, false, true, false])
})

test('a passthrough callback cannot acknowledge a different controlled frame', () => {
  const delivered: Array<{ data: string; frameId?: number }> = []
  const flow = new TerminalOutputFlowController(
    (data, frameId) => delivered.push({ data, frameId }),
    () => {},
    { highWatermarkChars: 4, lowWatermarkChars: 0 }
  )

  flow.push('old')
  flow.enable()
  flow.push('aaaa')
  flow.push('bbbb')

  flow.acknowledge(undefined)
  flow.acknowledge((delivered[1].frameId ?? 0) + 1)
  assert.deepEqual(
    delivered.map(({ data }) => data),
    ['old', 'aaaa']
  )

  flow.acknowledge(delivered[1].frameId)
  assert.deepEqual(
    delivered.map(({ data }) => data),
    ['old', 'aaaa', 'bbbb']
  )
})
