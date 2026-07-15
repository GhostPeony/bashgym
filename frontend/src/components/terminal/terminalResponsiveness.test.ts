import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import {
  buildPowerShellArgs,
  TerminalIpcBatcher,
  TerminalSizeTracker,
  TerminalScrollbackBuffer,
  type TerminalTimerApi,
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
    '.terminal-status-waiting-input',
  ]) {
    assert.doesNotMatch(cssRuleBody(css, selector), /\banimation\s*:/)
  }
})

test('terminal input uses fire-and-forget IPC while retaining reload compatibility', () => {
  const preload = readFrontendFile('electron/preload.ts')
  const main = readFrontendFile('electron/main.ts')

  assert.match(preload, /write:[\s\S]*?ipcRenderer\.send\('terminal:write'/)
  assert.match(preload, /resize:.*ipcRenderer\.invoke\('terminal:resize'/)
  assert.match(main, /ipcMain\.on\('terminal:write'/)
  assert.match(main, /ipcMain\.handle\('terminal:write'/)
  assert.match(main, /ipcMain\.handle\('terminal:resize'/)
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
    'Remove-Module PSReadLine -ErrorAction SilentlyContinue',
  ])
  assert.deepEqual(buildPowerShellArgs(true), ['-NoLogo'])
})

test('a fresh PTY receives its visible size before a coding agent launches', () => {
  const pane = readFrontendFile('src/components/terminal/TerminalPane.tsx')
  const createSuccess = pane.indexOf("if (!result.success)")
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
    },
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
