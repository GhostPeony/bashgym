import assert from 'node:assert/strict'
import test from 'node:test'
import {
  detectTerminalActivity,
  detectTerminalAgentKind,
  TerminalOutputBatcher,
  type TerminalOutputTimerApi,
  terminalRuntimeLabel
} from './terminalAgentRuntime'

test('detects Claude and Codex from their dedicated startup banner lines', () => {
  assert.equal(detectTerminalAgentKind('│ ✻ Welcome to Claude Code! │'), 'claude')
  assert.equal(detectTerminalAgentKind('│ >_ OpenAI Codex (v0.60.0) │'), 'codex')
})

test('does not infer an agent from ordinary shell output', () => {
  assert.equal(detectTerminalAgentKind('PS C:\\work> echo OpenAI Codex'), undefined)
  assert.equal(detectTerminalAgentKind('Using Claude Code to inspect this project'), undefined)
  assert.equal(terminalRuntimeLabel(), 'Bash shell')
})

test('classifies current Claude and Codex activity from the newest terminal line', () => {
  assert.deepEqual(detectTerminalActivity('✻ Thinking (12s · esc to interrupt)'), {
    status: 'running',
    summary: 'Thinking'
  })
  assert.deepEqual(detectTerminalActivity('• Ran Get-Content src/app.ts'), {
    status: 'tool_calling',
    currentTool: 'Bash',
    summary: 'Ran Get-Content src/app.ts',
    target: 'Get-Content src/app.ts'
  })
  assert.deepEqual(detectTerminalActivity('⏺ Read(src/app.ts)\n❯'), {
    status: 'waiting_input'
  })
  assert.deepEqual(detectTerminalActivity('PS C:\\work>'), {
    status: 'idle',
    shellCwd: 'C:\\work'
  })
})

test('flushes sustained terminal output at the maximum wait boundary', () => {
  let now = 0
  let nextId = 1
  const scheduled = new Map<number, { at: number; callback: () => void }>()
  const timerApi: TerminalOutputTimerApi = {
    set: (callback, delayMs) => {
      const id = nextId
      nextId += 1
      scheduled.set(id, { at: now + delayMs, callback })
      return id
    },
    clear: (handle) => scheduled.delete(handle as number)
  }
  const advance = (ms: number) => {
    const target = now + ms
    while (true) {
      const next = Array.from(scheduled.entries())
        .filter(([, task]) => task.at <= target)
        .sort((a, b) => a[1].at - b[1].at)[0]
      if (!next) break
      now = next[1].at
      scheduled.delete(next[0])
      next[1].callback()
    }
    now = target
  }

  const flushed: string[] = []
  const batcher = new TerminalOutputBatcher((output) => flushed.push(output), {
    quietMs: 80,
    maxWaitMs: 250,
    timerApi
  })

  batcher.push('a')
  advance(70)
  batcher.push('b')
  advance(70)
  batcher.push('c')
  advance(110)

  assert.deepEqual(flushed, ['abc'])
})
