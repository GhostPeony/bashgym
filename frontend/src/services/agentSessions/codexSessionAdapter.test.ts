import assert from 'node:assert/strict'
import test from 'node:test'
import { ingestCodexFile, resetCodexFile } from './codexSessionAdapter'

/**
 * Faithful in-memory copies of the sessions:readHead / sessions:readTail IPC
 * handlers in electron/main.ts, including the byte caps and last-newline
 * trimming. The point of the truncation semantics is that a session_meta first
 * line longer than the requested head cap comes back as an unparseable JSON
 * fragment — exactly the bug that dropped Codex sessions from the rail.
 */
const HARD_CAP = 262_144

function makeIpc(buffer: Buffer) {
  return {
    readHead: async (_path: string, maxBytes = 16_384) => {
      const cap = Math.min(Math.max(Number(maxBytes) || 16_384, 1_024), HARD_CAP)
      const end = Math.min(cap, buffer.length)
      const slice = buffer.subarray(0, end)
      const lastNewline = slice.lastIndexOf(0x0a)
      const stop = lastNewline === -1 ? end : lastNewline + 1
      return { success: true as const, data: buffer.toString('utf-8', 0, stop) }
    },
    readTail: async (_path: string, fromOffset: number, maxBytes = 131_072) => {
      const cap = Math.min(Math.max(Number(maxBytes) || 131_072, 4_096), HARD_CAP)
      const size = buffer.length
      let offset = Math.max(Number(fromOffset) || 0, 0)
      let reset = false
      if (size < offset) {
        offset = Math.max(0, size - cap)
        reset = true
      }
      const toRead = Math.min(cap, size - offset)
      if (toRead <= 0) return { success: true as const, data: '', newOffset: offset, size, reset }
      const slice = buffer.subarray(offset, offset + toRead)
      const lastNewline = slice.lastIndexOf(0x0a)
      if (lastNewline === -1) {
        if (toRead >= cap)
          return { success: true as const, data: '', newOffset: offset + toRead, size, reset }
        return { success: true as const, data: '', newOffset: offset, size, reset }
      }
      return {
        success: true as const,
        data: slice.toString('utf-8', 0, lastNewline + 1),
        newOffset: offset + lastNewline + 1,
        size,
        reset
      }
    }
  }
}

function installIpc(buffer: Buffer): void {
  ;(globalThis as { window?: unknown }).window = { bashgym: { sessions: makeIpc(buffer) } }
}

/** session_meta line padded past `padTo` bytes, mirroring real Codex growth */
function bigSessionMeta(cwd: string, padTo: number): string {
  const meta = {
    type: 'session_meta',
    payload: {
      session_id: 'test-session',
      cwd,
      model_provider: 'openai',
      git: { branch: 'main' },
      // base_instructions is what actually bloats real session_meta lines
      base_instructions: 'x'.repeat(padTo)
    }
  }
  return JSON.stringify(meta)
}

test('codex ingest recovers cwd when session_meta exceeds the old 16KB head cap', async () => {
  const cwd = 'C:\\Users\\Cade\\Projects\\ghostwork'
  const lines = [
    bigSessionMeta(cwd, 20_000), // ~20KB first line — over the retired 16KB cap
    JSON.stringify({ type: 'event_msg', payload: { type: 'task_started' } }),
    JSON.stringify({
      type: 'event_msg',
      timestamp: '2026-07-17T09:07:14.000Z',
      payload: {
        type: 'token_count',
        info: {
          total_token_usage: { input_tokens: 1000, output_tokens: 200, cached_input_tokens: 50 },
          last_token_usage: { input_tokens: 800, output_tokens: 100, cached_input_tokens: 40 },
          model_context_window: 200_000
        }
      }
    })
  ]
  // Pad past the 256KB tail window so the meta line lives ONLY in the head read.
  // Without this the tail read (from offset 0 on a small file) would recover cwd
  // anyway and the test would not exercise the head-cap bug at all.
  while (lines.join('\n').length < 300_000) {
    lines.push(JSON.stringify({ type: 'response_item', payload: { type: 'reasoning' } }))
  }
  const buffer = Buffer.from(lines.join('\n') + '\n', 'utf-8')
  const filePath = 'C:/codex/rollout-big-meta.jsonl'
  resetCodexFile(filePath)
  installIpc(buffer)

  const snap = await ingestCodexFile({ path: filePath, size: buffer.length, modified: Date.now() })

  assert.ok(snap, 'expected a snapshot')
  assert.equal(snap?.cwd, cwd, 'cwd must survive an oversized session_meta line')
  assert.equal(snap?.title, 'ghostwork')
  assert.equal(snap?.gitBranch, 'main')
})

test('codex ingest recovers cwd from a tail turn_context when the meta line is unreadably large', async () => {
  const cwd = 'C:\\Users\\Cade\\Projects\\ghostwork'
  // A meta line larger than the IPC hard cap can never be captured by the head
  // read; the session must still surface via turn_context events in the tail.
  const meta = bigSessionMeta('', HARD_CAP + 50_000).replace('"cwd":""', '"cwd":null')
  const turnContext = JSON.stringify({
    type: 'turn_context',
    payload: { turn_id: 't1', cwd, model: 'gpt-5.6' }
  })
  const tokenCount = JSON.stringify({
    type: 'event_msg',
    payload: {
      type: 'token_count',
      info: { total_token_usage: { input_tokens: 5, output_tokens: 5 } }
    }
  })
  const buffer = Buffer.from([meta, turnContext, tokenCount].join('\n') + '\n', 'utf-8')
  const filePath = 'C:/codex/rollout-huge-meta.jsonl'
  resetCodexFile(filePath)
  installIpc(buffer)

  const snap = await ingestCodexFile({ path: filePath, size: buffer.length, modified: Date.now() })

  assert.ok(snap, 'expected a snapshot')
  assert.equal(snap?.cwd, cwd, 'cwd must be recovered from a tail turn_context event')
  assert.equal(snap?.title, 'ghostwork')
})
