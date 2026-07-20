import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import type { TerminalSession } from '../../stores'
import { JournalSessionRow } from './JournalSessionRow'
import { SessionCard } from './SessionCard'

const session: TerminalSession = {
  id: 'terminal-1',
  title: 'Codex',
  cwd: 'F:\\File location\\repos\\ghostwork',
  isActive: true,
  attention: 'none',
  showBanner: false,
  status: 'running',
  lastActivity: Date.now(),
  agentKind: 'codex',
  taskSummary: 'Fix the agent session panel without letting this long title escape'
}

test('SessionCard is frameless, bounded, and shows folder-only project identity', () => {
  const markup = renderToStaticMarkup(
    createElement(SessionCard, {
      session,
      runtimeState: 'live',
      onOpenDetail: () => undefined
    })
  )

  assert.match(markup, /w-full min-w-0 overflow-hidden/)
  assert.doesNotMatch(markup, /class="card(?:\s|")/)
  assert.match(markup, />ghostwork</)
  assert.doesNotMatch(markup, /File location/)
  assert.match(markup, /OpenAI Codex/)
})

test('JournalSessionRow owns the available width and clips long content', () => {
  const markup = renderToStaticMarkup(
    createElement(JournalSessionRow, {
      snapshot: {
        filePath: 'F:/sessions/one.jsonl',
        fileMtime: Date.now(),
        kind: 'codex',
        topic: 'A very long historical session topic that should stay inside its rail',
        contextWindowApprox: true,
        totals: { input: 0, output: 0, cacheRead: 0, cacheCreate: 0 },
        perModel: {},
        recentFiles: [],
        totalsApprox: false,
        fileSize: 0
      },
      onOpenDetail: () => undefined
    })
  )

  assert.match(markup, /w-full min-w-0/)
  assert.match(markup, /overflow-hidden/)
  assert.match(markup, /truncate/)
})
