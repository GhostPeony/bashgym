import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type { ActivityEvent } from '../../stores/activityStore'
import { ActivityFeedPanel } from './ActivityFeed'

const event: ActivityEvent = {
  id: 1,
  type: 'campaign:action-failed',
  category: 'campaign',
  severity: 'error',
  title: 'Campaign action failed — action-1',
  detail: '{"campaign_id":"campaign-1","action_id":"action-1"}',
  destination: {
    label: 'Open AutoResearch',
    view: 'autoresearch',
    workspaceId: 'workspace-a',
    campaignId: 'campaign-1'
  },
  timestamp: Date.now()
}

test('global activity panel lets users inspect details before navigating', () => {
  const markup = renderToStaticMarkup(
    createElement(ActivityFeedPanel, {
      events: [event],
      enabledCategories: new Set<string>(),
      selectedEventId: event.id,
      onSelect: () => undefined,
      onNavigate: () => undefined,
      onToggleCategory: () => undefined,
      onClear: () => undefined,
      onClose: () => undefined
    })
  )
  assert.match(markup, /<button[^>]+aria-label="Inspect Campaign action failed/)
  assert.match(markup, /Event detail/)
  assert.match(markup, /campaign:action-failed/)
  assert.match(markup, /campaign-1/)
  assert.match(markup, />Open AutoResearch<svg/)
})

test('notification trigger lives in the top navigation, not the status bar', () => {
  const navigation = readFileSync(new URL('../layout/NavigationBar.tsx', import.meta.url), 'utf8')
  const status = readFileSync(new URL('../layout/StatusBar.tsx', import.meta.url), 'utf8')
  assert.match(navigation, /Bell/)
  assert.match(navigation, /useActivityStore/)
  assert.match(navigation, /Notifications/)
  assert.doesNotMatch(status, /Bell/)
  assert.doesNotMatch(status, /useActivityStore/)
})
