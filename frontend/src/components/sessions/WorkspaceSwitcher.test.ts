import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { useWorkspaceStore } from '../../stores'
import type { WorkspaceSessionGroup } from '../../stores'
import { WorkspaceSwitcher } from './WorkspaceSwitcher'

// renderToStaticMarkup reads zustand's server snapshot, which is the store's
// INITIAL state (setState does not affect it). So the switcher always renders
// the initial active workspace here — drive the fixtures off that.
const initial = useWorkspaceStore.getInitialState()
const activeWorkspace = initial.workspaces.find((w) => w.id === initial.activeWorkspaceId)!

function group(id: string, name: string, live: number, saved: number): WorkspaceSessionGroup {
  return {
    workspace: id === activeWorkspace.id ? activeWorkspace : { id, name, createdAt: 1, lastActiveAt: 1 },
    isActive: id === activeWorkspace.id,
    sessions: Array.from({ length: saved }, () => ({}) as never),
    runningCount: live,
    waitingCount: 0,
    errorCount: 0,
    liveCount: live
  }
}

test('WorkspaceSwitcher is collapsed by default: active workspace + count only, dropdown and other workspaces absent', () => {
  const groups = [
    group(activeWorkspace.id, activeWorkspace.name, 2, 2),
    group('other-workspace-id', 'HiddenWorkspace', 0, 3)
  ]

  const markup = renderToStaticMarkup(createElement(WorkspaceSwitcher, { groups }))

  // The single trigger row shows the active workspace and its live count
  assert.match(markup, new RegExp(`>${activeWorkspace.name}<`))
  assert.match(markup, /2 live/)
  // Collapsed: the dropdown list is not rendered, so it cannot push the feed
  // down, and other workspaces do not appear inline.
  assert.doesNotMatch(markup, /Workspaces</) // dropdown header absent
  assert.doesNotMatch(markup, /HiddenWorkspace/)
  assert.match(markup, /aria-expanded="false"/)
  assert.match(markup, /aria-haspopup="menu"/)
})

test('WorkspaceSwitcher shows a saved count when the active workspace has no live terminals', () => {
  const groups = [group(activeWorkspace.id, activeWorkspace.name, 0, 5)]

  const markup = renderToStaticMarkup(createElement(WorkspaceSwitcher, { groups }))

  assert.match(markup, /5 saved/)
  assert.doesNotMatch(markup, /live/)
})
