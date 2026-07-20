import assert from 'node:assert/strict'
import test from 'node:test'

import { useUIStore } from './uiStore'

interface NavigationHarness {
  pushes: string[]
  replacements: string[]
  restore: () => void
}

function navigationHarness(search = ''): NavigationHarness {
  const globals = globalThis as unknown as Record<string, unknown>
  const priorWindow = globals.window
  const pushes: string[] = []
  const replacements: string[] = []
  const location = { pathname: '/app', search, hash: '' }
  const apply = (target: string) => {
    const query = target.indexOf('?')
    location.search = query >= 0 ? target.slice(query) : ''
  }
  globals.window = {
    location,
    history: {
      pushState: (_state: unknown, _title: string, target: string) => {
        pushes.push(target)
        apply(target)
      },
      replaceState: (_state: unknown, _title: string, target: string) => {
        replacements.push(target)
        apply(target)
      }
    }
  }
  return {
    pushes,
    replacements,
    restore: () => {
      if (priorWindow === undefined) delete globals.window
      else globals.window = priorWindow
    }
  }
}

function reset() {
  useUIStore.setState({
    overlayView: 'home',
    trainingSubview: 'runs',
    trainingSelection: { workspaceId: null, campaignId: null }
  })
}

test('serializes canonical Runs and AutoResearch URLs with one user push', () => {
  reset()
  const nav = navigationHarness('?view=training&tab=runs')
  try {
    useUIStore.getState().openTraining('autoresearch', {
      workspaceId: 'workspace a',
      campaignId: 'campaign/1'
    })
    assert.equal(nav.pushes.length, 1)
    assert.match(nav.pushes[0] || '', /view=training/)
    assert.match(nav.pushes[0] || '', /tab=autoresearch/)
    assert.match(nav.pushes[0] || '', /workspace_id=workspace\+a/)
    assert.match(nav.pushes[0] || '', /campaign_id=campaign%2F1/)

    useUIStore.getState().openTraining('runs')
    assert.match(nav.pushes[1] || '', /view=training&tab=runs$/)
    assert.doesNotMatch(nav.pushes[1] || '', /campaign_id|workspace_id/)
  } finally {
    nav.restore()
  }
})

test('hydrates canonical selection without writing history', () => {
  reset()
  const nav = navigationHarness(
    '?view=training&tab=autoresearch&workspace_id=workspace-a&campaign_id=campaign-1'
  )
  try {
    useUIStore.getState().hydrateNavigationFromUrl()
    assert.equal(useUIStore.getState().overlayView, 'training')
    assert.equal(useUIStore.getState().trainingSubview, 'autoresearch')
    assert.deepEqual(useUIStore.getState().trainingSelection, {
      workspaceId: 'workspace-a',
      campaignId: 'campaign-1'
    })
    assert.deepEqual(nav.pushes, [])
    assert.deepEqual(nav.replacements, [])
  } finally {
    nav.restore()
  }
})

test('normalizes the legacy overlay with exactly one replacement', () => {
  reset()
  const nav = navigationHarness(
    '?view=autoresearch&workspace_id=workspace-a&campaign_id=campaign-1'
  )
  try {
    useUIStore.getState().hydrateNavigationFromUrl()
    assert.equal(useUIStore.getState().overlayView, 'training')
    assert.equal(useUIStore.getState().trainingSubview, 'autoresearch')
    assert.equal(nav.replacements.length, 1)
    assert.match(nav.replacements[0] || '', /view=training&tab=autoresearch/)
    assert.deepEqual(nav.pushes, [])
  } finally {
    nav.restore()
  }
})

test('invalid tabs and empty IDs recover to Runs with null selection', () => {
  reset()
  const nav = navigationHarness()
  try {
    useUIStore
      .getState()
      .hydrateNavigationFromUrl('?view=training&tab=unknown&workspace_id=&campaign_id=%20')
    assert.equal(useUIStore.getState().trainingSubview, 'runs')
    assert.deepEqual(useUIStore.getState().trainingSelection, {
      workspaceId: null,
      campaignId: null
    })
  } finally {
    nav.restore()
  }
})

test('legacy openOverlay compatibility normalizes with replacement', () => {
  reset()
  const nav = navigationHarness()
  try {
    useUIStore.getState().openOverlay('autoresearch')
    assert.equal(useUIStore.getState().overlayView, 'training')
    assert.equal(nav.replacements.length, 1)
  } finally {
    nav.restore()
  }
})

test('non-training navigation writes canonical URLs and hydration leaves Training', () => {
  reset()
  const nav = navigationHarness('?view=training&tab=autoresearch')
  try {
    useUIStore.getState().hydrateNavigationFromUrl()
    assert.equal(useUIStore.getState().overlayView, 'training')

    useUIStore.getState().openOverlay('home')
    assert.match(nav.pushes.at(-1) || '', /view=home$/)
    assert.equal(useUIStore.getState().overlayView, 'home')

    useUIStore.getState().hydrateNavigationFromUrl('?view=training&tab=runs')
    assert.equal(useUIStore.getState().overlayView, 'training')
    useUIStore.getState().hydrateNavigationFromUrl('?view=home')
    assert.equal(useUIStore.getState().overlayView, 'home')
  } finally {
    nav.restore()
  }
})

test('workspace has an explicit canonical URL and back hydration restores it', () => {
  reset()
  const nav = navigationHarness('?view=training&tab=runs')
  try {
    useUIStore.getState().closeOverlay()
    assert.match(nav.pushes.at(-1) || '', /view=workspace$/)
    assert.equal(useUIStore.getState().overlayView, null)

    useUIStore.getState().hydrateNavigationFromUrl('?view=training&tab=autoresearch')
    assert.equal(useUIStore.getState().overlayView, 'training')
    useUIStore.getState().hydrateNavigationFromUrl('?view=workspace')
    assert.equal(useUIStore.getState().overlayView, null)

    useUIStore.getState().openOverlay('workspace')
    assert.equal(useUIStore.getState().overlayView, null)
    assert.match(nav.pushes.at(-1) || '', /view=workspace$/)
  } finally {
    nav.restore()
  }
})

test('an empty or unknown URL hydrates to Home instead of retaining Training', () => {
  reset()
  const nav = navigationHarness('?view=training&tab=runs')
  try {
    useUIStore.getState().hydrateNavigationFromUrl()
    assert.equal(useUIStore.getState().overlayView, 'training')
    useUIStore.getState().hydrateNavigationFromUrl('')
    assert.equal(useUIStore.getState().overlayView, 'home')

    useUIStore.getState().hydrateNavigationFromUrl('?view=not-a-view')
    assert.equal(useUIStore.getState().overlayView, 'home')
  } finally {
    nav.restore()
  }
})
