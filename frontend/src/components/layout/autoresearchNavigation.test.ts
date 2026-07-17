import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

const sidebar = readFileSync(new URL('./Sidebar.tsx', import.meta.url), 'utf8')
const layout = readFileSync(new URL('./MainLayout.tsx', import.meta.url), 'utf8')
const navigationBar = readFileSync(new URL('./NavigationBar.tsx', import.meta.url), 'utf8')
const training = readFileSync(new URL('../training/TrainingDashboard.tsx', import.meta.url), 'utf8')
const uiStore = readFileSync(new URL('../../stores/uiStore.ts', import.meta.url), 'utf8')
const websocket = readFileSync(new URL('../../services/websocket.ts', import.meta.url), 'utf8')
const componentIndex = readFileSync(new URL('../autoresearch/index.ts', import.meta.url), 'utf8')
const storeIndex = readFileSync(new URL('../../stores/index.ts', import.meta.url), 'utf8')

test('AutoResearch is a first-class sidebar destination directly below Training', () => {
  const trainingItem = sidebar.indexOf('label="Training"')
  const autoResearchItem = sidebar.indexOf('label="AutoResearch"')

  assert.ok(trainingItem >= 0, 'Training should remain in primary navigation')
  assert.ok(autoResearchItem > trainingItem, 'AutoResearch should follow Training')
  assert.doesNotMatch(
    sidebar.slice(trainingItem, autoResearchItem),
    /<MenuItem[\s\S]*?<MenuItem/,
    'no other menu item should appear between Training and AutoResearch',
  )
  assert.match(sidebar.slice(autoResearchItem), /openTraining\(['"]autoresearch['"]\)/)
  assert.doesNotMatch(layout, /AutoResearchDashboard/)
  assert.doesNotMatch(layout, /overlayView\s*===\s*['"]autoresearch['"]/)
  assert.match(layout, /hydrateNavigationFromUrl/)
})

test('layout chooses the destination while Training keeps only its existing monitor header', () => {
  assert.match(layout, /AutoResearchControlRoom/)
  assert.match(layout, /trainingSubview\s*===\s*['"]autoresearch['"]/)
  assert.doesNotMatch(training, /AutoResearchControlRoom/)
  assert.doesNotMatch(training, /role=['"]tablist['"]/)
  assert.doesNotMatch(training, /Training views/)
  assert.equal((training.match(/>Training Monitor</g) || []).length, 1)
  assert.doesNotMatch(training, /Direct runs and durable AutoResearch campaigns/)
})

test('top navigation names AutoResearch as its own destination', () => {
  assert.match(navigationBar, /trainingSubview/)
  assert.match(navigationBar, /trainingSubview\s*===\s*['"]autoresearch['"]\s*\?\s*['"]AutoResearch['"]/)
})

test('legacy URLs remain redirect-only compatibility', () => {
  assert.match(uiStore, /['"]autoresearch['"]/)
  assert.match(uiStore, /replaceState/)
  assert.match(uiStore, /view.*training/)
})

test('official renderer ownership excludes the retired prototype dashboard and store', () => {
  assert.match(componentIndex, /AutoResearchControlRoom/)
  assert.doesNotMatch(componentIndex, /AutoResearchDashboard/)
  assert.doesNotMatch(websocket, /autoresearchStore|useAutoResearchStore/)
  assert.doesNotMatch(storeIndex, /autoresearchStore|useAutoResearchStore/)
})
