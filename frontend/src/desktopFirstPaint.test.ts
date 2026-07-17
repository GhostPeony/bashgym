import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const mainSource = readFileSync(new URL('../electron/main.ts', import.meta.url), 'utf8')

test('desktop waits for the first renderer paint and explicitly invalidates stale compositor surfaces', () => {
  assert.match(mainSource, /new BrowserWindow\(\{[\s\S]*?show:\s*false,/)
  assert.match(
    mainSource,
    /once\('ready-to-show',[\s\S]*?\.show\(\)[\s\S]*?webContents\.invalidate\(\)/,
  )
  assert.match(
    mainSource,
    /on\('did-finish-load',[\s\S]*?webContents\.invalidate\(\)/,
  )
  assert.doesNotMatch(mainSource, /disableHardwareAcceleration/)
})
