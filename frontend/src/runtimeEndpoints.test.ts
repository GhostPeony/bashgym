import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import {
  DEFAULT_DESKTOP_RUNTIME_ENDPOINTS,
  resolveDesktopRuntimeEndpoints
} from '../electron/runtimeEndpoints'

test('uses one portable API base and derives the desktop origin and websocket', () => {
  assert.deepEqual(DEFAULT_DESKTOP_RUNTIME_ENDPOINTS, {
    apiBase: 'http://127.0.0.1:8003/api',
    apiOrigin: 'http://127.0.0.1:8003',
    webSocketUrl: 'ws://127.0.0.1:8003/ws',
    devServerUrl: 'http://127.0.0.1:5173'
  })

  assert.deepEqual(
    resolveDesktopRuntimeEndpoints({
      BASHGYM_API_BASE: 'http://localhost:9010/api/',
      BASHGYM_DEV_SERVER_URL: 'http://localhost:4173'
    }),
    {
      apiBase: 'http://localhost:9010/api',
      apiOrigin: 'http://localhost:9010',
      webSocketUrl: 'ws://localhost:9010/ws',
      devServerUrl: 'http://localhost:4173'
    }
  )
})

test('keeps BASHGYM_API_URL only as a normalized compatibility alias', () => {
  assert.equal(
    resolveDesktopRuntimeEndpoints({
      BASHGYM_API_URL: 'http://127.0.0.1:8111'
    }).apiBase,
    'http://127.0.0.1:8111/api'
  )
  assert.equal(
    resolveDesktopRuntimeEndpoints({
      BASHGYM_API_BASE: 'http://127.0.0.1:8222/api',
      BASHGYM_API_URL: 'http://127.0.0.1:8111'
    }).apiBase,
    'http://127.0.0.1:8222/api'
  )
})

test('rejects ambiguous, credential-bearing, remote, and non-api desktop endpoints', () => {
  for (const value of [
    'https://127.0.0.1:8003/api',
    'http://example.test:8003/api',
    'http://user:secret@127.0.0.1:8003/api',
    'http://127.0.0.1:8003/v1',
    'http://127.0.0.1:8003/api?token=secret'
  ]) {
    assert.throws(() => resolveDesktopRuntimeEndpoints({ BASHGYM_API_BASE: value }))
  }
  assert.throws(() =>
    resolveDesktopRuntimeEndpoints({
      BASHGYM_DEV_SERVER_URL: 'https://example.test:5173'
    })
  )
})

test('main projects the normalized endpoints and renderer clients consume the projection', () => {
  const main = readFileSync(new URL('../electron/main.ts', import.meta.url), 'utf8')
  const preload = readFileSync(new URL('../electron/preload.ts', import.meta.url), 'utf8')
  const api = readFileSync(new URL('./services/api.ts', import.meta.url), 'utf8')
  const websocket = readFileSync(new URL('./services/websocket.ts', import.meta.url), 'utf8')

  assert.match(main, /additionalArguments:[\s\S]*?bashgym-api-base=[\s\S]*?bashgym-websocket-url=/)
  assert.match(preload, /runtime:[\s\S]*?apiBase:[\s\S]*?webSocketUrl:/)
  assert.match(api, /window\.bashgym\?\.runtime\?\.apiBase/)
  assert.match(websocket, /window\.bashgym\?\.runtime\?\.webSocketUrl/)
})
