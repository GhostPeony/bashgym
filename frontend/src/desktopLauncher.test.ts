import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const projectRoot = fileURLToPath(new URL('../../', import.meta.url))
const launcher = readFileSync(`${projectRoot}Start-BashGym.ps1`, 'utf8')

test('desktop launcher probes Python candidates without terminating on stderr', () => {
  assert.match(launcher, /System\.Diagnostics\.ProcessStartInfo/)
  assert.match(launcher, /CreateNoWindow\s*=\s*\$true/)
  assert.match(launcher, /RedirectStandardError\s*=\s*\$true/)
  assert.doesNotMatch(launcher, /& \$candidate -c 'import httpx, uvicorn'/)
})

test('desktop launcher persists a durable failure record before showing an error', () => {
  assert.match(launcher, /desktop-launcher\.log/)
  assert.match(launcher, /Add-Content -LiteralPath \$LauncherLog/)
  assert.match(launcher, /Show-LaunchError -Message/)
})

test('desktop launcher retries stubborn owned processes during shutdown', () => {
  assert.match(launcher, /AddSeconds\(20\)/)
  assert.match(
    launcher,
    /foreach \(\$remainingId in \$remaining\)[\s\S]*?Stop-Process -Id \$remainingId -Force/
  )
})

test('desktop launcher only claims recognizable process types', () => {
  assert.match(
    launcher,
    /\$isFrontendChild = \$process\.Name -in @\('node\.exe', 'cmd\.exe'\) -and/
  )
  assert.match(
    launcher,
    /\$isManagedBackend = \$process\.Name -in @\('python\.exe', 'pythonw\.exe'\) -and/
  )
})
