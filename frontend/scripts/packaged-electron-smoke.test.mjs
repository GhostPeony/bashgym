import assert from 'node:assert/strict'
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { buildSmokeEnvironment, findPackagedExecutable } from './packaged-electron-smoke.mjs'

function fixture() {
  return join(tmpdir(), `bashgym-packaged-smoke-test-${process.pid}-${Math.random()}`)
}

function executable(root, platform) {
  if (platform === 'win32') return join(root, 'win-unpacked', 'Bash Gym.exe')
  if (platform === 'darwin') return join(root, 'mac-arm64', 'Bash Gym.app', 'Contents', 'MacOS', 'Bash Gym')
  return join(root, 'linux-arm64-unpacked', 'bash-gym')
}

for (const platform of ['win32', 'darwin', 'linux']) {
  test(`finds the target-native unpacked ${platform} executable`, (t) => {
    const root = fixture()
    t.after(() => rmSync(root, { recursive: true, force: true }))
    const expected = executable(root, platform)
    mkdirSync(join(expected, '..'), { recursive: true })
    writeFileSync(expected, '')

    assert.equal(findPackagedExecutable(root, platform), expected)
  })
}

test('fails closed when an unpacked executable is missing or ambiguous', (t) => {
  const root = fixture()
  t.after(() => rmSync(root, { recursive: true, force: true }))
  mkdirSync(root, { recursive: true })
  assert.throws(() => findPackagedExecutable(root, 'linux'), /found none/)

  for (const directory of ['linux-unpacked', 'linux-arm64-unpacked']) {
    const candidate = join(root, directory, 'bash-gym')
    mkdirSync(join(candidate, '..'), { recursive: true })
    writeFileSync(candidate, '')
  }
  assert.throws(() => findPackagedExecutable(root, 'linux'), /found linux-arm64-unpacked\/bash-gym, linux-unpacked\/bash-gym/)
})

test('launch environment preserves host execution settings without forwarding secrets', () => {
  const profile = join(tmpdir(), 'bashgym-packaged-smoke-env')
  const env = buildSmokeEnvironment(profile, {
    PATH: '/usr/bin',
    DISPLAY: ':99',
    GH_TOKEN: 'do-not-forward',
    BASHGYM_API_KEY: 'do-not-forward',
    SERVICE_PASSWORD: 'do-not-forward',
  })

  assert.equal(env.PATH, '/usr/bin')
  assert.equal(env.DISPLAY, ':99')
  assert.equal(env.GH_TOKEN, undefined)
  assert.equal(env.BASHGYM_API_KEY, undefined)
  assert.equal(env.SERVICE_PASSWORD, undefined)
  assert.equal(env.BASHGYM_API_BASE, 'http://127.0.0.1:9/api')
  assert.equal(env.BASHGYM_PYTHON, join(profile, 'intentionally-missing-python'))
})
