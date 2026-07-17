import assert from 'node:assert/strict'
import path from 'node:path'
import test from 'node:test'

import {
  createRetryableInitializer,
  managedBackendStartAction,
  resolveBackendRoot,
} from '../electron/backendLifecycle'

test('finds an unpacked clone backend from the executable when cwd belongs to another app', () => {
  const cloneRoot = path.resolve('C:/Users/example/Projects/bashgym-clone')
  const executablePath = path.join(
    cloneRoot,
    'frontend',
    'release',
    'win-unpacked',
    'Bash Gym.exe',
  )
  const backendMarker = path.join(cloneRoot, 'bashgym', 'api', 'routes.py')

  const resolved = resolveBackendRoot({
    configuredRoot: undefined,
    cwd: path.resolve('C:/Program Files/Unrelated Launcher/app'),
    appPath: path.join(
      cloneRoot,
      'frontend',
      'release',
      'win-unpacked',
      'resources',
      'app.asar',
    ),
    resourcesPath: path.join(
      cloneRoot,
      'frontend',
      'release',
      'win-unpacked',
      'resources',
    ),
    executablePath,
    markerExists: (candidate) => path.normalize(candidate) === path.normalize(backendMarker),
  })

  assert.ok(resolved)
  assert.equal(path.normalize(resolved), path.normalize(cloneRoot))
})

test('uses an explicit configured backend root before inferred locations', () => {
  const configuredRoot = path.resolve('D:/portable/bashgym')
  const resolved = resolveBackendRoot({
    configuredRoot,
    cwd: path.resolve('C:/somewhere/else'),
    appPath: path.resolve('C:/package/resources/app.asar'),
    resourcesPath: path.resolve('C:/package/resources'),
    executablePath: path.resolve('C:/package/Bash Gym.exe'),
    markerExists: (candidate) => candidate === path.join(configuredRoot, 'bashgym', 'api', 'routes.py'),
  })

  assert.ok(resolved)
  assert.equal(resolved, configuredRoot)
})

test('uses the installed Python package when a packaged desktop has no source checkout', () => {
  const resolved = resolveBackendRoot({
    configuredRoot: undefined,
    cwd: path.resolve('C:/Users/example/AppData/Local/Bash Gym'),
    appPath: path.resolve('C:/Program Files/Bash Gym/resources/app.asar'),
    resourcesPath: path.resolve('C:/Program Files/Bash Gym/resources'),
    executablePath: path.resolve('C:/Program Files/Bash Gym/Bash Gym.exe'),
    markerExists: () => false,
  })

  assert.equal(resolved, undefined)
})

test('rejects an explicitly configured backend root that is not a BashGym checkout', () => {
  assert.throws(
    () => resolveBackendRoot({
      configuredRoot: path.resolve('D:/not-bashgym'),
      cwd: path.resolve('C:/somewhere/else'),
      appPath: path.resolve('C:/package/resources/app.asar'),
      resourcesPath: path.resolve('C:/package/resources'),
      executablePath: path.resolve('C:/package/Bash Gym.exe'),
      markerExists: () => false,
    }),
    /configured BashGym backend root is invalid/,
  )
})

test('a rejected managed-backend initialization can be retried', async () => {
  let attempts = 0
  const readiness = createRetryableInitializer(async () => {
    attempts += 1
    if (attempts === 1) throw new Error('backend root unavailable')
  })

  await assert.rejects(readiness.ensureReady(), /backend root unavailable/)
  await readiness.ensureReady()

  assert.equal(attempts, 2)
})

test('an exited managed backend invalidates prior readiness', async () => {
  let attempts = 0
  const readiness = createRetryableInitializer(async () => {
    attempts += 1
  })

  await readiness.ensureReady()
  readiness.invalidate()
  await readiness.ensureReady()

  assert.equal(attempts, 2)
})

test('reuses its own reachable child when retrying desktop authentication', () => {
  assert.equal(managedBackendStartAction(true, true), 'reuse')
  assert.equal(managedBackendStartAction(false, false), 'spawn')
})

test('rejects a reachable backend the desktop process does not own', () => {
  assert.throws(
    () => managedBackendStartAction(true, false),
    /port is already in use/,
  )
})
