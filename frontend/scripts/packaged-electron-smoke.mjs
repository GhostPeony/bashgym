import { randomUUID } from 'node:crypto'
import { existsSync, mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join, relative, sep } from 'node:path'
import { pathToFileURL } from 'node:url'

import { _electron as electron } from 'playwright-core'

/* global clearTimeout, console, setTimeout */

const LAUNCH_TIMEOUT_MS = 30_000
const WINDOW_TIMEOUT_MS = 20_000
const IPC_TIMEOUT_MS = 10_000
const CLOSE_TIMEOUT_MS = 10_000
const MAX_DIAGNOSTICS = 20

function walk(directory, depth = 0) {
  if (depth > 5 || !existsSync(directory)) return []
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    return entry.isDirectory() ? walk(path, depth + 1) : [path]
  })
}

function isPackagedExecutable(path, platform) {
  const normalized = path.split(sep).join('/')
  if (platform === 'win32') {
    return basename(path).toLowerCase() === 'bash gym.exe'
      && normalized.toLowerCase().includes('/win-unpacked/')
  }
  if (platform === 'darwin') {
    return normalized.endsWith('/Bash Gym.app/Contents/MacOS/Bash Gym')
  }
  if (platform === 'linux') {
    return basename(path) === 'bash-gym' && /\/linux(?:-[^/]+)?-unpacked\//.test(normalized)
  }
  return false
}

export function findPackagedExecutable(releaseDirectory, platform = process.platform) {
  const matches = walk(releaseDirectory).filter((path) => isPackagedExecutable(path, platform))
  if (matches.length !== 1) {
    const found = matches.length === 0
      ? 'none'
      : matches
        .map((path) => relative(releaseDirectory, path).split(sep).join('/'))
        .join(', ')
    throw new Error(
      `Expected exactly one ${platform} unpacked Bash Gym executable under ${releaseDirectory}; found ${found}`,
    )
  }
  return matches[0]
}

function withTimeout(promise, timeoutMs, label) {
  let timer
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} exceeded ${timeoutMs}ms`)), timeoutMs)
    }),
  ]).finally(() => clearTimeout(timer))
}

export function buildSmokeEnvironment(profileDirectory, sourceEnvironment = process.env) {
  const blocked = /(api[_-]?key|authorization|credential|password|secret|token)/i
  const env = Object.fromEntries(
    Object.entries(sourceEnvironment).filter(([key, value]) => value !== undefined && !blocked.test(key)),
  )
  return {
    ...env,
    BASHGYM_API_BASE: 'http://127.0.0.1:9/api',
    BASHGYM_PYTHON: join(profileDirectory, 'intentionally-missing-python'),
    BASHGYM_OPEN_DEVTOOLS: '0',
    ELECTRON_ENABLE_LOGGING: '1',
  }
}

function formatError(error, diagnostics) {
  const message = error instanceof Error ? error.stack || error.message : String(error)
  if (diagnostics.length === 0) return message
  return `${message}\nRecent renderer diagnostics:\n${diagnostics.join('\n')}`
}

export async function runPackagedElectronSmoke({
  releaseDirectory = join(process.cwd(), 'release'),
  platform = process.platform,
} = {}) {
  const executablePath = findPackagedExecutable(releaseDirectory, platform)
  const profileDirectory = mkdtempSync(join(tmpdir(), 'bashgym-electron-smoke-'))
  const terminalId = `ci-packaged-smoke-${randomUUID()}`
  const diagnostics = []
  let electronApplication
  let failure
  let passSummary

  const remember = (line) => {
    diagnostics.push(String(line).slice(0, 1_000))
    if (diagnostics.length > MAX_DIAGNOSTICS) diagnostics.shift()
  }

  try {
    console.log(`[packaged-smoke] launching ${relative(process.cwd(), executablePath)}`)
    electronApplication = await electron.launch({
      executablePath,
      cwd: process.cwd(),
      env: buildSmokeEnvironment(profileDirectory),
      args: [
        `--user-data-dir=${join(profileDirectory, 'user-data')}`,
        '--disable-gpu',
        '--disable-gpu-compositing',
        '--no-default-browser-check',
      ],
      offline: true,
      timeout: LAUNCH_TIMEOUT_MS,
    })

    electronApplication.process().stderr?.on('data', (chunk) => remember(chunk.toString().trim()))
    const window = await withTimeout(
      electronApplication.firstWindow(),
      WINDOW_TIMEOUT_MS,
      'first Electron window',
    )
    window.on('console', (message) => {
      if (message.type() === 'error' || message.type() === 'warning') remember(message.text())
    })
    window.on('pageerror', (error) => remember(`pageerror: ${error.message}`))

    const systemInfo = await withTimeout(
      window.evaluate(() => globalThis.window.bashgym?.system.info()),
      IPC_TIMEOUT_MS,
      'preload system.info()',
    )
    if (!systemInfo || systemInfo.platform !== platform || !systemInfo.electronVersion) {
      throw new Error(`Invalid preload system.info() result: ${JSON.stringify(systemInfo)}`)
    }

    const createResult = await withTimeout(
      window.evaluate((id) => globalThis.window.bashgym?.terminal.create(id), terminalId),
      IPC_TIMEOUT_MS,
      'PTY create',
    )
    if (!createResult?.success || createResult.id !== terminalId || createResult.attached !== false) {
      throw new Error(`PTY create failed: ${JSON.stringify(createResult)}`)
    }

    const sessions = await withTimeout(
      window.evaluate(() => globalThis.window.bashgym?.terminal.list()),
      IPC_TIMEOUT_MS,
      'PTY list',
    )
    const created = sessions?.find((session) => session.id === terminalId)
    if (!created || created.exited || !created.cwd) {
      throw new Error(`Created PTY was not live in terminal.list(): ${JSON.stringify(sessions)}`)
    }

    const killed = await withTimeout(
      window.evaluate((id) => globalThis.window.bashgym?.terminal.kill(id), terminalId),
      IPC_TIMEOUT_MS,
      'PTY kill',
    )
    if (killed !== true) throw new Error(`PTY kill returned ${JSON.stringify(killed)}`)

    const remaining = await withTimeout(
      window.evaluate(() => globalThis.window.bashgym?.terminal.list()),
      IPC_TIMEOUT_MS,
      'post-kill PTY list',
    )
    if (remaining?.some((session) => session.id === terminalId)) {
      throw new Error(`Killed PTY remained in terminal.list(): ${JSON.stringify(remaining)}`)
    }

    passSummary = `${systemInfo.platform}/${systemInfo.arch} Electron ${systemInfo.electronVersion}; preload and PTY lifecycle verified offline`
  } catch (error) {
    failure = error
  } finally {
    if (electronApplication) {
      const applicationProcess = electronApplication.process()
      try {
        await withTimeout(electronApplication.close(), CLOSE_TIMEOUT_MS, 'Electron shutdown')
      } catch (error) {
        remember(error)
        failure ||= error
        applicationProcess.kill()
      }
      if (applicationProcess.exitCode === null && applicationProcess.signalCode === null) {
        try {
          await withTimeout(
            new Promise((resolve) => applicationProcess.once('exit', resolve)),
            CLOSE_TIMEOUT_MS,
            'Electron process exit',
          )
        } catch (error) {
          remember(error)
          failure ||= error
          applicationProcess.kill()
        }
      }
    }
    try {
      rmSync(profileDirectory, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 })
    } catch (error) {
      remember(error)
      failure ||= error
    }
  }

  if (failure) throw new Error(formatError(failure, diagnostics))
  console.log(`[packaged-smoke] PASS ${passSummary}`)
}

async function main() {
  await runPackagedElectronSmoke()
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`[packaged-smoke] FAIL\n${error instanceof Error ? error.message : String(error)}`)
    process.exitCode = 1
  })
}
