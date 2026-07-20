import { app, BrowserWindow, ipcMain, shell, Menu, webContents, clipboard, nativeImage, safeStorage } from 'electron'
import path from 'path'
import fs from 'fs'
import os from 'os'
import { spawn, type ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import type { IPty } from 'node-pty'
import { buildAgentBridgeLaunchCommand, type AgentBridgeLaunchRequest } from './agentBridge'
import {
  buildBackendChildEnvironment,
  CampaignBridgeClient,
  MANAGED_BACKEND_STARTUP_TIMEOUT_MS,
  type CampaignBody,
  type CampaignMethod,
  type CampaignQuery,
  type CampaignRequestAuthority,
  createDesktopBootstrapToken,
  resolveCampaignApiOrigin,
} from './campaignBridge'
import {
  CampaignAgentHostController,
  type MainOwnedCampaignAgentIdentity,
} from './campaignAgentHost'
import { buildCodexCampaignAgentLaunch } from './campaignAgentLaunch'
import { CampaignAgentMcpHost } from './campaignAgentMcpHost'
import {
  createRetryableInitializer,
  managedBackendStartAction,
  resolveBackendRoot,
} from './backendLifecycle'
import {
  DEFAULT_DESKTOP_RUNTIME_ENDPOINTS,
  resolveDesktopRuntimeEndpoints,
} from './runtimeEndpoints'
import {
  buildPowerShellArgs,
  TerminalIpcBatcher,
  TerminalOutputFlowController,
  TerminalOutputFlowOwnership,
  TerminalScrollbackBuffer,
  TerminalSizeTracker,
} from './terminalTransport'

// Handle creating/removing shortcuts on Windows when installing/uninstalling
// This is only needed for production builds with Squirrel installer
// Removed: electron-squirrel-startup check (causes ESM/CJS issues in dev)

// Single instance lock - prevent multiple instances from running
const gotTheLock = app.requestSingleInstanceLock()

if (!gotTheLock) {
  // Another instance is already running, quit immediately
  app.quit()
}

let mainWindow: BrowserWindow | null = null

// PTY session registry — main process owns sessions so they survive renderer
// reloads and window close; scrollback is retained for re-attach replay.
interface PtySession {
  pty: IPty
  /** Raw output, bounded and replayed on re-attach. */
  scrollback: TerminalScrollbackBuffer
  /** Short-frame output batching prevents redraw IPC floods. */
  outputBatcher: TerminalIpcBatcher
  /** Parser acknowledgements keep xterm's redraw queue bounded. */
  outputFlow: TerminalOutputFlowController
  /** Ensures stale panes cannot acknowledge or release a replacement pane. */
  outputFlowOwnership: TerminalOutputFlowOwnership
  /** Last applied ConPTY size, used to suppress duplicate full-screen TUI redraws. */
  size: TerminalSizeTracker
  cwd: string
  createdAt: number
  exited: boolean
  exitCode: number | null
  /** Main-owned lifecycle generation; changes whenever a PTY id is replaced. */
  generation: string
  /** Set only by a main-owned Codex/Hermes launch path, never from attach IPC. */
  campaignAgentIdentity?: MainOwnedCampaignAgentIdentity
}

const MAX_BUFFER_BYTES = 400_000 // ~400KB of scrollback per session
const ptySessions: Map<string, PtySession> = new Map()
const campaignAgentMcpHosts: Map<string, CampaignAgentMcpHost> = new Map()
const campaignAgentLaunchScopes: Map<string, { workspaceId: string; campaignId: string }> = new Map()
const agentBridgeArtifacts: Map<string, Set<string>> = new Map()

function registerAgentBridgeArtifact(terminalId: string, artifactPath: string) {
  const artifacts = agentBridgeArtifacts.get(terminalId) ?? new Set<string>()
  artifacts.add(artifactPath)
  agentBridgeArtifacts.set(terminalId, artifacts)
}

function cleanupAgentBridgeArtifacts(terminalId: string) {
  const artifacts = agentBridgeArtifacts.get(terminalId)
  if (!artifacts) return
  artifacts.forEach((artifactPath) => {
    try { fs.unlinkSync(artifactPath) } catch { /* already removed */ }
  })
  agentBridgeArtifacts.delete(terminalId)
}

function revokeCampaignAgentTerminal(
  terminalId: string,
  reason: 'pty_exit' | 'pty_replacement' | 'renderer_reload' | 'app_shutdown' | 'explicit_revoke',
): Promise<void> {
  return campaignAgentHostController.teardownTerminal(terminalId, reason)
    .finally(() => closeCampaignAgentMcpHost(terminalId))
}

function closeCampaignAgentMcpHost(terminalId: string): Promise<void> {
  const host = campaignAgentMcpHosts.get(terminalId)
  campaignAgentLaunchScopes.delete(terminalId)
  if (!host) return Promise.resolve()
  campaignAgentMcpHosts.delete(terminalId)
  host.lock()
  return host.close()
}

async function teardownAllCampaignAgents(
  reason: 'renderer_reload' | 'app_shutdown',
): Promise<void> {
  try {
    await campaignAgentHostController.teardownAll(reason)
  } finally {
    await Promise.allSettled([...campaignAgentMcpHosts.keys()].map(closeCampaignAgentMcpHost))
  }
}

function killAllPtys() {
  ptySessions.forEach((s, id) => {
    s.outputBatcher.dispose()
    s.outputFlow.dispose()
    try { s.pty.kill() } catch { /* already dead */ }
    cleanupAgentBridgeArtifacts(id)
  })
  ptySessions.clear()
  campaignAgentMcpHosts.forEach((host) => {
    host.lock()
    void host.close()
  })
  campaignAgentMcpHosts.clear()
  campaignAgentLaunchScopes.clear()
  campaignAgentHostController.disposeLocal()
}

// Determine if we're in development
const isDev = !app.isPackaged
let runtimeEndpointConfigurationError: Error | null = null
let desktopRuntimeEndpoints = DEFAULT_DESKTOP_RUNTIME_ENDPOINTS
try {
  desktopRuntimeEndpoints = resolveDesktopRuntimeEndpoints(process.env)
} catch (error) {
  runtimeEndpointConfigurationError = asError(error)
}
const campaignApiOrigin = resolveCampaignApiOrigin(desktopRuntimeEndpoints.apiOrigin)
const desktopBootstrapToken = createDesktopBootstrapToken()
const campaignBridgeClient = new CampaignBridgeClient(
  campaignApiOrigin,
  desktopBootstrapToken,
)
const campaignAgentHostInstanceId = `desktop_${randomUUID().replaceAll('-', '')}`
const campaignAgentHostController = new CampaignAgentHostController({
  transport: {
    register: (body) => campaignBridgeClient.registerCampaignAgentHostSession(body),
    claim: (registrationId) => campaignBridgeClient.claimCampaignAgentDelivery(registrationId),
    revoke: (registrationId) => campaignBridgeClient.revokeCampaignAgentHostSession(registrationId),
    grant: (campaignId, body) => campaignBridgeClient.issueCampaignAgentGrant(campaignId, body),
    attach: (campaignId, body) => campaignBridgeClient.attachCampaignAgent(campaignId, body),
    revokeAttachment: (campaignId, attachmentId, body) => (
      campaignBridgeClient.revokeCampaignAgentAttachment(campaignId, attachmentId, body)
    ),
    heartbeat: (credential, body) => campaignBridgeClient.heartbeatCampaignAgent(credential, body),
    observe: (credential) => campaignBridgeClient.observeCampaignAsAgent(credential),
    artifacts: (credential, query) => campaignBridgeClient.listCampaignArtifactsAsAgent(credential, query),
  },
  resolveIdentity: (terminalId) => {
    const session = ptySessions.get(terminalId)
    if (!session?.campaignAgentIdentity) return null
    return {
      ...session.campaignAgentIdentity,
      live: !session.exited,
    }
  },
  onLifecycle: (event) => {
    if (event.kind === 'activated') return
    const host = campaignAgentMcpHosts.get(event.terminalId)
    if (event.kind === 'actions_locked') {
      host?.lock()
      void revokeCampaignAgentTerminal(event.terminalId, 'explicit_revoke').catch(() => undefined)
      return
    }
    campaignAgentLaunchScopes.delete(event.terminalId)
    if (!host) return
    campaignAgentMcpHosts.delete(event.terminalId)
    host.lock()
    void host.close()
  },
})
let backendProcess: ChildProcess | null = null
let lastCampaignBackendError: Error | null = runtimeEndpointConfigurationError

function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value || 'Unknown backend error'))
}

async function backendIsReachable(): Promise<boolean> {
  try {
    const response = await fetch(`${campaignApiOrigin}/api/health`, {
      signal: AbortSignal.timeout(500),
    })
    return response.ok
  } catch {
    return false
  }
}

async function startManagedBackend(): Promise<void> {
  const startAction = managedBackendStartAction(
    await backendIsReachable(),
    backendProcess !== null && backendProcess.exitCode === null,
  )
  if (startAction === 'reuse') return
  const root = resolveBackendRoot({
    configuredRoot: process.env.BASHGYM_PROJECT_ROOT,
    cwd: process.cwd(),
    appPath: app.getAppPath(),
    resourcesPath: process.resourcesPath,
    executablePath: process.execPath,
    markerExists: fs.existsSync,
  })
  const url = new URL(campaignApiOrigin)
  const pythonCommand = process.env.BASHGYM_PYTHON || 'python'
  let spawnFailed = false
  backendProcess = spawn(
    pythonCommand,
    [
      '-m',
      'uvicorn',
      'bashgym.api.routes:create_app',
      '--factory',
      '--host',
      url.hostname === 'localhost' ? '127.0.0.1' : url.hostname,
      '--port',
      url.port || '8003',
    ],
    {
      cwd: root,
      env: buildBackendChildEnvironment(
        {
          ...process.env,
          BASHGYM_API_BASE: desktopRuntimeEndpoints.apiBase,
          BASHGYM_API_URL: desktopRuntimeEndpoints.apiOrigin,
        },
        desktopBootstrapToken,
      ),
      stdio: 'ignore',
      windowsHide: true,
    },
  )
  backendProcess.once('error', () => {
    spawnFailed = true
  })
  backendProcess.once('exit', () => {
    backendProcess = null
    campaignBridgeClient.dispose()
    lastCampaignBackendError = new Error('BashGym managed backend exited')
    campaignBackendReadiness.invalidate()
  })
  const deadline = Date.now() + MANAGED_BACKEND_STARTUP_TIMEOUT_MS
  while (Date.now() < deadline) {
    if (spawnFailed || backendProcess?.exitCode !== null) {
      stopManagedBackend()
      throw new Error('BashGym managed backend exited during startup')
    }
    if (await backendIsReachable()) return
    await new Promise((resolve) => setTimeout(resolve, 150))
  }
  stopManagedBackend()
  throw new Error('BashGym managed backend did not become ready')
}

const campaignBackendReadiness = createRetryableInitializer(async () => {
  try {
    if (runtimeEndpointConfigurationError) throw runtimeEndpointConfigurationError
    await startManagedBackend()
    await campaignBridgeClient.initialize()
    lastCampaignBackendError = null
  } catch (error) {
    lastCampaignBackendError = asError(error)
    throw error
  }
})

function initializeManagedCampaignBackend(): Promise<void> {
  return campaignBackendReadiness.ensureReady()
}

function stopManagedBackend(): void {
  campaignBridgeClient.dispose()
  campaignBackendReadiness.invalidate()
  if (!backendProcess) return
  try { backendProcess.kill() } catch { /* already stopped */ }
  backendProcess = null
}

// Credentials directory for secure storage
const credentialsDir = path.join(app.getPath('userData'), 'credentials')

function resolveAppIconPath(): string | undefined {
  const candidates = isDev
    ? [
        path.join(process.cwd(), 'public', 'bashgym-peony.png'),
        path.join(__dirname, '../public/bashgym-peony.png')
      ]
    : [
        path.join(__dirname, '../dist/bashgym-peony.png'),
        path.join(process.resourcesPath, 'app.asar', 'dist', 'bashgym-peony.png')
      ]

  return candidates.find((candidate) => fs.existsSync(candidate))
}

function setupMenu() {
  const template: Electron.MenuItemConstructorOptions[] = [
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
      ]
    },
    {
      label: 'View',
      submenu: [
        // Reload deliberately omitted — Ctrl+R kills terminals and wipes state
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { role: 'togglefullscreen' },
      ]
    }
  ]
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

function createWindow() {
  const isWin = process.platform === 'win32'
  const appIconPath = resolveAppIconPath()

  mainWindow = new BrowserWindow({
    show: false,
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: 'Bash Gym',
    titleBarStyle: isWin ? 'hidden' : 'hiddenInset',
    ...(isWin ? {} : { trafficLightPosition: { x: 16, y: 16 } }),
    ...(appIconPath ? { icon: appIconPath } : {}),
    backgroundColor: '#0D0D0D',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      additionalArguments: [
        `--bashgym-api-base=${desktopRuntimeEndpoints.apiBase}`,
        `--bashgym-websocket-url=${desktopRuntimeEndpoints.webSocketUrl}`,
      ],
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false, // Required for node-pty
      webviewTag: true // Required for browser panel screenshots
    }
  })

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show()
    mainWindow?.webContents.invalidate()
  })

  // Content Security Policy
  const devSources = isDev ? " 'unsafe-inline' 'unsafe-eval' http://localhost:*" : " 'unsafe-inline'"
  const csp = [
    `default-src 'self'`,
    `script-src 'self'${devSources}`,
    `style-src 'self' 'unsafe-inline' https://fonts.googleapis.com`,
    `font-src 'self' https://fonts.gstatic.com`,
    `img-src 'self' data: blob: https:`,
    `connect-src 'self' http://localhost:* ws://localhost:* http://127.0.0.1:* ws://127.0.0.1:* https://api.vercel.com https://api.neon.tech https://fonts.googleapis.com https://fonts.gstatic.com`,
    `worker-src 'self' blob:`,
    `child-src 'self'`,
    `frame-src 'self' https:`,
  ].join('; ')

  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [csp]
      }
    })
  })

  let rendererHasLoaded = false
  mainWindow.webContents.on('did-start-navigation', (_event, _url, _isInPlace, isMainFrame) => {
    if (rendererHasLoaded && isMainFrame) {
      void teardownAllCampaignAgents('renderer_reload').catch(() => undefined)
    }
  })
  mainWindow.webContents.on('render-process-gone', () => {
    void teardownAllCampaignAgents('renderer_reload').catch(() => undefined)
  })

  // Load the app
  if (isDev) {
    mainWindow.loadURL(desktopRuntimeEndpoints.devServerUrl)
    // DevTools adds substantial input-tail latency to xterm/ConPTY traffic.
    // Keep it available for targeted debugging without attaching it to every launch.
    if (process.env.BASHGYM_OPEN_DEVTOOLS === '1') {
      mainWindow.webContents.openDevTools({ mode: 'detach' })
    }
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  // Inject electron-window class for platform-specific styling (e.g. window border on Windows)
  mainWindow.webContents.on('did-finish-load', () => {
    rendererHasLoaded = true
    mainWindow?.webContents.invalidate()
    mainWindow?.webContents.executeJavaScript(
      `document.documentElement.classList.add('electron-window')`
    )
  })

  // Block Ctrl+R / Ctrl+Shift+R / F5 from refreshing the app — kills terminals and wipes state
  mainWindow.webContents.on('before-input-event', (_event, input) => {
    if (input.type !== 'keyDown') return

    const isRefresh =
      (input.key === 'r' && input.control && !input.alt) ||
      (input.key === 'R' && input.control && input.shift && !input.alt) ||
      (input.key === 'F5' && !input.control && !input.alt)

    if (isRefresh) {
      // Forward Ctrl+R to renderer so browser panes can handle it
      mainWindow?.webContents.send('app-keydown', {
        key: input.key,
        ctrlKey: input.control,
        shiftKey: input.shift
      })
      _event.preventDefault()
    }
  })

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  mainWindow.on('closed', () => {
    void teardownAllCampaignAgents('renderer_reload').catch(() => undefined)
    mainWindow = null
    // PTY processes intentionally survive window close — they are cleaned up on app quit.
  })
}

// Only initialize if we got the lock
if (gotTheLock) {
  let campaignAgentShutdownStarted = false
  let campaignAgentShutdownReady = false
  // Handle second instance - focus existing window
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })

  // App lifecycle
  app.whenReady().then(() => {
    setupMenu()
    createWindow()
    void initializeManagedCampaignBackend().catch((error) => {
      // The app remains usable for non-campaign work; campaign IPC fails closed.
      console.error('BashGym managed campaign backend failed to initialize:', asError(error).message)
    })

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow()
      }
    })
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit()
    }
  })

  app.on('before-quit', (event) => {
    if (campaignAgentShutdownReady) return
    event.preventDefault()
    if (campaignAgentShutdownStarted) return
    campaignAgentShutdownStarted = true
    void teardownAllCampaignAgents('app_shutdown')
      .catch(() => undefined)
      .finally(() => {
        campaignAgentShutdownReady = true
        app.quit()
      })
  })

  app.on('will-quit', () => {
    stopManagedBackend()
    killAllPtys()
  })

  // Unclean-exit guards: ensure no orphaned PTY processes
  process.on('exit', () => { stopManagedBackend(); killAllPtys() })
  process.on('SIGTERM', () => { app.quit() })
  process.on('SIGINT', () => { app.quit() })
}

// IPC Handlers

ipcMain.handle(
  'campaign:request',
  async (
    event,
    method: CampaignMethod,
    route: string,
    body?: CampaignBody,
    query?: CampaignQuery,
    authority?: CampaignRequestAuthority,
  ) => {
    if (!mainWindow || event.sender !== mainWindow.webContents) {
      return { ok: false, status: 403, error: 'Campaign request sender is not permitted' }
    }
    try {
      await initializeManagedCampaignBackend()
    } catch {
      const sourceRootInvalid = lastCampaignBackendError?.message === 'The configured BashGym backend root is invalid'
      const configurationInvalid = sourceRootInvalid || runtimeEndpointConfigurationError !== null
      return {
        ok: false,
        status: 503,
        code: configurationInvalid
          ? 'campaign_backend_configuration_invalid'
          : 'campaign_backend_unavailable',
        error: sourceRootInvalid
          ? 'The configured BashGym source checkout is invalid. Remove BASHGYM_PROJECT_ROOT to use the package installed in BASHGYM_PYTHON, or point it at a valid checkout.'
          : configurationInvalid
            ? 'The configured BashGym API or development-server endpoint is invalid. Use a credential-free loopback URL and set BASHGYM_API_BASE to an /api URL.'
          : 'The authenticated campaign service could not start or reconnect.',
      }
    }
    try {
      return await campaignBridgeClient.request(method, route, body, query, authority)
    } catch {
      return { ok: false, status: 400, error: 'Campaign request was rejected' }
    }
  },
)

function campaignAgentHostSenderPermitted(event: Electron.IpcMainInvokeEvent): boolean {
  return Boolean(mainWindow && event.sender === mainWindow.webContents)
}

function campaignAgentHostFailure(error = 'Campaign agent host request was rejected') {
  return { success: false, error }
}

interface CampaignAgentLaunchRequest {
  workspaceId: string
  campaignId: string
  cwd?: string
}

interface CampaignAgentScopeRequest {
  workspaceId: string
  campaignId: string
}

interface CampaignAgentTerminalScopeRequest extends CampaignAgentScopeRequest {
  terminalId: string
}

function assertCampaignAgentScopeRequest(value: unknown): CampaignAgentScopeRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid campaign agent scope request')
  }
  const prototype = Object.getPrototypeOf(value)
  const record = value as Record<string, unknown>
  const keys = Object.keys(record)
  if ((prototype !== Object.prototype && prototype !== null)
    || keys.length !== 2
    || !Object.hasOwn(record, 'workspaceId')
    || !Object.hasOwn(record, 'campaignId')
    || typeof record.workspaceId !== 'string'
    || typeof record.campaignId !== 'string') {
    throw new Error('Invalid campaign agent scope request')
  }
  return { workspaceId: record.workspaceId, campaignId: record.campaignId }
}

function assertCampaignAgentLaunchRequest(value: unknown): CampaignAgentLaunchRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid campaign agent launch request')
  }
  const prototype = Object.getPrototypeOf(value)
  if (prototype !== Object.prototype && prototype !== null) {
    throw new Error('Invalid campaign agent launch request')
  }
  const record = value as Record<string, unknown>
  const keys = Object.keys(record)
  if (!Object.hasOwn(record, 'workspaceId')
    || !Object.hasOwn(record, 'campaignId')
    || keys.some((key) => !['workspaceId', 'campaignId', 'cwd'].includes(key))) {
    throw new Error('Invalid campaign agent launch request')
  }
  if (typeof record.workspaceId !== 'string'
    || typeof record.campaignId !== 'string'
    || (record.cwd !== undefined && typeof record.cwd !== 'string')) {
    throw new Error('Invalid campaign agent launch request')
  }
  return {
    workspaceId: record.workspaceId,
    campaignId: record.campaignId,
    ...(record.cwd !== undefined ? { cwd: record.cwd } : {}),
  }
}

function assertCampaignAgentTerminalScopeRequest(value: unknown): CampaignAgentTerminalScopeRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid campaign agent terminal request')
  }
  const prototype = Object.getPrototypeOf(value)
  const record = value as Record<string, unknown>
  const keys = Object.keys(record)
  if ((prototype !== Object.prototype && prototype !== null)
    || keys.length !== 3
    || !Object.hasOwn(record, 'terminalId')
    || !Object.hasOwn(record, 'workspaceId')
    || !Object.hasOwn(record, 'campaignId')
    || typeof record.terminalId !== 'string'
    || typeof record.workspaceId !== 'string'
    || typeof record.campaignId !== 'string') {
    throw new Error('Invalid campaign agent terminal request')
  }
  return {
    terminalId: record.terminalId,
    workspaceId: record.workspaceId,
    campaignId: record.campaignId,
  }
}

function assertActiveCampaignAgentTerminal(terminalId: string): void {
  if (!campaignAgentMcpHosts.has(terminalId) || !campaignAgentLaunchScopes.has(terminalId)) {
    throw new Error('Campaign agent MCP host is unavailable')
  }
}

function assertActiveCampaignAgentRequest(value: unknown): void {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid campaign agent request')
  }
  const record = value as Record<string, unknown>
  if (typeof record.terminalId !== 'string'
    || typeof record.workspaceId !== 'string'
    || typeof record.campaignId !== 'string') {
    throw new Error('Invalid campaign agent request')
  }
  assertActiveCampaignAgentTerminal(record.terminalId)
  const scope = campaignAgentLaunchScopes.get(record.terminalId)
  if (!scope
    || scope.workspaceId !== record.workspaceId
    || scope.campaignId !== record.campaignId) {
    throw new Error('Campaign agent request scope changed')
  }
}

function executableFile(candidate: string): boolean {
  try {
    return fs.statSync(candidate).isFile()
  } catch {
    return false
  }
}

function resolveCodexExecutable(): string {
  const executableName = process.platform === 'win32' ? 'codex.exe' : 'codex'
  const configured = process.env.BASHGYM_CODEX_EXECUTABLE?.trim()
  if (configured) {
    if (!path.isAbsolute(configured)
      || path.basename(configured).toLowerCase() !== executableName
      || !executableFile(configured)) {
      throw new Error('Configured Codex CLI executable is unavailable')
    }
    return configured
  }
  const pathValue = process.env.PATH ?? process.env.Path ?? ''
  for (const entry of pathValue.split(path.delimiter)) {
    const directory = entry.trim().replace(/^"|"$/g, '')
    if (!directory) continue
    const candidate = path.resolve(directory, executableName)
    if (executableFile(candidate)) return candidate
  }
  throw new Error('Codex CLI executable was not found')
}

ipcMain.handle('campaign-agent-host:launch', async (event, request: unknown) => {
  if (!campaignAgentHostSenderPermitted(event)) {
    return campaignAgentHostFailure('Campaign agent host sender is not permitted')
  }
  let host: CampaignAgentMcpHost | null = null
  let ptyProcess: IPty | null = null
  let launchedTerminalId: string | null = null
  try {
    const intent = assertCampaignAgentLaunchRequest(request)
    await initializeManagedCampaignBackend()
    const terminalId = `campaign_codex_${randomUUID().replaceAll('-', '')}`
    launchedTerminalId = terminalId
    const generation = `ptygen_${randomUUID().replaceAll('-', '')}`
    const resolvedCwd = resolveTerminalCwd(intent.cwd)
    host = new CampaignAgentMcpHost({
      terminalId,
      generation,
      scope: { workspaceId: intent.workspaceId, campaignId: intent.campaignId },
      observe: () => campaignAgentHostController.observe(terminalId),
      artifacts: (args) => campaignAgentHostController.artifacts(terminalId, args),
    })
    const mcpLaunch = await host.start()
    const launch = buildCodexCampaignAgentLaunch({
      intent: { ...intent, cwd: resolvedCwd },
      terminalId,
      generation,
      hostInstanceId: campaignAgentHostInstanceId,
      mcpLaunch,
    }, {
      resolveCodexExecutable,
      pathExists: fs.existsSync,
      defaultCwd: os.homedir(),
      sourceEnv: process.env,
    })
    const pty = await import('node-pty')
    ptyProcess = pty.spawn(launch.executable, [...launch.args], {
      name: 'xterm-256color',
      cols: 80,
      rows: 24,
      cwd: launch.cwd,
      env: { ...launch.env },
    })
    campaignAgentMcpHosts.set(terminalId, host)
    campaignAgentLaunchScopes.set(terminalId, {
      workspaceId: intent.workspaceId,
      campaignId: intent.campaignId,
    })
    registerPtySession(
      terminalId,
      ptyProcess,
      launch.cwd,
      generation,
      launch.identity,
    )
    return { success: true, terminalId, cwd: launch.cwd }
  } catch {
    try { ptyProcess?.kill() } catch { /* launch did not produce a live PTY */ }
    if (launchedTerminalId && campaignAgentMcpHosts.has(launchedTerminalId)) {
      await closeCampaignAgentMcpHost(launchedTerminalId).catch(() => undefined)
    } else {
      host?.lock()
      await host?.close().catch(() => undefined)
    }
    return campaignAgentHostFailure(
      'Codex campaign agent could not start. Ensure the Codex CLI is installed and the workspace is accessible.',
    )
  }
})

ipcMain.handle('campaign-agent-host:eligible', (event, request: unknown) => {
  if (!campaignAgentHostSenderPermitted(event)) return campaignAgentHostFailure('Campaign agent host sender is not permitted')
  try {
    const selectedScope = assertCampaignAgentScopeRequest(request)
    const identities = [...ptySessions.values()]
      .map((session) => session.campaignAgentIdentity
        ? { ...session.campaignAgentIdentity, live: !session.exited }
        : null)
      .filter((identity): identity is MainOwnedCampaignAgentIdentity => identity !== null)
      .filter((identity) => {
        const scope = campaignAgentLaunchScopes.get(identity.terminalId)
        return Boolean(scope
          && scope.workspaceId === selectedScope.workspaceId
          && scope.campaignId === selectedScope.campaignId)
      })
    return { success: true, sessions: campaignAgentHostController.eligibleSessions(identities) }
  } catch {
    return campaignAgentHostFailure()
  }
})

ipcMain.handle('campaign-agent-host:attach', async (event, request: unknown) => {
  if (!campaignAgentHostSenderPermitted(event)) return campaignAgentHostFailure('Campaign agent host sender is not permitted')
  try {
    assertActiveCampaignAgentRequest(request)
    await initializeManagedCampaignBackend()
    const status = await campaignAgentHostController.attach(request as never)
    return { success: true, status }
  } catch {
    return campaignAgentHostFailure()
  }
})

ipcMain.handle('campaign-agent-host:activate', async (event, request: unknown) => {
  if (!campaignAgentHostSenderPermitted(event)) return campaignAgentHostFailure('Campaign agent host sender is not permitted')
  try {
    const { terminalId, workspaceId, campaignId } = assertCampaignAgentTerminalScopeRequest(request)
    assertActiveCampaignAgentRequest({ terminalId, workspaceId, campaignId })
    await initializeManagedCampaignBackend()
    const current = campaignAgentHostController.status(terminalId)
    if (current?.state !== 'credential_ready') {
      await campaignAgentHostController.claim(terminalId)
    }
    const status = await campaignAgentHostController.activate(terminalId)
    return { success: true, status }
  } catch {
    return campaignAgentHostFailure()
  }
})

ipcMain.handle('campaign-agent-host:authorize', async (event, request: unknown) => {
  if (!campaignAgentHostSenderPermitted(event)) return campaignAgentHostFailure('Campaign agent host sender is not permitted')
  try {
    assertActiveCampaignAgentRequest(request)
    await initializeManagedCampaignBackend()
    const status = await campaignAgentHostController.authorize(request as never)
    return { success: true, status }
  } catch {
    return campaignAgentHostFailure()
  }
})

ipcMain.handle('campaign-agent-host:revoke', async (event, request: unknown) => {
  if (!campaignAgentHostSenderPermitted(event)) return campaignAgentHostFailure('Campaign agent host sender is not permitted')
  try {
    const { terminalId, workspaceId, campaignId } = assertCampaignAgentTerminalScopeRequest(request)
    assertActiveCampaignAgentRequest({ terminalId, workspaceId, campaignId })
    await revokeCampaignAgentTerminal(terminalId, 'explicit_revoke')
    return { success: true }
  } catch {
    return campaignAgentHostFailure()
  }
})

function safeBridgeArgument(value: string, label: string): string {
  if (!value || value.length > 512 || /[\r\n"&|<>^%]/.test(value)) {
    throw new Error(`Invalid ${label}`)
  }
  return value
}

function writeAgentBridgeWrapper(request: AgentBridgeLaunchRequest): string {
  const pythonCommand = safeBridgeArgument(
    request.pythonCommand || process.env.BASHGYM_PYTHON || 'python',
    'Python command'
  )
  const args = [
    '-m',
    'bashgym.mcp.skill_lab_server',
    '--workspace-id',
    safeBridgeArgument(request.workspaceId || 'default', 'workspace id'),
    '--origin-terminal-id',
    safeBridgeArgument(request.terminalId, 'terminal id'),
    '--agent',
    request.kind,
  ]
  if (request.panelId) {
    args.push('--origin-panel-id', safeBridgeArgument(request.panelId, 'panel id'))
  }
  const apiBase = request.apiBase || desktopRuntimeEndpoints.apiOrigin
  args.push('--api-base', safeBridgeArgument(apiBase, 'API URL'))

  const bridgeDir = path.join(os.homedir(), '.bashgym', 'agent_bridge')
  fs.mkdirSync(bridgeDir, { recursive: true })
  const basename = request.terminalId.replace(/[^A-Za-z0-9._-]/g, '_')
  const wrapperPath = path.join(bridgeDir, `${basename}.${process.platform === 'win32' ? 'cmd' : 'sh'}`)
  const quote = process.platform === 'win32'
    ? (value: string) => `"${value}"`
    : (value: string) => `'${value.replaceAll("'", `'"'"'`)}'`
  const quotedPython = quote(pythonCommand)
  const quotedArgs = args.map(quote).join(' ')
  const body = process.platform === 'win32'
    ? `@echo off\r\n${quotedPython} ${quotedArgs}\r\n`
    : `#!/bin/sh\nexec ${quotedPython} ${quotedArgs}\n`
  fs.writeFileSync(wrapperPath, body, { encoding: 'utf-8', mode: 0o700 })
  registerAgentBridgeArtifact(request.terminalId, wrapperPath)
  return wrapperPath
}

function writeClaudeAgentBridgeConfig(
  request: AgentBridgeLaunchRequest,
  wrapperPath: string,
): string {
  const bridgeDir = path.dirname(wrapperPath)
  const basename = request.terminalId.replace(/[^A-Za-z0-9._-]/g, '_')
  const configPath = path.join(bridgeDir, `${basename}.claude-mcp.json`)
  fs.writeFileSync(configPath, JSON.stringify({
    mcpServers: {
      bashgym: {
        command: wrapperPath,
        args: [],
      },
    },
  }, null, 2), 'utf-8')
  registerAgentBridgeArtifact(request.terminalId, configPath)
  return configPath
}

ipcMain.handle('agent-bridge:prepare-launch', (event, request: AgentBridgeLaunchRequest) => {
  try {
    if (!mainWindow || event.sender !== mainWindow.webContents) {
      throw new Error('Agent bridge sender is not permitted')
    }
    if (!request || !['claude', 'codex'].includes(request.kind)) {
      throw new Error('Unsupported agent bridge kind')
    }
    if (!request.terminalId?.trim()) throw new Error('Terminal id is required')
    const scopedRequest = {
      ...request,
      workspaceId: request.workspaceId?.trim() || 'default',
      apiBase: request.apiBase || desktopRuntimeEndpoints.apiOrigin,
      pythonCommand: request.pythonCommand || process.env.BASHGYM_PYTHON || 'python',
    }
    const wrapperPath = writeAgentBridgeWrapper(scopedRequest)
    const claudeConfigPath = scopedRequest.kind === 'claude'
      ? writeClaudeAgentBridgeConfig(scopedRequest, wrapperPath)
      : undefined
    const command = buildAgentBridgeLaunchCommand({
      ...scopedRequest,
      serverCommand: wrapperPath,
      serverArgs: [],
      claudeConfigPath,
    })
    return {
      success: true,
      command,
    }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

// Get fresh environment with updated PATH (especially important on Windows)
function getFreshEnv(): Record<string, string> {
  const env = { ...process.env } as Record<string, string>

  // Remove Claude Code's nesting guard so terminals can launch claude freely
  delete env.CLAUDECODE
  delete env.BASHGYM_DESKTOP_BOOTSTRAP_SECRET

  if (process.platform === 'win32') {
    // On Windows, add common tool paths that might have been installed after app launch
    const userProfile = process.env.USERPROFILE || ''
    const localAppData = process.env.LOCALAPPDATA || ''

    const additionalPaths = [
      `${localAppData}\\Programs\\Ollama`,
      `${localAppData}\\Programs\\OpenCode`,
      `${userProfile}\\.local\\bin`,
      `${userProfile}\\scoop\\shims`,
    ].filter(p => p)

    // Prepend to PATH so new tools are found first
    const currentPath = env.PATH || env.Path || ''
    env.PATH = [...additionalPaths, currentPath].join(';')
    env.Path = env.PATH
  }

  return env
}

function resolveTerminalCwd(cwd?: string): string {
  const requested = cwd?.trim()
  if (!requested || requested === '~') return os.homedir()
  if (requested.startsWith('~/') || requested.startsWith('~\\')) {
    return path.join(os.homedir(), requested.slice(2))
  }
  return requested
}

function registerPtySession(
  id: string,
  ptyProcess: IPty,
  resolvedCwd: string,
  generation: string,
  campaignAgentIdentity?: MainOwnedCampaignAgentIdentity,
): PtySession {
  const outputFlow = new TerminalOutputFlowController(
    (data, frameId) => mainWindow?.webContents.send(
      `terminal:data:${id}`,
      frameId === undefined ? data : { data, frameId },
    ),
    (paused) => {
      try {
        if (paused) ptyProcess.pause()
        else ptyProcess.resume()
      } catch {
        // The PTY may exit while a renderer acknowledgement is in flight.
      }
    },
  )
  const session: PtySession = {
    pty: ptyProcess,
    scrollback: new TerminalScrollbackBuffer(MAX_BUFFER_BYTES),
    outputBatcher: new TerminalIpcBatcher((data) => outputFlow.push(data)),
    outputFlow,
    outputFlowOwnership: new TerminalOutputFlowOwnership(outputFlow),
    size: new TerminalSizeTracker(80, 24),
    cwd: resolvedCwd,
    createdAt: Date.now(),
    exited: false,
    exitCode: null,
    generation,
    ...(campaignAgentIdentity ? { campaignAgentIdentity } : {}),
  }
  ptySessions.set(id, session)

  ptyProcess.onData((data: string) => {
    session.scrollback.append(data)
    session.outputBatcher.push(data)
  })
  ptyProcess.onExit(({ exitCode }) => {
    session.outputBatcher.dispose(true)
    session.exited = true
    session.exitCode = exitCode
    mainWindow?.webContents.send(`terminal:exit:${id}`, exitCode)
    void revokeCampaignAgentTerminal(id, 'pty_exit').catch(() => undefined)
    // Keep the entry and scrollback so a renderer can display the exit.
  })
  return session
}

// Terminal management — create-or-attach: a live PTY for this id is re-attached
// (with scrollback replay) instead of respawned, so renderer remounts never
// orphan or kill the underlying process.
// Creation is single-flight per terminal ID: the handler awaits between the
// existence check and registry insertion, so concurrent invokes (double
// restart clicks, remount races) would otherwise both spawn a PTY and orphan
// one of them.
const terminalCreateInFlight = new Map<string, Promise<unknown>>()

ipcMain.handle('terminal:create', (_, id: string, cwd?: string) => {
  const pending = terminalCreateInFlight.get(id)
  if (pending) return pending
  const create = createOrAttachTerminal(id, cwd).finally(() => {
    terminalCreateInFlight.delete(id)
  })
  terminalCreateInFlight.set(id, create)
  return create
})

async function createOrAttachTerminal(id: string, cwd?: string) {
  try {
    const existing = ptySessions.get(id)
    if (existing && !existing.exited) {
      existing.outputBatcher.flush()
      return {
        success: true,
        id,
        attached: true,
        buffer: existing.scrollback.toString(),
        cwd: existing.cwd
      }
    }
    // A dead session being recreated: drop the stale entry
    if (existing) {
      await revokeCampaignAgentTerminal(id, 'pty_replacement').catch(() => undefined)
      existing.outputBatcher.dispose()
      existing.outputFlow.dispose()
      ptySessions.delete(id)
      cleanupAgentBridgeArtifacts(id)
    }

    // Dynamic import for node-pty (native module)
    const pty = await import('node-pty')

    const shell = process.platform === 'win32'
      ? 'powershell.exe'
      : process.env.SHELL || '/bin/bash'

    // The embedded terminal favors immediate echo. Set
    // BASHGYM_ENABLE_PSREADLINE=1 to restore PowerShell's advanced line editor.
    const shellArgs = process.platform === 'win32'
      ? buildPowerShellArgs(process.env.BASHGYM_ENABLE_PSREADLINE === '1')
      : []

    const resolvedCwd = resolveTerminalCwd(cwd)

    const env = {
      ...getFreshEnv(),
      BASHGYM_TERMINAL_ID: id
    }

    const ptyProcess = pty.spawn(shell, shellArgs, {
      name: 'xterm-256color',
      cols: 80,
      rows: 24,
      cwd: resolvedCwd,
      env
    })

    registerPtySession(
      id,
      ptyProcess,
      resolvedCwd,
      `ptygen_${randomUUID().replaceAll('-', '')}`,
    )

    return { success: true, id, attached: false, cwd: resolvedCwd }
  } catch (error) {
    console.error('Failed to create terminal:', error)
    return { success: false, error: String(error) }
  }
}

function writeTerminalInput(id: string, data: string): boolean {
  const session = ptySessions.get(id)
  if (session && !session.exited) {
    session.pty.write(data)
    return true
  }
  return false
}

// New preload versions use fire-and-forget input so a renderer under TUI paint
// load never waits for an IPC response. Keep invoke support for renderer reloads
// that are still running the previous preload contract.
ipcMain.on('terminal:write', (_, id: string, data: string) => {
  writeTerminalInput(id, data)
})

ipcMain.handle('terminal:write', (_, id: string, data: string) => {
  return writeTerminalInput(id, data)
})

// Opt-in parser flow control keeps the existing terminal:data payload contract
// compatible with renderer/preload hot reloads. Old renderers never enable it.
ipcMain.on('terminal:output-flow', (_, id: string, owner: string, enabled: boolean) => {
  const session = ptySessions.get(id)
  if (!session) return
  if (enabled) session.outputFlowOwnership.acquire(owner)
  else session.outputFlowOwnership.release(owner)
})

ipcMain.on('terminal:output-ack', (_, id: string, owner: string, frameId: number) => {
  const session = ptySessions.get(id)
  session?.outputFlowOwnership.acknowledge(owner, frameId)
})

ipcMain.handle('terminal:resize', (_, id: string, cols: number, rows: number) => {
  const session = ptySessions.get(id)
  if (session && !session.exited) {
    if (session.size.update(cols, rows)) session.pty.resize(cols, rows)
    return true
  }
  return false
})

ipcMain.handle('terminal:kill', async (_, id: string) => {
  const session = ptySessions.get(id)
  if (session) {
    await revokeCampaignAgentTerminal(id, 'pty_exit').catch(() => undefined)
    session.outputBatcher.dispose()
    session.outputFlow.dispose()
    try { session.pty.kill() } catch { /* already dead */ }
    ptySessions.delete(id)
    cleanupAgentBridgeArtifacts(id)
    return true
  }
  cleanupAgentBridgeArtifacts(id)
  return false
})

// Enumerate sessions so a reloaded renderer can re-adopt live PTYs
ipcMain.handle('terminal:list', () => {
  return Array.from(ptySessions.entries()).map(([id, s]) => ({
    id,
    cwd: s.cwd,
    createdAt: s.createdAt,
    exited: s.exited,
    exitCode: s.exitCode
  }))
})

// Read-only tail of the scrollback ring buffer (raw ANSI; renderer strips)
ipcMain.handle('terminal:snapshot', (_, id: string, maxBytes?: number) => {
  const session = ptySessions.get(id)
  if (!session) return { success: false, error: 'No such terminal session' }
  const cap = Math.min(Math.max(maxBytes ?? 64_000, 1_000), MAX_BUFFER_BYTES)
  return {
    success: true,
    data: session.scrollback.tail(cap),
    cwd: session.cwd,
    exited: session.exited
  }
})

// System info
ipcMain.handle('system:info', () => {
  return {
    platform: process.platform,
    arch: process.arch,
    nodeVersion: process.version,
    electronVersion: process.versions.electron,
    cwd: process.cwd()
  }
})

// Window controls (for frameless window on Windows)
ipcMain.handle('window:minimize', () => {
  mainWindow?.minimize()
})

ipcMain.handle('window:maximize', () => {
  if (mainWindow?.isMaximized()) {
    mainWindow.unmaximize()
  } else {
    mainWindow?.maximize()
  }
})

ipcMain.handle('window:close', () => {
  mainWindow?.close()
})

ipcMain.handle('window:isMaximized', () => {
  return mainWindow?.isMaximized() ?? false
})

// Theme persistence
let currentTheme: 'light' | 'dark' = 'dark'

ipcMain.handle('theme:get', () => currentTheme)

ipcMain.handle('theme:set', (_, theme: 'light' | 'dark') => {
  currentTheme = theme
  return currentTheme
})

// API proxy (for development - avoids CORS issues)
ipcMain.handle('api:fetch', async (_, url: string, options?: RequestInit) => {
  try {
    // Ensure Content-Type header is set for POST/PUT requests with body
    const fetchOptions: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.BASHGYM_API_KEY || '',
        ...(options?.headers || {})
      }
    }
    const response = await fetch(url, fetchOptions)
    const text = await response.text()
    try {
      const data = JSON.parse(text)
      return { ok: response.ok, status: response.status, data }
    } catch {
      if (response.ok) return { ok: true, status: response.status, data: text }
      return { ok: false, status: response.status, error: text || `HTTP ${response.status}` }
    }
  } catch (error) {
    return { ok: false, error: String(error) }
  }
})

type ApiStreamEvent =
  | { type: 'headers'; ok: boolean; status: number }
  | { type: 'chunk'; data: string }
  | { type: 'done' }
  | { type: 'aborted' }
  | { type: 'error'; error: string }

const apiStreamControllers = new Map<string, AbortController>()

ipcMain.on(
  'api:stream:start',
  async (event, requestId: string, url: string, options?: RequestInit) => {
    apiStreamControllers.get(requestId)?.abort()
    const controller = new AbortController()
    apiStreamControllers.set(requestId, controller)

    const send = (payload: ApiStreamEvent) => {
      if (!event.sender.isDestroyed()) {
        event.sender.send('api:stream:event', requestId, payload)
      }
    }

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': process.env.BASHGYM_API_KEY || '',
          ...(options?.headers || {})
        }
      })
      send({ type: 'headers', ok: response.ok, status: response.status })

      if (!response.body) {
        send({ type: 'error', error: 'API stream returned no response body' })
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const data = decoder.decode(value, { stream: true })
        if (data) send({ type: 'chunk', data })
      }
      const remaining = decoder.decode()
      if (remaining) send({ type: 'chunk', data: remaining })
      send({ type: 'done' })
    } catch (error) {
      if (controller.signal.aborted) {
        send({ type: 'aborted' })
      } else {
        send({ type: 'error', error: String(error) })
      }
    } finally {
      if (apiStreamControllers.get(requestId) === controller) {
        apiStreamControllers.delete(requestId)
      }
    }
  }
)

ipcMain.on('api:stream:cancel', (_, requestId: string) => {
  apiStreamControllers.get(requestId)?.abort()
})

// File system handlers

interface FileInfo {
  name: string
  path: string
  type: 'file' | 'directory'
  size?: number
  modified?: number
}

ipcMain.handle('files:readDirectory', async (_, dirPath: string) => {
  try {
    const resolvedPath = dirPath.startsWith('~')
      ? path.join(os.homedir(), dirPath.slice(1))
      : dirPath

    const entries = await fs.promises.readdir(resolvedPath, { withFileTypes: true })

    const files: FileInfo[] = []

    for (const entry of entries) {
      // Skip hidden files on Unix (starting with .)
      if (entry.name.startsWith('.')) continue

      const fullPath = path.join(resolvedPath, entry.name)

      try {
        const stats = await fs.promises.stat(fullPath)
        files.push({
          name: entry.name,
          path: fullPath,
          type: entry.isDirectory() ? 'directory' : 'file',
          size: entry.isFile() ? stats.size : undefined,
          modified: stats.mtimeMs
        })
      } catch {
        // Skip files we can't stat (permission issues, etc.)
      }
    }

    // Sort: directories first, then by name
    files.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === 'directory' ? -1 : 1
      }
      return a.name.localeCompare(b.name)
    })

    return { success: true, files }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

ipcMain.handle('files:getHomeDirectory', () => {
  return os.homedir()
})

ipcMain.handle('files:getParentDirectory', (_, filePath: string) => {
  return path.dirname(filePath)
})

ipcMain.handle('files:readFile', async (_, filePath: string) => {
  try {
    const content = await fs.promises.readFile(filePath, 'utf-8')
    return { success: true, content }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

ipcMain.handle('files:exists', async (_, filePath: string) => {
  try {
    await fs.promises.access(filePath)
    return true
  } catch {
    return false
  }
})

// Browser screenshot via webContentsId — reliable alternative to webview.capturePage() in renderer
ipcMain.handle('browser:screenshot', async (_, webContentsId: number, rect?: { x: number; y: number; width: number; height: number; vpW: number; vpH: number }) => {
  try {
    const wc = webContents.fromId(webContentsId)
    if (!wc) return { success: false, error: 'WebContents not found' }
    const fullImage = await wc.capturePage()
    if (rect) {
      // Scale CSS pixel bounds to device pixels using viewport size
      const { width: imgW, height: imgH } = fullImage.getSize()
      const scaleX = imgW / rect.vpW
      const scaleY = imgH / rect.vpH
      const cropped = fullImage.crop({
        x: Math.round(rect.x * scaleX),
        y: Math.round(rect.y * scaleY),
        width: Math.max(1, Math.round(rect.width * scaleX)),
        height: Math.max(1, Math.round(rect.height * scaleY))
      })
      return { success: true, dataUrl: cropped.toDataURL() }
    }
    return { success: true, dataUrl: fullImage.toDataURL() }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

ipcMain.handle('files:writeTempFile', async (_, dataUrl: string, ext: string, basename?: string) => {
  try {
    const safeBase = (basename || 'bashgym_file').replace(/[^a-zA-Z0-9_-]/g, '_')
    const filename = `${safeBase}_${Date.now()}.${ext}`
    const filePath = path.join(os.tmpdir(), filename)
    const base64Data = dataUrl.replace(/^data:[^;]+;base64,/, '')
    await fs.promises.writeFile(filePath, Buffer.from(base64Data, 'base64'))
    return { success: true, path: filePath }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

ipcMain.handle('files:stat', async (_, filePath: string) => {
  try {
    const stats = await fs.promises.stat(filePath)
    return {
      success: true,
      stats: {
        isFile: stats.isFile(),
        isDirectory: stats.isDirectory(),
        size: stats.size,
        modified: stats.mtimeMs,
        created: stats.birthtimeMs
      }
    }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

// Session history handlers — read-only access to local agent-CLI session journals
// (Claude Code: ~/.claude/projects, Codex: ~/.codex/sessions). All reads stay on
// this machine; paths are restricted to the session roots and .jsonl files only.

const SESSION_ROOTS = [
  path.join(os.homedir(), '.claude', 'projects'),
  path.join(os.homedir(), '.codex', 'sessions')
]

function safeSessionFilePath(p: string): string {
  const resolved = path.resolve(p)
  const inRoot = SESSION_ROOTS.some((root) => resolved.startsWith(root + path.sep))
  if (!inRoot || !resolved.endsWith('.jsonl')) {
    throw new Error('Path outside session roots')
  }
  return resolved
}

interface SessionFileInfo {
  path: string
  size: number
  modified: number
}

async function listJsonlFiles(dir: string): Promise<SessionFileInfo[]> {
  const out: SessionFileInfo[] = []
  let entries: fs.Dirent[]
  try {
    entries = await fs.promises.readdir(dir, { withFileTypes: true })
  } catch {
    return out
  }
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.jsonl')) continue
    const filePath = path.join(dir, entry.name)
    try {
      const stats = await fs.promises.stat(filePath)
      out.push({ path: filePath, size: stats.size, modified: stats.mtimeMs })
    } catch {
      // File vanished between readdir and stat
    }
  }
  return out
}

const SCAN_MAX_FILES_PER_KIND = 400

ipcMain.handle('sessions:scan', async (_, lookbackDays = 14) => {
  try {
    const days = Math.min(Math.max(Number(lookbackDays) || 14, 1), 90)
    const cutoff = Date.now() - days * 86_400_000
    const byRecency = (files: SessionFileInfo[]) =>
      files
        .filter((f) => f.modified >= cutoff)
        .sort((a, b) => b.modified - a.modified)
        .slice(0, SCAN_MAX_FILES_PER_KIND)

    // Claude: every project dir's top-level *.jsonl. Enumerating the root (rather
    // than taking dir names from the renderer) sidesteps path-case mismatches on
    // Windows. Subagent files live in <session-id>/subagents/ subdirs and are
    // excluded by the top-level-only listing.
    const claude: SessionFileInfo[] = []
    const claudeRoot = SESSION_ROOTS[0]
    let projectDirs: fs.Dirent[] = []
    try {
      projectDirs = await fs.promises.readdir(claudeRoot, { withFileTypes: true })
    } catch {
      // No Claude Code installation
    }
    for (const entry of projectDirs) {
      if (!entry.isDirectory()) continue
      claude.push(...await listJsonlFiles(path.join(claudeRoot, entry.name)))
    }

    // Codex: rollout-*.jsonl under date dirs for the last N days
    const codex: SessionFileInfo[] = []
    const codexRoot = SESSION_ROOTS[1]
    for (let i = 0; i < days; i++) {
      const d = new Date(Date.now() - i * 86_400_000)
      const dir = path.join(
        codexRoot,
        String(d.getFullYear()),
        String(d.getMonth() + 1).padStart(2, '0'),
        String(d.getDate()).padStart(2, '0')
      )
      codex.push(...(await listJsonlFiles(dir)).filter((f) => path.basename(f.path).startsWith('rollout-')))
    }

    return { success: true, claude: byRecency(claude), codex: byRecency(codex) }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

const SESSION_TAIL_HARD_CAP = 262_144 // 256KB per read

ipcMain.handle('sessions:readTail', async (_, filePath: string, fromOffset: number, maxBytes = 131_072) => {
  let fd: fs.promises.FileHandle | null = null
  try {
    const resolved = safeSessionFilePath(filePath)
    const cap = Math.min(Math.max(Number(maxBytes) || 131_072, 4_096), SESSION_TAIL_HARD_CAP)
    const stats = await fs.promises.stat(resolved)

    // File truncated or rotated since last read: restart near the end
    let offset = Math.max(Number(fromOffset) || 0, 0)
    let reset = false
    if (stats.size < offset) {
      offset = Math.max(0, stats.size - cap)
      reset = true
    }

    const toRead = Math.min(cap, stats.size - offset)
    if (toRead <= 0) {
      return { success: true, data: '', newOffset: offset, size: stats.size, reset }
    }

    fd = await fs.promises.open(resolved, 'r')
    const buf = Buffer.alloc(toRead)
    const { bytesRead } = await fd.read(buf, 0, toRead, offset)

    // Trim at the last newline so the renderer never sees split lines or
    // split multibyte characters
    const lastNewline = buf.lastIndexOf(0x0a, bytesRead - 1)
    if (lastNewline === -1) {
      if (bytesRead >= cap) {
        // Single line longer than the cap: skip it to guarantee forward progress
        return { success: true, data: '', newOffset: offset + bytesRead, size: stats.size, reset }
      }
      // Partial final line still being written: wait for the next poll
      return { success: true, data: '', newOffset: offset, size: stats.size, reset }
    }

    const data = buf.toString('utf-8', 0, lastNewline + 1)
    return { success: true, data, newOffset: offset + lastNewline + 1, size: stats.size, reset }
  } catch (error) {
    return { success: false, error: String(error) }
  } finally {
    await fd?.close().catch(() => {})
  }
})

ipcMain.handle('sessions:readHead', async (_, filePath: string, maxBytes = 16_384) => {
  let fd: fs.promises.FileHandle | null = null
  try {
    const resolved = safeSessionFilePath(filePath)
    const cap = Math.min(Math.max(Number(maxBytes) || 16_384, 1_024), SESSION_TAIL_HARD_CAP)
    fd = await fs.promises.open(resolved, 'r')
    const buf = Buffer.alloc(cap)
    const { bytesRead } = await fd.read(buf, 0, cap, 0)
    // Trim at the last complete line for safe parsing
    const lastNewline = buf.lastIndexOf(0x0a, bytesRead - 1)
    const end = lastNewline === -1 ? bytesRead : lastNewline + 1
    return { success: true, data: buf.toString('utf-8', 0, end) }
  } catch (error) {
    return { success: false, error: String(error) }
  } finally {
    await fd?.close().catch(() => {})
  }
})

// Account info: parsed in the main process, only a whitelist of fields crosses
// IPC — the raw config file (which contains much more) never leaves this handler.
ipcMain.handle('sessions:readAccount', async () => {
  try {
    const raw = await fs.promises.readFile(path.join(os.homedir(), '.claude.json'), 'utf-8')
    const parsed = JSON.parse(raw)
    const acct = parsed?.oauthAccount
    if (!acct || typeof acct !== 'object') {
      return { success: false, error: 'No account info found' }
    }
    return {
      success: true,
      account: {
        emailAddress: acct.emailAddress ?? null,
        displayName: acct.displayName ?? null,
        organizationName: acct.organizationName ?? null,
        organizationRole: acct.organizationRole ?? null,
        billingType: acct.billingType ?? null,
        seatTier: acct.seatTier ?? null,
        userRateLimitTier: acct.userRateLimitTier ?? null,
        organizationRateLimitTier: acct.organizationRateLimitTier ?? null
      }
    }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

// Clipboard handlers — native Electron clipboard is reliable; navigator.clipboard.write() is not
ipcMain.handle('clipboard:writeImage', (_, dataUrl: string) => {
  try {
    const img = nativeImage.createFromDataURL(dataUrl)
    clipboard.writeImage(img)
    return { success: true }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

ipcMain.handle('clipboard:writeText', (_, text: string) => {
  try {
    clipboard.writeText(text)
    return { success: true }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

// Credential storage handlers — encrypt/decrypt via Electron safeStorage (OS-level encryption)

function safeCredentialPath(key: string): string {
  const sanitized = key.replace(/[^a-zA-Z0-9_-]/g, '')
  if (!sanitized) throw new Error('Invalid credential key')
  const resolved = path.join(credentialsDir, sanitized)
  if (!resolved.startsWith(credentialsDir)) throw new Error('Invalid credential path')
  return resolved
}

ipcMain.handle('credentials:store', async (_, key: string, value: string) => {
  try {
    if (!safeStorage.isEncryptionAvailable()) {
      return { success: false, error: 'Encryption is not available on this system' }
    }
    await fs.promises.mkdir(credentialsDir, { recursive: true })
    const encrypted = safeStorage.encryptString(value)
    await fs.promises.writeFile(safeCredentialPath(key), encrypted)
    return { success: true }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

ipcMain.handle('credentials:read', async (_, key: string) => {
  try {
    if (!safeStorage.isEncryptionAvailable()) {
      return { success: false, error: 'Encryption is not available on this system' }
    }
    const encrypted = await fs.promises.readFile(safeCredentialPath(key))
    const value = safeStorage.decryptString(encrypted)
    return { success: true, value }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})

ipcMain.handle('credentials:delete', async (_, key: string) => {
  try {
    await fs.promises.unlink(safeCredentialPath(key))
    return { success: true }
  } catch (error) {
    return { success: false, error: String(error) }
  }
})
