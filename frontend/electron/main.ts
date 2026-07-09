import { app, BrowserWindow, ipcMain, shell, Menu, webContents, clipboard, nativeImage, safeStorage } from 'electron'
import path from 'path'
import fs from 'fs'
import os from 'os'
import type { IPty } from 'node-pty'

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
  /** Raw output chunks, trimmed to MAX_BUFFER_BYTES — replayed on re-attach */
  buffer: string[]
  bufferBytes: number
  cwd: string
  createdAt: number
  exited: boolean
  exitCode: number | null
}

const MAX_BUFFER_BYTES = 400_000 // ~400KB of scrollback per session
const ptySessions: Map<string, PtySession> = new Map()

function appendToBuffer(session: PtySession, data: string) {
  session.buffer.push(data)
  session.bufferBytes += data.length
  while (session.bufferBytes > MAX_BUFFER_BYTES && session.buffer.length > 1) {
    const removed = session.buffer.shift()
    session.bufferBytes -= removed?.length ?? 0
  }
}

function killAllPtys() {
  ptySessions.forEach((s) => {
    try { s.pty.kill() } catch { /* already dead */ }
  })
  ptySessions.clear()
}

// Determine if we're in development
const isDev = !app.isPackaged

// Credentials directory for secure storage
const credentialsDir = path.join(app.getPath('userData'), 'credentials')

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

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: 'Bash Gym',
    titleBarStyle: isWin ? 'hidden' : 'hiddenInset',
    ...(isWin ? {} : { trafficLightPosition: { x: 16, y: 16 } }),
    backgroundColor: '#0D0D0D',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false, // Required for node-pty
      webviewTag: true // Required for browser panel screenshots
    }
  })

  // Content Security Policy
  const devSources = isDev ? " 'unsafe-inline' 'unsafe-eval' http://localhost:*" : " 'unsafe-inline'"
  const csp = [
    `default-src 'self'`,
    `script-src 'self'${devSources}`,
    `style-src 'self' 'unsafe-inline' https://fonts.googleapis.com`,
    `font-src 'self' https://fonts.gstatic.com`,
    `img-src 'self' data: blob: https:`,
    `connect-src 'self' http://localhost:* ws://localhost:* https://api.vercel.com https://api.neon.tech https://fonts.googleapis.com https://fonts.gstatic.com`,
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

  // Load the app
  if (isDev) {
    const devServerUrl = process.env.BASHGYM_DEV_SERVER_URL || 'http://localhost:5190'
    mainWindow.loadURL(devServerUrl)
    mainWindow.webContents.openDevTools() // TEMP: debug black screen
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  // Inject electron-window class for platform-specific styling (e.g. window border on Windows)
  mainWindow.webContents.on('did-finish-load', () => {
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
    mainWindow = null
    // PTY processes intentionally survive window close — they are cleaned up on app quit.
  })
}

// Only initialize if we got the lock
if (gotTheLock) {
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

  app.on('will-quit', () => {
    killAllPtys()
  })

  // Unclean-exit guards: ensure no orphaned PTY processes
  process.on('exit', () => killAllPtys())
  process.on('SIGTERM', () => { killAllPtys(); app.quit() })
  process.on('SIGINT', () => { killAllPtys(); app.quit() })
}

// IPC Handlers

// Get fresh environment with updated PATH (especially important on Windows)
function getFreshEnv(): Record<string, string> {
  const env = { ...process.env } as Record<string, string>

  // Remove Claude Code's nesting guard so terminals can launch claude freely
  delete env.CLAUDECODE

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

// Terminal management — create-or-attach: a live PTY for this id is re-attached
// (with scrollback replay) instead of respawned, so renderer remounts never
// orphan or kill the underlying process.
ipcMain.handle('terminal:create', async (_, id: string, cwd?: string) => {
  try {
    const existing = ptySessions.get(id)
    if (existing && !existing.exited) {
      return {
        success: true,
        id,
        attached: true,
        buffer: existing.buffer.join(''),
        cwd: existing.cwd
      }
    }
    // A dead session being recreated: drop the stale entry
    if (existing) ptySessions.delete(id)

    // Dynamic import for node-pty (native module)
    const pty = await import('node-pty')

    const shell = process.platform === 'win32'
      ? 'powershell.exe'
      : process.env.SHELL || '/bin/bash'

    // Args to pass to shell (-NoLogo removes PowerShell copyright text)
    const shellArgs = process.platform === 'win32'
      ? ['-NoLogo']
      : []

    const resolvedCwd = resolveTerminalCwd(cwd)

    const ptyProcess = pty.spawn(shell, shellArgs, {
      name: 'xterm-256color',
      cols: 80,
      rows: 24,
      cwd: resolvedCwd,
      env: getFreshEnv()
    })

    const session: PtySession = {
      pty: ptyProcess,
      buffer: [],
      bufferBytes: 0,
      cwd: resolvedCwd,
      createdAt: Date.now(),
      exited: false,
      exitCode: null
    }
    ptySessions.set(id, session)

    // Forward output to renderer and retain it for re-attach replay
    ptyProcess.onData((data: string) => {
      appendToBuffer(session, data)
      mainWindow?.webContents.send(`terminal:data:${id}`, data)
    })

    ptyProcess.onExit(({ exitCode }) => {
      session.exited = true
      session.exitCode = exitCode
      mainWindow?.webContents.send(`terminal:exit:${id}`, exitCode)
      // Entry (with buffer) is kept so a reloaded renderer can see it exited;
      // it is dropped on explicit kill or recreate.
    })

    return { success: true, id, attached: false, cwd: resolvedCwd }
  } catch (error) {
    console.error('Failed to create terminal:', error)
    return { success: false, error: String(error) }
  }
})

ipcMain.handle('terminal:write', (_, id: string, data: string) => {
  const session = ptySessions.get(id)
  if (session && !session.exited) {
    session.pty.write(data)
    return true
  }
  return false
})

ipcMain.handle('terminal:resize', (_, id: string, cols: number, rows: number) => {
  const session = ptySessions.get(id)
  if (session && !session.exited) {
    session.pty.resize(cols, rows)
    return true
  }
  return false
})

ipcMain.handle('terminal:kill', (_, id: string) => {
  const session = ptySessions.get(id)
  if (session) {
    try { session.pty.kill() } catch { /* already dead */ }
    ptySessions.delete(id)
    return true
  }
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
    data: session.buffer.join('').slice(-cap),
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
      return { ok: false, status: response.status, error: text || `HTTP ${response.status}` }
    }
  } catch (error) {
    return { ok: false, error: String(error) }
  }
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
